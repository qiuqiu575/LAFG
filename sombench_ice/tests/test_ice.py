import copy
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_origin
import torch

from ice_settings import SETTINGS
from sombench_ice.core.schema import CHANNELS, GROUP_INDICES, LAYERS, SCENARIOS
from sombench_ice.core.group_dropout import drop_one_group
from sombench_ice.data import IceDataset, prepare, read_metadata, split_lists
from sombench_ice.engine import evaluate, load_ice, loader, train
from sombench_ice.metrics import RegressionMetrics
from sombench_ice.model import IceModel, masked_sse, shared_key


def fixture(root):
    root.mkdir()
    y, x = np.mgrid[:35, :47].astype(np.float32)
    for number, split in enumerate(("train", "val", "test")):
        name = f"patch_0000_000{number}_80N"
        (root / f"{split}_filtered.txt").write_text(name + "\n", encoding="utf-8")
        for layer, bands in (*LAYERS, ("PRO", 1)):
            if layer == "PRO":
                data = ((x + y) / 100.)[None]
            else:
                data = np.stack([(x + y + band + number * 10) for band in range(bands)])
            data[:, 0, 0] = -9999.
            data[:, 0, 1] = 0.  # A valid physical zero, including in the target.
            with rasterio.open(root / f"{name}_{layer}.tif", "w", driver="GTiff", height=35, width=47,
                               count=bands, dtype="float32", nodata=-9999., crs="EPSG:3413",
                               transform=from_origin(number * 20000., 500000., 240., 240.)) as dst:
                dst.write(data)


class IceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="try_ice_test_")
        cls.root = Path(cls.temporary.name)
        cls.raw, cls.cache = cls.root / "raw", cls.root / "prepared"
        fixture(cls.raw)
        prepare(cls.raw, cls.cache, 32)
        cls.meta = read_metadata(cls.cache)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def config(self, name):
        cfg = copy.deepcopy(SETTINGS)
        cfg.update({"cache_dir": str(self.cache), "output_dir": str(self.root / name),
                    "output_root": str(self.root), "patch_size": 32})
        cfg["model"].update({"variant": "debug", "decoder_dim": 8})
        cfg["training"].update({"epochs": 2, "batch_size": 2, "grad_accum": 2, "patience": 0,
                                "warmup_epochs": 0, "lr": 0.001, "group_drop_prob": 0.5})
        cfg["runtime"].update({"device": "cpu", "precision": "fp32", "num_workers": 0,
                               "cpu_threads": 2, "deterministic": True})
        return cfg

    def test_official_split_padding_train_statistics_and_zero(self):
        train_data = IceDataset(self.cache, "train", self.meta)
        self.assertEqual(len(train_data), 4)
        counts = [r["valid_targets"] for r in train_data.rows]
        self.assertEqual(sum(counts), 35 * 47 - 1)
        first = train_data[0]
        self.assertTrue(first["target_valid"][0, 0, 1])
        self.assertFalse(first["target_valid"][0, 0, 0])
        self.assertEqual(first["target"][0, 0, 1], 0.)
        last = train_data[3]
        self.assertFalse(last["target_valid"][:, 3:, :].any())
        with rasterio.open(self.raw / "patch_0000_0000_80N_DICE.tif") as src:
            raw = src.read(1, masked=True)
        self.assertAlmostEqual(self.meta["statistics"]["DICE"]["mean"], float(raw.compressed().astype('float64').mean()))
        # Validation values differ by +10; they must not affect train normalization.
        self.assertEqual(self.meta["statistics"]["DICE"]["count"], 35 * 47 - 1)
        original = (self.raw / "val_filtered.txt").read_text()
        try:
            (self.raw / "val_filtered.txt").write_text("patch_0000_0000_80N\n")
            with self.assertRaisesRegex(ValueError, "Split leakage"):
                split_lists(self.raw)
        finally:
            (self.raw / "val_filtered.txt").write_text(original)

    def test_masks_dropout_no_target_leakage_and_gradients(self):
        cfg = self.config("not_written")
        model = IceModel(cfg["model"]).eval()
        pixels = torch.randn(6, 9, 32, 32)
        valid = torch.ones_like(pixels, dtype=torch.bool)
        valid[:, :2] = False
        changed = pixels.clone()
        changed[:, :2] = float("nan")
        with torch.no_grad():
            torch.testing.assert_close(model(pixels, valid), model(changed, valid), rtol=0, atol=0)
        target = torch.rand(6, 1, 32, 32)
        batch = {"pixels": pixels, "valid": torch.ones_like(valid), "target": target}
        dropped, flags = drop_one_group(batch, 1., torch.Generator().manual_seed(1))
        self.assertTrue((flags.sum(1) == 1).all())
        self.assertIs(dropped["target"], target)
        self.assertTrue(batch["valid"].all())
        self.assertFalse(dropped["valid"].all())
        one = torch.zeros_like(valid)
        one[:, :2] = True
        _, flags = drop_one_group({"pixels": pixels, "valid": one}, 1., torch.Generator().manual_seed(1))
        self.assertFalse(flags.any())
        model.train()
        prediction = model(dropped["pixels"], dropped["valid"])
        sse, count = masked_sse(prediction, target, torch.ones_like(target, dtype=torch.bool))
        (sse / count).backward()
        self.assertEqual(tuple(prediction.shape), (6, 1, 32, 32))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_global_r2_and_constant_targets(self):
        y = torch.tensor([0., 1., 2., 4.]).reshape(1, 1, 2, 2)
        p = torch.tensor([0.5, 1., 2.5, 3.]).reshape_as(y)
        mask = torch.ones_like(y, dtype=torch.bool)
        metrics = RegressionMetrics()
        metrics.update(p[..., :1, :], y[..., :1, :], mask[..., :1, :])
        metrics.update(p[..., 1:, :], y[..., 1:, :], mask[..., 1:, :])
        expected = 1 - float(((y-p)**2).sum() / ((y-y.mean())**2).sum())
        self.assertAlmostEqual(metrics.result()["r2"], expected, places=6)
        constant = RegressionMetrics()
        constant.update(torch.zeros_like(y), torch.ones_like(y), mask)
        self.assertIsNone(constant.result()["r2"])

    def test_persistent_worker_resume_sample_order(self):
        cfg = self.config("unused_loader")
        cfg["runtime"]["num_workers"] = 1
        dataset = IceDataset(self.cache, "train", self.meta)
        continuous = loader(dataset, cfg, shuffle=True)
        list(continuous)  # First epoch starts the persistent worker.
        continuous.sampler.generator.manual_seed(cfg["seed"] + 100003)
        expected = [int(i) for batch in continuous for i in batch["tile_index"]]
        restarted = loader(dataset, cfg, shuffle=True, epoch=1)
        actual = [int(i) for batch in restarted for i in batch["tile_index"]]
        self.assertEqual(expected, actual)

    def test_old_seven_group_transfer_and_isolation(self):
        from lunar_fusion.model import LunarMaskedModel
        from lunar_fusion.schema import N_CHANNELS as old_channels
        cfg = self.config("unused_transfer")
        original = LunarMaskedModel(cfg["model"])
        source = {"model": original.state_dict(), "config": {"model": cfg["model"]}, "kind": "pretrain", "epoch": 4}
        transferred = IceModel(cfg["model"], source)
        original_shared = {shared_key(k): v for k, v in original.encoder.backbone.state_dict().items()}
        for name in transferred.transfer_report["transferred_parameter_names"]:
            key = name.removeprefix("encoder.backbone.")
            torch.testing.assert_close(transferred.state_dict()[name], original_shared[shared_key(key)], rtol=0, atol=0)
        for stage in transferred.encoder.backbone.stages[:cfg["model"]["branch_mora"].get("stages", 1)]:
            for block in stage:
                self.assertEqual(block.bank.S.shape[0], 3)
                for group in range(3):
                    layer = block.base.pwconv1
                    torch.testing.assert_close(layer.effective_weight(group), layer.base.weight, rtol=0, atol=0)
        self.assertEqual(old_channels, 28)
        self.assertEqual(original.encoder.fusion.stems["surface"][0].in_channels, 8)
        self.assertEqual(transferred.encoder.fusion.stems["thermal"][0].in_channels, 4)

    def test_stage2_shared_transfer_and_regression_gradients(self):
        from lunar_fusion.model import LunarMaskedModel
        from sombench_ice.core.branch_mora import CrossGroupGram
        cfg = self.config("stage2_transfer")
        cfg["model"]["branch_mora"]["stages"] = 2
        original = LunarMaskedModel(cfg["model"])
        with torch.no_grad():
            # A distinctive transferred value verifies adapter anchors are not random.
            original.encoder.backbone.stages[1][0].base.pwconv1.base.weight.add_(.2)
        source = {"model": original.state_dict(), "config": {"model": cfg["model"]}, "kind": "pretrain"}
        model = IceModel(cfg["model"], source).train()
        source_weight = original.encoder.backbone.stages[1][0].base.pwconv1.base.weight
        layer = model.encoder.backbone.stages[1][0].base.pwconv1
        torch.testing.assert_close(layer.base.weight, source_weight, rtol=0, atol=0)
        for group in range(3):
            torch.testing.assert_close(layer.effective_weight(group), source_weight, rtol=0, atol=0)
        pixels = torch.randn(2, 9, 32, 32)
        valid = torch.ones_like(pixels, dtype=torch.bool)
        prediction = model(pixels, valid)
        self.assertEqual(tuple(prediction.shape), (2, 1, 32, 32))
        (prediction - .3).square().mean().backward()
        banks = [m for m in model.modules() if isinstance(m, CrossGroupGram)]
        self.assertEqual(len(banks), 2)
        self.assertTrue(all(float(bank.S.grad.norm()) > 0 for bank in banks))

    def test_epoch_resume_and_all_seven_scenarios(self):
        full_cfg = self.config("full")
        full = train(full_cfg)
        resumed_cfg = self.config("resumed")
        partial = train(resumed_cfg, stop_after_epoch=0)
        continued = train(resumed_cfg, resume=partial / "last.pt")
        a, b = load_ice(full / "last.pt"), load_ice(continued / "last.pt")
        for name in a["model"]:
            torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
        result_dir = evaluate(continued / "best.pt", split="test")
        with (result_dir / "comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([r["scenario"] for r in rows], list(SCENARIOS))
        self.assertEqual(len({r["valid_pixels"] for r in rows}), 1)
        self.assertEqual(int(rows[0]["valid_pixels"]), 35 * 47 - 1)
        for row in rows:
            self.assertAlmostEqual(float(row["delta_r2"]), float(rows[0]["r2"]) - float(row["r2"]))


if __name__ == "__main__":
    unittest.main()
