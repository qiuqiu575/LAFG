import copy
from pathlib import Path
import tempfile
import unittest

import torch

from lunar_fusion.branch_mora import BranchedBlock, CrossGroupGram, GramExchange
from lunar_fusion.checkpoints import initialize_encoder, load_checkpoint
from lunar_fusion.cli import resolve_training, training_parser
from lunar_fusion.config import DEFAULTS, load_config, model_defaults
from lunar_fusion.model import LunarEncoder, LunarMaskedModel, LunarTaskModel
from lunar_fusion.schema import GROUPS, GROUP_INDICES, TASK_OUTPUTS


def model_config(stages=2, shared=True, enabled=True, checkpointing=False, gram_context="static"):
    cfg = copy.deepcopy(DEFAULTS["model"])
    cfg.update(variant="debug", decoder_dim=8, gradient_checkpointing=checkpointing)
    cfg["branch_mora"].update(enabled=enabled, shared=shared, stages=stages, gram_context=gram_context)
    return cfg


def perturb_gram(model, scale=.5):
    """Move every Gram factor away from its anchor so the shared update is nonzero."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, CrossGroupGram):
                module.S.add_(scale * torch.randn_like(module.S))


class BranchStagesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_legacy_config_and_cli_architecture_inheritance(self):
        legacy = model_config(1)
        del legacy["branch_mora"]["stages"]
        self.assertEqual(model_defaults(legacy)["branch_mora"]["stages"], 1)
        parser = training_parser("pretrain")
        cfg = resolve_training("pretrain", parser.parse_args([]))
        self.assertEqual(cfg["model"]["branch_mora"]["stages"], 2)
        cfg = resolve_training("pretrain", parser.parse_args(["--branch-stages", "1"]))
        self.assertEqual(cfg["model"]["branch_mora"]["stages"], 1)
        for invalid in (0, 3, True, 1.5):
            cfg = copy.deepcopy(DEFAULTS)
            cfg["model"] = model_config(invalid)
            with self.assertRaisesRegex(ValueError, "stages"):
                load_config(base=cfg)
        with tempfile.TemporaryDirectory() as directory:
            for stage_count in (1, 2):
                pretrain_cfg = copy.deepcopy(DEFAULTS)
                pretrain_cfg["model"] = model_config(stage_count)
                if stage_count == 1:
                    del pretrain_cfg["model"]["branch_mora"]["stages"]
                source = LunarMaskedModel(pretrain_cfg["model"])
                path = Path(directory) / f"pretrain_s{stage_count}.pt"
                torch.save({"format_version": 1, "kind": "pretrain", "config": pretrain_cfg,
                            "model": source.state_dict()}, path)
                downstream_parser = training_parser("downstream")
                args = downstream_parser.parse_args(["--checkpoint", str(path), "--task", "mare"])
                resolved = resolve_training("downstream", args)
                self.assertEqual(resolved["model"]["branch_mora"]["stages"], stage_count)
                target = LunarTaskModel(resolved["model"], "mare", "finetune")
                initialize_encoder(target.encoder, load_checkpoint(path))
                for key, value in source.encoder.state_dict().items():
                    torch.testing.assert_close(value, target.encoder.state_dict()[key], rtol=0, atol=0)
                mismatch = model_config(3 - stage_count)
                with self.assertRaisesRegex(ValueError, "architecture"):
                    initialize_encoder(LunarMaskedModel(mismatch).encoder, load_checkpoint(path))

    def test_stage2_is_per_group_and_pyramid_keeps_four_resolutions(self):
        cfg = model_config()
        encoder = LunarMaskedModel(cfg).encoder.eval()
        calls = {"stage1": 0, "stage2": 0, "stage3": 0, "stage4": 0, "downsample2": 0}
        def counter(name):
            def hook(module, inputs, output):
                calls[name] += 1
            return hook
        handles = [encoder.backbone.stages[i][0].register_forward_hook(counter(f"stage{i+1}")) for i in range(4)]
        handles.append(encoder.backbone.downsample_layers[1].register_forward_hook(counter("downsample2")))
        pixels = torch.randn(2, 28, 64, 64)
        valid = torch.ones_like(pixels, dtype=torch.bool)
        valid[:, GROUP_INDICES["thermal"]] = False
        valid[:, :, :8, :8] = False
        valid[1] = False
        with torch.no_grad():
            result = encoder(pixels, valid)
        for handle in handles:
            handle.remove()
        self.assertEqual(calls, {"stage1": 7, "stage2": 7, "stage3": 1, "stage4": 1, "downsample2": 7})
        self.assertEqual([tuple(p.shape) for p in result["pyramid"]],
                         [(2, 8, 16, 16), (2, 16, 8, 8), (2, 32, 4, 4), (2, 64, 2, 2)])
        weights, coverage = result["fusion_weights"], result["coverage"]
        self.assertEqual(tuple(weights.shape), (2, 7, 8, 8))
        self.assertTrue(torch.equal(weights[coverage == 0], torch.zeros_like(weights[coverage == 0])))
        torch.testing.assert_close(weights.sum(1), (coverage > 0).any(1).float())
        for value in result["pyramid"]:
            self.assertTrue(torch.isfinite(value).all())

    def test_unobserved_values_cannot_change_prediction(self):
        pixels = torch.randn(2, 28, 64, 64)
        valid = torch.rand_like(pixels) > .4
        valid[:, GROUP_INDICES["thermal"]] = False
        valid[:, :, :16, :16] = False
        valid[1] = False
        changed = pixels.clone()
        changed[~valid] = float("nan")
        for mode in ("gated", "coverage", "mean"):
            cfg = model_config()
            cfg["fusion"] = mode
            model = LunarMaskedModel(cfg).eval()
            with torch.no_grad():
                reference, result = model(pixels, valid), model(changed, valid)
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(reference, result, rtol=0, atol=0)

    def test_both_gram_stages_learn_and_private_ablation_has_same_layout(self):
        pixels = torch.randn(2, 28, 64, 64)
        valid = torch.ones_like(pixels, dtype=torch.bool)
        for shared in (True, False):
            model = LunarMaskedModel(model_config(shared=shared)).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            loss = (model(pixels, valid) - pixels).square().mean()
            loss.backward()
            before = {}
            for stage in (0, 1):
                block = model.encoder.backbone.stages[stage][0]
                self.assertIsInstance(block, BranchedBlock)
                self.assertGreater(float(block.base.pwconv1.B.grad.norm()), 0)
                self.assertEqual(block.bank.S.requires_grad, shared)
                if shared:
                    self.assertGreater(float(block.bank.S.grad.norm()), 0)
                    before[stage] = block.bank.S.detach().clone()
                else:
                    self.assertIsNone(block.bank.S.grad)
            optimizer.step()
            for stage, value in before.items():
                self.assertFalse(torch.equal(value, model.encoder.backbone.stages[stage][0].bank.S))
        original = LunarMaskedModel(model_config(enabled=False))
        self.assertFalse(any(isinstance(m, CrossGroupGram) for m in original.modules()))
        self.assertEqual(tuple(original(pixels, valid).shape), tuple(pixels.shape))

    def test_stage2_receives_unfused_group_features(self):
        encoder = LunarMaskedModel(model_config()).encoder.eval()
        pixels = torch.randn(1, 28, 32, 32)
        valid = torch.ones_like(pixels, dtype=torch.bool)
        inputs = []
        def collect(module, args):
            inputs.append((args[1], args[0].detach().clone()))
        handle = encoder.backbone.stages[1][0].register_forward_pre_hook(collect)
        with torch.no_grad():
            encoder(pixels, valid)
            altered = pixels.clone()
            altered[:, GROUP_INDICES["thermal"]] += 3
            encoder(altered, valid)
        handle.remove()
        self.assertEqual(len(inputs), 2 * len(GROUPS))
        for first, second in zip(inputs[:len(GROUPS)], inputs[len(GROUPS):]):
            group, tensor = first
            self.assertEqual(group, second[0])
            if GROUPS[group] == "thermal":
                self.assertFalse(torch.equal(tensor, second[1]))
            else:
                torch.testing.assert_close(tensor, second[1], rtol=0, atol=0)

    def test_six_task_heads_finetune_and_frozen_encoder(self):
        pixels = torch.randn(2, 28, 32, 32)
        valid = torch.ones_like(pixels, dtype=torch.bool)
        cfg = model_config()
        source = LunarMaskedModel(cfg)
        payload = {"config": {"model": cfg}, "model": source.state_dict()}
        for task, outputs in TASK_OUTPUTS.items():
            model = LunarTaskModel(cfg, task, mode="finetune")
            initialize_encoder(model.encoder, payload)
            prediction = model(pixels, valid)
            expected = (2, 2, 32, 32) if task in ("mare", "craters") else (2, outputs)
            self.assertEqual(tuple(prediction.shape), expected)
            prediction.square().mean().backward()
            bank = model.encoder.backbone.stages[1][0].bank
            self.assertGreater(float(bank.S.grad.norm()), 0)
        frozen = LunarTaskModel(cfg, "geology", mode="linear").train()
        frozen(pixels, valid).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in frozen.encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in frozen.head.parameters()))

    def test_gradient_checkpointing_preserves_outputs_and_gradients(self):
        normal = LunarMaskedModel(model_config()).train()
        checked = LunarMaskedModel(model_config(checkpointing=True)).train()
        checked.load_state_dict(normal.state_dict(), strict=True)
        pixels = torch.randn(2, 28, 32, 32)
        valid = torch.rand_like(pixels) > .2
        a, b = normal(pixels, valid), checked(pixels, valid)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward()
        b.square().mean().backward()
        for (name, left), (_, right) in zip(normal.named_parameters(), checked.named_parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad, name)
            else:
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-8, msg=name)


class ObservedGramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(7)

    def test_static_effective_weight_matches_legacy_formula(self):
        model = LunarMaskedModel(model_config())
        perturb_gram(model)
        layer = model.encoder.backbone.stages[1][0].base.pwconv1
        with torch.no_grad():
            layer.B.normal_()
            layer.private_magnitude.mul_(1.3)
            layer.shared_magnitude.mul_(.7)

        def direction(weight):
            return weight / weight.norm(dim=0, keepdim=True).clamp_min(1e-5)

        def update(weight, delta, magnitude, initial):
            base = direction(weight)
            return magnitude * (direction(weight + delta) - base) + (magnitude - initial) * base

        with torch.no_grad():
            for group in range(len(GROUPS)):
                weight = layer.base.weight
                private = layer.alpha * (layer.B[group] @ layer.A[group])
                shared = layer.alpha * (weight @ layer.bank(group))
                expected = (weight + update(weight, private, layer.private_magnitude[group], layer.initial_magnitude)
                            + update(weight, shared, layer.shared_magnitude[group], layer.initial_magnitude))
                torch.testing.assert_close(layer.effective_weight(group), expected, rtol=0, atol=0)

    def test_observed_mode_keeps_state_layout_and_is_zero_update_at_init(self):
        for stages in (1, 2):
            static = LunarMaskedModel(model_config(stages)).eval()
            observed = LunarMaskedModel(model_config(stages, gram_context="observed")).eval()
            self.assertEqual(static.state_dict().keys(), observed.state_dict().keys())
            observed.load_state_dict(static.state_dict(), strict=True)
            pixels = torch.randn(2, 28, 64, 64)
            valid = torch.rand_like(pixels) > .3
            valid[0, GROUP_INDICES["radar"]] = False
            with torch.no_grad():
                torch.testing.assert_close(observed(pixels, valid), static(pixels, valid),
                                           rtol=1e-5, atol=1e-5)

    def test_observed_unobserved_values_cannot_change_prediction(self):
        pixels = torch.randn(2, 28, 64, 64)
        valid = torch.rand_like(pixels) > .4
        valid[:, GROUP_INDICES["thermal"]] = False
        valid[:, :, :16, :16] = False
        valid[1] = False
        changed = pixels.clone()
        changed[~valid] = float("nan")
        for stages in (1, 2):
            model = LunarMaskedModel(model_config(stages, gram_context="observed")).eval()
            perturb_gram(model)
            with torch.no_grad():
                reference, result = model(pixels, valid), model(changed, valid)
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(reference, result, rtol=0, atol=0)

    def _stage2_inputs(self, encoder, pixels, valid):
        inputs = []
        handle = encoder.backbone.stages[1][0].register_forward_pre_hook(
            lambda module, args: inputs.append((args[1], args[0].detach().clone())))
        with torch.no_grad():
            encoder(pixels, valid)
        handle.remove()
        return inputs

    def test_observed_gram_conditions_streams_on_other_groups(self):
        pixels = torch.randn(2, 28, 32, 32)
        valid = torch.ones_like(pixels, dtype=torch.bool)
        altered = pixels.clone()
        altered[:, GROUP_INDICES["thermal"]] += 3
        dropped = valid.clone()
        dropped[:, GROUP_INDICES["thermal"]] = False
        thermal = GROUPS.index("thermal")
        for context in ("static", "observed"):
            encoder = LunarMaskedModel(model_config(gram_context=context)).encoder.eval()
            perturb_gram(encoder)
            reference = self._stage2_inputs(encoder, pixels, valid)
            for variant in (self._stage2_inputs(encoder, altered, valid),
                            self._stage2_inputs(encoder, pixels, dropped)):
                self.assertEqual([g for g, _ in reference], [g for g, _ in variant])
                for (group, first), (_, second) in zip(reference, variant):
                    if group == thermal:
                        continue
                    if context == "static":
                        torch.testing.assert_close(first, second, rtol=0, atol=0)
                    else:
                        self.assertFalse(torch.allclose(first, second), GROUPS[group])

    def test_observed_gram_without_other_groups_is_private_only(self):
        pixels = torch.randn(1, 28, 32, 32)
        valid = torch.zeros_like(pixels, dtype=torch.bool)
        valid[:, GROUP_INDICES["surface"]] = True
        static = LunarMaskedModel(model_config(shared=False)).eval()
        observed = LunarMaskedModel(model_config(gram_context="observed")).eval()
        observed.load_state_dict(static.state_dict(), strict=True)
        perturb_gram(observed)
        with torch.no_grad():
            torch.testing.assert_close(observed(pixels, valid), static(pixels, valid), rtol=1e-5, atol=1e-5)

    def test_observed_gram_learns_and_reports_diagnostics(self):
        from lunar_fusion.branch_mora import diagnostics
        model = LunarMaskedModel(model_config(gram_context="observed")).train()
        pixels = torch.randn(2, 28, 32, 32)
        valid = torch.rand_like(pixels) > .2
        (model(pixels, valid) - pixels).square().mean().backward()
        for stage in (0, 1):
            block = model.encoder.backbone.stages[stage][0]
            self.assertGreater(float(block.bank.S.grad.norm()), 0)
            self.assertGreater(float(block.base.pwconv1.B.grad.norm()), 0)
        report = diagnostics(model)
        layers = [value for value in report.values() if "shared_rel" in value]
        self.assertTrue(layers)
        self.assertTrue(all(value["shared_rel"] == 0 for value in layers))
        perturb_gram(model)
        self.assertTrue(all(value["shared_rel"] > 0 for value in diagnostics(model).values()
                            if "shared_rel" in value))

    def test_observed_gradient_checkpointing_preserves_outputs_and_gradients(self):
        normal = LunarMaskedModel(model_config(gram_context="observed")).train()
        perturb_gram(normal)
        checked = LunarMaskedModel(model_config(checkpointing=True, gram_context="observed")).train()
        checked.load_state_dict(normal.state_dict(), strict=True)
        pixels = torch.randn(2, 28, 32, 32)
        valid = torch.rand_like(pixels) > .2
        a, b = normal(pixels, valid), checked(pixels, valid)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward()
        b.square().mean().backward()
        for (name, left), (_, right) in zip(normal.named_parameters(), checked.named_parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad, name)
            else:
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7, msg=name)

    def test_legacy_checkpoint_is_static_and_mode_mismatch_is_rejected(self):
        legacy = model_config()
        del legacy["branch_mora"]["gram_context"]
        self.assertEqual(model_defaults(legacy)["branch_mora"]["gram_context"], "static")
        source = LunarMaskedModel(legacy)
        payload = {"config": {"model": legacy}, "model": source.state_dict()}
        initialize_encoder(LunarTaskModel(model_config(), "mare", "finetune").encoder, payload)
        with self.assertRaisesRegex(ValueError, "architecture"):
            initialize_encoder(LunarTaskModel(model_config(gram_context="observed"), "mare").encoder, payload)
        cfg = copy.deepcopy(DEFAULTS)
        cfg["model"] = model_config(gram_context="dynamic")
        with self.assertRaisesRegex(ValueError, "gram_context"):
            load_config(base=cfg)
        parser = training_parser("pretrain")
        resolved = resolve_training("pretrain", parser.parse_args([]))
        self.assertEqual(resolved["model"]["branch_mora"]["gram_context"], "exchange")
        resolved = resolve_training("pretrain", parser.parse_args(["--gram-context", "static"]))
        self.assertEqual(resolved["model"]["branch_mora"]["gram_context"], "static")

    def test_observed_optimizer_exempts_anchored_parameters_from_decay(self):
        from lunar_fusion.engine import _optimizer
        for context in ("static", "observed"):
            cfg = copy.deepcopy(DEFAULTS)
            cfg["model"] = model_config(gram_context=context)
            model = LunarMaskedModel(cfg["model"])
            optimizer = _optimizer(model, cfg, "pretrain")
            exempt = {id(p) for group in optimizer.param_groups if group["weight_decay"] == 0
                      for p in group["params"]}
            total = sum(len(group["params"]) for group in optimizer.param_groups)
            self.assertEqual(total, len([p for p in model.parameters() if p.requires_grad]))
            bank = model.encoder.backbone.stages[0][0].bank
            layer = model.encoder.backbone.stages[0][0].base.pwconv1
            if context == "static":
                self.assertEqual(len(optimizer.param_groups), 1)
                self.assertFalse(exempt)
            else:
                self.assertIn(id(bank.S), exempt)
                self.assertIn(id(layer.shared_magnitude), exempt)
                self.assertNotIn(id(layer.A), exempt)
                self.assertNotIn(id(layer.base.weight), exempt)

    def test_finetune_group_dropout_allowed_for_every_task(self):
        cfg = copy.deepcopy(DEFAULTS)
        cfg["downstream"].update(task="mare", mode="finetune", train_group_drop_prob=.3, consistency_weight=1.)
        with self.assertRaisesRegex(ValueError, "consistency_weight"):
            load_config(base=cfg)
        for task in TASK_OUTPUTS:
            cfg = copy.deepcopy(DEFAULTS)
            cfg["downstream"].update(task=task, mode="finetune", train_group_drop_prob=.3)
            load_config(base=cfg)
        cfg["downstream"]["mode"] = "linear"
        with self.assertRaisesRegex(ValueError, "finetune"):
            load_config(base=cfg)


def perturb_exchange(model, scale=.2):
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GramExchange):
                module.output.add_(scale * torch.randn_like(module.output))


class GramExchangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(11)

    def test_exchange_closes_each_branched_stage_and_private_has_none(self):
        model = LunarMaskedModel(model_config(gram_context="exchange"))
        for stage in (0, 1):
            blocks = list(model.encoder.backbone.stages[stage])
            self.assertIsInstance(blocks[-1].exchange, GramExchange)
            self.assertTrue(all(not hasattr(block, "exchange") for block in blocks[:-1]))
        private = LunarMaskedModel(model_config(shared=False, gram_context="exchange"))
        self.assertFalse(any(isinstance(m, GramExchange) for m in private.modules()))

    def test_exchange_starts_as_independent_streams(self):
        static = LunarMaskedModel(model_config()).eval()
        exchange = LunarMaskedModel(model_config(gram_context="exchange")).eval()
        exchange.load_state_dict(static.state_dict(), strict=False)
        pixels = torch.randn(2, 28, 64, 64)
        valid = torch.rand_like(pixels) > .3
        with torch.no_grad():
            torch.testing.assert_close(exchange(pixels, valid), static(pixels, valid), rtol=1e-5, atol=1e-5)

    def test_exchange_never_reads_unobserved_values(self):
        pixels = torch.randn(2, 28, 64, 64)
        valid = torch.rand_like(pixels) > .4
        valid[:, GROUP_INDICES["thermal"]] = False
        valid[:, :, :16, :16] = False
        valid[1] = False
        changed = pixels.clone()
        changed[~valid] = float("nan")
        for stages in (1, 2):
            model = LunarMaskedModel(model_config(stages, gram_context="exchange")).eval()
            perturb_exchange(model)
            with torch.no_grad():
                reference, result = model(pixels, valid), model(changed, valid)
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(reference, result, rtol=0, atol=0)

    def test_exchange_messages_only_come_from_observed_groups(self):
        width = 16
        bank = CrossGroupGram(width, 4, True, "exchange")
        exchange = GramExchange(bank, width)
        with torch.no_grad():
            exchange.output.normal_()
        features = [torch.randn(1, width, 4, 4) for _ in GROUPS]
        supports = [torch.ones(1, 1, 4, 4, dtype=torch.bool) for _ in GROUPS]
        for group in range(1, len(GROUPS)):
            supports[group] = torch.zeros_like(supports[group])
        lonely = exchange(features, supports)
        torch.testing.assert_close(lonely[0], features[0], rtol=0, atol=0)
        self.assertTrue(all(torch.equal(out, torch.zeros_like(out)) for out in lonely[1:]))
        supports[1] = torch.ones_like(supports[1])
        before = exchange(features, supports)[0]
        features[2] = features[2] + 5
        torch.testing.assert_close(exchange(features, supports)[0], before, rtol=0, atol=0)
        features[1] = features[1] + 5
        self.assertFalse(torch.allclose(exchange(features, supports)[0], before))

    def test_exchange_learns_and_checkpointing_matches(self):
        normal = LunarMaskedModel(model_config(gram_context="exchange")).train()
        perturb_exchange(normal)
        checked = LunarMaskedModel(model_config(checkpointing=True, gram_context="exchange")).train()
        checked.load_state_dict(normal.state_dict(), strict=True)
        pixels = torch.randn(2, 28, 32, 32)
        valid = torch.rand_like(pixels) > .2
        a, b = normal(pixels, valid), checked(pixels, valid)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().mean().backward()
        b.square().mean().backward()
        for stage in (0, 1):
            block = normal.encoder.backbone.stages[stage][-1]
            self.assertGreater(float(block.bank.S.grad.norm()), 0)
            self.assertGreater(float(block.exchange.output.grad.norm()), 0)
        for (name, left), (_, right) in zip(normal.named_parameters(), checked.named_parameters()):
            if left.grad is None:
                self.assertIsNone(right.grad, name)
            else:
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-5, atol=1e-7, msg=name)

    def test_paired_views_match_separate_forwards_and_train(self):
        from lunar_fusion.engine import forward_loss
        from lunar_fusion.losses import feature_consistency
        cfg = copy.deepcopy(DEFAULTS)
        cfg["model"] = model_config(gram_context="exchange")
        model = LunarTaskModel(cfg["model"], "geology", "finetune").eval()
        pixels = torch.randn(3, 28, 32, 32)
        valid = torch.ones_like(pixels, dtype=torch.bool)
        valid[2, 4:] = False  # one group only: never dropped
        with torch.no_grad():
            joint, encoded = model(torch.cat([pixels, pixels]), torch.cat([valid, valid]), return_features=True)
            alone = model(pixels, valid)
        torch.testing.assert_close(joint[:3], alone, rtol=1e-5, atol=1e-5)
        self.assertLess(float(feature_consistency(encoded["pyramid"])), 1e-8)
        batch = {"pixels": pixels, "valid": valid, "target": torch.tensor([1, 2, 3]),
                 "target_valid": torch.ones(3, dtype=torch.bool)}
        for kind, section in (("downstream", "downstream"), ("pretrain", "masking")):
            cfg[section]["consistency_weight"] = 1.
            cfg["downstream"]["task"] = "geology"
            net = (LunarTaskModel(cfg["model"], "geology", "finetune") if kind == "downstream"
                   else LunarMaskedModel(cfg["model"])).train()
            perturb_exchange(net)
            loss, prediction, _ = forward_loss(net, batch, cfg, kind, pair_generator=torch.Generator().manual_seed(0))
            self.assertEqual(prediction.shape[0], 3)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(float(net.encoder.backbone.stages[1][-1].exchange.output.grad.norm()), 0)

    def test_grouped_path_matches_per_group_path(self):
        from unittest import mock
        pixels = torch.randn(2, 28, 64, 64)
        valid = torch.rand_like(pixels) > .3
        valid[0, GROUP_INDICES["radar"]] = False
        valid[:, :, :16, :16] = False
        for stages in (1, 2):
            for shared in (True, False):
                for checkpointing in (False, True):
                    cfg = model_config(stages, shared=shared, checkpointing=checkpointing,
                                       gram_context="exchange")
                    grouped = LunarMaskedModel(cfg).train()
                    perturb_exchange(grouped)
                    with torch.no_grad():
                        for module in grouped.modules():
                            if hasattr(module, "B") and hasattr(module, "private_magnitude"):
                                module.B.normal_()
                                module.shared_magnitude.mul_(1.1)
                    loop = LunarMaskedModel(cfg).train()
                    loop.load_state_dict(grouped.state_dict(), strict=True)
                    with mock.patch.object(BranchedBlock, "grouped", new=property(lambda self: False)):
                        reference = loop(pixels, valid)
                    result = grouped(pixels, valid)
                    torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-5)
                    result.square().mean().backward()
                    with mock.patch.object(BranchedBlock, "grouped", new=property(lambda self: False)):
                        reference.square().mean().backward()
                    for (name, left), (_, right) in zip(grouped.named_parameters(), loop.named_parameters()):
                        if left.grad is None:
                            self.assertIsNone(right.grad, name)
                        else:
                            torch.testing.assert_close(left.grad, right.grad, rtol=1e-3, atol=1e-6, msg=name)

    def test_legacy_configs_backfill_zero_consistency(self):
        from lunar_fusion.checkpoints import check_resume_compatibility
        legacy = load_config(base=copy.deepcopy(DEFAULTS))
        del legacy["masking"]["consistency_weight"]
        del legacy["downstream"]["consistency_weight"]
        current = load_config(base=copy.deepcopy(legacy))
        self.assertEqual(current["masking"]["consistency_weight"], 0.0)
        self.assertEqual(current["downstream"]["consistency_weight"], 0.0)
        check_resume_compatibility(legacy, current)


if __name__ == "__main__":
    unittest.main()
