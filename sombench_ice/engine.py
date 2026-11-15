import copy
import csv
from datetime import datetime
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .common import sha256, write_json
from .core.config import model_defaults
from .core.group_dropout import drop_one_group
from .core.schema import GROUPS, GROUP_INDICES, SCENARIOS, SCHEMA_ID
from .data import IceDataset, read_metadata
from .metrics import RegressionMetrics
from .model import IceModel, load_source, masked_sse


def source_hashes():
    root = Path(__file__).resolve().parent
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob("*.py"))}


def validate(cfg):
    t, r, m = cfg["training"], cfg["runtime"], cfg["model"]
    for name in ("epochs", "batch_size", "grad_accum"):
        if type(t[name]) is not int or t[name] < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    if not (0 < t["lr"] and 0 <= t["min_lr"] <= t["lr"] and t["encoder_lr_scale"] > 0):
        raise ValueError("Invalid learning rates")
    if not 0 <= t["group_drop_prob"] <= 1 or t["patience"] < 0:
        raise ValueError("Invalid group-drop probability or patience")
    if t["warmup_epochs"] < 0 or t["clip_grad"] <= 0 or t["weight_decay"] < 0:
        raise ValueError("Invalid warmup/gradient clipping/weight decay")
    if r["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("precision must be fp32, fp16 or bf16")
    if r["num_workers"] < 0 or r["cpu_threads"] < 1 or r["log_every"] < 1:
        raise ValueError("Invalid workers/threads/logging interval")
    if m["fusion"] not in ("gated", "coverage", "mean") or not 0 <= m["drop_path"] < 1:
        raise ValueError("Invalid fusion or drop_path")
    if not math.isfinite(m["branch_mora"]["alpha"]) or m["branch_mora"]["alpha"] <= 0:
        raise ValueError("branch_mora.alpha must be finite and positive")
    branch = model_defaults(m)["branch_mora"]
    if type(branch["stages"]) is not int or branch["stages"] not in (1, 2):
        raise ValueError("branch_mora.stages must be 1 or 2")
    if branch["gram_context"] not in ("static", "observed", "exchange"):
        raise ValueError("branch_mora.gram_context must be static, observed or exchange")


def setup(cfg):
    r = cfg["runtime"]
    torch.set_num_threads(r["cpu_threads"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg["seed"])
    if r["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(r["deterministic"])
    torch.backends.cudnn.benchmark = False
    name = ("cuda" if torch.cuda.is_available() else "cpu") if r["device"] == "auto" else r["device"]
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu, cuda, cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu for a small smoke test")
    precision = r["precision"] if device.type == "cuda" else "fp32"
    if precision == "bf16":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise RuntimeError("This GPU does not support bf16; choose --precision fp32 or fp16")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    return device, dtype, precision


def loader(dataset, cfg, shuffle=False, epoch=0):
    # Worker seeds must not consume the sampler RNG: persistent and newly
    # spawned workers otherwise produce different orders after resuming.
    generator = torch.Generator().manual_seed(cfg["seed"] + 1234567)
    sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(
        cfg["seed"] + 100003 * epoch)) if shuffle else None
    workers = cfg["runtime"]["num_workers"]
    return DataLoader(dataset, batch_size=cfg["training"]["batch_size"], sampler=sampler,
                      num_workers=workers, pin_memory=cfg["runtime"]["device"] != "cpu",
                      generator=generator, drop_last=False,
                      **({"prefetch_factor": 1, "persistent_workers": True} if workers else {}))


def to_device(batch, device):
    return {name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for name, value in batch.items()}


def hide_groups(batch, groups):
    pixels, valid = batch["pixels"].clone(), batch["valid"].clone()
    for group in groups:
        pixels[:, GROUP_INDICES[group]] = 0
        valid[:, GROUP_INDICES[group]] = False
    return {**batch, "pixels": pixels, "valid": valid}


@torch.inference_mode()
def measure(model, data_loader, device, dtype, groups=()):
    model.eval()
    metrics = RegressionMetrics()
    for batch in data_loader:
        batch = hide_groups(to_device(batch, device), groups)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            prediction = model(batch["pixels"], batch["valid"])
        metrics.update(prediction.float(), batch["target"], batch["target_valid"])
    return metrics.result()


def save_checkpoint(path, payload):
    temporary = Path(path).with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_ice(path):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(saved, dict) or saved.get("schema") != SCHEMA_ID or saved.get("kind") != "ice_dense_regression":
        raise ValueError("Use a SomBench ICE checkpoint here; old six-task checkpoints are finetune sources only")
    return saved


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def epoch_lr(epoch, cfg):
    t = cfg["training"]
    warmup = min(t["warmup_epochs"], t["epochs"])
    if epoch < warmup:
        return t["lr"] * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, t["epochs"] - warmup - 1)
    return t["min_lr"] + (t["lr"] - t["min_lr"]) * (1 + math.cos(math.pi * progress)) / 2


def train(cfg, mode="scratch", checkpoint=None, resume=None, stop_after_epoch=None):
    """stop_after_epoch is a validation hook; the full planned schedule stays fixed."""
    cfg = copy.deepcopy(cfg)
    cfg["cache_dir"] = str(Path(cfg["cache_dir"]).resolve())
    saved = load_ice(resume) if resume else None
    if saved:
        # CLI passes the original training settings; only data location/runtime may change.
        for section in ("seed", "model", "training", "patch_size"):
            current, previous = cfg[section], saved["config"][section]
            if section == "model":
                current, previous = model_defaults(current), model_defaults(previous)
            if current != previous:
                raise ValueError(f"Resume would change {section}; start a new run instead")
    source = None
    if mode == "finetune" and not saved:
        if not checkpoint:
            raise ValueError("finetune requires --checkpoint or PRETRAIN_CHECKPOINT in ice_settings.py")
        source = load_source(checkpoint)
        # Inherit actual architecture from weights; no assumption about Pico versus Nano.
        source_model = source["config"]["model"]
        for key in ("variant", "fusion", "branch_mora"):
            if key in source_model:
                cfg["model"][key] = copy.deepcopy(source_model[key])
        if "branch_mora" not in source_model:
            cfg["model"]["branch_mora"]["enabled"] = False
    cfg["model"] = model_defaults(cfg["model"])
    validate(cfg)
    device, dtype, precision = setup(cfg)
    meta = read_metadata(cfg["cache_dir"])
    if cfg["patch_size"] != meta["patch_size"]:
        raise ValueError("Configured patch_size differs from prepared cache")
    if saved and saved["cache_signature"] != meta["signature"]:
        raise ValueError("Resume cache/splits/normalization differ from training")
    model = IceModel(cfg["model"], source, checkpoint).to(device)
    report = saved["transfer_report"] if saved else model.transfer_report
    transferred = set(report["transferred_parameter_names"])
    old_params, new_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            (old_params if name in transferred else new_params).append(param)
    groups = [{"params": new_params, "lr_scale": 1.}]
    if old_params:
        groups.append({"params": old_params, "lr_scale": cfg["training"]["encoder_lr_scale"]})
    optimizer = torch.optim.AdamW(groups, lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    train_data = IceDataset(cfg["cache_dir"], "train", meta)
    val_loader = loader(IceDataset(cfg["cache_dir"], "val", meta), cfg)
    best, stale, start = float("inf"), 0, 0
    if saved:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        best, stale, start = saved["best_rmse"], saved["no_improve"], saved["epoch"] + 1
        output = Path(resume).resolve().parent
        # Prevent appending older epochs to an already newer training run.
        if (output / "last.pt").exists():
            latest = load_ice(output / "last.pt")
            if latest["epoch"] != saved["epoch"]:
                raise ValueError("Resume from last.pt; this checkpoint is older than the run's latest epoch")
        restore_rng(saved["rng"])
    else:
        output = Path(cfg.get("output_dir") or (Path(cfg["output_root"]) /
                      (datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + mode)))
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "config.json", cfg)
        write_json(output / "transfer_report.json", report)
        write_json(output / "data_protocol.json", {
            "cache_signature": meta["signature"], "channels": meta["channels"], "groups": meta["groups"],
            "statistics": meta["statistics"], "splits": meta["splits"],
            "metric_mask": meta["metric_mask"], "tiling": meta["tiling"],
            "target": "model-derived PRO prospectivity; not measured ice concentration"})
    provenance = {"torch": torch.__version__, "numpy": np.__version__, "device": str(device),
                  "effective_precision": precision, "source_sha256": source_hashes()}
    write_json(output / ("resume_runtime.json" if saved else "runtime.json"), provenance)
    branch = cfg["model"]["branch_mora"]
    print(f"ICE {report['mode']} | variant={cfg['model']['variant']} | device={device} | precision={precision}\n"
          f"branch_stages={branch['stages'] if branch['enabled'] else 0} gram={branch['enabled'] and branch['shared']} "
          f"gram_context={branch['gram_context']}\n"
          f"train_tiles={len(train_data)} val_tiles={len(val_loader.dataset)}\n"
          f"Best checkpoint: {output / 'best.pt'}", flush=True)
    if start >= cfg["training"]["epochs"] or (cfg["training"]["patience"] and stale >= cfg["training"]["patience"]):
        print("This run has already finished its schedule or reached early stopping.", flush=True)
        return output
    train_loader = loader(train_data, cfg, shuffle=True)
    for epoch in range(start, cfg["training"]["epochs"]):
        epoch_start = time.monotonic()
        # Dataset has no random transforms. Reuse workers; epoch-keyed shuffle
        # and dropout streams make epoch-boundary resume reproducible.
        train_loader.sampler.generator.manual_seed(cfg["seed"] + 100003 * epoch)
        drop_generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 7919 * (epoch + 1))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        metrics = RegressionMetrics()
        counts = torch.zeros(len(GROUPS), device=device, dtype=torch.int64)
        lr = epoch_lr(epoch, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr * group["lr_scale"]
        window_pixels = 0
        nominal_pixels = cfg["training"]["batch_size"] * cfg["patch_size"] ** 2 * cfg["training"]["grad_accum"]
        for step, batch in enumerate(train_loader, 1):
            batch, flags = drop_one_group(to_device(batch, device), cfg["training"]["group_drop_prob"], drop_generator)
            counts += flags.sum(0)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
                prediction = model(batch["pixels"], batch["valid"])
            sse, n = masked_sse(prediction, batch["target"], batch["target_valid"])
            scaler.scale(sse / nominal_pixels).backward()
            window_pixels += n
            metrics.update(prediction.detach().float(), batch["target"], batch["target_valid"])
            if step % cfg["training"]["grad_accum"] == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                # True pixel-weighted MSE, also for short accumulation windows/padded tiles.
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.mul_(nominal_pixels / window_pixels)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["clip_grad"], error_if_nonfinite=precision != "fp16")
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                window_pixels = 0
            if step == 1 or step % cfg["runtime"]["log_every"] == 0 or step == len(train_loader):
                print(f"epoch {epoch+1}/{cfg['training']['epochs']} step {step}/{len(train_loader)} "
                      f"train_rmse={metrics.result()['rmse']:.6f} elapsed_s={time.monotonic()-epoch_start:.1f}", flush=True)
        training_metrics = metrics.result()
        validation_metrics = measure(model, val_loader, device, dtype)
        improved = validation_metrics["rmse"] < best
        best, stale = (validation_metrics["rmse"], 0) if improved else (best, stale + 1)
        payload = {"schema": SCHEMA_ID, "kind": "ice_dense_regression", "config": cfg,
                   "cache_signature": meta["signature"], "transfer_report": report, "epoch": epoch,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                   "best_rmse": best, "no_improve": stale, "rng": rng_state(),
                   "train_metrics": training_metrics, "val_metrics": validation_metrics,
                   "runtime": provenance}
        if improved:
            save_checkpoint(output / "best.pt", payload)
        save_checkpoint(output / "last.pt", payload)
        row = {"epoch": epoch + 1, "lr": lr, "train_rmse": training_metrics["rmse"],
               "val_rmse": validation_metrics["rmse"], "val_mae": validation_metrics["mae"],
               "val_r2": validation_metrics["r2"], "best_val_rmse": best,
               "seconds": time.monotonic() - epoch_start,
               **{f"dropped_{g}": int(n) for g, n in zip(GROUPS, counts.cpu().tolist())}}
        metrics_file = output / "metrics.csv"
        exists = metrics_file.exists()
        with metrics_file.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"epoch {epoch+1}: val_rmse={validation_metrics['rmse']:.6f} "
              f"val_r2={validation_metrics['r2']} best={best:.6f}", flush=True)
        if (stop_after_epoch is not None and epoch >= stop_after_epoch) or (
                cfg["training"]["patience"] and stale >= cfg["training"]["patience"]):
            break
    return output


def evaluate(checkpoint, cache_dir=None, split="test", output_dir=None, runtime=None, batch_size=None):
    saved = load_ice(checkpoint)
    cfg = copy.deepcopy(saved["config"])
    if runtime:
        cfg["runtime"].update(runtime)
    if cache_dir:
        cfg["cache_dir"] = str(cache_dir)
    if batch_size is not None:
        cfg["training"]["batch_size"] = batch_size
    validate(cfg)
    device, dtype, precision = setup(cfg)
    meta = read_metadata(cfg["cache_dir"])
    if meta["signature"] != saved["cache_signature"]:
        raise ValueError("Evaluation cache/splits/normalization differ from training")
    dataset = IceDataset(cfg["cache_dir"], split, meta)
    data_loader = loader(dataset, cfg)
    model = IceModel(cfg["model"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    output = Path(output_dir) if output_dir else Path(checkpoint).resolve().parent / "evaluations" / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + split)
    output.mkdir(parents=True, exist_ok=False)
    rows, details, original = [], {}, None
    for scenario, dropped in SCENARIOS.items():
        result = measure(model, data_loader, device, dtype, dropped)
        if original is None:
            original = result
        if result["valid_pixels"] != original["valid_pixels"] or result["target_sst"] != original["target_sst"]:
            raise AssertionError("Evaluation target population changed across missing-group scenarios")
        row = {"scenario": scenario, "dropped_groups": "+".join(dropped),
               "rmse": result["rmse"], "mae": result["mae"], "r2": result["r2"],
               "delta_rmse": result["rmse"] - original["rmse"],
               "delta_mae": result["mae"] - original["mae"],
               "delta_r2": original["r2"] - result["r2"] if original["r2"] is not None else None,
               "valid_pixels": result["valid_pixels"], "tiles": len(dataset), "split": split}
        rows.append(row)
        details[scenario] = result
        print(f"{scenario}: RMSE={row['rmse']:.6f} MAE={row['mae']:.6f} R2={row['r2']}", flush=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "results.json", {
        "schema": SCHEMA_ID, "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256(checkpoint), "epoch": saved["epoch"] + 1,
        "selection": "original_validation_RMSE", "cache_signature": meta["signature"],
        "split": split, "config": cfg, "effective_precision": precision,
        "source_sha256": source_hashes(), "results": details,
        "delta_convention": "positive means worse: RMSE/MAE = scenario-original; R2 = original-scenario"})
    print(f"Evaluation saved: {output / 'comparison.csv'}", flush=True)
    return output
