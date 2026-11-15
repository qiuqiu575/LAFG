from contextlib import nullcontext
from datetime import datetime
import itertools
import math
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from .backbone import load_official_weights
from .checkpoints import (check_resume_compatibility, initialize_encoder, load_checkpoint,
                          restore_rng, save_checkpoint)
from .data import BenchmarkDataset, RandomCropDataset
from .group_dropout import drop_one_group
from .losses import feature_consistency, reconstruction_loss, supervised_loss
from .masking import make_hidden_mask
from .metrics import Metrics
from .model import LunarMaskedModel, LunarTaskModel, make_task_head
from .runlog import RunLogger, write_json
from .schema import GROUPS


def setup(cfg):
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(cfg["runtime"]["deterministic"], warn_only=False)
    torch.backends.cudnn.benchmark = not cfg["runtime"]["deterministic"]
    requested = cfg["runtime"]["device"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type not in ("cuda", "cpu"):
        raise ValueError("This implementation supports a single CUDA GPU or CPU")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        device = torch.device("cuda", device.index if device.index is not None else 0)
        torch.cuda.set_device(device)
    return device


class Precision:
    def __init__(self, device, requested):
        self.device = device
        self.enabled = device.type == "cuda" and requested != "fp32"
        self.dtype = torch.bfloat16 if requested == "bf16" else torch.float16
        if self.enabled and self.dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unavailable; set optim.precision=fp16 or fp32")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enabled and self.dtype == torch.float16)
        self.name = requested if self.enabled else "fp32"

    def context(self):
        return torch.autocast("cuda", dtype=self.dtype) if self.enabled else nullcontext()


def make_loader(dataset, cfg, device, shuffle, epoch=0):
    # A fresh worker set each epoch receives the updated deterministic crop epoch.
    generator = torch.Generator().manual_seed(cfg["seed"] + epoch)
    workers = cfg["data"]["num_workers"]
    options = ({"prefetch_factor": cfg["runtime"].get("loader_prefetch_factor", 2)}
               if workers > 0 else {})
    return DataLoader(dataset, batch_size=cfg["optim"]["batch_size"], shuffle=shuffle,
                      num_workers=workers, pin_memory=device.type == "cuda",
                      drop_last=False, persistent_workers=False, generator=generator, **options)


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()}


def learning_rate(update, total, warmup, base, minimum):
    if warmup and update < warmup:
        return base * (update + 1) / warmup
    progress = (update - warmup) / max(total - warmup - 1, 1)
    progress = max(0.0, min(1.0, progress))
    return minimum + 0.5 * (base - minimum) * (1 + math.cos(math.pi * progress))


def limited_batches(loader, maximum):
    return min(len(loader), maximum) if maximum is not None else len(loader)


def batch_examples(batch):
    """Actual batch length, including a final incomplete batch."""
    return len(batch["pixels"])


def consistency_weight(cfg, kind):
    section = cfg["masking"] if kind == "pretrain" else cfg["downstream"]
    return section.get("consistency_weight", 0.0)


def _paired_forward(model, pixels, observed, pair_generator):
    """One forward over [full; one more group removed]; samples stay independent.

    Every normalization in the encoder and heads is per sample, so concatenating
    the two views along the batch is exactly two separate forwards.
    """
    dropped, _ = drop_one_group({"pixels": pixels, "valid": observed}, 1.0, pair_generator)
    prediction, encoded = model(torch.cat([pixels, pixels]), torch.cat([observed, dropped["valid"]]),
                                return_features=True)
    full, reduced = prediction.chunk(2)
    return full, reduced, dropped["valid"], feature_consistency(encoded["pyramid"])


def forward_loss(model, batch, cfg, kind, generator=None, pair_generator=None):
    """pair_generator (training only) enables the paired full/group-dropped objective."""
    weight = consistency_weight(cfg, kind) if pair_generator is not None else 0.0
    if kind == "pretrain":
        masks = cfg["masking"]
        hidden = make_hidden_mask(batch["valid"], masks["ratio"], masks["block_size"],
                                  masks["modality_drop_prob"], generator)
        observed = batch["valid"] & ~hidden
        if weight:
            prediction, reduced, reduced_observed, consistency = _paired_forward(
                model, batch["pixels"], observed, pair_generator)
        else:
            prediction = model(batch["pixels"], observed)
        loss, statistics = reconstruction_loss(prediction, batch["pixels"], batch["valid"], hidden)
        if not sum(count for _, count in statistics.values()):
            raise RuntimeError("Batch has no valid hidden targets; check validity masks / masking")
        if weight:
            # The removed group becomes an extra cross-modal reconstruction target.
            reduced_loss, _ = reconstruction_loss(reduced, batch["pixels"], batch["valid"], ~reduced_observed)
            loss = .5 * (loss + reduced_loss) + weight * consistency
        return loss, prediction, statistics
    if not batch["target_valid"].any():
        raise RuntimeError("Batch has no valid supervised targets")
    task, smoothing = cfg["downstream"]["task"], cfg["downstream"]["label_smoothing"]
    if weight:
        prediction, reduced, _, consistency = _paired_forward(
            model, batch["pixels"], batch["valid"], pair_generator)
        loss = (.5 * (supervised_loss(prediction, batch, task, smoothing)
                      + supervised_loss(reduced, batch, task, smoothing)) + weight * consistency)
        return loss, prediction, None
    prediction = model(batch["pixels"], batch["valid"])
    loss = supervised_loss(prediction, batch, task, smoothing)
    return loss, prediction, None


def update_meter(meter, batch, loss, prediction, statistics):
    meter.add_loss(float(loss.detach()), batch_examples(batch))
    if statistics is not None:
        meter.add_reconstruction(statistics)
    else:
        meter.add_predictions(prediction, batch["target"], batch["target_valid"])


@torch.no_grad()
def _validate_impl(model, dataset, cfg, device, precision, kind, maximum=None):
    model.eval()
    task = None if kind == "pretrain" else cfg["downstream"]["task"]
    meter = Metrics(task)
    # Validation masking is independent of training RNG and identical each epoch.
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 1_000_003)
    loader = make_loader(dataset, cfg, device, shuffle=False)
    for batch in itertools.islice(loader, limited_batches(loader, maximum)):
        batch = move_batch(batch, device)
        with precision.context():
            loss, prediction, statistics = forward_loss(model, batch, cfg, kind, generator)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        update_meter(meter, batch, loss, prediction, statistics)
    return meter.compute()


@torch.no_grad()
def validate(model, dataset, cfg, device, precision, kind, maximum=None):
    if cfg["model"].get("branch_mora", {}).get("enabled", False) or cfg["runtime"].get("strict_validation", False):
        from .numerics import strict_numeric
        with strict_numeric():
            return _validate_impl(model, dataset, cfg, device, Precision(device, "fp32"), kind, maximum)
    return _validate_impl(model, dataset, cfg, device, precision, kind, maximum)


def _datasets(cfg, kind):
    if cfg["runtime"].get("feature_cache_dir"):
        raise ValueError("Legacy feature-cache training is not included. Use raw mmap/HDF5 inputs.")
    if kind == "pretrain":
        hint = cfg["runtime"].get("mmap_read_hint", "default")
        return (RandomCropDataset(cfg["data"], "train", cfg["seed"], mmap_read_hint=hint),
                RandomCropDataset(cfg["data"], "val", cfg["seed"] + 100_003,
                                  mmap_read_hint=hint))
    options = cfg["downstream"]
    return tuple(BenchmarkDataset(
        cfg["data"], options["task"], split, options["drop_groups"],
        options["few_shot"], cfg["seed"]) for split in ("train", "val"))


def _build_model(cfg, kind, logger, resume=False):
    if kind == "pretrain":
        model = LunarMaskedModel(cfg["model"])
    else:
        task = cfg["downstream"]
        model = LunarTaskModel(cfg["model"], task["task"], task["mode"], task["feature_dropout"])
        if not resume:
            # The same task gets identical fresh head weights for either encoder.
            # Keep the rest of the training RNG unchanged.
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(cfg["seed"] + 700_003)
                model.head = make_task_head(model.encoder, cfg["model"], task["task"], task["feature_dropout"])
            if task.get("init_checkpoint"):
                raise ValueError("Use pretraining weights for a fresh task, or --resume for an existing run")
            elif task["mode"] != "scratch":
                if not task["checkpoint"]:
                    raise ValueError("linear/finetune requires downstream.checkpoint from lunar pretraining")
                pretrained = load_checkpoint(task["checkpoint"])
                if pretrained["kind"] != "pretrain":
                    raise ValueError("downstream.checkpoint must be a lunar pretraining checkpoint")
                initialize_encoder(model.encoder, pretrained)
                logger.info("Loaded complete lunar encoder from %s", task["checkpoint"])
            elif task["checkpoint"] or cfg["runtime"]["init_weights"]:
                raise ValueError("scratch must start without pretrained weights")
    if not resume and cfg["runtime"]["init_weights"]:
        if kind != "pretrain":
            raise ValueError("runtime.init_weights is for pretraining initialization only")
        report = load_official_weights(model.encoder.backbone, cfg["runtime"]["init_weights"])
        logger.info("Official backbone import: %s", report)
        write_json(logger.directory / "weight_import.json", report)
    return model


def _decay_exempt(name, parameter):
    # S and the magnitudes are anchored at their initial values, so decay toward
    # zero would move the effective weights away from the base kernels.
    return (name.endswith(".bank.S") or name.endswith("_magnitude") or ".grn." in name
            or parameter.ndim <= 1 or name.endswith("empty_token"))


def _optimizer(model, cfg, kind):
    optim = cfg["optim"]
    if kind == "downstream" and cfg["downstream"]["mode"] == "finetune":
        groups = [
            {"params": list(model.encoder.named_parameters()), "lr_scale": cfg["downstream"]["encoder_lr_scale"]},
            {"params": list(model.head.named_parameters()), "lr_scale": 1.0},
        ]
    else:
        groups = [{"params": [(n, p) for n, p in model.named_parameters() if p.requires_grad], "lr_scale": 1.0}]
    branch = cfg["model"].get("branch_mora", {})
    if branch.get("enabled") and branch.get("gram_context", "static") in ("observed", "exchange"):
        split = []
        for group in groups:
            decay = [p for n, p in group["params"] if not _decay_exempt(n, p)]
            exempt = [p for n, p in group["params"] if _decay_exempt(n, p)]
            split += [{"params": decay, "lr_scale": group["lr_scale"]},
                      {"params": exempt, "lr_scale": group["lr_scale"], "weight_decay": 0.0}]
        groups = split
    else:
        groups = [{**group, "params": [p for _, p in group["params"]]} for group in groups]
    return torch.optim.AdamW(groups, lr=optim["lr"], betas=(0.9, 0.95),
                             weight_decay=optim["weight_decay"])


def selection_score(metrics, task):
    if task in ("mare", "craters"):
        return metrics["mean_iou"]
    if task in ("geology", "age"):
        return metrics["macro_f1"]
    if task in ("composition", "crossmodal"):
        return -metrics["rmse"] if metrics["rmse"] is not None else None
    return -metrics["loss"] if metrics["loss"] is not None else None


def train(cfg, kind="pretrain", resume_path=None):
    if kind != "downstream" and (cfg["downstream"].get("init_checkpoint") or
                                  cfg["downstream"].get("train_group_drop_prob", 0)):
        raise ValueError("Downstream continuation/group dropout cannot be used for pretraining")
    device = setup(cfg)
    precision = Precision(device, cfg["optim"]["precision"])
    if cfg["runtime"].get("feature_cache_dir"):
        raise ValueError("Legacy feature-cache training is not included; use the original project to resume it.")
    if resume_path:
        checkpoint = load_checkpoint(resume_path)
        if checkpoint["kind"] != kind:
            raise ValueError("Checkpoint stage differs from requested training stage")
        check_resume_compatibility(checkpoint["config"], cfg)
        run_dir = Path(resume_path).resolve().parent
        patience = cfg["downstream"]["patience"] if kind == "downstream" else 0
        if checkpoint["epoch"] >= cfg["optim"]["epochs"] or (
                patience and checkpoint["no_improve"] >= patience):
            print(f"Already completed or early-stopped: {run_dir}; no training restarted.")
            return run_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        task = "" if kind == "pretrain" else "_" + cfg["downstream"]["task"] + "_" + cfg["downstream"]["mode"]
        run_dir = Path(cfg["runtime"]["output_root"]) / f"{stamp}_{kind}{task}_{cfg['name']}"
        run_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = None
    logger = RunLogger(run_dir)
    train_dataset = val_dataset = None
    try:
        logger.info("RUN %s | directory=%s", kind, run_dir)
        logger.info("device=%s | precision=%s | seed=%d", device, precision.name, cfg["seed"])
        logger.info("epoch-boundary resume=%s | weights initialization=%s",
                    bool(resume_path), cfg["downstream"].get("init_checkpoint") or
                    cfg["runtime"]["init_weights"] or "random / lunar encoder")
        environment = {
            "python": platform.python_version(), "torch": torch.__version__,
            "numpy": np.__version__, "platform": platform.platform(), "device": str(device),
            "cuda_build": torch.version.cuda, "precision": precision.name,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        }
        write_json(run_dir / "config.json", cfg)
        write_json(run_dir / "environment.json", environment)
        if kind == "pretrain":
            logger.info("PRETRAIN IO | mmap_read_hint=%s | workers=%d | prefetch_factor=%s",
                        cfg["runtime"].get("mmap_read_hint", "default"),
                        cfg["data"]["num_workers"],
                        cfg["runtime"].get("loader_prefetch_factor", 2)
                        if cfg["data"]["num_workers"] else "disabled")
            logger.info("pretrain pool=%s | split=%s | samples/epoch=%d",
                        cfg["data"]["pretrain_pool"], cfg["data"]["split_protocol"],
                        cfg["data"]["train_samples"])
            if cfg["data"]["pretrain_pool"] == "all":
                logger.info("PROTOCOL: global/transductive pretraining; crops may overlap val/test regions")
        train_dataset, val_dataset = _datasets(cfg, kind)
        if kind == "downstream":
            logger.info("benchmark backend=%s | normalization stats SHA256=%s",
                        cfg["data"].get("benchmark_backend", "mmap"),
                        cfg["data"].get("normalization_stats_sha256") or "not applicable")
        model = _build_model(cfg, kind, logger, resume=bool(checkpoint)).to(device)
        optimizer = _optimizer(model, cfg, kind)
        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("parameters=%s | trainable=%s | train=%d | val=%d",
                    f"{total_params:,}", f"{trainable:,}", len(train_dataset), len(val_dataset))
        logger.info("batch=%d | accumulation=%d | nominal effective batch=%d",
                    cfg["optim"]["batch_size"], cfg["optim"]["grad_accum"],
                    cfg["optim"]["batch_size"] * cfg["optim"]["grad_accum"])
        start_epoch, update, best_score, no_improve = 0, 0, float("-inf"), 0
        if checkpoint:
            model.load_state_dict(checkpoint["model"], strict=True)
            optimizer.load_state_dict(checkpoint["optimizer"])
            precision.scaler.load_state_dict(checkpoint["scaler"])
            start_epoch, update = checkpoint["epoch"], checkpoint["update"]
            best_score, no_improve = checkpoint["best_score"], checkpoint["no_improve"]
            restore_rng(checkpoint["rng"])
            logger.info("Restored epoch=%d | update=%d | best score=%.6f", start_epoch, update, best_score)
            del checkpoint
        epochs = cfg["optim"]["epochs"]
        if start_epoch >= epochs:
            logger.info("Checkpoint has already completed the configured %d epochs", epochs)
            return run_dir
        loader = make_loader(train_dataset, cfg, device, shuffle=True)
        batches = limited_batches(loader, cfg["runtime"]["max_train_batches"])
        accumulation = cfg["optim"]["grad_accum"]
        updates_per_epoch = math.ceil(batches / accumulation)
        total_updates = epochs * updates_per_epoch
        warmup = min(cfg["optim"]["warmup_epochs"] * updates_per_epoch, max(total_updates - 1, 0))
        task = None if kind == "pretrain" else cfg["downstream"]["task"]
        drop_probability = cfg["downstream"].get("train_group_drop_prob", 0) if kind == "downstream" else 0
        if kind == "downstream":
            logger.info("TRAIN GROUP DROP | probability=%.3f | at most one available group per sample | validation off",
                        drop_probability)
        paired = consistency_weight(cfg, kind) > 0
        logger.info("PAIRED VIEWS | %s | consistency weight=%.3f | full + one group removed, 2x forward batch",
                    "on" if paired else "off", consistency_weight(cfg, kind))
        parameters = [p for p in model.parameters() if p.requires_grad]
        branch_enabled = cfg["model"].get("branch_mora", {}).get("enabled", False)
        if branch_enabled:
            branch = cfg["model"]["branch_mora"]
            logger.info("BRANCH MORA | stage 1: seven streams; stages 2-4: shared | all base weights trainable")
            logger.info("GRAM | shared=%s | context=%s", branch.get("shared", True),
                        branch.get("gram_context", "static"))
            logger.info("Selection uses deterministic FP32 validation; original train precision is retained")
        if resume_path is None and (branch_enabled or cfg["runtime"].get("strict_validation", False)):
            # Resume even if interrupted before the first epoch boundary.
            save_checkpoint(run_dir / "last.pt", model, optimizer, precision.scaler, cfg, kind,
                            0, 0, best_score, no_improve, {})
        for epoch in range(start_epoch, epochs):
            if hasattr(train_dataset, "set_epoch"):
                train_dataset.set_epoch(epoch)
            loader = make_loader(train_dataset, cfg, device, shuffle=True, epoch=epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            meter = Metrics(task)
            # Epoch-local augmentation RNG makes boundary resume reproducible and
            # leaves model dropout and sample order on the same RNG stream as A0.
            drop_generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 2_000_003 + epoch)
            drop_counts = torch.zeros(len(GROUPS), dtype=torch.long, device=device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            epoch_start = previous_end = time.perf_counter()
            epoch_data_seconds = epoch_step_seconds = 0.0
            for batch_index, batch in enumerate(itertools.islice(loader, batches)):
                data_seconds = time.perf_counter() - previous_end
                step_start = time.perf_counter()
                batch = move_batch(batch, device)
                if drop_probability:
                    batch, dropped = drop_one_group(batch, drop_probability, drop_generator)
                    drop_counts += dropped.sum(0)
                if batch_index % accumulation == 0:
                    current_lr = learning_rate(update, total_updates, warmup,
                                               cfg["optim"]["lr"], cfg["optim"]["min_lr"])
                    for group in optimizer.param_groups:
                        group["lr"] = current_lr * group["lr_scale"]
                # Weight each microbatch by its actual size, including final short batches.
                group_start = batch_index // accumulation * accumulation
                group_end = min(group_start + accumulation, batches)
                group_examples = sum(min(cfg["optim"]["batch_size"],
                                         len(train_dataset) - j * cfg["optim"]["batch_size"])
                                     for j in range(group_start, group_end))
                scale = batch_examples(batch) / group_examples
                with precision.context():
                    loss, prediction, statistics = forward_loss(
                        model, batch, cfg, kind,
                        pair_generator=drop_generator if paired else None)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at epoch={epoch + 1}, batch={batch_index + 1}")
                precision.scaler.scale(loss * scale).backward()
                if batch_index + 1 == group_end:
                    precision.scaler.unscale_(optimizer)
                    if branch_enabled and update == 0:
                        from .branch_mora import gradient_report
                        write_json(run_dir / "first_update_gradients.json", gradient_report(model))
                    torch.nn.utils.clip_grad_norm_(parameters, cfg["optim"]["clip_grad"],
                                                   error_if_nonfinite=True)
                    precision.scaler.step(optimizer)
                    precision.scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    update += 1
                update_meter(meter, batch, loss, prediction, statistics)
                # Synchronization makes timings explicit and comparable on one GPU.
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                step_seconds = time.perf_counter() - step_start
                epoch_data_seconds += data_seconds
                epoch_step_seconds += step_seconds
                if batch_index == 0 or (batch_index + 1) % cfg["runtime"]["log_every"] == 0 or batch_index + 1 == batches:
                    logger.progress(epoch + 1, epochs, batch_index + 1, batches, update, {
                        "loss": float(loss.detach()), "lr": current_lr,
                        "samples_per_s": batch_examples(batch) / max(data_seconds + step_seconds, 1e-9),
                        "data_seconds": data_seconds, "step_seconds": step_seconds,
                        "gpu_gb": torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == "cuda" else 0,
                    })
                previous_end = time.perf_counter()
            train_metrics = meter.compute()
            train_metrics["seconds"] = time.perf_counter() - epoch_start
            # Totals include every batch, even those not printed in step logs.
            train_metrics["data_seconds"] = epoch_data_seconds
            train_metrics["step_seconds"] = epoch_step_seconds
            train_metrics["samples_per_s"] = train_metrics["examples"] / max(train_metrics["seconds"], 1e-9)
            if kind == "downstream":
                counts = drop_counts.tolist()
                train_metrics["group_drop_counts"] = dict(zip(GROUPS, counts))
                train_metrics["group_drop_samples"] = sum(counts)
                train_metrics["group_drop_fraction"] = sum(counts) / max(train_metrics["examples"], 1)
                logger.info("GROUP DROP | samples=%d/%d | groups=%s",
                            sum(counts), train_metrics["examples"], dict(zip(GROUPS, counts)))
            if branch_enabled:
                from .branch_mora import diagnostics
                train_metrics["branch_mora"] = diagnostics(model)
            logger.summary("train_epoch", epoch + 1, update, train_metrics)
            val_metrics = validate(model, val_dataset, cfg, device, precision, kind,
                                   cfg["runtime"]["max_val_batches"])
            logger.summary("val_epoch", epoch + 1, update, val_metrics)
            score = selection_score(val_metrics, task)
            if score is None or not math.isfinite(score):
                raise RuntimeError("Validation metric is undefined; inspect valid targets / validation coverage")
            improved = score > best_score
            if improved:
                best_score, no_improve = score, 0
            else:
                no_improve += 1
            args = (model, optimizer, precision.scaler, cfg, kind, epoch + 1,
                    update, best_score, no_improve, val_metrics)
            if improved:
                save_checkpoint(run_dir / "best.pt", *args)
                logger.info("Saved BEST complete model | score=%.6f", score)
            # Publish the boundary last, after a new best has been committed.
            save_checkpoint(run_dir / "last.pt", *args)
            if (epoch + 1) % cfg["runtime"]["save_every"] == 0:
                save_checkpoint(run_dir / f"epoch_{epoch + 1:04d}.pt", *args)
            logger.info("Saved LAST checkpoint | epoch %03d complete", epoch + 1)
            patience = cfg["downstream"]["patience"] if kind == "downstream" else 0
            if patience and no_improve >= patience:
                logger.info("Early stopping: no validation improvement for %d epochs", no_improve)
                break
        write_json(run_dir / "summary.json", {
            "kind": kind, "best_score": best_score, "completed_epoch": epoch + 1,
            "optimizer_updates": update, "best_checkpoint": str(run_dir / "best.pt"),
        })
        logger.info("Training finished. Evaluate best.pt on the test split only after model selection.")
        return run_dir
    except BaseException:
        logger.logger.exception("Run stopped; last.pt remains the last completed epoch")
        raise
    finally:
        logger.close()
        for dataset in (train_dataset, val_dataset):
            if dataset is not None:
                if hasattr(dataset, "close"):
                    dataset.close()
                else:
                    dataset.store.close()
