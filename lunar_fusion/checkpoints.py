import os
from pathlib import Path
import random

import numpy as np
import torch

from .config import downstream_defaults, masking_defaults, model_defaults


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, optimizer, scaler, config, kind, epoch, update,
                    best_score, no_improve, metrics):
    path = Path(path)
    payload = {
        "format_version": 1, "kind": kind, "config": config,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(), "epoch": epoch, "update": update,
        "best_score": best_score, "no_improve": no_improve,
        "metrics": metrics, "rng": rng_state(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path):
    # Own checkpoints include Python/NumPy RNG state. Load only trusted local files.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("Expected a checkpoint produced by this lunar_fusion project")
    return payload


def initialize_encoder(encoder, checkpoint):
    """Load a lunar encoder exactly; decoder/head tensors are not transferred."""
    saved_branch = model_defaults(checkpoint["config"]["model"])["branch_mora"]
    if saved_branch != encoder.branch_options:
        raise ValueError("Encoder architecture/Gram mode differs from pretraining checkpoint")
    state = {key.removeprefix("encoder."): value
             for key, value in checkpoint["model"].items() if key.startswith("encoder.")}
    encoder.load_state_dict(state, strict=True)


def check_resume_compatibility(saved, current):
    # These change sampling, model semantics, optimizer schedule or supervision.
    for section in ("seed", "data", "model", "masking", "optim", "downstream"):
        old, new = saved[section], current[section]
        if section == "data":
            defaults = {"benchmark_backend": "mmap", "normalization_stats": None,
                        "normalization_stats_sha256": None}
            old, new = {**defaults, **old}, {**defaults, **new}
            if saved["runtime"].get("strict_validation", False) and current["runtime"].get("strict_validation", False):
                old.pop("num_workers", None)
                new.pop("num_workers", None)
        if section == "model":
            old, new = model_defaults(old), model_defaults(new)
        if section == "downstream":
            old, new = downstream_defaults(old), downstream_defaults(new)
        if section == "masking":
            old, new = masking_defaults(old), masking_defaults(new)
        if old != new:
            raise ValueError(
                f"Resume configuration changed: {section}. Resume with the saved "
                "configuration; use a new run for a changed training protocol.")

    for key in ("max_train_batches", "max_val_batches", "deterministic", "feature_cache_dir", "strict_validation"):
        default = False if key == "strict_validation" else None
        if saved["runtime"].get(key, default) != current["runtime"].get(key, default):
            raise ValueError(f"Resume configuration changed: runtime.{key}")
