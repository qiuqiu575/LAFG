import copy
import hashlib
import json
import math
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULTS = {
    "name": "pico",
    "seed": 42,
    "data": {
        "mmap_dir": "/root/autodl-tmp/Moonstone/mmap",
        "h5_path": "/root/autodl-tmp/Moonstone/benchmark/lunar_patches_v4.h5",
        "label_dir": "/root/autodl-tmp/Moonstone/benchmark",
        "patch_size": 256,
        "ppd": 128,
        "split_protocol": "stored",
        "pretrain_pool": "train",
        "latitude_range": [-70.0, 70.0],
        "min_surface_fraction": 0.3,
        "train_samples": 20000,
        "val_samples": 256,
        "num_workers": 0,
        "allow_missing_channels": False,
        "benchmark_backend": "mmap",
        "normalization_stats": None,
        "normalization_stats_sha256": None,
    },
    "model": {
        "variant": "pico",
        "fusion": "gated",
        "decoder_dim": 64,
        "drop_path": 0.0,
        "gradient_checkpointing": False,
        # Missing stages / gram_context in historical checkpoints must retain
        # Stage-1 fusion and the parameter-only Gram.
        "branch_mora": {"enabled": False, "rank": 4, "alpha": 0.01, "shared": True, "stages": 1,
                        "gram_context": "static"},
    },
    # consistency_weight > 0: paired full / one-group-removed views per batch.
    "masking": {"ratio": 0.75, "block_size": 16, "modality_drop_prob": 0.2, "consistency_weight": 0.0},
    "optim": {
        "epochs": 20,
        "batch_size": 2,
        "grad_accum": 8,
        "lr": 0.00015,
        "min_lr": 0.000001,
        "weight_decay": 0.05,
        "warmup_epochs": 2,
        "clip_grad": 1.0,
        "precision": "bf16",
    },
    "downstream": {
        "task": "geology",
        "mode": "linear",
        "checkpoint": None,
        "encoder_lr_scale": 0.1,
        "label_smoothing": 0.1,
        "feature_dropout": 0.1,
        "patience": 5,
        "few_shot": 0,
        "drop_groups": [],
        "init_checkpoint": None,
        "init_checkpoint_sha256": None,
        "train_group_drop_prob": 0.0,
        "consistency_weight": 0.0,
    },
    "runtime": {
        "device": "auto",
        "output_root": "runs",
        "log_every": 10,
        "save_every": 5,
        "init_weights": None,
        "feature_cache_dir": None,
        "max_train_batches": None,
        "max_val_batches": None,
        "deterministic": False,
        "strict_validation": False,
        "cpu_threads": 4,
        "mmap_read_hint": "default",
        "loader_prefetch_factor": 2,
    },
}


def _merge(target, changes, prefix=""):
    for key, value in changes.items():
        if key not in target:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(target[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key} must be an object")
            _merge(target[key], value, prefix + key + ".")
        else:
            target[key] = value


def downstream_defaults(options):
    """Backfill continuation fields without changing old checkpoint semantics."""
    return {"init_checkpoint": None, "init_checkpoint_sha256": None,
            "train_group_drop_prob": 0.0, "consistency_weight": 0.0, **options}


def masking_defaults(options):
    return {"consistency_weight": 0.0, **options}


def model_defaults(options):
    return {**options, "branch_mora": {**DEFAULTS["model"]["branch_mora"],
                                      **options.get("branch_mora", {})}}


def validate(cfg):
    from .backbone import VARIANTS
    if cfg["model"]["variant"] not in VARIANTS:
        raise ValueError(f"Unknown backbone variant: {cfg['model']['variant']}")
    branch = model_defaults(cfg["model"])["branch_mora"]
    if set(branch) != {"enabled", "rank", "alpha", "shared", "stages", "gram_context"}:
        raise ValueError("Unknown branch_mora option")
    if type(branch["stages"]) is not int or branch["stages"] not in (1, 2):
        raise ValueError("branch_mora.stages must be 1 or 2")
    if branch["gram_context"] not in ("static", "observed", "exchange"):
        raise ValueError("branch_mora.gram_context must be static, observed or exchange")
    if type(branch["enabled"]) is not bool or type(branch["shared"]) is not bool:
        raise ValueError("branch_mora.enabled/shared must be boolean")
    if type(branch["rank"]) is not int or branch["rank"] < 1:
        raise ValueError("branch_mora.rank must be a positive integer")
    if not isinstance(branch["alpha"], (int, float)) or not math.isfinite(branch["alpha"]) or branch["alpha"] <= 0:
        raise ValueError("branch_mora.alpha must be finite and positive")
    if branch["enabled"]:
        if branch["rank"] > VARIANTS[cfg["model"]["variant"]][1][0]:
            raise ValueError("branch rank exceeds stage-1 width")
        if cfg["runtime"].get("init_weights"):
            raise ValueError("Branched experiment starts from scratch; official weight import is unsupported")
    d, m, o, r = (cfg[k] for k in ("data", "masking", "optim", "runtime"))
    downstream = downstream_defaults(cfg["downstream"])
    probability = downstream["train_group_drop_prob"]
    if not 0 <= probability <= 1:
        raise ValueError("downstream.train_group_drop_prob must be in [0,1]")
    if probability and downstream["mode"] != "finetune":
        raise ValueError("Training group dropout requires finetune mode")
    if downstream["init_checkpoint"] and (downstream["task"] != "geology" or downstream["mode"] != "finetune"):
        raise ValueError("Continuation requires geology finetune")
    for name, weight in (("masking", masking_defaults(m)["consistency_weight"]),
                         ("downstream", downstream["consistency_weight"])):
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"{name}.consistency_weight must be finite and nonnegative")
    if downstream["consistency_weight"]:
        if probability:
            raise ValueError("Use train_group_drop_prob OR consistency_weight; the paired view already removes a group")
    if probability or downstream["init_checkpoint"] or downstream["consistency_weight"]:
        if r.get("feature_cache_dir"):
            raise ValueError("Group dropout/continuation cannot use frozen feature caches")
    if downstream["init_checkpoint"] and (downstream["checkpoint"] or r["init_weights"]):
        raise ValueError("Full downstream initialization cannot be combined with pretraining weights")
    if downstream["init_checkpoint_sha256"] and not downstream["init_checkpoint"]:
        raise ValueError("init_checkpoint_sha256 requires init_checkpoint")
    if d["patch_size"] < 32 or d["patch_size"] % 32:
        raise ValueError("data.patch_size must be a positive multiple of 32")
    if m["block_size"] < 4 or m["block_size"] % 4 or d["patch_size"] % m["block_size"]:
        raise ValueError("masking.block_size must be a multiple of 4 and divide patch_size")
    if not 0 < m["ratio"] < 1 or not 0 <= m["modality_drop_prob"] <= 1:
        raise ValueError("Mask ratio must be in (0,1), modality drop in [0,1]")
    if d["patch_size"] // m["block_size"] < 2:
        raise ValueError("Use at least two mask blocks per spatial dimension")
    for key in ("epochs", "batch_size", "grad_accum"):
        if o[key] < 1:
            raise ValueError(f"optim.{key} must be positive")
    if o["lr"] <= 0 or not 0 <= o["min_lr"] <= o["lr"]:
        raise ValueError("Require 0 <= min_lr <= lr and lr > 0")
    if o["warmup_epochs"] < 0 or o["weight_decay"] < 0 or o["clip_grad"] <= 0:
        raise ValueError("Invalid optimizer configuration")
    if o["precision"] not in ("fp32", "bf16", "fp16"):
        raise ValueError("precision must be fp32, bf16 or fp16")
    if d["split_protocol"] not in ("stored", "latitude"):
        raise ValueError("split_protocol must be stored or latitude")
    if d["pretrain_pool"] not in ("train", "all"):
        raise ValueError("pretrain_pool must be train or all")
    if d.get("benchmark_backend", "mmap") not in ("mmap", "hdf5"):
        raise ValueError("data.benchmark_backend must be mmap or hdf5")
    if d.get("benchmark_backend") == "hdf5" and not d.get("normalization_stats"):
        raise ValueError("HDF5 benchmark input requires the mmap's normalization_stats file")
    if d["num_workers"] < 0 or min(d["train_samples"], d["val_samples"]) < 1:
        raise ValueError("Invalid dataset length or worker count")
    if not -90 <= d["latitude_range"][0] < d["latitude_range"][1] <= 90:
        raise ValueError("Invalid latitude range")
    if d["ppd"] <= 0 or not 0 <= d["min_surface_fraction"] <= 1:
        raise ValueError("Invalid grid or validity threshold")
    if cfg["model"]["fusion"] not in ("gated", "mean", "coverage"):
        raise ValueError("fusion must be gated, mean or coverage")
    if cfg["model"]["decoder_dim"] < 4 or not 0 <= cfg["model"]["drop_path"] < 1:
        raise ValueError("Invalid model width or drop_path")
    if r["log_every"] < 1 or r["save_every"] < 1 or r["cpu_threads"] < 1:
        raise ValueError("Logging, saving and thread intervals must be positive")
    if r.get("mmap_read_hint", "default") not in ("default", "random"):
        raise ValueError("runtime.mmap_read_hint must be default or random")
    prefetch = r.get("loader_prefetch_factor", 2)
    if type(prefetch) is not int or prefetch < 1:
        raise ValueError("runtime.loader_prefetch_factor must be a positive integer")
    for key in ("max_train_batches", "max_val_batches"):
        if r[key] is not None and r[key] < 1:
            raise ValueError(f"runtime.{key} must be null or positive")
    from .schema import GROUPS, TASK_OUTPUTS
    task = cfg["downstream"]
    if task["task"] not in TASK_OUTPUTS or task["mode"] not in ("scratch", "linear", "finetune"):
        raise ValueError("Invalid downstream task/mode")
    if set(task["drop_groups"]) - set(GROUPS):
        raise ValueError("Unknown group in downstream.drop_groups")
    if task["few_shot"] < 0 or task["patience"] < 0:
        raise ValueError("few_shot and patience must be nonnegative")
    if task["few_shot"] and task["task"] not in ("geology", "age"):
        raise ValueError("few_shot is K examples per class for geology/age only")
    if not 0 <= task["feature_dropout"] < 1 or not 0 <= task["label_smoothing"] < 1:
        raise ValueError("Invalid dropout / label smoothing")
    if r.get("feature_cache_dir") and (
        task["task"] != "geology" or task["mode"] != "linear" or task["few_shot"] != 0
    ):
        raise ValueError("Feature caching supports only geology / linear / few_shot=0")
    if task["encoder_lr_scale"] <= 0:
        raise ValueError("encoder_lr_scale must be positive")


def apply_overrides(cfg, overrides):
    """Apply explicit dotted keys in place; reject unknown/misspelled options."""
    for expression in overrides:
        if "=" not in expression:
            raise ValueError(f"Override must be key=value: {expression}")
        dotted, raw = expression.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target = cfg
        keys = dotted.split(".")
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise ValueError(f"Unknown configuration key: {dotted}")
            target = target[key]
        _merge(target, {keys[-1]: value}, ".".join(keys[:-1]) + ".")


def load_config(path=None, overrides=(), base=None):
    cfg = copy.deepcopy(DEFAULTS if base is None else base)
    cfg["downstream"] = downstream_defaults(cfg["downstream"])
    cfg["masking"] = masking_defaults(cfg["masking"])
    cfg["model"] = model_defaults(cfg["model"])
    # Preserve old checkpoint semantics while backfilling optional I/O fields.
    for key in ("benchmark_backend", "normalization_stats", "normalization_stats_sha256"):
        cfg["data"].setdefault(key, DEFAULTS["data"][key])
    cfg["runtime"].setdefault("feature_cache_dir", None)
    cfg["runtime"].setdefault("strict_validation", False)
    for key in ("mmap_read_hint", "loader_prefetch_factor"):
        cfg["runtime"].setdefault(key, DEFAULTS["runtime"][key])
    if path:
        with open(path, encoding="utf-8-sig") as handle:
            _merge(cfg, json.load(handle))
    apply_overrides(cfg, overrides)
    validate(cfg)
    for section, keys in {
        "data": ("mmap_dir", "h5_path", "label_dir", "normalization_stats"),
        "runtime": ("output_root", "init_weights", "feature_cache_dir"),
        "downstream": ("checkpoint", "init_checkpoint"),
    }.items():
        for key in keys:
            if cfg[section].get(key):
                value = Path(os.path.expandvars(cfg[section][key])).expanduser()
                cfg[section][key] = str(value if value.is_absolute() else PROJECT_DIR / value)
    if cfg["data"].get("benchmark_backend") == "hdf5":
        digest = hashlib.sha256(Path(cfg["data"]["normalization_stats"]).read_bytes()).hexdigest()
        expected = cfg["data"].get("normalization_stats_sha256")
        if expected and expected != digest:
            raise ValueError("Normalization statistics changed since this configuration was saved")
        cfg["data"]["normalization_stats_sha256"] = digest
    return cfg
