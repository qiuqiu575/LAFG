import copy
import csv
from datetime import datetime
from pathlib import Path

import default as settings
from .checkpoints import load_checkpoint
from .config import _merge, apply_overrides, load_config
from .data import BenchmarkDataset, FixedPatchDataset
from .engine import Precision, setup, validate
from .model import LunarMaskedModel, LunarTaskModel
from .runlog import RunLogger, write_json
from .schema import GROUPS

# Changing labels, masking, task, learned architecture or metric protocol would
# make the checkpoint's reported results incomparable. Only relocate data / tune I/O.
EVALUATION_KEYS = {
    "data": {"mmap_dir", "h5_path", "label_dir", "normalization_stats", "num_workers",
             "benchmark_backend"},
    "optim": {"batch_size", "precision"},
    "runtime": {"device", "output_root", "cpu_threads", "mmap_read_hint", "loader_prefetch_factor"},
}


def _check_override_keys(options):
    for section, values in options.items():
        if section not in EVALUATION_KEYS or not isinstance(values, dict):
            raise ValueError(f"Evaluation cannot override {section}")
        unknown = set(values) - EVALUATION_KEYS[section]
        if unknown:
            raise ValueError(f"Evaluation cannot override {section}.{sorted(unknown)}")


def resolve_evaluation(args, options):
    _check_override_keys(options)
    payload = load_checkpoint(args.checkpoint)
    if payload["kind"] not in ("pretrain", "downstream"):
        raise ValueError("Unsupported checkpoint kind")
    cfg = copy.deepcopy(payload["config"])
    # Shared path settings permit moving a checkpoint to another server.
    for key in ("mmap_dir", "h5_path", "label_dir"):
        cfg["data"][key] = settings.COMMON["data"][key]
    cfg["data"]["num_workers"] = settings.EVALUATE["num_workers"]
    cfg["optim"]["batch_size"] = settings.EVALUATE["batch_size"]
    cfg["optim"]["precision"] = "fp32"
    cfg["runtime"]["device"] = settings.COMMON["runtime"]["device"]
    cfg["runtime"]["output_root"] = settings.COMMON["runtime"]["output_root"]
    cfg["runtime"]["feature_cache_dir"] = None
    cfg["runtime"]["strict_validation"] = True
    # Backfill legacy keys before applying strictly validated overrides.
    from .config import DEFAULTS
    for section in EVALUATION_KEYS:
        for key in EVALUATION_KEYS[section]:
            cfg[section].setdefault(key, DEFAULTS[section][key])
    _merge(cfg, options)
    for expression in args.set:
        dotted = expression.partition("=")[0].split(".")
        if len(dotted) != 2 or dotted[0] not in EVALUATION_KEYS or dotted[1] not in EVALUATION_KEYS[dotted[0]]:
            raise ValueError(f"Evaluation cannot override {'.'.join(dotted)}")
    apply_overrides(cfg, args.set)
    for value, section, key in (
        (args.workers, "data", "num_workers"), (args.batch_size, "optim", "batch_size"),
        (args.device, "runtime", "device"), (args.output_root, "runtime", "output_root"),
    ):
        if value is not None:
            cfg[section][key] = str(value) if isinstance(value, Path) else value
    return load_config(base=cfg), payload


def _comparison_row(metrics, original, metadata):
    """Positive drop/increase always denotes deterioration from original."""
    row = dict(metadata)
    row.update({key: value for key, value in metrics.items()
                if value is None or isinstance(value, (int, float, str, bool))})
    bounded_scores = ("accuracy", "macro_f1", "mean_iou", "foreground_iou",
                      "foreground_f1", "mare_f1", "crater_f1")
    for key, value in metrics.items():
        base = original.get(key)
        if not isinstance(value, (int, float)) or not isinstance(base, (int, float)):
            continue
        if key in bounded_scores:
            row[key + "_drop_pp"] = 100 * (base - value)
        elif key == "r2" or key.startswith("r2_"):
            row[key + "_drop"] = base - value
        elif key == "rmse" or key.startswith("rmse_") or key == "loss":
            row[key + "_increase"] = value - base
    return row


def evaluate_suite(checkpoint_path, payload, cfg, split="val", all_drops=False,
                   drop_groups=(), max_batches=None):
    if split not in ("val", "test"):
        raise ValueError("Evaluation requires val or test")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    if set(drop_groups) - set(GROUPS):
        raise ValueError("Unknown group deletion")
    kind = payload["kind"]
    if kind == "pretrain" and (all_drops or drop_groups):
        raise ValueError("Group-deletion evaluation is for downstream checkpoints")
    device = setup(cfg)
    if kind == "pretrain":
        task = "pretrain"
        model = LunarMaskedModel(cfg["model"])
        dataset = FixedPatchDataset(cfg["data"], split)
    else:
        options = cfg["downstream"]
        task = options["task"]
        model = LunarTaskModel(cfg["model"], task, options["mode"], options["feature_dropout"])
        dataset = BenchmarkDataset(cfg["data"], task, split)
    logger = None
    try:
        model.load_state_dict(payload["model"], strict=True)
        model.to(device)
        precision = Precision(device, cfg["optim"]["precision"])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        destination = Path(cfg["runtime"]["output_root"]) / "evaluate" / task / (stamp + "_" + split)
        destination.mkdir(parents=True, exist_ok=False)
        logger = RunLogger(destination)
        write_json(destination / "config.json", cfg)
        conditions = [("original", ())]
        if all_drops:
            conditions += [("drop_" + group, (group,)) for group in GROUPS]
        elif drop_groups:
            groups = tuple(dict.fromkeys(drop_groups))
            conditions.append(("drop_" + "+".join(groups), groups))
        reports, rows, original = {}, [], None
        checkpoint_path = str(Path(checkpoint_path).resolve())
        for condition, groups in conditions:
            if kind == "downstream":
                dataset.drop_groups = groups
            logger.info("EVALUATE | task=%s | split=%s | condition=%s | available=%d",
                        task, split, condition, len(dataset))
            metrics = validate(model, dataset, cfg, device, precision, kind, max_batches)
            if original is None:
                original = metrics
            if metrics["examples"] != original["examples"]:
                raise RuntimeError("Evaluation conditions have different sample counts")
            metadata = {
                "task": task, "condition": condition, "split": split,
                "checkpoint": checkpoint_path, "epoch": payload["epoch"],
                "available_examples": len(dataset),
                "partial_evaluation": metrics["examples"] < len(dataset),
                "already_withheld": task == "crossmodal" and groups == ("thermal",),
            }
            report = {**metadata, "dropped_groups": list(groups), "metrics": metrics}
            reports[condition] = report
            write_json(destination / (condition + ".json"), report)
            logger.summary(condition, payload["epoch"], payload["update"], metrics)
            rows.append(_comparison_row(metrics, original, metadata))
        write_json(destination / "results.json", {"config": cfg, "conditions": reports})
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (destination / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Saved comparison: %s", destination / "comparison.csv")
        return destination, reports
    finally:
        dataset.close() if hasattr(dataset, "close") else dataset.store.close()
        if logger is not None:
            logger.close()
