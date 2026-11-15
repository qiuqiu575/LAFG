import csv
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import sys

from .schema import GROUPS

CSV_FIELDS = [
    "timestamp", "phase", "epoch", "batch", "batches", "update", "loss", "lr",
    "samples_per_s", "data_seconds", "step_seconds", "gpu_gb",
    "accuracy", "macro_f1", "mean_iou", "foreground_iou", "foreground_f1", "rmse", "r2",
] + [f"mse_{g}" for g in GROUPS]


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return json_safe(value.item())
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, ensure_ascii=False, allow_nan=False)
    temporary.replace(path)


class RunLogger:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"lunar_fusion.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                      datefmt="%Y-%m-%d %H:%M:%S")
        for handler in (logging.StreamHandler(sys.stdout),
                        logging.FileHandler(self.directory / "train.log", encoding="utf-8")):
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.jsonl = (self.directory / "metrics.jsonl").open("a", encoding="utf-8", buffering=1)
        csv_path = self.directory / "metrics.csv"
        new_csv = not csv_path.exists() or csv_path.stat().st_size == 0
        self.csv_handle = csv_path.open("a", encoding="utf-8", newline="")
        self.csv_writer = csv.DictWriter(self.csv_handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_csv:
            self.csv_writer.writeheader()
            self.csv_handle.flush()

    def info(self, text, *args):
        self.logger.info(text, *args)

    def record(self, phase, epoch, update, values, batch=None, batches=None):
        row = json_safe({"timestamp": datetime.now(timezone.utc).isoformat(),
                         "phase": phase, "epoch": epoch, "update": update,
                         "batch": batch, "batches": batches, **values})
        self.jsonl.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        self.jsonl.flush()
        self.csv_writer.writerow(row)
        self.csv_handle.flush()

    def progress(self, epoch, epochs, batch, batches, update, values):
        self.record("train_step", epoch, update, values, batch, batches)
        self.info(
            "epoch %03d/%03d | batch %05d/%05d | update %06d | "
            "loss %9.5f | lr %.3e | %6.2f sample/s | "
            "data %.3fs | step %.3fs | GPU %.2f GiB",
            epoch, epochs, batch, batches, update, values["loss"], values["lr"],
            values["samples_per_s"], values["data_seconds"], values["step_seconds"],
            values["gpu_gb"])

    def summary(self, phase, epoch, update, values):
        self.record(phase, epoch, update, values)
        scalar = " | ".join(f"{k}={values[k]:.6f}" for k in
                            ("loss", "accuracy", "macro_f1", "mean_iou", "rmse", "r2")
                            if values.get(k) is not None)
        self.info("%s epoch %03d | %s", phase.upper(), epoch, scalar)
        if phase == "train_epoch" and "samples_per_s" in values:
            self.info("TRAIN TIMING | wall %.3fs | data_wait %.3fs | step %.3fs | %.2f sample/s",
                      values["seconds"], values["data_seconds"], values["step_seconds"],
                      values["samples_per_s"])
        if "mse_surface" in values:
            self.info("  %-16s | %12s | %14s", "modality", "MSE", "target pixels")
            for group in GROUPS:
                value = values.get(f"mse_{group}")
                text = f"{value:.6f}" if value is not None else "N/A"
                self.info("  %-16s | %12s | %14d", group, text,
                          values.get(f"valid_pixels_{group}", 0))

    def close(self):
        self.jsonl.close()
        self.csv_handle.close()
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
