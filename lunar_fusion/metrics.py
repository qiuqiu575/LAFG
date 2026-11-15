import math
import numpy as np
import torch
from .schema import GROUPS, TASK_OUTPUTS, REGRESSION_CHANNELS


class Metrics:
    def __init__(self, task=None):
        self.task = task
        self.loss_sum, self.examples = 0.0, 0
        self.groups = {g: [0.0, 0] for g in GROUPS}
        if task in REGRESSION_CHANNELS:
            n = TASK_OUTPUTS[task]
            self.count = np.zeros(n, dtype=np.float64)
            self.sum_y = np.zeros(n, dtype=np.float64)
            self.sum_y2 = np.zeros(n, dtype=np.float64)
            self.sse = np.zeros(n, dtype=np.float64)
        elif task:
            self.confusion = np.zeros((TASK_OUTPUTS[task], TASK_OUTPUTS[task]), dtype=np.int64)

    def add_loss(self, loss, examples):
        self.loss_sum += float(loss) * examples
        self.examples += examples

    def add_reconstruction(self, statistics):
        for group, (total, count) in statistics.items():
            self.groups[group][0] += total
            self.groups[group][1] += count

    def add_predictions(self, prediction, target, valid):
        if self.task in REGRESSION_CHANNELS:
            p, y = prediction.detach().double().cpu().numpy(), target.double().cpu().numpy()
            v = valid.cpu().numpy().astype(bool)
            self.count += v.sum(axis=0)
            self.sum_y += np.where(v, y, 0).sum(axis=0)
            self.sum_y2 += np.where(v, y * y, 0).sum(axis=0)
            self.sse += np.where(v, (p - y) ** 2, 0).sum(axis=0)
        else:
            p = prediction.detach().argmax(dim=1).flatten()
            y, v = target.flatten(), valid.flatten().bool()
            classes = self.confusion.shape[0]
            v = v & (y >= 0) & (y < classes)
            encoded = y[v].long() * classes + p[v].long()
            matrix = torch.bincount(encoded, minlength=classes * classes)
            self.confusion += matrix.reshape(classes, classes).cpu().numpy()

    def compute(self):
        result = {"loss": self.loss_sum / max(self.examples, 1), "examples": self.examples}
        if self.task is None:
            active = []
            for group, (total, count) in self.groups.items():
                value = total / count if count else None
                result[f"mse_{group}"] = value
                result[f"valid_pixels_{group}"] = int(count)
                if value is not None:
                    active.append(value)
            result["loss"] = sum(active) / len(active) if active else None
        elif self.task in REGRESSION_CHANNELS:
            r2s, mses = [], []
            for i, name in enumerate(REGRESSION_CHANNELS[self.task]):
                n = self.count[i]
                mse = self.sse[i] / n if n else None
                variance = self.sum_y2[i] - self.sum_y[i] ** 2 / max(n, 1)
                r2 = 1 - self.sse[i] / variance if n >= 2 and variance > 1e-12 else None
                result[f"r2_{name}"] = r2
                result[f"rmse_{name}"] = math.sqrt(mse) if mse is not None else None
                result[f"target_count_{name}"] = int(n)
                if mse is not None:
                    mses.append(mse)
                if r2 is not None:
                    r2s.append(r2)
            result["rmse"] = math.sqrt(sum(mses) / len(mses)) if mses else None
            # Equal-output mean, NOT a flattened, variance-weighted R2.
            result["r2"] = sum(r2s) / len(r2s) if r2s else None
        else:
            tp = np.diag(self.confusion).astype(np.float64)
            predicted, actual = self.confusion.sum(0), self.confusion.sum(1)
            f1 = 2 * tp / np.maximum(predicted + actual, 1)
            union = predicted + actual - tp
            iou = tp / np.maximum(union, 1)
            result["accuracy"] = float(tp.sum() / max(self.confusion.sum(), 1))
            result["macro_f1"] = float(f1.mean())
            result["per_class_f1"] = f1.tolist()
            result["confusion_matrix"] = self.confusion.tolist()
            if self.task in ("mare", "craters"):
                result["mean_iou"] = float(iou.mean())
                result["foreground_iou"] = float(iou[1])
                result["per_class_iou"] = iou.tolist()
                result["foreground_f1"] = float(f1[1])
                result["mare_f1" if self.task == "mare" else "crater_f1"] = float(f1[1])
        return result
