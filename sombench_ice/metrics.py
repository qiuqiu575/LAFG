import math

import torch


class RegressionMetrics:
    def __init__(self):
        self.count, self.mean, self.m2, self.sse, self.sae = 0, 0., 0., 0., 0.

    @torch.no_grad()
    def update(self, prediction, target, valid):
        if prediction.shape != target.shape or target.shape != valid.shape:
            raise ValueError("Mismatched metric tensor shapes")
        y, p = target[valid].double(), prediction[valid].double()
        if not y.numel():
            return
        if not torch.isfinite(y).all() or not torch.isfinite(p).all():
            raise FloatingPointError("Nonfinite supervised prediction/target")
        n, mean = y.numel(), y.mean().item()
        delta = mean - self.mean
        self.m2 += (y - mean).square().sum().item() + delta * delta * self.count * n / (self.count + n)
        self.mean += delta * n / (self.count + n)
        self.count += n
        self.sse += (p - y).square().sum().item()
        self.sae += (p - y).abs().sum().item()

    def result(self):
        if not self.count:
            raise ValueError("No valid pixels to evaluate")
        return {"rmse": math.sqrt(self.sse / self.count), "mae": self.sae / self.count,
                "r2": 1. - self.sse / self.m2 if self.m2 > 0. else None,
                "valid_pixels": self.count, "target_mean": self.mean,
                "target_sst": self.m2,
                "aggregation": "global_valid_pixels", "constant_target_r2": "null"}
