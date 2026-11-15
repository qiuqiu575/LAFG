import torch
from torch.nn import functional as F
from .schema import GROUP_INDICES


def reconstruction_loss(prediction, target, valid, hidden):
    """Uniform mean of active group MSEs, with exact per-pixel validity."""
    selected = valid.bool() & hidden.bool() & torch.isfinite(target)
    safe_target = torch.where(selected, target, 0).float()
    difference = torch.where(selected, prediction.float() - safe_target, 0)
    square = difference.square()
    losses, statistics = [], {}
    for group, indices in GROUP_INDICES.items():
        count = selected[:, indices].sum()
        numerator = square[:, indices].sum()
        statistics[group] = (float(numerator.detach()), int(count))
        if count.item() > 0:
            losses.append(numerator / count)
    # Keep a differentiable zero for diagnostic all-missing inputs.
    loss = torch.stack(losses).mean() if losses else prediction.sum() * 0
    return loss, statistics


def feature_consistency(pyramid):
    """Pull the group-dropped half of each fused level toward the full-input half.

    Each pyramid level holds [full; dropped] along the batch. The full view is a
    stop-gradient teacher, so it is never degraded toward the dropped view.
    Channels are normalized per position so every level contributes equally.
    """
    losses = []
    for level in pyramid:
        full, dropped = level.float().chunk(2)
        full = F.layer_norm(full.detach().permute(0, 2, 3, 1), full.shape[1:2])
        dropped = F.layer_norm(dropped.permute(0, 2, 3, 1), dropped.shape[1:2])
        losses.append((dropped - full).square().mean())
    return torch.stack(losses).mean()


def supervised_loss(prediction, batch, task, label_smoothing=0.1):
    target, valid = batch["target"], batch["target_valid"].bool()
    if task in ("composition", "crossmodal"):
        error = torch.where(valid, prediction.float() - target.float(), 0)
        return error.square().sum() / valid.sum().clamp_min(1)
    if task in ("mare", "craters"):
        target = torch.where(valid, target, torch.full_like(target, -100))
        values = F.cross_entropy(prediction.float(), target.long(),
                                 label_smoothing=label_smoothing,
                                 ignore_index=-100, reduction="none")
        return values.sum() / valid.sum().clamp_min(1)
    return F.cross_entropy(prediction.float(), target.long(),
                           label_smoothing=label_smoothing)
