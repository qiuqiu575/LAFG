import torch
from .schema import GROUP_INDICES


def make_hidden_mask(valid, ratio=0.75, block_size=16, modality_drop_prob=0.2,
                     generator=None):
    """Return [B,C,H,W] bool; True means intentionally hidden.

    Each group has its own block mask, shared across its channels. Whole-group
    dropout adds cross-modal reconstruction examples. If masking eliminates all
    real observations of a sample, one actually observed block is restored.
    Natural missingness is never changed; callers use valid & hidden for loss.
    """
    b, c, h, w = valid.shape
    if h % block_size or w % block_size or not 0 < ratio < 1:
        raise ValueError("Invalid spatial mask size or ratio")
    gh, gw = h // block_size, w // block_size
    total = gh * gw
    if total < 2:
        raise ValueError("At least two spatial mask blocks are required")
    n_hide = max(1, min(total - 1, round(ratio * total)))
    noise = torch.rand((b, len(GROUP_INDICES), total), device=valid.device,
                       generator=generator)
    blocks = torch.zeros_like(noise, dtype=torch.bool)
    blocks.scatter_(2, noise.argsort(dim=-1)[:, :, :n_hide], True)
    drops = torch.rand((b, len(GROUP_INDICES)), device=valid.device,
                       generator=generator) < modality_drop_prob
    blocks = blocks | drops.unsqueeze(-1)
    hidden = torch.zeros_like(valid, dtype=torch.bool)
    for gi, indices in enumerate(GROUP_INDICES.values()):
        spatial = blocks[:, gi].reshape(b, 1, gh, gw)
        spatial = spatial.repeat_interleave(block_size, -2).repeat_interleave(block_size, -1)
        # CUDA deterministic index_put can fail when broadcasting into an
        # advanced channel index. Basic slice copies avoid that kernel, with
        # exactly the same group mask and no additional random draws.
        for channel in indices:
            hidden[:, channel:channel + 1].copy_(spatial)
    # The loop is only for the rare all-observations-hidden case.
    needs_restore = valid.flatten(1).any(1) & ~(valid & ~hidden).flatten(1).any(1)
    for sample in torch.where(needs_restore)[0].tolist():
        channel, row, col = valid[sample].nonzero()[0].tolist()
        group = next(indices for indices in GROUP_INDICES.values() if channel in indices)
        top, left = row // block_size * block_size, col // block_size * block_size
        for channel in group:
            hidden[sample, channel, top:top + block_size, left:left + block_size].fill_(False)
    return hidden


def hide_groups(pixels, valid, groups):
    """Controlled inference ablation; targets must be formed BEFORE this call."""
    pixels, valid = pixels.clone(), valid.clone()
    for group in groups:
        indices = GROUP_INDICES[group]
        for channel in indices:
            pixels[:, channel:channel + 1].zero_()
            valid[:, channel:channel + 1].fill_(False)
    return pixels, valid
