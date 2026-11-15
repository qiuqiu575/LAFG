import torch

from .schema import GROUP_INDICES, GROUPS, N_CHANNELS


def drop_one_group(batch, probability, generator):
    """Return a new batch and Bx7 drop flags; labels and the input are untouched.

    Availability means at least one finite, valid pixel in the group. Samples
    with fewer than two available groups are unchanged. Conditional on a drop,
    iid uniform scores choose uniformly among available groups. Randomness uses
    a dedicated generator, never the global model/data-loader RNG.
    """
    if not 0 <= probability <= 1:
        raise ValueError("Group drop probability must be in [0,1]")
    pixels, valid = batch["pixels"], batch["valid"]
    if pixels.ndim != 4 or pixels.shape[1] != N_CHANNELS or pixels.shape != valid.shape:
        raise ValueError("Expected matching Bx28xHxW pixels and validity masks")
    flags = torch.zeros((len(pixels), len(GROUPS)), dtype=torch.bool, device=pixels.device)
    if probability == 0:
        return batch, flags
    if generator is None:
        raise ValueError("Training group dropout requires its own generator")
    present = (valid & torch.isfinite(pixels)).flatten(2).any(2)
    available = torch.stack([present[:, GROUP_INDICES[g]].any(1) for g in GROUPS], 1)
    eligible = available.sum(1) >= 2
    requested = torch.rand(len(pixels), device=pixels.device, generator=generator) < probability
    scores = torch.rand(available.shape, device=pixels.device, generator=generator)
    selected = scores.masked_fill(~available, -1).argmax(1)
    flags.scatter_(1, selected[:, None], (eligible & requested)[:, None])
    channel_groups = torch.tensor([g for g, name in enumerate(GROUPS)
                                   for _ in GROUP_INDICES[name]], device=pixels.device)
    hidden = flags[:, channel_groups, None, None]
    return {**batch, "pixels": pixels.masked_fill(hidden, 0),
            "valid": valid.masked_fill(hidden, False)}, flags
