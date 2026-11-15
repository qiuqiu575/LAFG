import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

VARIANTS = {
    "debug": ((1, 1, 1, 1), (8, 16, 32, 64)),
    "atto": ((2, 2, 6, 2), (40, 80, 160, 320)),
    "femto": ((2, 2, 6, 2), (48, 96, 192, 384)),
    "pico": ((2, 2, 6, 2), (64, 128, 256, 512)),
    "nano": ((2, 2, 8, 2), (80, 160, 320, 640)),
    "tiny": ((3, 3, 9, 3), (96, 192, 384, 768)),
}


class LayerNorm(nn.Module):
    def __init__(self, channels, channels_first=False, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.channels_first = channels_first
        self.eps = eps

    def forward(self, x):
        if self.channels_first:
            x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (len(self.weight),), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2) if self.channels_first else x


class GRN(nn.Module):
    """Global response normalization; reduction in FP32 for mixed precision."""
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, channels))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, channels))

    def forward(self, x):
        magnitude = torch.linalg.vector_norm(x.float(), dim=(1, 2), keepdim=True)
        relative = magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-6)
        return x + self.gamma.to(x.dtype) * x * relative.to(x.dtype) + self.beta.to(x.dtype)


class DropPath(nn.Module):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((len(x), 1, 1, 1)).bernoulli_(keep)
        return x * mask / keep


class Block(nn.Module):
    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.GELU()
        self.grn = GRN(4 * channels)
        self.pwconv2 = nn.Linear(4 * channels, channels)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.dwconv(x).permute(0, 2, 3, 1)
        x = self.pwconv2(self.grn(self.act(self.pwconv1(self.norm(x)))))
        return residual + self.drop_path(x.permute(0, 3, 1, 2))


class ConvNeXtV2Stages(nn.Module):
    """The input is already a stride-4 feature map from the fusion module."""
    def __init__(self, variant="pico", drop_path=0.0, gradient_checkpointing=False):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}; choose {tuple(VARIANTS)}")
        depths, self.dims = VARIANTS[variant]
        self.gradient_checkpointing = gradient_checkpointing
        # Keep official indices 1..3 to enable official weight import.
        self.downsample_layers = nn.ModuleList([nn.Identity()])
        for before, after in zip(self.dims[:-1], self.dims[1:]):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(before, channels_first=True),
                nn.Conv2d(before, after, 2, stride=2),
            ))
        rates = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        offset = 0
        for depth, width in zip(depths, self.dims):
            self.stages.append(nn.Sequential(
                *[Block(width, rates[offset + j]) for j in range(depth)]))
            offset += depth
        self.norm = nn.LayerNorm(self.dims[-1], eps=1e-6)
        self.apply(initialize_weights)

    def forward(self, x, start_stage=0, pyramid=None):
        pyramid = [] if pyramid is None else list(pyramid)
        if start_stage not in (0, 1, 2) or len(pyramid) != start_stage:
            raise ValueError("Invalid backbone entry stage/pyramid")
        for downsample, stage in list(zip(self.downsample_layers, self.stages))[start_stage:]:
            x = downsample(x)
            for block in stage:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    x = checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)
            pyramid.append(x)
        return {"global": self.norm(x.mean(dim=(-2, -1))), "pyramid": pyramid}


def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def load_official_weights(backbone, path):
    """Import dense or official sparse FCMAE weights; never silently accept a wrong size.

    The RGB stem/head and reconstruction decoder are intentionally excluded.
    FCMAE may not contain the final global pooling norm, which stays initialized.
    No network access or automatic downloads occur here.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload.get("model", payload.get("state_dict", payload))
    converted = {}
    ignored = []
    for original, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = original.removeprefix("module.").removeprefix("encoder.")
        if not key.startswith(("stages.", "downsample_layers.", "norm.")):
            ignored.append(original)
            continue
        if key.startswith("downsample_layers.0."):
            ignored.append(original)
            continue
        # Official MinkowskiEngine -> dense conversion (same axis convention).
        if key.endswith(".kernel"):
            key = key[:-len("kernel")] + "weight"
            side = math.isqrt(value.shape[0])
            if side * side != value.shape[0]:
                raise ValueError(f"Non-square convolution kernel: {original}")
            if value.ndim == 3:
                _, in_dim, out_dim = value.shape
                value = value.permute(2, 1, 0).reshape(out_dim, in_dim, side, side).transpose(3, 2)
            elif value.ndim == 2:
                value = value.T.reshape(value.shape[1], 1, side, side).transpose(3, 2)
            else:
                raise ValueError(f"Unsupported sparse kernel: {original}")
        key = key.replace(".ln.", ".").replace(".linear.", ".")
        if ".grn." in key:
            value = value.reshape(1, 1, 1, -1)
        elif key.endswith(".bias"):
            value = value.reshape(-1)
        converted[key] = value
    expected = backbone.state_dict()
    required = {k for k in expected if not k.startswith("norm.")}
    missing = sorted(required - converted.keys())
    mismatched = [k for k in converted if k in expected and converted[k].shape != expected[k].shape]
    if missing or mismatched:
        raise ValueError(
            "Weights do not match the selected ConvNeXt V2 variant. "
            f"Missing: {missing[:5]}; shape mismatch: {mismatched[:5]}")
    accepted = {k: v for k, v in converted.items() if k in expected}
    result = backbone.load_state_dict(accepted, strict=False)
    return {"loaded_tensors": len(accepted), "initialized": result.missing_keys,
            "ignored_tensors": len(ignored), "source": str(path)}
