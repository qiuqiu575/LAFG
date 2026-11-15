import re
from pathlib import Path

import torch
from torch import nn

from .common import sha256
from .core.branch_mora import attach_branch_stage
from .core.config import model_defaults
from .core.model import GroupedFusion, LunarEncoder, PyramidSegmentationHead


def shared_key(key):
    """Undo branched-stage wrappers, excluding modality-specific adapter tensors."""
    key = re.sub(r"^(stages\.\d+\.\d+)\.base\.", r"\1.", key)
    key = key.replace(".pwconv1.base.", ".pwconv1.").replace(".pwconv2.base.", ".pwconv2.")
    return key


def load_source(path):
    # Only load trusted user-produced checkpoints: old files contain NumPy RNG state.
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("Expected a TRY checkpoint with model and config fields")
    if not isinstance(payload.get("config", {}).get("model"), dict):
        raise ValueError("The source checkpoint must include its model configuration")
    if not any(k.removeprefix("module.").startswith("encoder.backbone.") for k in payload["model"]):
        raise ValueError("Checkpoint does not contain a TRY encoder.backbone")
    return payload


class IceModel(nn.Module):
    def __init__(self, model_config, source=None, source_path=None):
        super().__init__()
        options = model_defaults(model_config)
        # Initialize shared weights BEFORE constructing directional adapters.
        # This anchors new-group magnitudes to the transferred weights, not random ones.
        self.encoder = LunarEncoder(**{k: options[k] for k in (
            "variant", "fusion", "drop_path", "gradient_checkpointing")},
            branch_mora={**options["branch_mora"], "enabled": False})
        self.transfer_report = {"mode": "scratch", "loaded": [], "transferred_parameter_names": []}
        loaded = {}
        if source is not None:
            expected = self.encoder.backbone.state_dict()
            for name, value in source["model"].items():
                name = name.removeprefix("module.")
                if name.startswith("encoder.backbone."):
                    key = shared_key(name.removeprefix("encoder.backbone."))
                    if key in expected:
                        if value.shape != expected[key].shape:
                            raise ValueError(f"Shared backbone shape mismatch: {name}; check variant")
                        if key in loaded:
                            raise ValueError(f"Duplicate shared parameter: {key}")
                        loaded[key] = value
            missing = sorted(set(expected) - set(loaded))
            if missing:
                raise ValueError(f"Incomplete shared-backbone transfer: {missing[:8]}")
            self.encoder.backbone.load_state_dict(loaded, strict=True)
            self.transfer_report = {
                "mode": "shared_backbone_transfer", "source_path": str(Path(source_path).resolve()) if source_path else None,
                "source_sha256": sha256(source_path) if source_path else None,
                "source_kind": source.get("kind"), "source_epoch": source.get("epoch"),
                "loaded": sorted(loaded),
                "reset": ["input stems", "fusion gates", "modality-specific adapters and Gram banks",
                          "ice regression head"],
            }
        self.encoder.branch_options = options["branch_mora"]
        if options["branch_mora"]["enabled"]:
            if options["branch_mora"]["stages"] == 2:
                self.encoder.fusion = GroupedFusion(
                    self.encoder.dims[0], options["fusion"], self.encoder.dims[1])
            attach_branch_stage(self.encoder.backbone, options["branch_mora"])
        self.head = PyramidSegmentationHead(self.encoder.dims, options["decoder_dim"], classes=1)
        self.transfer_report["transferred_parameter_names"] = [
            "encoder.backbone." + name for name, _ in self.encoder.backbone.named_parameters()
            if shared_key(name) in loaded]

    def forward(self, pixels, observed):
        if pixels.shape != observed.shape or pixels.ndim != 4 or pixels.shape[1] != 9:
            raise ValueError("ICE requires matching Bx9xHxW pixels and observation masks")
        features = self.encoder(pixels, observed)
        return self.head(features["pyramid"], pixels.shape[-2:])


def masked_sse(prediction, target, valid):
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError("Prediction, target and mask must have matching Bx1xHxW shapes")
    if not valid.any():
        raise ValueError("No valid target pixels in batch")
    prediction, target = prediction.float()[valid], target.float()[valid]
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Nonfinite prediction or supervised target")
    return (prediction - target).square().sum(), prediction.numel()
