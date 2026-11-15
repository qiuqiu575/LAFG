import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .backbone import ConvNeXtV2Stages, LayerNorm, Block, initialize_weights
from .schema import GROUPS, GROUP_INDICES, N_CHANNELS, TASK_OUTPUTS


class GroupedFusion(nn.Module):
    """Fuse aligned maps after stems, Stage 1, or Stage 2.

    observed is the actual input visibility, not the reconstruction target mask.
    Values are removed BEFORE the first spatial convolution. A channel-level
    observation mask is concatenated to each group input, distinguishing valid
    zero from unavailable zero. Gates never receive hidden target values.
    """
    def __init__(self, width, mode="gated", fusion_width=None):
        super().__init__()
        self.mode = mode
        fusion_width = width if fusion_width is None else fusion_width
        self.stems = nn.ModuleDict({
            group: nn.Sequential(
                nn.Conv2d(2 * len(indices), width, 4, stride=4),
                LayerNorm(width, channels_first=True),
            ) for group, indices in GROUP_INDICES.items()
        })
        self.gates = nn.ModuleDict({
            group: nn.Conv2d(fusion_width + 1, 1, 1) for group in GROUPS
        }) if mode == "gated" else nn.ModuleDict()
        self.empty_token = nn.Parameter(torch.zeros(1, fusion_width, 1, 1))
        if fusion_width != width:
            self.skip_empty_token = nn.Parameter(torch.zeros(1, width, 1, 1))
        self.apply(initialize_weights)
        # Start with coverage weighting, then learn group/position preferences.
        for gate in self.gates.values():
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def _group_features(self, pixels, observed):
        if pixels.ndim != 4 or pixels.shape[1] != N_CHANNELS or pixels.shape != observed.shape:
            raise ValueError(f"Expected pixels and observed of shape [B,{N_CHANNELS},H,W]")
        if pixels.shape[-1] % 32 or pixels.shape[-2] % 32:
            raise ValueError("Spatial dimensions must be divisible by 32")
        observed = observed.bool() & torch.isfinite(pixels)
        safe = torch.where(observed, pixels, 0.0)
        for group_index, (group, indices) in enumerate(GROUP_INDICES.items()):
            group_mask = observed[:, indices].float()
            feature = self.stems[group](torch.cat(
                [safe[:, indices], group_mask.to(safe.dtype)], dim=1))
            fraction = F.avg_pool2d(group_mask.mean(dim=1, keepdim=True), 4, 4)
            yield group_index, group, feature, fraction

    def _run_stage(self, stage, feature, group_index, fraction, gradient_checkpointing):
        support = fraction > 0
        for block in stage:
            if gradient_checkpointing and self.training and torch.is_grad_enabled():
                feature = checkpoint(block, feature, group_index, support, use_reentrant=False)
            else:
                feature = block(feature, group_index, support)
        return feature

    @staticmethod
    def _lockstep(stage):
        return getattr(stage[0], "lockstep", False)

    @staticmethod
    def _grouped(stage):
        return getattr(stage[0], "grouped", False)

    def _stacked_stems(self, pixels, observed):
        groups = list(self._group_features(pixels, observed))
        return (torch.stack([feature for _, _, feature, _ in groups]),
                torch.stack([fraction for _, _, _, fraction in groups]))

    def _run_stage_grouped(self, stage, features, fractions, gradient_checkpointing):
        """features [G,B,C,H,W] -> [G,B,C,H,W]; groups folded into one batched pass.

        Shared layers see a G*B batch; per-group pointwise weights use one bmm.
        Every operation is per sample, so this equals running the groups one by one.
        """
        groups = features.shape[0]
        x, support = features.flatten(0, 1), (fractions > 0).flatten(0, 1)
        use_checkpoint = gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in stage:
            x = (checkpoint(block.forward_grouped, x, support, groups, use_reentrant=False)
                 if use_checkpoint else block.forward_grouped(x, support, groups))
            exchange = getattr(block, "exchange", None)
            if exchange is not None:
                x = (checkpoint(exchange.forward_grouped, x, support, groups, use_reentrant=False)
                     if use_checkpoint else exchange.forward_grouped(x, support, groups))
        return x.unflatten(0, features.shape[:2])

    def _scores(self, maps, fractions):
        return [self._score(group, feature, fraction)
                for group, feature, fraction in zip(GROUPS, maps, fractions)]

    def _run_stage_lockstep(self, stage, features, fractions, gradient_checkpointing):
        """Observed / exchange Gram: every group finishes block k before block k+1.

        observed: each block's weight context comes from all groups' block inputs.
        exchange: a GramExchange after the stage's last block passes per-pixel
        messages between observed groups; fusion still follows the stage.
        """
        supports = [fraction > 0 for fraction in fractions]
        use_checkpoint = gradient_checkpointing and self.training and torch.is_grad_enabled()
        for block in stage:
            context = block.bank.context(features, supports) if block.needs_context else None
            features = [
                checkpoint(block, feature, group_index, support, context, use_reentrant=False)
                if use_checkpoint else block(feature, group_index, support, context)
                for group_index, (feature, support) in enumerate(zip(features, supports))]
            exchange = getattr(block, "exchange", None)
            if exchange is not None:
                features = (checkpoint(exchange, features, supports, use_reentrant=False)
                            if use_checkpoint else exchange(features, supports))
        return features

    def _score(self, group, feature, fraction):
        if self.mode == "gated":
            return self.gates[group](torch.cat([feature, fraction.to(feature.dtype)], dim=1)).float()
        return torch.zeros_like(fraction)

    def forward(self, pixels, observed, branch_stage=None, gradient_checkpointing=False):
        """Original early-fusion and Stage-1 paths, with unchanged state keys."""
        if branch_stage is not None and self._grouped(branch_stage):
            features, fractions = self._stacked_stems(pixels, observed)
            maps = list(self._run_stage_grouped(branch_stage, features, fractions,
                                                gradient_checkpointing).unbind(0))
            fractions = list(fractions.unbind(0))
            return self._fuse(maps, fractions, self._scores(maps, fractions), self.empty_token)
        if branch_stage is not None and self._lockstep(branch_stage):
            groups = list(self._group_features(pixels, observed))
            fractions = [fraction for _, _, _, fraction in groups]
            maps = self._run_stage_lockstep(
                branch_stage, [feature for _, _, feature, _ in groups], fractions, gradient_checkpointing)
            scores = [self._score(group, feature, fraction)
                      for (_, group, _, _), feature, fraction in zip(groups, maps, fractions)]
            return self._fuse(maps, fractions, scores, self.empty_token)
        maps, fractions, scores = [], [], []
        for group_index, group, feature, fraction in self._group_features(pixels, observed):
            if branch_stage is not None:
                feature = self._run_stage(branch_stage, feature, group_index, fraction, gradient_checkpointing)
            maps.append(feature)
            fractions.append(fraction)
            scores.append(self._score(group, feature, fraction))
        return self._fuse(maps, fractions, scores, self.empty_token)

    def forward_two_stage(self, pixels, observed, stages, downsample, gradient_checkpointing=False):
        """Each group crosses both stages before fusion at stride 8.

        A stride-4 skip is aggregated using upsampled Stage-2 gate logits and
        fine-scale coverage. It is only a decoder skip, never fed into Stage 2.
        This preserves all four resolutions for segmentation and ICE heads.
        """
        if self._grouped(stages[0]):
            return self._forward_two_stage_grouped(
                pixels, observed, stages, downsample, gradient_checkpointing)
        if self._lockstep(stages[0]):
            return self._forward_two_stage_lockstep(
                pixels, observed, stages, downsample, gradient_checkpointing)
        skip_maps, skip_fractions, maps, fractions, scores = [], [], [], [], []
        for group_index, group, feature, fraction in self._group_features(pixels, observed):
            feature = self._run_stage(stages[0], feature, group_index, fraction, gradient_checkpointing)
            skip_maps.append(feature)
            skip_fractions.append(fraction)
            if gradient_checkpointing and self.training and torch.is_grad_enabled():
                feature = checkpoint(downsample, feature, use_reentrant=False)
            else:
                feature = downsample(feature)
            # The stride-2 convolution and this pooling have matching footprints.
            # BranchedBlock masks convolution/normalization biases at unsupported sites.
            fraction = F.avg_pool2d(fraction, 2, 2)
            feature = self._run_stage(stages[1], feature, group_index, fraction, gradient_checkpointing)
            maps.append(feature)
            fractions.append(fraction)
            scores.append(self._score(group, feature, fraction))
        return self._finish_two_stage(skip_maps, skip_fractions, maps, fractions, scores)

    def _forward_two_stage_lockstep(self, pixels, observed, stages, downsample, gradient_checkpointing):
        groups = list(self._group_features(pixels, observed))
        skip_fractions = [fraction for _, _, _, fraction in groups]
        skip_maps = self._run_stage_lockstep(
            stages[0], [feature for _, _, feature, _ in groups], skip_fractions, gradient_checkpointing)
        if gradient_checkpointing and self.training and torch.is_grad_enabled():
            features = [checkpoint(downsample, feature, use_reentrant=False) for feature in skip_maps]
        else:
            features = [downsample(feature) for feature in skip_maps]
        fractions = [F.avg_pool2d(fraction, 2, 2) for fraction in skip_fractions]
        maps = self._run_stage_lockstep(stages[1], features, fractions, gradient_checkpointing)
        scores = [self._score(group, feature, fraction)
                  for (_, group, _, _), feature, fraction in zip(groups, maps, fractions)]
        return self._finish_two_stage(skip_maps, skip_fractions, maps, fractions, scores)

    def _forward_two_stage_grouped(self, pixels, observed, stages, downsample, gradient_checkpointing):
        features, skip_fractions = self._stacked_stems(pixels, observed)
        skip = self._run_stage_grouped(stages[0], features, skip_fractions, gradient_checkpointing)
        flat = skip.flatten(0, 1)
        if gradient_checkpointing and self.training and torch.is_grad_enabled():
            flat = checkpoint(downsample, flat, use_reentrant=False)
        else:
            flat = downsample(flat)
        fractions = F.avg_pool2d(skip_fractions.flatten(0, 1), 2, 2).unflatten(0, skip_fractions.shape[:2])
        maps = self._run_stage_grouped(stages[1], flat.unflatten(0, skip.shape[:2]), fractions,
                                       gradient_checkpointing)
        maps, fractions = list(maps.unbind(0)), list(fractions.unbind(0))
        return self._finish_two_stage(list(skip.unbind(0)), list(skip_fractions.unbind(0)),
                                      maps, fractions, self._scores(maps, fractions))

    def _finish_two_stage(self, skip_maps, skip_fractions, maps, fractions, scores):
        fused, weights, coverage = self._fuse(maps, fractions, scores, self.empty_token)
        skip_scores = [F.interpolate(score, size=skip_maps[0].shape[-2:], mode="nearest") for score in scores]
        skip, _, _ = self._fuse(skip_maps, skip_fractions, skip_scores, self.skip_empty_token)
        return [skip, fused], weights, coverage

    def _fuse(self, maps, fractions, scores, empty_token):
        # [B,G,1,h,w]; perform probability normalization in FP32.
        coverage = torch.stack(fractions, dim=1)
        available = coverage > 0
        logits = torch.stack(scores, dim=1)
        if self.mode in ("gated", "coverage"):
            logits = logits + coverage.clamp_min(1e-12).log()
        logits = logits.masked_fill(~available, float("-inf"))
        any_available = available.any(dim=1, keepdim=True)
        logits = torch.where(any_available, logits, torch.zeros_like(logits))
        weights = logits.softmax(dim=1) * available.float()
        stacked = torch.stack(maps, dim=1)
        fused = (stacked * weights.to(stacked.dtype)).sum(dim=1)
        fused = torch.where(any_available[:, 0], fused, empty_token.to(fused.dtype))
        return fused, weights.squeeze(2), coverage.squeeze(2)


class LunarEncoder(nn.Module):
    def __init__(self, variant="pico", fusion="gated", drop_path=0.0,
                 gradient_checkpointing=False, branch_mora=None):
        super().__init__()
        self.backbone = ConvNeXtV2Stages(variant, drop_path, gradient_checkpointing)
        from .config import model_defaults
        self.branch_options = model_defaults({"branch_mora": branch_mora or {}})["branch_mora"]
        stages = self.branch_options["stages"]
        if type(stages) is not int or stages not in (1, 2):
            raise ValueError("branch_mora.stages must be 1 or 2")
        fusion_stage = stages - 1 if self.branch_options["enabled"] else 0
        self.fusion = GroupedFusion(self.backbone.dims[0], fusion, self.backbone.dims[fusion_stage])
        self.dims = self.backbone.dims
        self.feature_dim = self.dims[-1]
        if self.branch_options["enabled"]:
            from .branch_mora import attach_branch_stage
            attach_branch_stage(self.backbone, self.branch_options)

    def forward(self, pixels, observed):
        branched = self.branch_options["enabled"]
        if branched and self.branch_options["stages"] == 2:
            pyramid, weights, coverage = self.fusion.forward_two_stage(
                pixels, observed, self.backbone.stages[:2], self.backbone.downsample_layers[1],
                self.backbone.gradient_checkpointing)
            result = self.backbone(pyramid[-1], start_stage=2, pyramid=pyramid)
            result["fusion_weights"] = weights
            result["coverage"] = coverage
            return result
        fused, weights, coverage = self.fusion(
            pixels, observed, self.backbone.stages[0] if branched else None,
            self.backbone.gradient_checkpointing)
        result = self.backbone(fused, start_stage=1, pyramid=[fused]) if branched else self.backbone(fused)
        result["fusion_weights"] = weights
        result["coverage"] = coverage
        return result


def build_encoder(model_cfg):
    from .config import model_defaults
    options = model_defaults(model_cfg)
    return LunarEncoder(**{k: options[k] for k in
                           ("variant", "fusion", "drop_path", "gradient_checkpointing", "branch_mora")})


class ReconstructionHead(nn.Module):
    """Light decoder from the deepest map; deliberately no shallow skip path."""
    def __init__(self, in_dim, width=64):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, width, 1)
        self.refine = nn.Sequential(Block(width), Block(width))
        self.heads = nn.ModuleDict({
            group: nn.Sequential(nn.Conv2d(width, len(indices) * 16, 1),
                                 nn.PixelShuffle(4))
            for group, indices in GROUP_INDICES.items()
        })
        self.apply(initialize_weights)

    def forward(self, deepest, output_size):
        x = self.refine(self.proj(deepest))
        x = F.interpolate(x, size=(output_size[0] // 4, output_size[1] // 4),
                          mode="bilinear", align_corners=False)
        return torch.cat([self.heads[g](x) for g in GROUPS], dim=1)


class LunarMaskedModel(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.encoder = build_encoder(model_cfg)
        self.decoder = ReconstructionHead(self.encoder.feature_dim, model_cfg["decoder_dim"])

    def forward(self, pixels, observed, return_features=False):
        encoded = self.encoder(pixels, observed)
        prediction = self.decoder(encoded["pyramid"][-1], pixels.shape[-2:])
        return (prediction, encoded) if return_features else prediction


class PyramidSegmentationHead(nn.Module):
    """Small top-down pyramid decoder for two-class dense prediction."""
    def __init__(self, dims, width=64, classes=2):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(d, width, 1) for d in dims])
        self.refine = nn.ModuleList([nn.Conv2d(width, width, 3, padding=1) for _ in dims])
        self.output = nn.Conv2d(width, classes, 1)
        self.apply(initialize_weights)

    def forward(self, pyramid, output_size):
        x = self.lateral[-1](pyramid[-1])
        x = F.gelu(self.refine[-1](x))
        for i in range(len(pyramid) - 2, -1, -1):
            x = F.interpolate(x, size=pyramid[i].shape[-2:], mode="bilinear", align_corners=False)
            x = F.gelu(self.refine[i](x + self.lateral[i](pyramid[i])))
        return F.interpolate(self.output(x), size=output_size,
                             mode="bilinear", align_corners=False)


def make_task_head(encoder, model_cfg, task, feature_dropout):
    """Build a fresh dense or patch-level head without changing the encoder."""
    if task in ("mare", "craters"):
        return PyramidSegmentationHead(encoder.dims, model_cfg["decoder_dim"])
    dim = encoder.feature_dim
    return nn.Sequential(
        nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(),
        nn.Dropout(feature_dropout), nn.Linear(dim // 2, TASK_OUTPUTS[task]))


class LunarTaskModel(nn.Module):
    """A 'linear' run freezes the encoder and trains this MLP / segmentation head."""
    def __init__(self, model_cfg, task, mode="linear", feature_dropout=0.1):
        super().__init__()
        self.encoder = build_encoder(model_cfg)
        self.task = task
        self.frozen = mode == "linear"
        self.head = make_task_head(self.encoder, model_cfg, task, feature_dropout)
        if self.frozen:
            self.encoder.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.encoder.eval()
        return self

    def forward(self, pixels, observed, return_features=False):
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen):
            encoded = self.encoder(pixels, observed)
        if self.task in ("mare", "craters"):
            prediction = self.head(encoded["pyramid"], pixels.shape[-2:])
        else:
            prediction = self.head(encoded["global"])
        return (prediction, encoded) if return_features else prediction
