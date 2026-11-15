import math
import torch
from torch import nn
from torch.nn import functional as F
from .schema import GROUPS


def direction(weight):
    # Column norms over the input axis; also valid for per-sample [B,out,in] weights.
    return weight / weight.norm(dim=-2, keepdim=True).clamp_min(1e-5)


def centered_update(weight, delta, magnitude, initial):
    base = direction(weight)
    return magnitude * (direction(weight + delta) - base) + (magnitude - initial) * base


class CrossGroupGram(nn.Module):
    """Rank-r cross-group interaction shared by a block's two pointwise layers.

    static:   group g sees the parameter Grams S_h^T S_h of all other groups.
    observed: group g sees S_h^T Sigma_h S_h, the second moment of group h's
              normalized block input on its supported pixels, averaged only over
              the other groups observed in the same sample. Sigma_h = I recovers
              the static form; no other observed group gives a zero update.
    exchange: S no longer modulates weights; it is the key/query basis of
              GramExchange, a per-pixel message pass between observed groups.
    """
    def __init__(self, width, rank, shared, gram_context="static"):
        super().__init__()
        self.shared = shared
        self.observed = gram_context == "observed"
        self.exchange = gram_context == "exchange"
        factor = torch.randn(len(GROUPS), width, rank) / math.sqrt(rank)
        self.S = nn.Parameter(factor, requires_grad=shared)
        self.register_buffer("initial_S", factor.detach().clone())

    @staticmethod
    def cross(factors, group):
        grams = factors.transpose(1, 2) @ factors
        other = (grams.sum(0) - grams[group]) / (len(GROUPS) - 1)
        return factors[group] @ other @ factors[group].T

    @staticmethod
    def observed_cross(factor, grams, available, group):
        weights = available.float().clone()
        weights[:, group] = 0
        count = weights.sum(1).clamp_min(1)
        other = (grams * weights[:, :, None, None]).sum(1) / count[:, None, None]
        return factor @ other @ factor.T

    def context(self, features, supports):
        """Per-sample [B,G,r,r] projected Grams (current and anchor) and [B,G] availability."""
        with torch.autocast(device_type=features[0].device.type, enabled=False):
            grams, anchors, available = [], [], []
            for group, (x, support) in enumerate(zip(features, supports)):
                support = support.permute(0, 2, 3, 1)
                x = F.layer_norm(x.float().permute(0, 2, 3, 1), x.shape[1:2])
                x = torch.where(support, x, 0.)
                count = support.flatten(1).sum(1).clamp_min(1).float()[:, None, None]
                projected = x @ self.S[group]
                anchor = x @ self.initial_S[group]
                grams.append(torch.einsum("bhwi,bhwj->bij", projected, projected) / count)
                anchors.append(torch.einsum("bhwi,bhwj->bij", anchor, anchor) / count)
                available.append(support.flatten(1).any(1))
            return torch.stack(grams, 1), torch.stack(anchors, 1), torch.stack(available, 1)

    def forward(self, group, context=None):
        # Recompute the anchor on the same device; exactly zero at initialization.
        if context is None:
            return self.cross(self.S, group) - self.cross(self.initial_S, group)
        grams, anchors, available = context
        return (self.observed_cross(self.S[group], grams, available, group)
                - self.observed_cross(self.initial_S[group], anchors, available, group))


class GroupDirectionalLinear(nn.Module):
    def __init__(self, base, bank, rank, alpha, side):
        super().__init__()
        self.base, self.alpha, self.side = base, alpha, side
        object.__setattr__(self, "bank", bank)  # registered once per block
        count, rows, columns = len(GROUPS), *base.weight.shape
        self.A = nn.Parameter(torch.randn(count, rank, columns) / math.sqrt(rank))
        self.B = nn.Parameter(torch.zeros(count, rows, rank))
        magnitude = .5 * base.weight.detach().norm(dim=0, keepdim=True)
        self.register_buffer("initial_magnitude", magnitude.clone())
        self.private_magnitude = nn.Parameter(magnitude.repeat(count, 1, 1))
        self.shared_magnitude = nn.Parameter(magnitude.repeat(count, 1, 1))

    def update_terms(self, group, context=None):
        """Private and shared directional updates; shared is [B,out,in] with an observed context."""
        weight = self.base.weight
        private = self.alpha * (self.B[group] @ self.A[group])
        if self.bank.shared and not self.bank.exchange:
            cross = self.bank(group, context)
            # pwconv1 [4d,d] and pwconv2 [d,4d] share a d-dimensional bank.
            shared = weight @ cross if self.side == "input" else cross @ weight
            shared = self.alpha * shared
        else:
            shared = torch.zeros_like(weight)
        return (centered_update(weight, private, self.private_magnitude[group], self.initial_magnitude),
                centered_update(weight, shared, self.shared_magnitude[group], self.initial_magnitude))

    def effective_weight(self, group, context=None):
        """[out,in] for the static Gram, per-sample [B,out,in] with an observed context."""
        with torch.autocast(device_type=self.base.weight.device.type, enabled=False):
            private, shared = self.update_terms(group, context)
            return self.base.weight + private + shared

    def grouped_weight(self):
        """All groups' [G,out,in] weights in one pass; no weight-modulating Gram term."""
        if self.bank.shared and not self.bank.exchange:
            raise RuntimeError("Grouped weights exclude the static/observed Gram update")
        with torch.autocast(device_type=self.base.weight.device.type, enabled=False):
            weight = self.base.weight
            base = direction(weight)
            private = self.alpha * (self.B @ self.A)
            private = (self.private_magnitude * (direction(weight + private) - base)
                       + (self.private_magnitude - self.initial_magnitude) * base)
            shared = (self.shared_magnitude - self.initial_magnitude) * base
            return weight + private + shared

    def forward_grouped(self, x, groups):
        """x is [G*B,H,W,in] with group-major batch order."""
        total, h, w, _ = x.shape
        y = torch.baddbmm(self.base.bias, x.reshape(groups, total // groups * h * w, -1),
                          self.grouped_weight().transpose(1, 2))
        return y.reshape(total, h, w, -1)

    def forward(self, x, group, context=None):
        weight = self.effective_weight(group, context)
        if weight.ndim == 2:
            return F.linear(x, weight, self.base.bias)
        b, h, w, _ = x.shape
        y = torch.baddbmm(self.base.bias, x.reshape(b, h * w, -1), weight.transpose(1, 2))
        return y.reshape(b, h, w, -1)


class GramExchange(nn.Module):
    """Per-pixel message passing between modality streams, weighted by Gram affinity.

    For group g at pixel p, the affinity to another group h is the rank-r Gram
    entry <S_g^T x_g(p), S_h^T x_h(p)> of the normalized features, reusing the
    block bank's factors. Only groups observed at p are sources; a pixel with no
    other observed group receives no message, and unobserved targets stay zero.
    The output projection is zero-initialized, so training starts from the
    independent-stream model.
    """
    def __init__(self, bank, width):
        super().__init__()
        object.__setattr__(self, "bank", bank)  # registered by the owning block
        count, rank = len(GROUPS), bank.S.shape[-1]
        self.value_rank = max(rank, width // 4)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.value = nn.Parameter(torch.randn(count, width, self.value_rank) / math.sqrt(width))
        self.output = nn.Parameter(torch.zeros(count, self.value_rank, width))

    def stacked(self, features, supports):
        """features [G,B,C,H,W], supports [G,B,1,H,W] bool -> updated [G,B,C,H,W]."""
        groups, width, rank = features.shape[0], features.shape[2], self.bank.S.shape[-1]
        x = self.norm(features.permute(1, 0, 3, 4, 2))  # B,G,H,W,C
        available = supports[:, :, 0].permute(1, 2, 3, 0)  # B,H,W,G
        query = torch.einsum("bghwc,gcr->bghwr", x, self.bank.S.to(x.dtype)) / math.sqrt(width)
        logits = torch.einsum("bghwr,bkhwr->bhwgk", query, query).float() * math.sqrt(rank)
        eye = torch.eye(groups, dtype=torch.bool, device=x.device)
        allowed = available[..., None, :] & ~eye
        logits = logits.masked_fill(~allowed, float("-inf"))
        has_source = allowed.any(-1, keepdim=True)
        weights = torch.where(has_source, logits, 0.).softmax(-1) * allowed
        values = torch.einsum("bghwc,gcm->bghwm", x, self.value.to(x.dtype))
        message = torch.einsum("bhwgk,bkhwm->bghwm", weights.to(values.dtype), values)
        update = torch.einsum("bghwm,gmc->gbchw", message, self.output.to(message.dtype))
        return torch.where(supports, features + update.to(features.dtype), 0.)

    def forward_grouped(self, x, support, groups):
        """x [G*B,C,H,W], support [G*B,1,H,W], group-major batch order."""
        shape = (groups, x.shape[0] // groups)
        return self.stacked(x.unflatten(0, shape), support.unflatten(0, shape)).flatten(0, 1)

    def forward(self, features, supports):
        return list(self.stacked(torch.stack(features), torch.stack(supports)).unbind(0))


class BranchedBlock(nn.Module):
    def __init__(self, base, rank, alpha, shared, gram_context="static", exchange=False):
        super().__init__()
        self.base = base
        self.bank = CrossGroupGram(base.pwconv1.in_features, rank, shared, gram_context)
        base.pwconv1 = GroupDirectionalLinear(base.pwconv1, self.bank, rank, alpha, "input")
        base.pwconv2 = GroupDirectionalLinear(base.pwconv2, self.bank, rank, alpha, "output")
        if exchange:
            self.exchange = GramExchange(self.bank, base.pwconv1.base.in_features)

    @property
    def needs_context(self):
        return self.bank.shared and self.bank.observed

    @property
    def lockstep(self):
        return self.bank.shared and (self.bank.observed or self.bank.exchange)

    @property
    def grouped(self):
        # exchange runs (gram or private) fold the groups into the batch dimension.
        return self.bank.exchange

    def forward_grouped(self, x, support, groups):
        """All groups at once: x [G*B,C,H,W]; same math as `forward` per group."""
        x = torch.where(support, x, 0.)
        residual = x
        y = self.base.dwconv(x).permute(0, 2, 3, 1)
        y = self.base.act(self.base.pwconv1.forward_grouped(self.base.norm(y), groups))
        y = torch.where(support.permute(0, 2, 3, 1), y, 0.)
        y = self.base.pwconv2.forward_grouped(self.base.grn(y), groups).permute(0, 3, 1, 2)
        return torch.where(support, residual + self.base.drop_path(y), 0.)

    def forward(self, x, group, support, context=None):
        # A missing observation is never replaced by a stem/adapter bias signal.
        x = torch.where(support, x, 0.)
        residual = x
        y = self.base.dwconv(x).permute(0, 2, 3, 1)
        y = self.base.act(self.base.pwconv1(self.base.norm(y), group, context))
        y = torch.where(support.permute(0, 2, 3, 1), y, 0.)
        y = self.base.pwconv2(self.base.grn(y), group, context).permute(0, 3, 1, 2)
        return torch.where(support, residual + self.base.drop_path(y), 0.)


def attach_branch_stage(backbone, options):
    count = options.get("stages", 1)
    if type(count) is not int or count not in (1, 2):
        raise ValueError("branch_mora.stages must be 1 or 2")
    if type(options["rank"]) is not int or not 1 <= options["rank"] <= min(backbone.dims[:count]):
        raise ValueError("branch_mora.rank must be within every branched stage's width")
    for index in range(count):
        if any(isinstance(block, BranchedBlock) for block in backbone.stages[index]):
            raise ValueError("Branch stage is already attached")
        context = options.get("gram_context", "static")
        blocks = list(backbone.stages[index])
        # One exchange closes each branched stage, after its last block.
        backbone.stages[index] = nn.Sequential(*[
            BranchedBlock(block, options["rank"], options["alpha"], options["shared"], context,
                          exchange=options["shared"] and context == "exchange" and j == len(blocks) - 1)
            for j, block in enumerate(blocks)])


def gradient_report(model):
    report = {"base_stage1": 0., "base_later_stages": 0., "private_B": 0., "shared_S": 0.}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().norm())
        if not math.isfinite(value):
            raise FloatingPointError(f"Nonfinite gradient: {name}")
        if name.endswith(".bank.S"):
            key = "shared_S"
        elif name.endswith(".B"):
            key = "private_B"
        elif "backbone.stages.0." in name and (".base.weight" in name or ".dwconv.weight" in name):
            key = "base_stage1"
        elif "backbone.stages." in name and "backbone.stages.0." not in name:
            key = "base_later_stages"
        else:
            continue
        report[key] += value
    return report


@torch.no_grad()
def diagnostics(model):
    """Gram factor drift and, per layer, mean update norms relative to the base weight.

    The shared norm uses the identity feature context (Sigma_h = I, all groups
    observed), i.e. the static Gram; a near-zero ratio means the Gram is idle.
    """
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, CrossGroupGram):
            result[name] = {"shared": module.shared, "observed": module.observed,
                            "exchange": module.exchange,
                            "factor_change_l2": float((module.S - module.initial_S).norm())}
        elif isinstance(module, GramExchange):
            # Zero at initialization; growth shows the streams are exchanging information.
            result[name] = {"output_l2": float(module.output.norm()),
                            "value_l2": float(module.value.norm())}
        elif isinstance(module, GroupDirectionalLinear):
            base = float(module.base.weight.norm().clamp_min(1e-12))
            terms = [module.update_terms(group) for group in range(len(GROUPS))]
            private = sum(float(p.norm()) for p, _ in terms) / len(terms) / base
            shared = sum(float(s.norm()) for _, s in terms) / len(terms) / base
            result[name] = {"private_rel": private, "shared_rel": shared,
                            "shared_to_private": shared / max(private, 1e-12)}
    return result
