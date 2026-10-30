"""Random directions in weight space, normalized as in Li et al. (2018).

A direction is a list of tensors aligned with `model.parameters()`. Each
filter of the direction (a slice along dim 0: a conv kernel, or one row of a
linear weight) is rescaled to the norm of the matching filter of the weights,
so that a step of size alpha means the same relative change in every layer
regardless of how the network is parametrized.
"""

from collections.abc import Mapping, Sequence

import torch
from torch import nn

EPS = 1e-10


def check_direction(model: nn.Module, direction: Sequence[torch.Tensor]) -> None:
    params = list(model.named_parameters())
    for i, (name, param) in enumerate(params):
        if i >= len(direction):
            raise ValueError(
                f"direction is missing parameter {name!r} at index {i}: "
                f"expected {len(params)} tensors, got {len(direction)}"
            )
        tensor = direction[i]
        if tensor.shape != param.shape:
            raise ValueError(
                f"direction for parameter {name!r} has shape {tuple(tensor.shape)}; "
                f"expected {tuple(param.shape)}"
            )
        if tensor.dtype != param.dtype:
            raise ValueError(
                f"direction for parameter {name!r} has dtype {tensor.dtype}; "
                f"expected {param.dtype}"
            )
    if len(direction) > len(params):
        raise ValueError(
            f"direction has an extra tensor at index {len(params)} with no matching parameter: "
            f"expected {len(params)} tensors, got {len(direction)}"
        )


def normalize_direction(
    direction: Sequence[torch.Tensor],
    weights: Sequence[torch.Tensor],
    bias_bn: str = "zero",
) -> list[torch.Tensor]:
    """Filter-wise normalization of `direction` against `weights`.

    Tensors with fewer than two dims (biases, BatchNorm scale and shift) have
    no filters. With bias_bn="zero" they get no perturbation, which is what
    the paper does. With "keep" the direction is the weight itself, so the
    parameter is scaled by (1 + alpha) along that axis.
    """
    if bias_bn not in ("zero", "keep"):
        raise ValueError(f"bias_bn must be 'zero' or 'keep', got {bias_bn!r}")
    if len(direction) != len(weights):
        raise ValueError("direction and weights have different lengths")

    out = []
    for d, w in zip(direction, weights):
        if d.shape != w.shape:
            raise ValueError(f"shape mismatch: {tuple(d.shape)} vs {tuple(w.shape)}")
        if d.dim() <= 1:
            out.append(torch.zeros_like(w) if bias_bn == "zero" else w.detach().clone())
            continue
        flat_d = d.reshape(d.shape[0], -1)
        flat_w = w.detach().reshape(w.shape[0], -1)
        scale = flat_w.norm(dim=1, keepdim=True) / (flat_d.norm(dim=1, keepdim=True) + EPS)
        out.append((flat_d * scale).reshape(d.shape))
    return out


def random_direction(
    model: nn.Module,
    seed: int | None = None,
    bias_bn: str = "zero",
) -> list[torch.Tensor]:
    """Gaussian direction with the same shapes as the parameters, filter-normalized.

    Sampling uses its own generator, so it never touches the global RNG.
    """
    params = [p.detach() for p in model.parameters()]
    gen = torch.Generator(device="cpu")
    if seed is None:
        gen.seed()
    else:
        gen.manual_seed(seed)
    raw = [torch.randn(p.shape, generator=gen, dtype=p.dtype).to(p.device) for p in params]
    return normalize_direction(raw, params, bias_bn)


def direction_between(
    state_a: Mapping[str, torch.Tensor],
    state_b: Mapping[str, torch.Tensor],
    model: nn.Module,
) -> list[torch.Tensor]:
    """Return b - a, filter-normalized to the model's current weights.

    Buffers are ignored, tied parameters appear once, and biases/BN are zeroed.
    Unchanged filters stay zero. Neither checkpoint nor the model is modified.
    """
    params = list(model.named_parameters())
    states = []
    for label, state in (("state_a", state_a), ("state_b", state_b)):
        for name, _ in params:
            if name not in state:
                raise ValueError(f"{label} checkpoint is missing parameter {name!r}")
        tensors = [state[name] for name, _ in params]
        check_direction(model, tensors)
        states.append([t.detach().to(p.device) for t, (_, p) in zip(tensors, params)])
    raw = [b - a for a, b in zip(*states)]
    return normalize_direction(raw, [p for _, p in params])


def _orthogonal_component(direction, against):
    if len(direction) != len(against):
        raise ValueError("direction and against have different lengths")
    for i, (d, a) in enumerate(zip(direction, against)):
        if d.shape != a.shape or d.dtype != a.dtype or d.device != a.device:
            raise ValueError(f"direction and against tensor {i} must match shape, dtype and device")
    if not direction:
        return []
    d = torch.cat([t.detach().reshape(-1) for t in direction])
    a = torch.cat([t.detach().reshape(-1) for t in against])
    if not d.is_floating_point() or not torch.isfinite(d).all() or not torch.isfinite(a).all():
        raise ValueError("directions must contain finite floating-point values")
    # Scale first so the reference norm cannot overflow or underflow.
    unit = a / a.abs().max() if torch.count_nonzero(a) else torch.zeros_like(a)
    if torch.count_nonzero(unit):
        unit = unit / unit.norm()
    residual = d - torch.dot(d, unit) * unit
    if residual.norm() <= 4 * torch.finfo(d.dtype).eps * d.norm():
        residual = torch.zeros_like(d)
    return [r.reshape(t.shape) for r, t in zip(residual.split([t.numel() for t in direction]), direction)]


def orthogonalize(
    direction: Sequence[torch.Tensor], against: Sequence[torch.Tensor],
) -> list[torch.Tensor]:
    """Project off `against` in flat space, then restore `direction` filter norms.

    Per-filter rescaling can destroy flat orthogonality. Scalars and 1D tensors
    are each treated as one filter. Collapsed filters (including parallel
    directions within floating-point tolerance) stay zero; a zero `against`
    leaves the direction unchanged. Inputs are never modified.
    """
    residual = _orthogonal_component(direction, against)
    if not any(torch.count_nonzero(t) for t in against):
        return [t.detach().clone() for t in direction]
    out = []
    for raw, original in zip(residual, direction):
        rows = raw.reshape(raw.shape[0], -1) if raw.dim() > 1 else raw.reshape(1, -1)
        target = original.detach().reshape(rows.shape).norm(dim=1, keepdim=True)
        norm = rows.norm(dim=1, keepdim=True)
        collapsed = norm <= 4 * torch.finfo(raw.dtype).eps * target
        scale = target / norm.masked_fill(collapsed, 1)
        out.append((rows * scale).masked_fill(collapsed, 0).reshape(original.shape))
    return out
