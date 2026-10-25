"""Random directions in weight space, normalized as in Li et al. (2018).

A direction is a list of tensors aligned with `model.parameters()`. Each
filter of the direction (a slice along dim 0: a conv kernel, or one row of a
linear weight) is rescaled to the norm of the matching filter of the weights,
so that a step of size alpha means the same relative change in every layer
regardless of how the network is parametrized.
"""

from collections.abc import Sequence

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
