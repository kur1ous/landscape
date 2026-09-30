"""One-dimensional slices: along a direction, or between two checkpoints."""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .evaluate import Evaluator, LossFn, flatten


@dataclass
class Curve:
    alphas: np.ndarray
    losses: np.ndarray
    marks: tuple[float, ...] = (0.0,)  # alphas that correspond to real weights
    mark_labels: tuple[str, ...] = ()
    xlabel: str = "α"


def _alphas(span: tuple[float, float], steps: int) -> np.ndarray:
    if steps < 2:
        raise ValueError("steps must be at least 2")
    return np.linspace(span[0], span[1], steps)


def along_direction(
    model: nn.Module,
    loss_fn: LossFn,
    loader,
    direction,
    span: tuple[float, float] = (-1.0, 1.0),
    steps: int = 51,
) -> Curve:
    """Loss of theta + alpha * direction, with theta the model's current weights."""
    ev = Evaluator(model, loss_fn, loader)
    alphas = _alphas(span, steps)
    d = flatten(direction).to(ev.device)
    thetas = ev.theta0[None, :] + torch.as_tensor(alphas, dtype=d.dtype)[:, None] * d[None, :]
    return Curve(alphas, ev.losses(thetas).numpy())


def between_checkpoints(
    model: nn.Module,
    loss_fn: LossFn,
    loader,
    start: Mapping[str, torch.Tensor],
    end: Mapping[str, torch.Tensor],
    span: tuple[float, float] = (0.0, 1.0),
    steps: int = 51,
) -> Curve:
    """Loss of (1 - alpha) * start + alpha * end for two state dicts.

    Only parameters are interpolated. Buffers (BatchNorm running statistics)
    stay as they are in `model`, so load the checkpoint you want them from
    before calling.
    """
    ev = Evaluator(model, loss_fn, loader)
    names = [n for n, _ in model.named_parameters()]
    a = flatten([start[n] for n in names]).to(ev.device)
    b = flatten([end[n] for n in names]).to(ev.device)
    alphas = torch.as_tensor(_alphas(span, steps), dtype=a.dtype)[:, None]
    thetas = (1 - alphas) * a[None, :] + alphas * b[None, :]
    return Curve(
        alphas.squeeze(1).numpy(),
        ev.losses(thetas).numpy(),
        marks=(0.0, 1.0),
        mark_labels=("start", "end"),
        xlabel="interpolation coefficient α",
    )
