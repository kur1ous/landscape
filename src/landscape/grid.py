"""Two-dimensional slices of the loss surface."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .directions import check_direction
from .evaluate import Evaluator, LossFn, flatten


@dataclass
class Surface:
    alphas: np.ndarray  # x axis, length nx
    betas: np.ndarray  # y axis, length ny
    losses: np.ndarray  # shape (ny, nx), losses[j, i] at (alphas[i], betas[j])
    metric: np.ndarray | None = None  # same shape as losses


def plane_slice(
    model: nn.Module,
    loss_fn: LossFn,
    loader,
    dx,
    dy,
    xspan: tuple[float, float] = (-1.0, 1.0),
    yspan: tuple[float, float] = (-1.0, 1.0),
    steps: int | tuple[int, int] = 41,
    chunk: int = 64,
    vectorize: bool = True,
    *,
    metric_fn: LossFn | None = None,
) -> Surface:
    """Loss of theta + alpha * dx + beta * dy on a regular grid.

    theta is the model's current weights. Points are evaluated `chunk` at a
    time so the (points x n_weights) matrix never has to exist in full.
    `chunk` must be at least 1.
    """
    nx, ny = (steps, steps) if isinstance(steps, int) else steps
    if min(nx, ny) < 2:
        raise ValueError("steps must be at least 2")
    if chunk < 1:
        raise ValueError("chunk must be at least 1")
    check_direction(model, dx)
    check_direction(model, dy)
    ev = Evaluator(model, loss_fn, loader, metric_fn=metric_fn)
    alphas = np.linspace(*xspan, nx)
    betas = np.linspace(*yspan, ny)
    a, b = np.meshgrid(alphas, betas)  # (ny, nx)
    a, b = a.ravel(), b.ravel()
    vx = flatten(dx).to(ev.device)
    vy = flatten(dy).to(ev.device)

    flat = np.empty(a.size)
    metric = None if metric_fn is None else np.empty(a.size)
    for start in range(0, a.size, chunk):
        sl = slice(start, start + chunk)
        ca = torch.as_tensor(a[sl], dtype=vx.dtype, device=ev.device)
        cb = torch.as_tensor(b[sl], dtype=vx.dtype, device=ev.device)
        thetas = ev.theta0[None, :] + ca[:, None] * vx[None, :] + cb[:, None] * vy[None, :]
        losses, values = ev.evaluate(thetas, vectorize=vectorize)
        flat[sl] = losses.numpy()
        if metric is not None:
            metric[sl] = values.numpy()
    return Surface(alphas, betas, flat.reshape(ny, nx),
                   None if metric is None else metric.reshape(ny, nx))
