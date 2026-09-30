"""Loss of a model at arbitrary points in weight space."""

from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager

import torch
from torch import nn

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def flatten(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1) for t in tensors])


class Evaluator:
    """Evaluates `loss_fn` over a fixed dataset at flat weight vectors.

    The loader is read once and kept in memory, so shuffling and augmentation
    are frozen: every point on a landscape sees exactly the same data. The
    model is evaluated in eval mode (BatchNorm uses running statistics,
    dropout is off) and is put back exactly as it was afterwards.

    `loss_fn(output, target)` must return a mean over the batch.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: LossFn,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device | str | None = None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.params = list(model.parameters())
        self.device = torch.device(device) if device else self.params[0].device
        self.batches = [(x.to(self.device), y.to(self.device)) for x, y in loader]
        if not self.batches:
            raise ValueError("loader yielded no batches")
        self.n = sum(len(y) for _, y in self.batches)
        self.theta0 = flatten(self.params)

    @contextmanager
    def _eval_state(self):
        saved_params = [p.detach().clone() for p in self.params]
        saved_modes = [(m, m.training) for m in self.model.modules()]
        self.model.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            with torch.no_grad():
                for p, q in zip(self.params, saved_params):
                    p.copy_(q)
            for m, mode in saved_modes:
                m.training = mode

    def _assign(self, theta: torch.Tensor) -> None:
        if theta.numel() != self.theta0.numel():
            raise ValueError(f"expected {self.theta0.numel()} weights, got {theta.numel()}")
        offset = 0
        for p in self.params:
            k = p.numel()
            p.copy_(theta[offset : offset + k].view_as(p))
            offset += k

    def _loss_here(self) -> float:
        total = 0.0
        for x, y in self.batches:
            total += self.loss_fn(self.model(x), y).item() * len(y)
        return total / self.n

    def loss(self, theta: torch.Tensor) -> float:
        return float(self.losses(theta.unsqueeze(0))[0])

    def losses(self, thetas: torch.Tensor) -> torch.Tensor:
        """Loss at each row of `thetas`, shape (points, n_weights)."""
        out = torch.empty(len(thetas), dtype=torch.float64)
        with self._eval_state():
            for i, theta in enumerate(thetas):
                self._assign(theta.to(self.device))
                out[i] = self._loss_here()
        return out
