"""Loss of a model at arbitrary points in weight space."""

import random
import warnings
from bisect import bisect_left
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager

import torch
from torch import nn
from torch.func import functional_call, vmap

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def accuracy(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean top-1 accuracy for scores (N, C) and integer class labels (N,)."""
    if output.ndim != 2 or target.ndim != 1 or output.shape[0] != target.shape[0]:
        raise ValueError("accuracy expects scores (N, C) and class labels (N,)")
    if output.shape[0] == 0 or output.shape[1] == 0:
        raise ValueError("accuracy needs at least one example and one class")
    return (output.argmax(dim=1) == target).float().mean()


def flatten(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.detach().reshape(-1) for t in tensors])


def subsample(
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]], n: int, seed: int = 0,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Sample exactly n examples without replacement, preserving loader order.

    The seed selects positions in the loader's example stream, independently
    of batch boundaries. Repeated calls need the same input order and data.
    Empty batches are omitted; n must be positive and no larger than the data.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    batches = list(loader)
    if any(len(x) != len(y) for x, y in batches):
        raise ValueError("inputs and targets must have the same batch length")
    total = sum(len(y) for _, y in batches)
    if n > total:
        raise ValueError(f"requested {n} examples, but loader yielded {total}")
    selected = sorted(random.Random(seed).sample(range(total), n))
    result = []
    offset = start = 0
    for x, y in batches:
        end = bisect_left(selected, offset + len(y), lo=start)
        if end > start:
            indices = [i - offset for i in selected[start:end]]
            result.append((x[indices], y[indices]))
        offset += len(y)
        start = end
    return result


class Evaluator:
    """Evaluates `loss_fn` over a fixed dataset at flat weight vectors.

    The loader is read once and kept in memory, so shuffling and augmentation
    are frozen: every point on a landscape sees exactly the same data. The
    model is evaluated in eval mode (BatchNorm uses running statistics,
    dropout is off) and is put back exactly as it was afterwards.

    `loss_fn(output, target)` and optional `metric_fn(output, target)` must
    return scalar tensors containing means over the batch. Empty batches
    are skipped; input and target batch lengths must match, and a loader
    with no examples is rejected. `max_examples` caps
    the cached dataset using `subsample` and `seed` (default 0).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: LossFn,
        loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        device: torch.device | str | None = None,
        *,
        max_examples: int | None = None,
        seed: int = 0,
        metric_fn: LossFn | None = None,
    ):
        if max_examples is not None and (
            isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples <= 0
        ):
            raise ValueError("max_examples must be a positive integer")
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self._functions = (loss_fn,) if metric_fn is None else (loss_fn, metric_fn)
        self.params = list(model.parameters())
        self.device = torch.device(device) if device else self.params[0].device
        self.batches = list(loader)
        if not self.batches:
            raise ValueError("loader yielded no batches")
        if any(len(x) != len(y) for x, y in self.batches):
            raise ValueError("inputs and targets must have the same batch length")
        self.batches = [(x, y) for x, y in self.batches if len(y) > 0]
        if not self.batches:
            raise ValueError("loader yielded no examples")
        self.n = sum(len(y) for _, y in self.batches)
        if max_examples is not None:
            self.batches = subsample(self.batches, min(max_examples, self.n), seed)
            self.n = min(max_examples, self.n)
        self.batches = [(x.to(self.device), y.to(self.device)) for x, y in self.batches]
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

    def _evaluate_here(self) -> tuple[float, ...]:
        totals = [0.0] * len(self._functions)
        for x, y in self.batches:
            output = self.model(x)
            for i, fn in enumerate(self._functions):
                totals[i] += fn(output, y).item() * len(y)
        return tuple(total / self.n for total in totals)

    def loss(self, theta: torch.Tensor) -> float:
        return float(self.losses(theta.unsqueeze(0))[0])

    def _evaluate_vmapped(self, thetas: torch.Tensor) -> tuple[torch.Tensor, ...]:
        names = [n for n, _ in self.model.named_parameters()]
        shapes = [p.shape for p in self.params]
        sizes = [p.numel() for p in self.params]
        buffers = {n: b.detach() for n, b in self.model.named_buffers()}

        def one(theta, x, y):
            params = {n: t.view(s) for n, t, s in zip(names, theta.split(sizes), shapes)}
            output = functional_call(self.model, (params, buffers), (x,))
            return tuple(fn(output, y) for fn in self._functions)

        batched = vmap(one, in_dims=(0, None, None))
        totals = tuple(torch.zeros(len(thetas), dtype=torch.float64) for _ in self._functions)
        for x, y in self.batches:
            for total, values in zip(totals, batched(thetas, x, y)):
                total += values.double().cpu() * len(y)
        return tuple(total / self.n for total in totals)

    def losses(self, thetas: torch.Tensor, vectorize: bool = True) -> torch.Tensor:
        """Loss at each row of `thetas`; see `evaluate` for vectorization."""
        return self.evaluate(thetas, vectorize=vectorize)[0]

    def evaluate(
        self, thetas: torch.Tensor, vectorize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return (losses, metrics) at rows of `thetas`, shape (points, n_weights).

        Results are CPU float64 tensors of shape (points,); metrics is None
        without `metric_fn`. Both means are weighted by batch size and share
        each forward pass.

        With vectorize=True all rows go through the model at once via vmap, so
        the cost in memory is (points x activations); pass smaller blocks for
        big models. A callback or model that vmap cannot trace falls back to the
        plain per-point loop with a warning.
        """
        thetas = thetas.to(self.device)
        if thetas.shape[1] != self.theta0.numel():
            raise ValueError(f"expected {self.theta0.numel()} weights, got {thetas.shape[1]}")
        with self._eval_state():
            if vectorize:
                try:
                    values = self._evaluate_vmapped(thetas)
                    return values[0], values[1] if self.metric_fn is not None else None
                except (RuntimeError, ValueError, TypeError) as err:
                    warnings.warn(f"vmap failed ({err}); evaluating point by point", RuntimeWarning)
            out = tuple(torch.empty(len(thetas), dtype=torch.float64) for _ in self._functions)
            for i, theta in enumerate(thetas):
                self._assign(theta)
                for values, value in zip(out, self._evaluate_here()):
                    values[i] = value
            return out[0], out[1] if self.metric_fn is not None else None
