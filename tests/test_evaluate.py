import warnings

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from landscape import Evaluator, check_direction, random_direction
from landscape.evaluate import flatten


class TiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(3, 2)
        self.output = nn.Linear(2, 3, bias=False)
        self.output.weight = self.embedding.weight

    def forward(self, x):
        return self.output(self.embedding(x))


@pytest.fixture(params=["tied", "frozen"])
def parameter_case(request):
    if request.param == "tied":
        model = TiedModel()
        x = torch.tensor([0, 2, 1, 0, 2])
        y = torch.zeros(5, 3)
        names = ["embedding.weight"]

        def predict(theta):
            weight = theta.reshape(3, 2)
            return F.linear(F.embedding(x, weight), weight)

    else:
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
        model[0].requires_grad_(False)
        x = torch.tensor([[1., 2.], [0., 1.], [-1., 2.], [2., 0.], [1., 1.]])
        y = torch.zeros(5, 1)
        names = ["0.weight", "0.bias", "1.weight", "1.bias"]

        def predict(theta):
            hidden = F.linear(x, theta[:4].reshape(2, 2), theta[4:6])
            return F.linear(hidden, theta[6:8].reshape(1, 2), theta[8:])

    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.arange(1, p.numel() + 1).reshape_as(p) / 10)
    return model, x, y, names, predict


@pytest.mark.parametrize("bias_bn", ["zero", "keep"])
def test_direction_counts_shared_weights_once_and_includes_frozen(parameter_case, bias_bn):
    model, x, y, names, _ = parameter_case
    params = list(model.parameters())
    evaluator = Evaluator(model, nn.MSELoss(), [(x, y)])
    direction = random_direction(model, seed=7, bias_bn=bias_bn)

    assert [name for name, _ in model.named_parameters()] == names
    assert len(direction) == len(params) == len(names)
    assert all(a is b for a, b in zip(evaluator.params, params))
    assert evaluator.theta0.numel() == flatten(direction).numel() == sum(p.numel() for p in params)
    check_direction(model, direction)
    for p, d in zip(params, direction):
        assert d.shape == p.shape
        if p.ndim > 1:
            torch.testing.assert_close(d.norm(dim=1), p.detach().norm(dim=1))
        else:
            expected = torch.zeros_like(p) if bias_bn == "zero" else p.detach()
            torch.testing.assert_close(d, expected)


@pytest.mark.parametrize("vectorize", [True, False])
def test_losses_with_tied_or_frozen_parameters(parameter_case, vectorize):
    model, x, y, _, predict = parameter_case
    list(model.children())[0].eval()
    params = list(model.parameters())
    flags = [p.requires_grad for p in params]
    modes = [m.training for m in model.modules()]
    before = {name: value.clone() for name, value in model.state_dict().items()}
    for p in params:
        if p.requires_grad:
            p.grad = torch.ones_like(p)
    grads = [None if p.grad is None else p.grad.clone() for p in params]
    evaluator = Evaluator(model, nn.MSELoss(), [(x[:2], y[:2]), (x[2:], y[2:])])
    direction = flatten(random_direction(model, seed=7, bias_bn="keep"))
    # Move only the first weight: it is shared in one model and frozen in the other.
    changed = evaluator.theta0.clone()
    changed[:params[0].numel()] += 0.3
    thetas = torch.stack([evaluator.theta0, evaluator.theta0 + 0.4 * direction, changed])
    expected = torch.stack([F.mse_loss(predict(theta), y) for theta in thetas]).double()
    assert not torch.isclose(expected[0], expected[2])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = evaluator.losses(thetas, vectorize=vectorize)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    assert all(a is b for a, b in zip(model.parameters(), params))
    assert [p.requires_grad for p in params] == flags
    assert [m.training for m in model.modules()] == modes
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for p, grad in zip(params, grads):
        if grad is None:
            assert p.grad is None
        else:
            torch.testing.assert_close(p.grad, grad, rtol=0, atol=0)
    if isinstance(model, TiedModel):
        assert model.embedding.weight is model.output.weight


@pytest.mark.parametrize("vectorize", [True, False])
def test_empty_batches_do_not_poison_mean_loss(vectorize):
    model = nn.Linear(1, 1, bias=False)
    x = torch.tensor([[1.], [2.], [3.]])
    y = torch.zeros_like(x)
    empty = (x[:0], y[:0])
    batches = [empty, (x[:1], y[:1]), empty, (x[1:], y[1:]), empty]

    def loss_fn(output, target):
        assert len(target) > 0
        return F.mse_loss(output, target)

    evaluator = Evaluator(model, loss_fn, (batch for batch in batches))
    thetas = torch.tensor([[2.], [3.]])
    expected = torch.tensor([4., 9.], dtype=torch.float64) * (14 / 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        actual = evaluator.losses(thetas, vectorize=vectorize)
    assert evaluator.n == 3
    torch.testing.assert_close(actual, expected)


def test_loader_with_only_empty_batches_is_rejected():
    model = nn.Linear(1, 1)
    empty = torch.empty(0, 1)
    loader = ((empty, empty) for _ in range(2))
    with pytest.raises(ValueError, match="loader yielded no examples"):
        Evaluator(model, nn.MSELoss(), loader)
