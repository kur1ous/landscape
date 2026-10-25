import pytest
import torch
from torch import nn

from landscape import along_direction, check_direction, normalize_direction, plane_slice, random_direction


def conv_bn_net():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 5, 3),
        nn.BatchNorm2d(5),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(5 * 6 * 6, 4),
    )


def filter_norms(t):
    return t.reshape(t.shape[0], -1).norm(dim=1)


def test_filter_norms_match_weights():
    model = conv_bn_net()
    d = random_direction(model, seed=1)
    for (name, p), di in zip(model.named_parameters(), d):
        if p.dim() > 1:
            torch.testing.assert_close(filter_norms(di), filter_norms(p.detach()), msg=name)


def test_bias_and_bn_are_zeroed_by_default():
    model = conv_bn_net()
    d = random_direction(model, seed=1)
    seen = 0
    for (name, p), di in zip(model.named_parameters(), d):
        if p.dim() <= 1:
            assert torch.count_nonzero(di) == 0, name
            seen += 1
    assert seen == 4  # conv bias, bn scale, bn shift, linear bias


def test_bias_bn_keep_copies_the_weights():
    model = conv_bn_net()
    d = random_direction(model, seed=1, bias_bn="keep")
    for p, di in zip(model.parameters(), d):
        if p.dim() <= 1:
            torch.testing.assert_close(di, p.detach())


def test_bad_bias_bn_option():
    with pytest.raises(ValueError):
        random_direction(conv_bn_net(), bias_bn="ignore")


def test_seed_controls_direction():
    model = conv_bn_net()
    a, b, c = (random_direction(model, seed=s) for s in (3, 3, 4))
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert not all(torch.equal(x, y) for x, y in zip(a, c))


def test_does_not_touch_model_or_global_rng():
    model = conv_bn_net()
    before = [p.clone() for p in model.parameters()]
    torch.manual_seed(123)
    expected = torch.rand(3)
    torch.manual_seed(123)
    random_direction(model, seed=9)
    assert torch.equal(torch.rand(3), expected)
    assert all(torch.equal(p, q) for p, q in zip(model.parameters(), before))


def test_zero_filter_stays_zero():
    w = torch.ones(3, 4)
    w[1] = 0
    (d,) = normalize_direction([torch.randn(3, 4)], [w])
    assert torch.count_nonzero(d[1]) == 0
    assert d[0].norm() == pytest.approx(w[0].norm())


def test_mismatched_shapes_rejected():
    with pytest.raises(ValueError):
        normalize_direction([torch.zeros(2, 2)], [torch.zeros(3, 2)])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_check_direction_accepts_matching_parameters(dtype):
    model = conv_bn_net().to(dtype=dtype)
    direction = random_direction(model, seed=1)
    assert check_direction(model, tuple(direction)) is None


@pytest.mark.parametrize(
    "problem, message",
    [
        ("missing", r"missing parameter '0.bias' at index 1: expected 2 tensors, got 1"),
        ("extra", r"extra tensor at index 2 with no matching parameter: expected 2 tensors, got 3"),
        ("shape", r"parameter '0.weight' has shape \(1, 4\); expected \(2, 2\)"),
        ("dtype", r"parameter '0.bias' has dtype torch.float64; expected torch.float32"),
    ],
)
@pytest.mark.parametrize("entrypoint", ["check", "curve", "plane_x", "plane_y"])
def test_bad_directions_rejected_before_reading_loader(problem, message, entrypoint):
    model = nn.Sequential(nn.Linear(2, 2))
    valid = random_direction(model, seed=0)
    direction = list(valid)
    if problem == "missing":
        direction.pop()
    elif problem == "extra":
        direction.append(torch.empty(0))
    elif problem == "shape":
        direction[0] = direction[0].reshape(1, 4)
    else:
        direction[1] = direction[1].double()

    batches_read = []

    def loader():
        batches_read.append(True)
        yield torch.ones(2, 2), torch.zeros(2, 2)

    with pytest.raises(ValueError, match=message):
        if entrypoint == "check":
            check_direction(model, direction)
        elif entrypoint == "curve":
            along_direction(model, nn.MSELoss(), loader(), direction, steps=2)
        else:
            dx, dy = (direction, valid) if entrypoint == "plane_x" else (valid, direction)
            plane_slice(model, nn.MSELoss(), loader(), dx, dy, steps=2)

    assert batches_read == []


def test_check_direction_names_first_bad_parameter():
    model = nn.Sequential(nn.Linear(2, 2))
    direction = [torch.zeros(4), torch.zeros(3)]
    with pytest.raises(ValueError, match="parameter '0.weight' has shape"):
        check_direction(model, direction)
