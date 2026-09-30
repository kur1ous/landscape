import importlib.util
from pathlib import Path

import torch
from torch import nn

spec = importlib.util.spec_from_file_location(
    "spirals", Path(__file__).parent.parent / "examples" / "spirals.py"
)
spirals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spirals)


def test_spirals_are_balanced_and_deterministic():
    x, y = spirals.make_spirals(n_per_arm=50)
    assert x.shape == (100, 2) and y.sum() == 50
    x2, _ = spirals.make_spirals(n_per_arm=50)
    assert torch.equal(x, x2)


def test_a_few_steps_of_training_beat_chance():
    torch.manual_seed(0)
    x, y = spirals.make_spirals(n_per_arm=50)
    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 2))
    assert spirals.train(model, x, y, steps=60) > 0.6
