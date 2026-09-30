"""Train a small MLP on two interleaved spirals, then draw its loss landscape.

    python examples/spirals.py            # writes outputs/spirals.png
    python examples/spirals.py --out docs # somewhere else

Runs in well under a minute on a laptop CPU.
"""

import argparse
import copy
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

import landscape as ls


def make_spirals(n_per_arm=200, turns=1.5, noise=0.08, seed=0):
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0.15, 1.0, n_per_arm)
    angle = turns * 2 * math.pi * t
    arm = torch.stack([t * torch.cos(angle), t * torch.sin(angle)], dim=1)
    x = torch.cat([arm, -arm]) + noise * torch.randn(2 * n_per_arm, 2, generator=g)
    y = torch.cat([torch.zeros(n_per_arm), torch.ones(n_per_arm)]).long()
    return x, y


def train(model, x, y, steps=800, lr=0.02):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs", type=Path)
    args = ap.parse_args()

    torch.manual_seed(0)
    x, y = make_spirals()
    model = nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2))
    init = copy.deepcopy(model.state_dict())
    acc = train(model, x, y)
    trained = copy.deepcopy(model.state_dict())
    print(f"train accuracy {acc:.3f}")

    loader = [(x, y)]
    loss_fn = nn.CrossEntropyLoss()

    path = ls.between_checkpoints(model, loss_fn, loader, init, trained, span=(-0.25, 1.25))
    surface = ls.plane_slice(
        model,
        loss_fn,
        loader,
        ls.random_direction(model, seed=1),
        ls.random_direction(model, seed=2),
        xspan=(-0.6, 0.6),
        yspan=(-0.6, 0.6),
        steps=61,
    )

    with ls.style.context():
        fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1.15, 1]})
        ls.plot_curve(path, ax=a)
        a.set_title("init to trained")
        ls.plot_surface(surface, ax=b)
        b.set_title("filter-normalized random plane")

    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "spirals.png"
    fig.savefig(dest)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
