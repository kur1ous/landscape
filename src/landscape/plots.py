import matplotlib.pyplot as plt
import numpy as np

from . import style
from .interpolate import Curve


def plot_curve(curves, labels=None, ax=None, logy=False):
    """Draw one Curve or several on shared axes. Returns the Axes."""
    curves = [curves] if isinstance(curves, Curve) else list(curves)
    with style.context():
        if ax is None:
            _, ax = plt.subplots(figsize=(4.6, 3.0))
        for i, c in enumerate(curves):
            (line,) = ax.plot(c.alphas, c.losses, label=labels[i] if labels else None)
            if c.marks:
                ax.plot(
                    c.marks,
                    np.interp(c.marks, c.alphas, c.losses),
                    "o",
                    ms=4.5,
                    mfc=style.PAPER,
                    mec=line.get_color(),
                    mew=1.3,
                    zorder=3,
                )
        ax.set_xlabel(curves[0].xlabel)
        ax.set_ylabel("loss")
        if logy:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.margins(x=0.02)
        if labels:
            ax.legend(loc="upper center")
    return ax


def _nice_levels(vmin, vmax, n):
    # denser near the minimum, where the interesting structure is
    raw = vmin + (vmax - vmin) * np.linspace(0, 1, n) ** 2
    return np.unique([float(f"{v:.2g}") for v in raw])


def plot_surface(surface, levels=10, vmax=None, ax=None):
    """Filled contours with thin labelled lines. Returns the Axes.

    Colour is clipped at `vmax` (default: the 97th percentile of the grid) so
    a few huge losses at the border do not flatten the basin.
    """
    z = surface.losses
    vmin = float(z.min())
    vmax = float(np.percentile(z, 97)) if vmax is None else vmax
    lv = _nice_levels(vmin, vmax, levels)
    with style.context():
        if ax is None:
            _, ax = plt.subplots(figsize=(4.6, 4.0))
        ax.contourf(surface.alphas, surface.betas, z, levels=lv, cmap=style.CMAP, extend="max")
        lines = ax.contour(
            surface.alphas, surface.betas, z, levels=lv, colors=style.INK, linewidths=0.45, alpha=0.7
        )
        ax.clabel(lines, fmt="%g", fontsize=6.5, inline_spacing=4)
        ax.plot(0, 0, "o", ms=5, mfc=style.ACCENT, mec=style.PAPER, mew=1.0, zorder=4)
        ax.set_xlabel("α")
        ax.set_ylabel("β")
        ax.set_aspect("equal")
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.7)
    return ax
