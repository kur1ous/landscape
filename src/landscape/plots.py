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
