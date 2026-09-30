"""Plot style. Survey-map look: warm paper, ink lines, one vermilion accent.

Every plotting function draws inside `context()`, so nothing leaks into the
caller's global matplotlib state.
"""

from contextlib import contextmanager

import matplotlib as mpl
import numpy as np
from matplotlib.colors import ListedColormap

PAPER = "#f6f3ec"
INK = "#1e252b"
MUTED = "#69727a"
HAIRLINE = "#d9d5ca"
ACCENT = "#c2410c"
SERIES = (INK, ACCENT, "#3d7a94", "#8b7330")

# cividis is perceptually uniform and readable with colour-vision deficiency;
# the ends are trimmed so the darkest and lightest shades do not fight the paper.
CMAP = ListedColormap(mpl.colormaps["cividis"](np.linspace(0.06, 0.88, 256)), name="basin")

RC = {
    "figure.facecolor": PAPER,
    "figure.dpi": 110,
    "savefig.facecolor": PAPER,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.facecolor": PAPER,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "axes.labelcolor": INK,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "regular",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica Neue", "Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "image.cmap": "basin",
}

if "basin" not in mpl.colormaps:
    mpl.colormaps.register(CMAP)


@contextmanager
def context():
    with mpl.rc_context(RC):
        yield
