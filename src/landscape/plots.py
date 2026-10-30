import matplotlib.pyplot as plt
import numpy as np

from . import style
from .interpolate import Curve


def plot_curve(curves, labels=None, ax=None, logy=False):
    """Draw curves with optional metrics on a right axis. Returns the loss Axes."""
    curves = [curves] if isinstance(curves, Curve) else list(curves)
    with style.context():
        if ax is None:
            _, ax = plt.subplots(figsize=(4.6, 3.0))
        metric_ax = None
        for i, c in enumerate(curves):
            (line,) = ax.plot(c.alphas, c.losses, label=labels[i] if labels else None)
            if c.metric is not None:
                if metric_ax is None:
                    metric_ax = ax.twinx()
                    metric_ax.set_ylabel("metric", color=style.MUTED)
                    metric_ax.tick_params(axis="y", colors=style.MUTED)
                    metric_ax.spines["right"].set_visible(True)
                    metric_ax.spines["right"].set_color(style.MUTED)
                    metric_ax.spines["right"].set_linewidth(style.RC["axes.linewidth"])
                metric_ax.plot(
                    c.alphas, c.metric, "--", color=style.MUTED,
                    label=f"{labels[i]} metric" if labels else "metric",
                )
            if c.marks:
                ys = np.interp(c.marks, c.alphas, c.losses)
                ax.plot(c.marks, ys, "o", ms=4.5, mfc=style.PAPER, mec=line.get_color(), mew=1.3, zorder=3)
                for x, y, text in zip(c.marks, ys, c.mark_labels):
                    ax.annotate(
                        text, (x, y), (0, 8), textcoords="offset points",
                        ha="center", fontsize=7.5, color=style.MUTED,
                    )
        ax.set_xlabel(curves[0].xlabel)
        ax.set_ylabel("loss")
        if logy:
            ax.set_yscale("log")
        else:
            top = ax.get_ylim()[1]
            ax.set_ylim(-0.03 * top, top)  # keep a minimum at zero off the axis line
        ax.margins(x=0.02)
        if labels:
            handles, legend_labels = ax.get_legend_handles_labels()
            if metric_ax is not None:
                metric_handles, metric_labels = metric_ax.get_legend_handles_labels()
                handles += metric_handles
                legend_labels += metric_labels
            ax.legend(handles, legend_labels, loc="upper center")
    return ax


def _nice_levels(vmin, vmax, n):
    # log-spaced above the minimum: basins are steep, so linear levels
    # would put one line at the bottom and a crowd at the rim
    raw = vmin + (vmax - vmin) * np.geomspace(0.01, 1, n)
    return np.concatenate([[vmin], np.unique([float(f"{v:.2g}") for v in raw])])


def _label_colors(levels, cmap, vmin, vmax):
    rgb = cmap((np.asarray(levels) - vmin) / (vmax - vmin))[:, :3]
    dark = rgb @ np.array([0.2126, 0.7152, 0.0722]) < 0.45
    return [style.PAPER if d else style.INK for d in dark]


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
        line_levels = lv[1:]  # no line at the exact minimum
        lines = ax.contour(
            surface.alphas, surface.betas, z, levels=line_levels,
            colors=style.INK, linewidths=0.45, alpha=0.7,
        )
        ax.clabel(
            lines, fmt="%g", fontsize=6.5, inline_spacing=3,
            colors=_label_colors(line_levels, style.CMAP, vmin, vmax),
        )
        ax.plot(0, 0, "o", ms=5, mfc=style.ACCENT, mec=style.PAPER, mew=1.0, zorder=4)
        ax.set_xlabel("α")
        ax.set_ylabel("β")
        ax.set_aspect("equal")
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.7)
    return ax
