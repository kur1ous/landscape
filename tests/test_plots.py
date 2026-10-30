import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from landscape import Curve, plot_curve, style


@pytest.mark.parametrize("logy", [False, True])
def test_curve_metric_has_its_own_scale_and_style(logy, tmp_path):
    curve = Curve(
        np.array([-1.0, 0.0, 1.0]), np.array([10.0, 2.0, 12.0]),
        metric=np.array([0.2, 0.8, 0.1]),
    )
    fig, ax = plt.subplots()
    try:
        assert plot_curve(curve, ax=ax, logy=logy) is ax
        assert len(fig.axes) == 2
        metric_ax = fig.axes[1]
        assert ax.get_shared_x_axes().joined(ax, metric_ax)
        assert ax.get_ylabel() == "loss"
        assert metric_ax.get_ylabel() == "metric"
        assert ax.get_yscale() == ("log" if logy else "linear")
        assert metric_ax.get_yscale() == "linear"
        assert metric_ax.get_ylim()[1] < 1.0
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), curve.losses)
        metric_line, = metric_ax.lines
        np.testing.assert_array_equal(metric_line.get_xdata(), curve.alphas)
        np.testing.assert_array_equal(metric_line.get_ydata(), curve.metric)
        assert metric_line.get_linestyle() == "--"
        assert metric_line.get_color() == style.MUTED
        assert metric_ax.yaxis.label.get_color() == style.MUTED
        assert metric_ax.spines["right"].get_visible()
        assert metric_ax.spines["right"].get_edgecolor() == to_rgba(style.MUTED)
        assert metric_ax.spines["right"].get_linewidth() == style.RC["axes.linewidth"]
        assert not metric_ax.spines["top"].get_visible()
        fig.savefig(tmp_path / "metric.png")
        assert all(t.get_color() == style.MUTED for t in metric_ax.get_yticklabels())
        assert all(t.tick2line.get_color() == style.MUTED for t in metric_ax.yaxis.get_major_ticks())
    finally:
        plt.close(fig)


def test_curves_share_one_metric_axis_and_label_each_line():
    alphas = np.array([0.0, 1.0])
    curves = [
        Curve(alphas, np.array([3.0, 1.0]), marks=()),
        Curve(alphas, np.array([4.0, 2.0]), marks=(), metric=np.array([0.4, 0.8])),
        Curve(alphas, np.array([5.0, 3.0]), marks=(), metric=np.array([0.3, 0.7])),
    ]
    ax = plot_curve(curves, labels=["a", "b", "c"])
    try:
        assert len(ax.figure.axes) == 2
        assert len(ax.lines) == 3
        metric_ax = ax.figure.axes[1]
        assert len(metric_ax.lines) == 2
        for line, curve in zip(metric_ax.lines, curves[1:]):
            np.testing.assert_array_equal(line.get_ydata(), curve.metric)
        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "a", "b", "c", "b metric", "c metric",
        ]
    finally:
        plt.close(ax.figure)


def test_curve_without_metric_keeps_one_axis():
    curve = Curve(np.array([0.0, 1.0]), np.array([3.0, 1.0]), marks=())
    ax = plot_curve(curve, labels=["loss only"])
    try:
        assert ax.figure.axes == [ax]
        assert len(ax.lines) == 1
        assert [t.get_text() for t in ax.get_legend().get_texts()] == ["loss only"]
    finally:
        plt.close(ax.figure)
