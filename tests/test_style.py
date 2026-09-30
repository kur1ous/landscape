import matplotlib as mpl
import numpy as np

from landscape import style


def _lightness(rgba):
    c = rgba[:, :3]
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    return np.where(y > 0.008856, 116 * np.cbrt(y) - 16, 903.3 * y)


def test_cmap_never_gets_darker_as_loss_rises():
    lstar = _lightness(style.CMAP(np.linspace(0, 1, 256)))
    assert np.all(np.diff(lstar) >= 0)
    assert lstar[-1] - lstar[0] > 40


def test_context_restores_rcparams():
    before = mpl.rcParams["figure.facecolor"]
    with style.context():
        assert mpl.rcParams["figure.facecolor"] == style.PAPER
        assert not mpl.rcParams["axes.spines.top"]
    assert mpl.rcParams["figure.facecolor"] == before


def test_cmap_is_registered():
    assert mpl.colormaps["basin"].N == 256
