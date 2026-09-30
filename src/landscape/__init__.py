from . import style
from .directions import normalize_direction, random_direction
from .evaluate import Evaluator
from .grid import Surface, plane_slice
from .interpolate import Curve, along_direction, between_checkpoints
from .plots import plot_curve, plot_surface

__version__ = "0.1.0"

__all__ = [
    "Curve",
    "Evaluator",
    "Surface",
    "along_direction",
    "between_checkpoints",
    "normalize_direction",
    "plane_slice",
    "plot_curve",
    "plot_surface",
    "random_direction",
    "style",
]
