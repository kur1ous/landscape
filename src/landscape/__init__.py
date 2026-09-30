from . import style
from .directions import normalize_direction, random_direction
from .evaluate import Evaluator
from .interpolate import Curve, along_direction, between_checkpoints
from .plots import plot_curve

__version__ = "0.1.0"

__all__ = [
    "Curve",
    "Evaluator",
    "along_direction",
    "between_checkpoints",
    "normalize_direction",
    "plot_curve",
    "random_direction",
    "style",
]
