"""Simulation-based Typical Meteorological Year generator."""

from importlib import metadata as _metadata

__all__ = ["generate_tmy", "FSResult", "TMYResult"]

from .fs_rank import fs_rank_month, FSResult
from .selector import TMYResult, generate_tmy

try:
    __version__ = _metadata.version(__name__)
except _metadata.PackageNotFoundError:
    __version__ = "0+unknown"
