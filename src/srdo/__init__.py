"""SRDO — Bilevel Semantic Rate-Distortion Optimization.

Analysis phases 0-2 (CPU-only). See ../publication-strategy.md for the phase
plan and kill gates, and ../section3-srdo.md for the mathematics these modules
implement.
"""

from .config import CONFIG, Config, ensure_dirs

__all__ = ["CONFIG", "Config", "ensure_dirs"]
__version__ = "0.1.0"
