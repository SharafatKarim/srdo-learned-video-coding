"""Bjontegaard delta metrics, plus the utility-domain variant this paper needs.

BD-BR answers "how many fewer bits for equal quality", by integrating the gap
between two RD curves over the quality range they share. The standard form
measures quality as PSNR. For a codec optimized against a task, PSNR is the
wrong axis to be judged on and reporting only BD-BR/BD-PSNR would understate or
flatter the method arbitrarily: SRDO spends bits where a detector looks and
withholds them elsewhere, which is a PSNR loss by construction. We therefore
also compute BD-Rate against task utility on exactly the same machinery — the
integration does not care what the quality axis means, only that it is
monotone in rate over the interval compared.

Both are reported. BD-BR(PSNR) is what a compression audience expects and the
honest place to disclose the fidelity cost; BD-BR(U) is what the claim is about.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RDCurve:
    """One codec's operating points. `rate` in bpp, `quality` in dB or utility."""

    name: str
    rate: np.ndarray
    quality: np.ndarray

    def __post_init__(self) -> None:
        r = np.asarray(self.rate, dtype=float)
        q = np.asarray(self.quality, dtype=float)
        if r.shape != q.shape:
            raise ValueError(f"{self.name}: rate and quality differ in shape")
        if r.size < 4:
            raise ValueError(
                f"{self.name}: cubic interpolation needs >=4 points, got {r.size}")
        if np.any(r <= 0):
            raise ValueError(f"{self.name}: non-positive rate cannot be log-scaled")
        object.__setattr__(self, "rate", r)
        object.__setattr__(self, "quality", q)


def _integrate_cubic(x: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Definite integral of the cubic least-squares fit y(x) over [lo, hi]."""
    coeffs = np.polyfit(x, y, 3)
    poly = np.polyint(coeffs)
    return float(np.polyval(poly, hi) - np.polyval(poly, lo))


def bd_rate(anchor: RDCurve, test: RDCurve) -> float:
    """Average bitrate difference of `test` relative to `anchor`, in percent.

    Negative means `test` needs fewer bits for the same quality, which is the
    favourable direction. Quality is the integration variable, so the two curves
    are compared only over the quality interval both actually reach — extending
    beyond it would extrapolate a cubic fit, which is where BD numbers most often
    go quietly wrong.
    """
    lo = max(anchor.quality.min(), test.quality.min())
    hi = min(anchor.quality.max(), test.quality.max())
    if hi <= lo:
        raise ValueError(
            f"no overlapping quality range between {anchor.name} "
            f"[{anchor.quality.min():.3f}, {anchor.quality.max():.3f}] and "
            f"{test.name} [{test.quality.min():.3f}, {test.quality.max():.3f}]")

    int_a = _integrate_cubic(anchor.quality, np.log10(anchor.rate), lo, hi)
    int_t = _integrate_cubic(test.quality, np.log10(test.rate), lo, hi)

    avg_log_diff = (int_t - int_a) / (hi - lo)
    return float((10.0 ** avg_log_diff - 1.0) * 100.0)


def bd_quality(anchor: RDCurve, test: RDCurve) -> float:
    """Average quality difference at equal rate (BD-PSNR when quality is PSNR).

    Positive favours `test`. Integration runs over log-rate here, so the shared
    interval is a rate interval rather than a quality one.
    """
    la, lt = np.log10(anchor.rate), np.log10(test.rate)
    lo, hi = max(la.min(), lt.min()), min(la.max(), lt.max())
    if hi <= lo:
        raise ValueError(f"no overlapping rate range between {anchor.name} and {test.name}")

    int_a = _integrate_cubic(la, anchor.quality, lo, hi)
    int_t = _integrate_cubic(lt, test.quality, lo, hi)
    return float((int_t - int_a) / (hi - lo))


def overlap_fraction(anchor: RDCurve, test: RDCurve) -> float:
    """Share of the anchor's quality span that the comparison actually covers.

    A BD number computed over a sliver of the anchor's range is not wrong so
    much as uninformative, and the fraction is rarely reported. Below roughly
    0.5 the operating points should be re-chosen rather than the number quoted.
    """
    lo = max(anchor.quality.min(), test.quality.min())
    hi = min(anchor.quality.max(), test.quality.max())
    span = anchor.quality.max() - anchor.quality.min()
    return float(max(hi - lo, 0.0) / span) if span > 0 else 0.0


def bd_report(anchor: RDCurve, test: RDCurve) -> dict:
    return {
        "anchor": anchor.name,
        "test": test.name,
        "bd_rate_pct": bd_rate(anchor, test),
        "bd_quality": bd_quality(anchor, test),
        "overlap_fraction": overlap_fraction(anchor, test),
        "n_points_anchor": int(anchor.rate.size),
        "n_points_test": int(test.rate.size),
    }


def psnr(mse: float | np.ndarray, peak: float = 1.0) -> float | np.ndarray:
    """PSNR in dB from MSE on images scaled to [0, peak]."""
    mse = np.maximum(np.asarray(mse, dtype=float), 1e-12)
    return 10.0 * np.log10(peak ** 2 / mse)
