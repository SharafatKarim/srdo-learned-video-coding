"""Omega-weighted degradation and semantic distortion measurement.

Assumption A1 (section 3.4) states that utility is non-increasing in semantic
distortion S = sum_i Omega_i D_i. That is a property of the (Omega, U) pair and
of distortion itself — NOT of the codec that produced the distortion. So A1 can
be probed with no training and no GPU: induce distortion, weight it spatially by
Omega, sweep the strength, and measure S and U directly.

THREE DISTORTION FAMILIES, each probing a different geometry:

  blur  — smoothing / loss of high frequency
  noise — additive high-frequency corruption
  codec — a real x265 round-trip at varying CRF

The first two bracket the extremes; only the third is what a codec actually
does. A1 must hold in the regime the codec occupies, so it is tested there.

PAIRED CONTROLS. Every family runs twice: weighted by Omega ('fg') and by
(1 - Omega) ('bg'). The control must use the SAME distortion family as the arm
it controls — comparing blur-in-foreground against noise-in-background would
confound distortion geometry with spatial placement, which is precisely the
variable under test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from .video import encode_decode

FAMILIES = ("blur", "noise", "codec")
WEIGHTS = ("fg", "bg")


@dataclass(frozen=True)
class DistortionStats:
    s_semantic: float       # S = mean_i(Omega_i * D_i)   -- the constrained quantity
    d_total: float          # mean_i(D_i)                 -- conventional distortion
    s_normalised: float     # S / mean(Omega), i.e. mean D inside the prior
    psnr: float
    mean_omega: float


def _blurred(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return x.copy()
    # Spatial only: blurring across time would inject motion artefacts the
    # downstream model reads as different content.
    return gaussian_filter(x, sigma=(0, sigma, sigma, 0), mode="nearest")


def _noised(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return x.copy()
    return x + rng.normal(0.0, sigma / 255.0, size=x.shape).astype(np.float32)


def degrade(
    frames: np.ndarray,
    omega: np.ndarray,
    strength: float,
    family: str,
    weight: str = "fg",
    max_blur_sigma: float = 6.0,
    max_noise_sigma: float = 48.0,
    crf_range: tuple[int, int] = (20, 51),
    seed: int = 0,
) -> np.ndarray:
    """Apply spatially-weighted degradation of one family.

    frames : (T, H, W, 3) uint8
    omega  : (T, H, W) float32 in [0, 1]
    family : 'blur' | 'noise' | 'codec'
    weight : 'fg' (weight by Omega) | 'bg' (weight by 1 - Omega)
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    if weight not in WEIGHTS:
        raise ValueError(f"unknown weight {weight!r}; expected one of {WEIGHTS}")
    if strength <= 0:
        return frames.copy()

    if family == "codec":
        lo, hi = crf_range
        crf = int(round(lo + (hi - lo) * strength))
        deg = encode_decode(frames, crf).astype(np.float32) / 255.0
    else:
        rng = np.random.default_rng(seed)
        x0 = frames.astype(np.float32) / 255.0
        deg = (_blurred(x0, max_blur_sigma * strength) if family == "blur"
               else _noised(x0, max_noise_sigma * strength, rng))

    x = frames.astype(np.float32) / 255.0
    w = np.clip(omega if weight == "fg" else 1.0 - omega, 0.0, 1.0)[..., None]
    out = x * (1.0 - w) + deg * w
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def measure(frames: np.ndarray, degraded: np.ndarray, omega: np.ndarray) -> DistortionStats:
    """Compute S, total D, and PSNR between a clean and a degraded clip."""
    a = frames.astype(np.float32) / 255.0
    b = degraded.astype(np.float32) / 255.0

    d_map = ((a - b) ** 2).mean(axis=-1)       # (T, H, W) per-location distortion D_i
    mean_omega = float(omega.mean())

    s_semantic = float((omega * d_map).mean())
    d_total = float(d_map.mean())
    s_norm = s_semantic / mean_omega if mean_omega > 1e-8 else float("nan")
    psnr = float("inf") if d_total <= 1e-12 else float(10.0 * np.log10(1.0 / d_total))

    return DistortionStats(
        s_semantic=s_semantic,
        d_total=d_total,
        s_normalised=s_norm,
        psnr=psnr,
        mean_omega=mean_omega,
    )
