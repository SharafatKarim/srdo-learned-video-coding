"""Semantic prior Omega. Implements section 3.7.

Omega is supporting machinery, not the contribution: any detector emitting
spatial confidence can be substituted by replacing `detect_masks`. Three terms
are fused:

  1. Phi_boundary — soft spatial support with exponential decay away from
     detections, avoiding the hard mask edge that causes QP/rate cliffs.
  2. Temporal smoothing — P_hat_t = alpha * P_t + (1 - alpha) * P_hat_{t-1}.
  3. H_t — local entropy of the inter-frame difference (structural variation).

NOTE ON THE BOUNDARY TERM. Section 3.7 writes
`Phi(x, y) = exp(-kappa * min_b ||(x,y) - b||)` over detection boundaries b.
Read literally that peaks ON the boundary and decays both inward and outward,
which would down-weight the interior of a large object — the opposite of the
intent. We implement the sensible reading: unity inside the detection, decaying
outward. Update the manuscript to match, or this code and the paper disagree.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

from .config import CONFIG, OmegaConfig


@lru_cache(maxsize=2)
def _load_detector(weights: str):
    from ultralytics import YOLO

    return YOLO(weights)


def detect_masks(frames: np.ndarray, cfg: OmegaConfig | None = None) -> np.ndarray:
    """Binary semantic support per frame from a person detector.

    frames: (T, H, W, 3) uint8 RGB.
    returns: (T, H, W) float32 in {0, 1}.
    """
    cfg = cfg or CONFIG.omega
    model = _load_detector(cfg.detector_weights)

    t, h, w, _ = frames.shape
    masks = np.zeros((t, h, w), dtype=np.float32)

    # Ultralytics expects BGR when handed raw arrays.
    results = model.predict(
        [f[:, :, ::-1] for f in frames],
        conf=cfg.detector_conf,
        classes=list(cfg.detector_classes),
        imgsz=cfg.detector_imgsz,
        verbose=False,
    )

    for i, res in enumerate(results):
        boxes = getattr(res, "boxes", None)
        if boxes is None or boxes.xyxy is None or len(boxes) == 0:
            continue
        for x1, y1, x2, y2 in boxes.xyxy.cpu().numpy():
            xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
            xi2, yi2 = min(int(x2), w), min(int(y2), h)
            if xi2 > xi1 and yi2 > yi1:
                masks[i, yi1:yi2, xi1:xi2] = 1.0
    return masks


def boundary_support(mask: np.ndarray, kappa: float) -> np.ndarray:
    """Phi: 1 inside the detection, exp(-kappa * d) outside."""
    if mask.max() <= 0:
        return np.zeros_like(mask, dtype=np.float32)
    # Distance from every background pixel to the nearest foreground pixel.
    dist = distance_transform_edt(mask <= 0).astype(np.float32)
    return np.exp(-kappa * dist).astype(np.float32)


def temporal_entropy(frames_gray: np.ndarray, cfg: OmegaConfig) -> np.ndarray:
    """H_t: local entropy of inter-frame intensity differences, in [0, 1].

    Quantises |I_t - I_{t-1}| into `entropy_bins` levels and computes the
    Shannon entropy of the level distribution inside a sliding window. Regions
    of genuine structural change score high; static background and uniform
    global noise score low.
    """
    t, h, w = frames_gray.shape
    out = np.zeros((t, h, w), dtype=np.float32)
    if t < 2:
        return out

    diff = np.abs(np.diff(frames_gray.astype(np.int16), axis=0)).astype(np.float32)
    levels = np.clip((diff / 256.0 * cfg.entropy_bins).astype(np.int32),
                     0, cfg.entropy_bins - 1)

    win = cfg.entropy_window
    for i in range(t - 1):
        ent = np.zeros((h, w), dtype=np.float32)
        for b in range(cfg.entropy_bins):
            # Local probability of bin b == windowed mean of its indicator.
            p = uniform_filter((levels[i] == b).astype(np.float32), size=win)
            np.add(ent, -p * np.log(np.maximum(p, 1e-8)), out=ent)
        out[i + 1] = ent

    out[0] = out[1]
    max_ent = float(np.log(cfg.entropy_bins))
    return np.clip(out / max_ent, 0.0, 1.0)


def compute_omega(
    frames: np.ndarray,
    cfg: OmegaConfig | None = None,
    masks: np.ndarray | None = None,
) -> np.ndarray:
    """Full Omega pipeline. frames: (T, H, W, 3) uint8 -> (T, H, W) float32 [0,1]."""
    cfg = cfg or CONFIG.omega
    if masks is None:
        masks = detect_masks(frames, cfg)

    gray = frames.astype(np.float32).mean(axis=-1)
    ent = temporal_entropy(gray, cfg)

    t = frames.shape[0]
    omega = np.zeros_like(ent, dtype=np.float32)
    smoothed = None

    for i in range(t):
        phi = boundary_support(masks[i], cfg.kappa)
        # Temporal smoothing suppresses prior flicker, which would otherwise
        # propagate into rate oscillation through (1 + mu * Omega_i).
        smoothed = phi if smoothed is None else cfg.alpha * phi + (1 - cfg.alpha) * smoothed
        # Multiplicative fusion with a floor: a stationary person must keep a
        # high prior even when H_t is near zero.
        h_factor = cfg.entropy_floor + (1.0 - cfg.entropy_floor) * ent[i]
        omega[i] = np.clip(smoothed * h_factor, 0.0, 1.0)

    return omega


@dataclass(frozen=True)
class DensityResult:
    clip: str
    action: str
    source: str
    mean_omega: float   # k for this clip
    frac_above_half: float
    max_omega: float
    n_frames: int


def clip_density(
    path, action: str, source: str, crop, cfg: OmegaConfig | None = None,
    max_frames: int = 32,
) -> DensityResult:
    """Measure semantic density k = E[Omega] for one clip.

    This is the quantity asserted as k < 0.15 in section 3.9 and never
    previously measured.
    """
    from .video import read_frames

    cfg = cfg or CONFIG.omega
    frames = read_frames(path, crop=crop, height=cfg.work_height, max_frames=max_frames)
    omega = compute_omega(frames, cfg)
    return DensityResult(
        clip=str(path).rsplit("/", 1)[-1],
        action=action,
        source=source,
        mean_omega=float(omega.mean()),
        frac_above_half=float((omega > 0.5).mean()),
        max_omega=float(omega.max()),
        n_frames=int(frames.shape[0]),
    )
