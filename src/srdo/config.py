"""Central configuration for the SRDO analysis pipeline (phases 0-2).

Every tunable that appears in the paper lives here so that the manuscript and
the code cannot drift apart. Symbols referenced in comments map to
`section3-srdo.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "kaggle"
OUT_ROOT = REPO_ROOT / "outputs"

MANIFEST_CSV = OUT_ROOT / "phase0_manifest.csv"
DENSITY_CSV = OUT_ROOT / "phase1_density.csv"
FEATURES_NPZ = OUT_ROOT / "phase2_features.npz"
PROBE_CSV = OUT_ROOT / "phase2_a1_probe.csv"
FIGURE_DIR = OUT_ROOT / "figures"


@dataclass(frozen=True)
class Phase0Config:
    """Dataset characterisation and usable-subset selection."""

    data_root: Path = DATA_ROOT
    # cropdetect: limit, round, reset. limit=24 on 0-255 catches the near-black
    # pillarbox bars seen in the UCF-Crime clips without eating dark content.
    crop_limit: int = 24
    crop_round: int = 2
    crop_probe_frames: int = 120

    # TWO usability tiers, because the two constraints are genuinely different.
    #
    # codec tier (phase 4-5, GPU): bound by the latent grid. A stride-16 codec
    # turns a 240-line frame into 15 latent rows, far too coarse for the spatial
    # weight (1 + mu * Omega_i) to act on. This tier is strict.
    #
    # probe tier (phase 1-2, CPU): the A1 probe uses SYNTHETIC degradation at
    # work_height, never a codec, so the latent grid is irrelevant here. Only
    # the detector and the action classifier need enough pixels. Applying the
    # codec threshold to the probe would discard all of UCF-Crime and with it
    # three action classes and two of the three domains — exactly the
    # cross-domain variation the transfer experiment depends on.
    min_crop_height: int = 352      # codec tier
    min_probe_height: int = 150     # probe tier
    min_frames: int = 16


@dataclass(frozen=True)
class OmegaConfig:
    """Semantic prior Omega. See section 3.7."""

    detector_weights: str = "yolov8n.pt"
    detector_classes: tuple[int, ...] = (0,)  # COCO person
    detector_conf: float = 0.25
    detector_imgsz: int = 640

    # Phi_boundary(x, y) = exp(-kappa * dist_to_detection). kappa is in units of
    # 1/pixel at the working resolution below.
    kappa: float = 0.06
    # Temporal smoothing: P_hat_t = alpha * P_t + (1 - alpha) * P_hat_{t-1}.
    alpha: float = 0.35
    # Local entropy window for H_t (temporal structural variation).
    entropy_window: int = 16
    entropy_bins: int = 16
    # Floor on the H_t factor. Pure multiplicative fusion would drive Omega to
    # zero on a stationary-but-important object (a person standing still), which
    # is exactly the forensic content we must not discard.
    entropy_floor: float = 0.5

    work_height: int = 256  # Omega is computed at this height, then resized.


@dataclass(frozen=True)
class UtilityConfig:
    """Downstream task utility U. See section 3.6."""

    clip_frames: int = 16
    crop_size: int = 112
    backbone: str = "r3d_18"  # torchvision, Kinetics-400 pretrained
    # A linear probe is fitted on frozen backbone features over CLEAN clips,
    # then held fixed. Degraded clips are only ever evaluated, never trained on.
    probe_c: float = 1.0
    probe_max_iter: int = 2000
    test_size: float = 0.3
    seed: int = 0


@dataclass(frozen=True)
class ProbeConfig:
    """Assumption A1 probe: is U non-increasing in S? See section 3.4."""

    # Degradation strengths s. s=0 is the clean anchor and must be included:
    # it pins the left end of the U-S curve.
    strengths: tuple[float, ...] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)
    # Three distortion geometries. blur and noise bracket the extremes; only
    # 'codec' is what an encoder actually produces, and A1 must hold there.
    families: tuple[str, ...] = ("blur", "noise", "codec")
    max_blur_sigma: float = 6.0
    max_noise_sigma: float = 48.0  # on 0-255
    crf_range: tuple[int, int] = (20, 51)  # x265: near-transparent -> worst
    n_clips: int = 120
    seed: int = 0


@dataclass(frozen=True)
class Config:
    phase0: Phase0Config = field(default_factory=Phase0Config)
    omega: OmegaConfig = field(default_factory=OmegaConfig)
    utility: UtilityConfig = field(default_factory=UtilityConfig)
    probe: ProbeConfig = field(default_factory=ProbeConfig)


CONFIG = Config()


def ensure_dirs() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
