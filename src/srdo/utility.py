"""Downstream task utility U. Implements the utility model of section 3.6.

The task is action recognition over the 13 dataset classes (the ALPR framing
requires plate data this corpus does not contain; the mathematics is unchanged
by the swap — see publication-strategy.md section 9).

U comes in two forms, exactly as the paper specifies:

  hard U  — top-1 accuracy. REPORTED metric.
  soft U  — probability assigned to the ground-truth class. CONTROL signal for
            the dual update, because hard accuracy is quantised and high
            variance, which makes Eq. (11) a random walk.

The classifier is fitted once on CLEAN clips and then frozen. Degraded clips are
only ever evaluated. Training on degraded input would let the utility model
adapt to the distortion and destroy the measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import CONFIG, UtilityConfig

# Kinetics-400 normalisation for torchvision video backbones.
_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def _center_crop_resize(frames: np.ndarray, size: int) -> np.ndarray:
    """(T, H, W, 3) uint8 -> (T, size, size, 3) float32 in [0, 1]."""
    t, h, w, _ = frames.shape
    scale = size / min(h, w)
    new_h, new_w = max(int(round(h * scale)), size), max(int(round(w * scale)), size)

    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float() / 255.0
    x = nn.functional.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)

    top, left = (new_h - size) // 2, (new_w - size) // 2
    x = x[:, :, top:top + size, left:left + size]
    return x.permute(0, 2, 3, 1).numpy()


def preprocess(frames: np.ndarray, cfg: UtilityConfig | None = None) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (1, 3, T, S, S) normalised tensor."""
    cfg = cfg or CONFIG.utility
    x = _center_crop_resize(frames, cfg.crop_size)
    x = (x - _MEAN) / _STD
    return torch.from_numpy(x).permute(3, 0, 1, 2).unsqueeze(0).float()


class FeatureExtractor:
    """Frozen Kinetics-pretrained backbone, classification head removed."""

    def __init__(self, cfg: UtilityConfig | None = None):
        from torchvision.models import video as tvv

        self.cfg = cfg or CONFIG.utility
        ctor = getattr(tvv, self.cfg.backbone)
        weights = "DEFAULT"
        self.model = ctor(weights=weights)
        self.feature_dim = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.model.eval()
        torch.set_grad_enabled(False)

    @torch.no_grad()
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        x = preprocess(frames, self.cfg)
        return self.model(x).squeeze(0).numpy()

    @torch.no_grad()
    def batch(self, clips: list[np.ndarray]) -> np.ndarray:
        x = torch.cat([preprocess(c, self.cfg) for c in clips], dim=0)
        return self.model(x).numpy()


@dataclass
class UtilityModel:
    """Frozen linear probe over frozen backbone features."""

    classes: list[str]
    probe: object  # sklearn LogisticRegression
    extractor: FeatureExtractor

    def predict(self, frames: np.ndarray) -> tuple[str, np.ndarray]:
        feat = self.extractor(frames).reshape(1, -1)
        proba = self.probe.predict_proba(feat)[0]
        return self.classes[int(np.argmax(proba))], proba

    def utility(self, frames: np.ndarray, true_label: str) -> tuple[float, float]:
        """Returns (hard_U, soft_U) for one clip.

        hard_U in {0, 1}: correct top-1.
        soft_U in [0, 1]: probability mass on the true class.
        """
        pred, proba = self.predict(frames)
        soft = float(proba[self.classes.index(true_label)])
        return float(pred == true_label), soft


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    classes: list[str],
    extractor: FeatureExtractor,
    cfg: UtilityConfig | None = None,
) -> tuple[UtilityModel, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    cfg = cfg or CONFIG.utility
    x_tr, x_te, y_tr, y_te = train_test_split(
        features, labels, test_size=cfg.test_size,
        random_state=cfg.seed, stratify=labels,
    )
    # multinomial is the default and the `multi_class` argument was removed in
    # scikit-learn 1.7.
    clf = LogisticRegression(C=cfg.probe_c, max_iter=cfg.probe_max_iter)
    clf.fit(x_tr, y_tr)

    stats = {
        "train_acc": float(clf.score(x_tr, y_tr)),
        "test_acc": float(clf.score(x_te, y_te)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_classes": len(classes),
        "chance": 1.0 / len(classes),
    }
    return UtilityModel(classes=classes, probe=clf, extractor=extractor), stats


def save_features(path: Path, features: np.ndarray, labels: np.ndarray,
                  clips: np.ndarray, classes: list[str]) -> None:
    np.savez_compressed(path, features=features, labels=labels,
                        clips=clips, classes=np.array(classes))


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    d = np.load(path, allow_pickle=True)
    return d["features"], d["labels"], d["clips"], list(d["classes"])
