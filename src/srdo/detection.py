"""Detection-based task utility U.

Replaces the action-recognition utility, which failed the Phase 2 gate: whole-frame
action classifiers read global motion and scene context, so task-relevant
information is NOT concentrated in the semantic prior and Omega-weighted bit
allocation has almost nothing to exploit (measured selectivity ratio ~1.5x, far
short of the 5-10x the method needs).

Object detection is spatially localised BY CONSTRUCTION — degrade the region a
detector fires on and detection degrades — so Omega and U are definitionally
aligned. It is also the standard MPEG-VCM evaluation task.

Ground truth is the detector's own output on CLEAN video ("pseudo-GT"), the
standard VCM protocol. Consequences worth stating in the paper:

  * U(clean) = 1 by construction, which pins the left end of the U-S curve.
  * U measures *preservation of detectability under compression*, not absolute
    detector accuracy. That is the right quantity for a codec, but it must be
    described as such and not as detection performance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CONFIG, OmegaConfig
from .omega import _load_detector


@dataclass(frozen=True)
class Detections:
    """Per-frame boxes. `boxes[i]` is (N_i, 4) xyxy, `scores[i]` is (N_i,)."""

    boxes: list[np.ndarray]
    scores: list[np.ndarray]

    @property
    def n_total(self) -> int:
        return int(sum(b.shape[0] for b in self.boxes))


def detect(frames: np.ndarray, cfg: OmegaConfig | None = None,
           conf: float | None = None) -> Detections:
    """Run the detector over (T, H, W, 3) uint8 RGB frames."""
    cfg = cfg or CONFIG.omega
    model = _load_detector(cfg.detector_weights)
    results = model.predict(
        [f[:, :, ::-1] for f in frames],  # ultralytics wants BGR arrays
        conf=cfg.detector_conf if conf is None else conf,
        classes=list(cfg.detector_classes),
        imgsz=cfg.detector_imgsz,
        verbose=False,
    )

    boxes, scores = [], []
    for res in results:
        b = getattr(res, "boxes", None)
        if b is None or b.xyxy is None or len(b) == 0:
            boxes.append(np.zeros((0, 4), dtype=np.float32))
            scores.append(np.zeros((0,), dtype=np.float32))
        else:
            boxes.append(b.xyxy.cpu().numpy().astype(np.float32))
            scores.append(b.conf.cpu().numpy().astype(np.float32))
    return Detections(boxes=boxes, scores=scores)


def masks_from(dets: Detections, shape: tuple[int, int, int]) -> np.ndarray:
    """Binary support masks from detections, for reuse as the Omega input.

    Sharing one detector pass between Omega and the pseudo-GT guarantees the
    prior and the utility refer to exactly the same regions.
    """
    t, h, w = shape
    masks = np.zeros((t, h, w), dtype=np.float32)
    for i, bx in enumerate(dets.boxes[:t]):
        for x1, y1, x2, y2 in bx:
            xi1, yi1 = max(int(x1), 0), max(int(y1), 0)
            xi2, yi2 = min(int(x2), w), min(int(y2), h)
            if xi2 > xi1 and yi2 > yi1:
                masks[i, yi1:yi2, xi1:xi2] = 1.0
    return masks


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (Na,4) and (Nb,4) xyxy boxes."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)

    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]

    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def _match_greedy(iou: np.ndarray, thr: float) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching by descending IoU."""
    pairs: list[tuple[int, int, float]] = []
    if iou.size == 0:
        return pairs
    work = iou.copy()
    while True:
        idx = int(np.argmax(work))
        g, p = divmod(idx, work.shape[1])
        best = work[g, p]
        if best < thr:
            break
        pairs.append((g, p, float(best)))
        work[g, :] = -1.0
        work[:, p] = -1.0
        if not (work > thr).any():
            break
    return pairs


def evaluate(ref: Detections, hyp: Detections, iou_thr: float = 0.5) -> tuple[float, float]:
    """Compare detections on degraded video against clean-video pseudo-GT.

    Returns (hard_U, soft_U):

      hard_U — F1 at the IoU threshold. Bounded, interpretable, REPORTED.
      soft_U — mean over ground-truth boxes of (best IoU x that detection's
               confidence), 0 for unmatched GT. Degrades smoothly as boxes drift
               and confidence decays instead of stepping when a match crosses the
               threshold, which is what the dual update needs.

    Frames with no pseudo-GT are skipped: with nothing to detect, detection
    quality is undefined and averaging zeros in would fabricate a signal.
    """
    tp = fp = fn = 0
    soft_vals: list[float] = []

    n = min(len(ref.boxes), len(hyp.boxes))
    for i in range(n):
        gt, pr = ref.boxes[i], hyp.boxes[i]
        if gt.shape[0] == 0:
            continue

        iou = _iou_matrix(gt, pr)
        pairs = _match_greedy(iou, iou_thr)
        tp += len(pairs)
        fn += gt.shape[0] - len(pairs)
        fp += pr.shape[0] - len(pairs)

        # Soft score uses the unthresholded best overlap, so it keeps moving
        # below the IoU threshold where the hard metric has already saturated.
        best_iou = iou.max(axis=1) if pr.shape[0] else np.zeros(gt.shape[0], np.float32)
        best_j = iou.argmax(axis=1) if pr.shape[0] else None
        for g in range(gt.shape[0]):
            if best_j is None:
                soft_vals.append(0.0)
            else:
                soft_vals.append(float(best_iou[g] * hyp.scores[i][best_j[g]]))

    if not soft_vals:
        return float("nan"), float("nan")

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return float(f1), float(np.mean(soft_vals))


class DetectionUtility:
    """Utility backend with the interface run_probe expects."""

    name = "detection"

    def __init__(self, cfg: OmegaConfig | None = None, iou_thr: float = 0.5,
                 eval_stride: int = 4):
        self.cfg = cfg or CONFIG.omega
        self.iou_thr = iou_thr
        # Evaluating every frame triples the sweep cost for no extra signal;
        # detections are highly correlated between adjacent frames at 30 fps.
        self.eval_stride = eval_stride

    def reference(self, frames: np.ndarray) -> Detections:
        return detect(frames[:: self.eval_stride], self.cfg)

    def utility(self, frames: np.ndarray, ref: Detections) -> tuple[float, float]:
        hyp = detect(frames[:: self.eval_stride], self.cfg)
        return evaluate(ref, hyp, self.iou_thr)
