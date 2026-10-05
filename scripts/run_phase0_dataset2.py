"""Phase 0 characterisation for the multi-camera 1080p corpus in dataset-2/.

The existing corpus fails two requirements at once. Its source rate (0.041 bpp
on YouTube, 0.086 on NTU) is low enough that any BD-BR measured against it is
contaminated by the source encoder, and its institutional-CCTV domain tops out
at 240 lines, below the codec tier's floor. This corpus is 1080p at ~0.52 bpp,
which clears both by a wide margin, so it is worth characterising before it is
trained on rather than after.

Two questions decide how it gets used:

  1. Does it need the scale-to-256 the current loader applies? At 1080p a
     256x256 crop can be taken at NATIVE resolution, which is a different and
     better experiment: the latent grid then covers real detail instead of a
     4.2x downsample, and the semantic prior sits on pixels a detector can
     actually resolve. Omega is measured under both regimes below.

  2. How much usable material is here? Nineteen files understates it. The
     current loader reads only the first 8 frames of each clip and never any
     others, so 784 YouTube clips supply 6272 distinct frames. These clips run
     360-650 frames each, so sampling across time is what turns 19 files into
     a comparable pool.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srdo.config import CONFIG, OUT_ROOT  # noqa: E402
from srdo.detection import detect, masks_from  # noqa: E402
from srdo.omega import compute_omega  # noqa: E402
from srdo.video import Crop, detect_crop, probe, read_frames  # noqa: E402

DATA = ROOT / "dataset-2"
OUT_CSV = OUT_ROOT / "phase0_dataset2.csv"

PATCH = 256
GOP = 8
# Sample GOPs spread across the clip, not just the opening, so the measurement
# reflects the material a temporally-sampling loader would actually see.
GOP_STARTS_FRAC = (0.1, 0.35, 0.6, 0.85)


def centre_crop(a: np.ndarray, size: int) -> np.ndarray:
    _, h, w = a.shape[:3]
    top = (h - size) // 2 if h >= size else 0
    left = (w - size) // 2 if w >= size else 0
    return a[:, top:top + size, left:left + size]


def omega_stats(frames: np.ndarray) -> tuple[float, float, bool]:
    omega = compute_omega(frames, masks=masks_from(detect(frames), frames.shape[:3]))
    crop = centre_crop(omega, PATCH)
    return float(omega.mean()), float(crop.mean()), bool(crop.max() <= 1e-6)


def read_at(path: pathlib.Path, crop: Crop | None, start: int,
            height: int | None) -> np.ndarray:
    """Decode GOP frames beginning at `start`, optionally rescaled."""
    import subprocess

    filters = []
    if crop is not None:
        filters.append(crop.ffmpeg)
    if height is not None:
        filters.append(f"scale=-2:{height}")

    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    # Decode from the head and slice, rather than seeking: MPEG-2 seeking lands
    # on the nearest I-frame, which would silently shift every sampled GOP.
    cmd += ["-frames:v", str(start + GOP),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode()[-300:])

    meta = probe(path)
    w, h = meta.width, meta.height
    if crop is not None:
        w, h = crop.w, crop.h
    if height is not None:
        w = int(round(w * height / h / 2)) * 2
        h = height

    buf = np.frombuffer(res.stdout, dtype=np.uint8)
    fb = w * h * 3
    n = buf.size // fb
    frames = buf[: n * fb].reshape(n, h, w, 3)
    return frames[start:start + GOP]


def main() -> None:
    clips = sorted(DATA.rglob("*.mpeg"))
    print(f"{len(clips)} clips in {DATA}\n")

    rows = []
    for path in clips:
        meta = probe(path)
        c = detect_crop(path, CONFIG.phase0.crop_limit, CONFIG.phase0.crop_round,
                        CONFIG.phase0.crop_probe_frames)
        crop = c or Crop(meta.width, meta.height, 0, 0)
        border = 1.0 - (crop.w * crop.h) / (meta.width * meta.height)

        native, scaled = [], []
        for frac in GOP_STARTS_FRAC:
            start = int(frac * max(meta.nb_frames - GOP, 0))
            try:
                # (1) native resolution, no downscale: what 1080p makes possible
                native.append(omega_stats(read_at(path, crop, start, None)))
                # (2) the current loader's regime, for comparability with phase 1
                scaled.append(omega_stats(read_at(path, crop, start, PATCH)))
            except Exception as exc:
                print(f"  {path.name} @{start}: {type(exc).__name__}: {exc}")

        if not native:
            continue

        rows.append({
            "clip": path.name,
            "scene": path.parent.name,
            "cam": path.stem.rsplit("_", 1)[-1],
            "width": meta.width, "height": meta.height,
            "fps": round(meta.fps, 2), "nb_frames": meta.nb_frames,
            "duration": round(meta.duration, 1),
            "src_bpp": round(meta.bpp, 4),
            "crop_w": crop.w, "crop_h": crop.h, "border_frac": round(border, 4),
            "usable_codec": bool(crop.h >= CONFIG.phase0.min_crop_height),
            "n_gops_available": max(meta.nb_frames // GOP, 0),
            "k_native_full": float(np.mean([s[0] for s in native])),
            "k_native_crop": float(np.mean([s[1] for s in native])),
            "empty_native": float(np.mean([s[2] for s in native])),
            "k_scaled_full": float(np.mean([s[0] for s in scaled])),
            "k_scaled_crop": float(np.mean([s[1] for s in scaled])),
            "empty_scaled": float(np.mean([s[2] for s in scaled])),
        })
        r = rows[-1]
        print(f"{path.name:26s} crop {crop.w}x{crop.h} border {border:.3f}  "
              f"k_native_crop={r['k_native_crop']:.4f} (empty {r['empty_native']:.0%})  "
              f"k_scaled_crop={r['k_scaled_crop']:.4f} (empty {r['empty_scaled']:.0%})")

    df = pd.DataFrame(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}\n")

    print(f"clips                 {len(df)}   scenes {df.scene.nunique()}  "
          f"cams/scene {df.groupby('scene').size().tolist()}")
    print(f"codec-tier            {int(df.usable_codec.sum())}/{len(df)}")
    print(f"src_bpp               {df.src_bpp.mean():.4f}  "
          f"(YOUTUBE 0.0410, NTU 0.0864)")
    print(f"border_frac           median {df.border_frac.median():.4f}")
    print(f"distinct GOPs         {int(df.n_gops_available.sum())} "
          f"({int(df.nb_frames.sum())} frames)")
    print()
    print(f"k, native crop        {df.k_native_crop.mean():.4f}  "
          f"empty {df.empty_native.mean():.1%}  -> mu_ref {1 / df.k_native_crop.mean():.2f}")
    print(f"k, scale-to-256 crop  {df.k_scaled_crop.mean():.4f}  "
          f"empty {df.empty_scaled.mean():.1%}  -> mu_ref {1 / df.k_scaled_crop.mean():.2f}")
    print(f"                      (YOUTUBE codec-tier crop k = 0.0876, empty 19.1%)")


if __name__ == "__main__":
    main()
