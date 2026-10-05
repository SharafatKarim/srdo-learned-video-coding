#!/usr/bin/env python3
"""Phase 2a — fit the frozen utility model on CLEAN clips.

    uv run python scripts/run_phase2_fit.py [--n 900] [--batch 8]

Extracts frozen r3d_18 features over clean clips and fits a linear probe over
the 13 action classes. The probe is then held fixed for the whole A1 sweep:
fitting it on degraded input would let the utility model adapt to the
distortion and destroy the measurement.

The reported test accuracy sets the ceiling for the A1 probe. If it is close to
chance (1/13 = 0.077), the utility signal is too weak to drive a dual update
and no A1 conclusion is possible.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srdo.config import CONFIG, FEATURES_NPZ, OUT_ROOT, ensure_dirs  # noqa: E402
from srdo.manifest import load_manifest  # noqa: E402
from srdo.probe import select_clips  # noqa: E402
from srdo.utility import FeatureExtractor, fit_probe, save_features  # noqa: E402
from srdo.video import Crop, sample_clip  # noqa: E402

PROBE_PKL = OUT_ROOT / "phase2_utility_probe.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900, help="clean clips for probe fitting")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reuse-features", action="store_true",
                    help="skip extraction if the cache exists")
    args = ap.parse_args()

    manifest = load_manifest()
    ensure_dirs()
    extractor = FeatureExtractor()

    if args.reuse_features and FEATURES_NPZ.exists():
        from srdo.utility import load_features
        feats, labels, clip_names, classes = load_features(FEATURES_NPZ)
        print(f"reused {len(labels)} cached features from {FEATURES_NPZ}")
    else:
        clips = select_clips(manifest, args.n, args.seed)
        classes = sorted(manifest[manifest["usable"]]["action"].unique().tolist())
        print(f"extracting features: {len(clips)} clips, {len(classes)} classes")
        print(f"backbone={CONFIG.utility.backbone} "
              f"frames={CONFIG.utility.clip_frames} crop={CONFIG.utility.crop_size}\n")

        feats, labels, clip_names = [], [], []
        buf, meta = [], []

        def flush():
            if not buf:
                return
            for vec, m in zip(extractor.batch(buf), meta):
                feats.append(vec)
                labels.append(m[0])
                clip_names.append(m[1])
            buf.clear()
            meta.clear()

        for _, row in tqdm(clips.iterrows(), total=len(clips), desc="phase2a features"):
            crop = Crop(int(row["crop_w"]), int(row["crop_h"]), int(row["crop_x"]), int(row["crop_y"]))
            try:
                arr = sample_clip(row["path"], CONFIG.utility.clip_frames, crop=crop,
                                  height=CONFIG.omega.work_height)
            except Exception as exc:  # noqa: BLE001
                print("  skip {}: {}".format(row["clip"], exc), file=sys.stderr)
                continue
            buf.append(arr)
            meta.append((row["action"], row["clip"]))
            if len(buf) >= args.batch:
                flush()
        flush()

        feats = np.asarray(feats, dtype=np.float32)
        labels = np.asarray(labels)
        clip_names = np.asarray(clip_names)
        save_features(FEATURES_NPZ, feats, labels, clip_names, classes)
        print(f"\nfeatures cached to {FEATURES_NPZ}  shape={feats.shape}")

    model, stats = fit_probe(feats, labels, classes, extractor)

    print("\n=== Phase 2a: utility model ===")
    for k, v in stats.items():
        print(f"{k:12s}: {v}")

    margin = stats["test_acc"] - stats["chance"]
    print(f"\nabove chance : {margin:+.4f}")
    if margin < 0.10:
        print("WARNING: utility signal is weak. The A1 probe needs headroom to "
              "measure a drop; a near-chance classifier cannot show one.",
              file=sys.stderr)

    with open(PROBE_PKL, "wb") as fh:
        pickle.dump({"probe": model.probe, "classes": model.classes, "stats": stats}, fh)
    print(f"\nprobe written to {PROBE_PKL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
