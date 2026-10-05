#!/usr/bin/env python3
"""Phase 1 — semantic density k = E[Omega]. Needs torch + ultralytics (CPU).

    uv run python scripts/run_phase1.py [--n 300] [--max-frames 24]

Measures the quantity asserted as k < 0.15 in section 3.9 and never previously
measured, and produces the eta_mu calibration constant mu_ref = 1 / mean(Omega)
required by section 3.6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srdo.config import CONFIG, DENSITY_CSV, ensure_dirs  # noqa: E402
from srdo.manifest import load_manifest  # noqa: E402
from srdo.omega import clip_density  # noqa: E402
from srdo.probe import select_clips  # noqa: E402
from srdo.video import Crop  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="clips to sample")
    ap.add_argument("--max-frames", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    manifest = load_manifest()
    clips = select_clips(manifest, args.n, args.seed)
    print(f"measuring Omega density over {len(clips)} clips\n")

    rows = []
    for _, row in tqdm(clips.iterrows(), total=len(clips), desc="phase1 density"):
        crop = Crop(int(row["crop_w"]), int(row["crop_h"]), int(row["crop_x"]), int(row["crop_y"]))
        try:
            res = clip_density(row["path"], row["action"], row["source"], crop,
                               max_frames=args.max_frames)
        except Exception as exc:  # noqa: BLE001
            print("  skip {}: {}".format(row["clip"], exc), file=sys.stderr)
            continue
        rows.append(res.__dict__)

    if not rows:
        print("error: no clips measured", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    ensure_dirs()
    df.to_csv(DENSITY_CSV, index=False)

    k = float(df["mean_omega"].mean())
    mu_ref = 1.0 / k if k > 0 else float("inf")

    print("\n=== Phase 1: semantic density ===")
    print(f"clips measured        : {len(df)}")
    print(f"k = E[Omega]          : {k:.4f}")
    print(f"  median              : {df['mean_omega'].median():.4f}")
    print(f"  p10 / p90           : {df['mean_omega'].quantile(0.1):.4f} / "
          f"{df['mean_omega'].quantile(0.9):.4f}")
    print(f"frac(Omega > 0.5)     : {df['frac_above_half'].mean():.4f}")
    print()
    print("--- by source ---")
    print(df.groupby("source")["mean_omega"].agg(["size", "mean", "median"]).to_string())
    print()
    print("--- by action ---")
    print(df.groupby("action")["mean_omega"].mean().sort_values().to_string())
    print()
    print(f"mu_ref = 1 / k        : {mu_ref:.2f}   <- eta_mu calibration, section 3.6")
    print(f"section 3.9 claim k < 0.15 : {'HOLDS' if k < 0.15 else 'FALSE — soften or drop'}")
    print(f"\nwritten to {DENSITY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
