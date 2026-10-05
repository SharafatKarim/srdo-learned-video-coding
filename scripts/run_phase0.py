#!/usr/bin/env python3
"""Phase 0 — dataset characterisation. CPU only, no torch required.

    uv run python scripts/run_phase0.py [--workers 8] [--data-root ../kaggle]

Outputs outputs/phase0_manifest.csv and prints the dataset table for the paper.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srdo.config import CONFIG, MANIFEST_CSV  # noqa: E402
from srdo.manifest import build_manifest, summarise  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=CONFIG.phase0.data_root)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if not args.data_root.exists():
        print(f"error: data root not found: {args.data_root}", file=sys.stderr)
        return 1

    df = build_manifest(data_root=args.data_root, workers=args.workers)
    print()
    print(summarise(df))
    print(f"\nmanifest written to {MANIFEST_CSV}")

    usable = int(df["usable"].sum())
    if usable == 0:
        print("\nWARNING: zero usable clips. Lower Phase0Config.min_crop_height "
              "or accept that this corpus cannot support the spatial weight.",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
