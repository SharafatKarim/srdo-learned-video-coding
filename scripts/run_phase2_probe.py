#!/usr/bin/env python3
"""Phase 2b — the Assumption A1 probe. THE PRIMARY KILL GATE.

    uv run python scripts/run_phase2_probe.py [--n 120]

Sweeps Omega-weighted degradation strength, measures S = sum_i Omega_i D_i and
utility U, and tests whether U is non-increasing in S (Assumption A1, section
3.4). Also runs the background-weighted control that establishes Omega actually
localises task-relevant information.

Exit codes: 0 = A1 holds, 3 = A1 violated (stop, redesign Omega).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srdo.config import CONFIG, OUT_ROOT, PROBE_CSV  # noqa: E402
from srdo.manifest import load_manifest  # noqa: E402
from srdo.probe import run_probe, selectivity, test_monotonicity, verdict  # noqa: E402
from srdo.utility import FeatureExtractor, UtilityModel  # noqa: E402

PROBE_PKL = OUT_ROOT / "phase2_utility_probe.pkl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=CONFIG.probe.n_clips)
    ap.add_argument("--seed", type=int, default=CONFIG.probe.seed)
    ap.add_argument("--utility-col", default="soft_u", choices=["soft_u", "hard_u"])
    ap.add_argument("--utility", default="detection", choices=["detection", "action"],
                    help="detection = person-detection preservation vs clean-video "
                         "pseudo-GT (default); action = frozen action classifier")
    args = ap.parse_args()

    if args.utility == "detection":
        from srdo.detection import DetectionUtility

        model = DetectionUtility()
        print("utility backend: detection (pseudo-GT from clean video, IoU 0.5)")
    else:
        if not PROBE_PKL.exists():
            print(f"error: {PROBE_PKL} missing — run scripts/run_phase2_fit.py first",
                  file=sys.stderr)
            return 1
        with open(PROBE_PKL, "rb") as fh:
            blob = pickle.load(fh)
        model = UtilityModel(classes=blob["classes"], probe=blob["probe"],
                             extractor=FeatureExtractor())
        print(f"utility backend: action (test_acc={blob['stats']['test_acc']:.4f}, "
              f"chance={blob['stats']['chance']:.4f})")

    cfg = replace(CONFIG.probe, n_clips=args.n, seed=args.seed)
    manifest = load_manifest()

    df = run_probe(manifest, model, cfg=cfg)
    errors = int((df["error"].fillna("") != "").sum()) if "error" in df else 0
    print(f"\nprobe rows: {len(df)}  ({errors} clip failures)")
    print(f"written to {PROBE_CSV}")

    mono = test_monotonicity(df, utility_col=args.utility_col)
    sel = selectivity(df, utility_col=args.utility_col)
    print()
    print(verdict(mono, sel))

    fg = mono[mono["weight"] == "fg"]
    a1_pass = bool(len(fg)) and bool(
        (fg["rho_within_median"] < -0.5).all()
        and (fg["frac_clips_negative"] >= 0.8).all()
    )
    valid = sel[sel["overlap"]] if not sel.empty else sel
    sel_ok = valid.empty or bool((valid["selectivity_gap"] > 0.01).all())
    return 0 if (a1_pass and sel_ok) else 3


if __name__ == "__main__":
    raise SystemExit(main())
