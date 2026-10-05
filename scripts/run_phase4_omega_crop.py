"""Measure E[Omega] on the pixels training actually sees.

Phase 1 measured k = E[Omega] on the full border-cropped frame scaled to
height 256. Phase 4 trains on a 256x256 CENTRE CROP of that frame, so for a
16:9 source it discards roughly 44% of the width. The two populations are not
the same, and k is what sets mu_ref = 1/E[Omega], which is what eta_mu was
calibrated against. If the centre crop is denser or sparser than the full
frame, every step size derived from the phase 1 number is off by that ratio.

This script recomputes Omega through the identical path used by the training
loader — same border crop, same scale-to-256, same detector pass shared with
the pseudo-GT — and reports the full-frame and centre-crop means side by side.

Restricted to the codec tier, since that is the only pool phase 4 draws from.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srdo.config import CONFIG, MANIFEST_CSV, OUT_ROOT  # noqa: E402
from srdo.detection import detect, masks_from  # noqa: E402
from srdo.omega import compute_omega  # noqa: E402
from srdo.video import Crop, read_frames  # noqa: E402

DATA_ROOT = ROOT / "kaggle"
OUT_CSV = OUT_ROOT / "phase4_omega_crop.csv"

PATCH = 256      # CFG["patch"] in every phase 4 run
GOP = 8          # load_gop's n; Omega is computed over the same 8 frames


def centre_crop(a: np.ndarray, size: int) -> np.ndarray:
    """(T, H, W) -> (T, size, size), matching load_gop's slicing exactly."""
    _, h, w = a.shape
    top = (h - size) // 2 if h >= size else 0
    left = (w - size) // 2 if w >= size else 0
    return a[:, top:top + size, left:left + size]


def measure(row: pd.Series) -> dict | None:
    path = DATA_ROOT / str(row["action"]) / str(row["clip"])
    if not path.exists():
        return None
    crop = Crop(int(row["crop_w"]), int(row["crop_h"]), int(row["crop_x"]), int(row["crop_y"]))

    frames = read_frames(path, crop=crop, height=PATCH, max_frames=GOP)
    # One detector pass, reused for the prior — same as training.
    omega = compute_omega(frames, masks=masks_from(detect(frames), frames.shape[:3]))
    om_c = centre_crop(omega, PATCH)

    return {
        "clip": str(row["clip"]),
        "action": str(row["action"]),
        "source": str(row["source"]),
        "full_h": int(omega.shape[1]),
        "full_w": int(omega.shape[2]),
        "crop_h": int(om_c.shape[1]),
        "crop_w": int(om_c.shape[2]),
        "width_kept": om_c.shape[2] / omega.shape[2],
        "mean_omega_full": float(omega.mean()),
        "mean_omega_crop": float(om_c.mean()),
        "frac_above_half_full": float((omega > 0.5).mean()),
        "frac_above_half_crop": float((om_c > 0.5).mean()),
        "max_omega_crop": float(om_c.max()),
        # An all-zero crop contributes no semantic gradient at all: (1+mu*Omega)
        # collapses to 1 and the step is indistinguishable from mu=0.
        "crop_is_empty": bool(om_c.max() <= 1e-6),
        "n_frames": int(omega.shape[0]),
    }


def figure(df: pd.DataFrame, path: pathlib.Path) -> None:
    """Two facts Section 6 has to disclose, on one axis pair."""
    import matplotlib.pyplot as plt

    from srdo import plots as P

    P.apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.1))
    colours = {"YOUTUBE": P.PALETTE["blur"], "NTU": P.PALETTE["noise"]}

    ax = axes[0]
    for src, g in df.groupby("source"):
        ax.scatter(g["mean_omega_full"], g["mean_omega_crop"], s=11, alpha=0.45,
                   color=colours.get(src, P.INK_MUTED), edgecolor="none", label=src)
    hi = float(max(df["mean_omega_full"].max(), df["mean_omega_crop"].max()))
    ax.plot([0, hi], [0, hi], color=P.INK_MUTED, linewidth=0.9, linestyle=(0, (5, 4)),
            label="crop = full frame")
    ax.set_xlabel(r"$\bar\Omega$, full border-cropped frame")
    ax.set_ylabel(r"$\bar\Omega$, 256$\times$256 centre crop")
    ax.set_title(r"(a) the training crop is denser than the frame", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    P._tidy(ax)

    ax = axes[1]
    for src, g in df.groupby("source"):
        ne = g[~g["crop_is_empty"]]["mean_omega_crop"]
        ax.hist(ne, bins=40, alpha=0.55, color=colours.get(src, P.INK_MUTED),
                label=f"{src}: {g['crop_is_empty'].mean():.0%} of clips have "
                      r"$\Omega\equiv 0$")
    ax.set_xlabel(r"$\bar\Omega$ on the training crop (clips with any detection)")
    ax.set_ylabel("clips")
    ax.set_title(r"(b) a fifth of YouTube clips carry no prior", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    P._tidy(ax)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(path.with_suffix(f".{ext}"))
    plt.close(fig)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("source")
    out = pd.DataFrame({
        "n": g.size(),
        "k_full": g["mean_omega_full"].mean(),
        "k_crop": g["mean_omega_crop"].mean(),
        "k_crop_median": g["mean_omega_crop"].median(),
        "empty_crop_frac": g["crop_is_empty"].mean(),
    })
    out["ratio_crop_over_full"] = out["k_crop"] / out["k_full"]
    # mu_ref = 1/E[Omega]: the mu at which the semantic term doubles the mean
    # weight. Section 3.6 uses it to set eta_mu, so it inherits this error.
    out["mu_ref_full"] = 1.0 / out["k_full"]
    out["mu_ref_crop"] = 1.0 / out["k_crop"]
    # Clips with no detection anywhere contribute a constant weight of 1, so
    # they train as if mu=0 and are skipped at evaluation for want of pseudo-GT.
    # The scale mu actually acts on is the one over the clips that have a prior.
    live = df[~df["crop_is_empty"]].groupby("source")["mean_omega_crop"].mean()
    out["k_crop_nonempty"] = live
    out["mu_ref_effective"] = 1.0 / live
    return out.round(4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="YOUTUBE,NTU",
                    help="comma-separated; only these have codec-tier clips")
    ap.add_argument("--limit", type=int, default=0, help="clips per source, 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    man = pd.read_csv(MANIFEST_CSV)
    pool = man[man.usable_codec & man.source.isin(args.sources.split(","))]

    if args.limit:
        pool = (pool.groupby("source", group_keys=False)
                    .apply(lambda g: g.sample(min(len(g), args.limit), random_state=args.seed)))
    pool = pool.reset_index(drop=True)
    print(f"{len(pool)} codec-tier clips: {pool.source.value_counts().to_dict()}")

    rows, t0 = [], time.time()
    for i, r in pool.iterrows():
        try:
            m = measure(r)
        except Exception as exc:  # a single unreadable clip must not lose the run
            print(f"  skip {r['clip']}: {type(exc).__name__}: {exc}")
            continue
        if m is None:
            print(f"  missing on disk: {r['clip']}")
            continue
        rows.append(m)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(pool)}  {time.time() - t0:.0f}s")

    df = pd.DataFrame(rows)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nmeasured {len(df)} clips in {time.time() - t0:.0f}s -> {OUT_CSV}\n")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(summarise(df).to_string())

    figure(df, OUT_ROOT / "figures" / "phase4_omega_crop")
    print(f"\nwrote {OUT_ROOT / 'figures' / 'phase4_omega_crop.pdf'}")

    print(f"\nkappa={CONFIG.omega.kappa}  entropy_floor={CONFIG.omega.entropy_floor}"
          f"  work_height={PATCH}  gop={GOP}")


if __name__ == "__main__":
    main()
