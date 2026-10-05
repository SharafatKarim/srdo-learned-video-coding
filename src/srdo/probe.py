"""Phase 2 — the Assumption A1 probe. THE PRIMARY KILL GATE.

For each sampled clip:
  1. decode clean frames,
  2. compute Omega ONCE on the clean frames (the prior is not a function of the
     degradation),
  3. for each mode and strength, degrade, measure S, and evaluate U with the
     frozen utility model.

Outcome is binary:
  monotone     -> A1 holds, Eq. (8) is valid, the D_s binding argument stands.
  non-monotone -> Omega is wrong. Redesign it before spending a GPU-hour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from tqdm import tqdm

from .config import CONFIG, PROBE_CSV, ProbeConfig, ensure_dirs
from .degrade import degrade, measure
from .omega import compute_omega
from .utility import UtilityModel
from .video import Crop, read_frames


def select_clips(manifest: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Stratified sample over action classes from the usable subset."""
    usable = manifest[manifest["usable"]].copy()
    if usable.empty:
        raise ValueError("no usable clips in manifest — check Phase0Config thresholds")

    per_class = max(1, n // usable["action"].nunique())
    # Explicit concat rather than groupby.apply: since pandas 2.2 the grouping
    # column is excluded from the frame handed to apply, which silently drops
    # 'action' from the result.
    parts = [
        g.sample(min(len(g), per_class), random_state=seed)
        for _, g in usable.groupby("action", sort=True)
    ]
    sampled = pd.concat(parts, ignore_index=True)
    if len(sampled) > n:
        sampled = sampled.sample(n, random_state=seed).reset_index(drop=True)
    return sampled


SCHEMA = [
    "clip", "action", "source", "family", "weight", "mode", "strength",
    "s_semantic", "s_normalised", "d_total", "psnr", "mean_omega",
    "hard_u", "soft_u", "error",
]


def _load_checkpoint() -> tuple[pd.DataFrame, set[str]]:
    """Load prior partial results if they match the current schema.

    The sweep is long (real encoder round-trips) and has repeatedly been killed
    mid-run. Results are therefore appended per clip and completed clips are
    skipped on restart, so an interruption costs at most one clip rather than
    the whole run.
    """
    if not PROBE_CSV.exists():
        return pd.DataFrame(columns=SCHEMA), set()
    prev = pd.read_csv(PROBE_CSV)
    if not set(SCHEMA).issubset(prev.columns):
        # Stale schema from an earlier probe design — archive, do not merge.
        PROBE_CSV.rename(PROBE_CSV.with_suffix(".superseded.csv"))
        return pd.DataFrame(columns=SCHEMA), set()
    return prev, set(prev["clip"].astype(str))


def _append(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows).reindex(columns=SCHEMA)
    header = not PROBE_CSV.exists()
    df.to_csv(PROBE_CSV, mode="a", header=header, index=False)


def run_probe(
    manifest: pd.DataFrame,
    model: UtilityModel,
    cfg: ProbeConfig | None = None,
    omega_cfg=None,
    resume: bool = True,
) -> pd.DataFrame:
    cfg = cfg or CONFIG.probe
    omega_cfg = omega_cfg or CONFIG.omega
    ensure_dirs()

    prev, done = _load_checkpoint() if resume else (pd.DataFrame(columns=SCHEMA), set())
    if not resume and PROBE_CSV.exists():
        PROBE_CSV.unlink()

    clips = select_clips(manifest, cfg.n_clips, cfg.seed)
    # Every family runs foreground- and background-weighted, so each arm is
    # controlled by the SAME distortion geometry. See degrade.py.
    arms = [(f, w) for f in cfg.families for w in ("fg", "bg")]
    detection_mode = getattr(model, "name", "") == "detection"
    if done:
        clips = clips[~clips["clip"].astype(str).isin(done)]
        print(f"resuming: {len(done)} clips already done, {len(clips)} remaining")

    for _, row in tqdm(clips.iterrows(), total=len(clips), desc="phase2 A1 probe"):
        rows: list[dict] = []
        crop = Crop(int(row["crop_w"]), int(row["crop_h"]), int(row["crop_x"]), int(row["crop_y"]))
        try:
            frames = read_frames(row["path"], crop=crop, height=omega_cfg.work_height,
                                 max_frames=CONFIG.utility.clip_frames * 2)
            if detection_mode:
                # One detector pass feeds BOTH the pseudo-GT and Omega, so the
                # prior and the utility refer to exactly the same regions.
                from .detection import detect, masks_from

                dets = detect(frames, omega_cfg)
                if dets.n_total == 0:
                    rows.append({"clip": row["clip"],
                                 "error": "no detections on clean video"})
                    _append(rows)
                    continue
                omega = compute_omega(frames, omega_cfg,
                                      masks=masks_from(dets, frames.shape[:3]))
                reference = model.reference(frames)
            else:
                omega = compute_omega(frames, omega_cfg)
                reference = row["action"]
        except Exception as exc:  # noqa: BLE001 - skip unreadable clips, keep the sweep alive
            rows.append({"clip": row["clip"], "error": str(exc)})
            _append(rows)
            continue

        for family, weight in arms:
            for s in cfg.strengths:
                try:
                    deg = degrade(
                        frames, omega, s, family, weight,
                        cfg.max_blur_sigma, cfg.max_noise_sigma,
                        cfg.crf_range, seed=cfg.seed,
                    )
                except Exception as exc:  # noqa: BLE001 - e.g. encoder failure
                    rows.append({"clip": row["clip"], "family": family,
                                 "weight": weight, "strength": s,
                                 "error": str(exc)})
                    continue
                stats = measure(frames, deg, omega)
                hard_u, soft_u = model.utility(deg, reference)

                rows.append({
                    "clip": row["clip"],
                    "action": row["action"],
                    "source": row["source"],
                    "family": family,
                    "weight": weight,
                    "mode": f"{family}_{weight}",
                    "strength": s,
                    "s_semantic": stats.s_semantic,
                    "s_normalised": stats.s_normalised,
                    "d_total": stats.d_total,
                    "psnr": stats.psnr,
                    "mean_omega": stats.mean_omega,
                    "hard_u": hard_u,
                    "soft_u": soft_u,
                    "error": "",
                })

        _append(rows)

    return pd.read_csv(PROBE_CSV)


def test_monotonicity(df: pd.DataFrame, utility_col: str = "soft_u") -> pd.DataFrame:
    """Per-mode monotonicity of U against S.

    Spearman/Kendall are the right tools here: A1 claims a monotone
    relationship, not a linear one. Strongly negative correlation supports A1.
    """
    out = []
    ok = df[df["error"].fillna("") == ""] if "error" in df else df
    # Detection utility returns NaN for clips/frames with no pseudo-GT to detect.
    # Those rows carry no information and must not reach the rank correlations,
    # which propagate NaN rather than skipping it.
    ok = ok[ok[utility_col].notna()]

    for (family, weight), g in ok.groupby(["family", "weight"]):
        curve = g.groupby("strength").agg(
            s_semantic=("s_semantic", "mean"),
            u=(utility_col, "mean"),
        ).reset_index().sort_values("s_semantic")

        rho, p_rho = spearmanr(g["s_semantic"], g[utility_col])
        tau, p_tau = kendalltau(g["s_semantic"], g[utility_col])

        # A1 is a WITHIN-clip claim: raising S on a given clip lowers its utility.
        # Pooling all rows conflates that with between-clip variance ("clip A is
        # intrinsically harder than clip B"), which is irrelevant and badly
        # dilutes the statistic. The paired per-clip correlation is the correct
        # test; the pooled one is retained only for reference.
        per_clip = []
        for _, gc in g.groupby("clip"):
            if gc["s_semantic"].nunique() > 2:
                r = spearmanr(gc["s_semantic"], gc[utility_col])[0]
                if not np.isnan(r):
                    per_clip.append(float(r))
        per_clip_arr = np.array(per_clip) if per_clip else np.array([np.nan])

        u = curve["u"].to_numpy()
        diffs = np.diff(u)
        out.append({
            "family": family,
            "weight": weight,
            "rho_within_median": float(np.nanmedian(per_clip_arr)),
            "frac_clips_negative": float(np.mean(per_clip_arr < 0)) if per_clip else float("nan"),
            "n_clips": len(per_clip),
            "rho_pooled": float(rho),
            "spearman_p": float(p_rho),
            "kendall_tau": float(tau),
            "kendall_p": float(p_tau),
            # Fraction of adjacent steps on the mean curve that move the wrong
            # way. Small positive values are sampling noise; large ones are a
            # genuine A1 violation.
            "violation_frac": float((diffs > 0).mean()) if diffs.size else float("nan"),
            "u_clean": float(u[0]),
            "u_worst": float(u[-1]),
            "u_drop": float(u[0] - u[-1]),
            "s_max": float(curve["s_semantic"].iloc[-1]),
        })
    return pd.DataFrame(out).sort_values(["family", "weight"]).reset_index(drop=True)


def selectivity(df: pd.DataFrame, utility_col: str = "soft_u",
                n_grid: int = 12) -> pd.DataFrame:
    """Compare foreground vs background degradation AT MATCHED TOTAL DISTORTION.

    Comparing the two at matched *strength* is invalid. With semantic density
    k ~ 0.1, weighting by (1 - Omega) perturbs roughly nine times more area than
    weighting by Omega, so the background arm produces far more total distortion
    and would 'win' even if Omega were perfect. The meaningful question is:

        for the same total distortion budget D_bar, does spending it inside
        Omega cost more utility than spending it outside?

    Each mode's utility is interpolated onto a shared D_bar grid spanning the
    overlap of the two ranges, so the comparison is like-for-like.
    """
    ok = df[df["error"].fillna("") == ""] if "error" in df else df
    ok = ok[ok[utility_col].notna()]

    rows = []
    for family, gf in ok.groupby("family"):
        curves = {}
        for weight, g in gf.groupby("weight"):
            curves[weight] = g.groupby("strength").agg(
                d=("d_total", "mean"), u=(utility_col, "mean"),
            ).reset_index().sort_values("d")

        if "fg" not in curves or "bg" not in curves:
            rows.append({"family": family, "overlap": False})
            continue

        fg, bg = curves["fg"], curves["bg"]
        lo = max(fg["d"].min(), bg["d"].min())
        hi = min(fg["d"].max(), bg["d"].max())
        if not (hi > lo):
            rows.append({"family": family, "overlap": False})
            continue

        grid = np.linspace(lo, hi, n_grid)
        u_fg = np.interp(grid, fg["d"], fg["u"])
        u_bg = np.interp(grid, bg["d"], bg["u"])

        rows.append({
            "family": family,
            "overlap": True,
            "d_lo": float(lo),
            "d_hi": float(hi),
            "u_fg_mean": float(u_fg.mean()),
            "u_bg_mean": float(u_bg.mean()),
            # Positive => at equal total distortion, spending it inside Omega
            # costs more utility than spending it outside. Omega localises.
            "selectivity_gap": float((u_bg - u_fg).mean()),
            "frac_grid_fg_worse": float((u_fg < u_bg).mean()),
        })
    return pd.DataFrame(rows)


def verdict(mono: pd.DataFrame, sel: pd.DataFrame | None = None,
            rho_threshold: float = -0.5, frac_threshold: float = 0.8,
            gap_threshold: float = 0.01) -> str:
    """Plain-language gate decision.

    A1 is judged on the WITHIN-clip (paired) correlation, not the pooled one —
    see test_monotonicity.
    """
    fg = mono[mono["weight"] == "fg"]

    lines = ["=== Phase 2: Assumption A1 gate ===", "", mono.to_string(index=False), ""]

    a1_pass = bool(len(fg)) and bool(
        (fg["rho_within_median"] < rho_threshold).all()
        and (fg["frac_clips_negative"] >= frac_threshold).all()
    )
    lines.append(
        f"A1 (U non-increasing in S) : {'PASS' if a1_pass else 'FAIL'}  "
        f"[within-clip median rho < {rho_threshold} and "
        f">={frac_threshold:.0%} of clips negative]"
    )

    sel_pass = None
    if sel is not None and not sel.empty:
        valid = sel[sel["overlap"]]
        lines += ["", "--- selectivity: fg vs bg within family, matched total distortion ---",
                  sel.to_string(index=False)]
        if not valid.empty:
            sel_pass = bool((valid["selectivity_gap"] > gap_threshold).all())
            lines.append(
                f"\nOmega selectivity          : {'PASS' if sel_pass else 'FAIL'}  "
                f"[gap > {gap_threshold} for every distortion family]"
            )
            if not sel_pass:
                lines.append(
                    "  -> At equal total distortion, degrading inside Omega costs no "
                    "more utility than degrading outside it. Omega is not localising "
                    "task-relevant information, so the spatial weight has nothing to "
                    "exploit."
                )
        else:
            lines.append("\nOmega selectivity          : INCONCLUSIVE — no overlapping "
                         "total-distortion range. Widen the strength sweep.")

    if not a1_pass or sel_pass is False:
        lines += [
            "",
            "STOP. Eq. (8) is invalid and the D_s binding argument collapses.",
            "Redesign Omega (kappa, entropy_floor, detector) and re-probe before",
            "proceeding to any GPU work.",
        ]
    return "\n".join(lines)
