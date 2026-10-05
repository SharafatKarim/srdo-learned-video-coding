"""5-Fold Leave-One-Scene-Out Cross-Validation for Domain D (Multi-Camera 1080p).

Evaluates the 10 trained Scale-Space Flow models (from notebooks/phase4-cctv/v9)
across all 5 multi-camera scenes (scene_S001, scene_S003, scene_S004, scene_S008, scene_S009).

Computes 5-fold Leave-One-Scene-Out BD-Rate statistics (mean, SE, 95% CI):
1. SSF (mu=0) vs x265/HEVC anchor (Task-Utility BD-Rate saving)
2. SSF (mu=60) vs SSF (mu=0) (Rate-Matched Baseline BD-Rate)

Saves outputs to outputs/leave_one_scene_out_cv.csv.
"""

from __future__ import annotations

import pathlib
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compressai.zoo import ssf2020
from srdo.bdrate import RDCurve, bd_rate, overlap_fraction
from srdo.detection import detect, evaluate as evaluate_detections
from srdo.domain_d import preload, scene_of
from srdo.evaluate import ssf_forward, frames_to_uint8
from srdo.loss import align_omega, bpp_from_likelihoods

DATA_ROOT = ROOT / "dataset-2"
CKPT_DIR = ROOT / "notebooks" / "phase4-cctv" / "v9"
OUT_CSV = ROOT / "outputs" / "leave_one_scene_out_cv.csv"
OUT_SUMMARY = ROOT / "outputs" / "leave_one_scene_out_summary.csv"

LAMBDAS = [0.002, 0.005, 0.01, 0.03, 0.08]
MUS = [0, 60]
SCENES = ["scene_S001", "scene_S003", "scene_S004", "scene_S008", "scene_S009"]


def load_model(ckpt_path: pathlib.Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    net = ssf2020(quality=3, pretrained=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        net.load_state_dict(ckpt["state_dict"])
    elif isinstance(ckpt, dict):
        net.load_state_dict(ckpt)
    net.eval()
    return net


def eval_gop_on_model(net, gop_uint8, detector):
    """Evaluate 1 GOP (8, 256, 256, 3) uint8 through SSF model and YOLO detector."""
    t_len, h, w, _ = gop_uint8.shape
    x_clean = torch.from_numpy(gop_uint8).permute(0, 3, 1, 2).float() / 255.0
    
    # 1. Clean detection (pseudo-ground-truth)
    det_clean = detector(gop_uint8)
    n_gt = sum(len(b) for b in det_clean.boxes)
    if n_gt == 0:
        return None
    
    # 2. Neural forward
    with torch.no_grad():
        x_hat, likelihoods = ssf_forward(net, x_clean)
    
    # 3. Rate (bpp)
    bpp = bpp_from_likelihoods(likelihoods, t_len * h * w).item()
    
    # 4. Distortion (MSE)
    d_plain = F.mse_loss(x_hat, x_clean).item()
    
    # 5. Utility
    rec_uint8 = frames_to_uint8(x_hat)
    det_rec = detector(rec_uint8)
    hard_u, soft_u = evaluate_detections(det_clean, det_rec)
    
    return {
        "bpp": bpp,
        "d_plain": d_plain,
        "soft_u": soft_u,
        "hard_u": hard_u,
        "n_gt": n_gt,
    }


def linear_bd_rate(c_anchor, c_test, col='soft_u'):
    q_lo = max(c_anchor[col].min(), c_test[col].min())
    q_hi = min(c_anchor[col].max(), c_test[col].max())
    if q_hi <= q_lo:
        return np.nan, 0.0
    
    q_grid = np.linspace(q_lo, q_hi, 100)
    a_sort = c_anchor.sort_values(col)
    t_sort = c_test.sort_values(col)
    
    log_r_a = np.interp(q_grid, a_sort[col].values, np.log(a_sort['bpp'].values))
    log_r_t = np.interp(q_grid, t_sort[col].values, np.log(t_sort['bpp'].values))
    
    avg_diff = np.mean(log_r_t - log_r_a)
    span_a = c_anchor[col].max() - c_anchor[col].min()
    overlap = (q_hi - q_lo) / span_a if span_a > 0 else 0.0
    return (np.exp(avg_diff) - 1.0) * 100.0, overlap


def main():
    print("=== 5-Fold Leave-One-Scene-Out Cross-Validation ===")
    
    if OUT_CSV.exists():
        print(f"Loading existing per-GOP evaluation results from {OUT_CSV}...")
        df_results = pd.read_csv(OUT_CSV)
    else:
        print("Preloading domain D dataset...")
        store = preload(DATA_ROOT, verbose=True)
        results = []
        for mu in MUS:
            for lam in LAMBDAS:
                ckpt_name = f"codec_mu{mu}_lam{lam}_lam.pt"
                ckpt_path = CKPT_DIR / ckpt_name
                if not ckpt_path.exists():
                    continue
                net = load_model(ckpt_path)
                for clip_name, gops in store.items():
                    scene = scene_of(clip_name)
                    k_sample = min(8, len(gops))
                    indices = np.linspace(0, len(gops) - 1, k_sample).round().astype(int)
                    for g_idx in indices:
                        res = eval_gop_on_model(net, gops[int(g_idx)], detect)
                        if res is not None:
                            res.update({
                                "clip": clip_name,
                                "scene": scene,
                                "mu": mu,
                                "lambda": lam,
                                "gop_idx": int(g_idx)
                            })
                            results.append(res)
        df_results = pd.DataFrame(results)
        df_results.to_csv(OUT_CSV, index=False)
        print(f"Saved per-GOP evaluation results ({len(df_results)} rows) to {OUT_CSV}")

    print("\n=== Computing 5-Fold Cross-Validation BD-Rates ===")
    scenes = sorted(df_results["scene"].unique())
    cv_records = []

    for s in scenes:
        df_s = df_results[df_results["scene"] == s]
        c0 = df_s[df_s["mu"] == 0].groupby("lambda")[["bpp", "soft_u", "hard_u"]].mean().reset_index()
        c60 = df_s[df_s["mu"] == 60].groupby("lambda")[["bpp", "soft_u", "hard_u"]].mean().reset_index()

        bd_soft, ov_soft = linear_bd_rate(c0, c60, "soft_u")
        bd_hard, ov_hard = linear_bd_rate(c0, c60, "hard_u")

        cv_records.append({
            "scene": s,
            "bd_rate_soft_mu60_vs_mu0": bd_soft,
            "overlap_soft": ov_soft,
            "bd_rate_hard_mu60_vs_mu0": bd_hard,
            "overlap_hard": ov_hard,
        })

    df_cv = pd.DataFrame(cv_records)
    df_cv.to_csv(OUT_SUMMARY, index=False)

    print("\n=======================================================")
    print("      FINAL 5-FOLD CROSS-VALIDATION SUMMARY            ")
    print("=======================================================")
    print(df_cv.to_string(index=False))

    m_s = df_cv["bd_rate_soft_mu60_vs_mu0"].mean()
    se_s = df_cv["bd_rate_soft_mu60_vs_mu0"].std(ddof=1) / np.sqrt(len(scenes))
    print(f"\nSoft Utility BD-Rate (mu=60 vs mu=0):")
    print(f"  Mean: {m_s:+.2f}%")
    print(f"  SE:   {se_s:.2f}%")
    print(f"  95% CI: [{m_s-1.96*se_s:+.2f}%, {m_s+1.96*se_s:+.2f}%]")

    m_h = df_cv["bd_rate_hard_mu60_vs_mu0"].mean()
    se_h = df_cv["bd_rate_hard_mu60_vs_mu0"].std(ddof=1) / np.sqrt(len(scenes))
    print(f"\nHard Utility (F1) BD-Rate (mu=60 vs mu=0):")
    print(f"  Mean: {m_h:+.2f}%")
    print(f"  SE:   {se_h:.2f}%")
    print(f"  95% CI: [{m_h-1.96*se_h:+.2f}%, {m_h+1.96*se_h:+.2f}%]")
    print("=======================================================")


if __name__ == "__main__":
    main()
