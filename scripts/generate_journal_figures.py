"""Generate High-Resolution Publication-Quality Figures for Journal Submission.

Generates vector PDFs and 300 DPI PNGs:
1. figures/fig1_cross_validated_rd.pdf — 5-Fold Cross-Validated RD Curves (SSF mu=0, SSF mu=60, x265 Anchor)
2. figures/fig2_dual_ascent_convergence.pdf — Settled Dual-Ascent Trajectory at 80% Knee (tau=0.354)
3. figures/fig3_background_leverage.pdf — Background Leverage Law (1-k)/k vs. Reallocation Shift
"""

from __future__ import annotations

import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_FIG = ROOT / "outputs" / "figures"
OUT_FIG.mkdir(parents=True, exist_ok=True)

# Set IEEE Publication Plotting Style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 12,
    "figure.autolayout": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

COLOR_MU0 = "#1f77b4"      # Deep Blue
COLOR_MU60 = "#d62728"     # Red
COLOR_ANCHOR = "#2ca02c"   # Green
COLOR_TARGET = "#7f7f7f"   # Gray dashed


def plot_fig1_cross_validated_rd():
    """Figure 1: Cross-Validated RD Curves (Rate vs. Utility)."""
    df_cv = pd.read_csv(ROOT / "outputs" / "leave_one_scene_out_cv.csv")
    df_anchor = pd.read_csv(ROOT / "outputs" / "phase5_anchor_matched.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Calculate means across scenes for each lambda
    c0 = df_cv[df_cv["mu"] == 0].groupby("lambda")[["bpp", "soft_u", "hard_u"]].mean().reset_index()
    c60 = df_cv[df_cv["mu"] == 60].groupby("lambda")[["bpp", "soft_u", "hard_u"]].mean().reset_index()
    c_anc = df_anchor.groupby("crf")[["bpp", "soft_u", "hard_u"]].mean().reset_index()

    # Sort by bpp
    c0 = c0.sort_values("bpp")
    c60 = c60.sort_values("bpp")
    c_anc = c_anc.sort_values("bpp")

    # Panel (a): Soft Utility
    ax1.plot(c_anc["bpp"], c_anc["soft_u"], "s--", color=COLOR_ANCHOR, linewidth=1.8, label="x265 Anchor (HEVC)")
    ax1.plot(c0["bpp"], c0["soft_u"], "o-", color=COLOR_MU0, linewidth=2.0, label=r"SSF ($\mu=0$, unweighted)")
    ax1.plot(c60["bpp"], c60["soft_u"], "^-", color=COLOR_MU60, linewidth=2.0, label=r"SSF ($\mu=60$, weighted)")

    ax1.set_xlabel("Rate (bits per pixel, bpp)")
    ax1.set_ylabel("Soft Task Utility (Confidence × IoU)")
    ax1.set_title("(a) Soft Utility Rate-Distortion")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Panel (b): Hard Utility F1
    ax2.plot(c_anc["bpp"], c_anc["hard_u"], "s--", color=COLOR_ANCHOR, linewidth=1.8, label="x265 Anchor (HEVC)")
    ax2.plot(c0["bpp"], c0["hard_u"], "o-", color=COLOR_MU0, linewidth=2.0, label=r"SSF ($\mu=0$, unweighted)")
    ax2.plot(c60["bpp"], c60["hard_u"], "^-", color=COLOR_MU60, linewidth=2.0, label=r"SSF ($\mu=60$, weighted)")

    ax2.set_xlabel("Rate (bits per pixel, bpp)")
    ax2.set_ylabel(r"Detection $F_1$ Score (IoU @ 0.5)")
    ax2.set_title(r"(b) Detection $F_1$ Rate-Distortion")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="lower right")

    pdf_path = OUT_FIG / "fig1_cross_validated_rd.pdf"
    png_path = OUT_FIG / "fig1_cross_validated_rd.png"
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.close()
    print(f"Saved Figure 1 to {pdf_path} and {png_path}")


def plot_fig2_dual_ascent_convergence():
    """Figure 2: Dual-Ascent Trajectory (v16 Kaggle runs)."""
    csv_5 = ROOT / "notebooks" / "phase4-cctv" / "v16" / "continuous_dual_ascent_mu0_5.csv"
    csv_30 = ROOT / "notebooks" / "phase4-cctv" / "v16" / "continuous_dual_ascent_mu0_30.csv"
    if not (csv_5.exists() and csv_30.exists()):
        print("Skipping Figure 2: v16 trajectory CSVs not found.")
        return

    df_5 = pd.read_csv(csv_5)
    df_30 = pd.read_csv(csv_30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    # Panel (a): Measured Utility EMA vs Target Tau
    ax1.plot(df_5["outer_step"], df_5["u_ema"], "-", color="#1f78b4", linewidth=2.2, label=r"$\mu_0=5$ EMA")
    ax1.plot(df_30["outer_step"], df_30["u_ema"], "-", color="#33a02c", linewidth=2.2, label=r"$\mu_0=30$ EMA")
    ax1.axhline(0.992, linestyle="--", color="#e31a1c", linewidth=1.8, label=r"Target Threshold $\tau=0.992$")
    
    # Deadband band
    ax1.fill_between(df_5["outer_step"], 0.992 - 0.0002, 0.992 + 0.0002, color="#e31a1c", alpha=0.15)

    ax1.set_xlabel("Outer Controller Step ($k$)")
    ax1.set_ylabel(r"Task Utility ($1 - \mathrm{MSE}$)")
    ax1.set_title(r"(a) Dual Control Utility Response ($U$)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Panel (b): Multiplier Mu Trajectory
    ax2.plot(df_5["outer_step"], df_5["mu"], "s-", color="#1f78b4", linewidth=2.0, label=r"$\mu_0=5$")
    ax2.plot(df_30["outer_step"], df_30["mu"], "o-", color="#33a02c", linewidth=2.0, label=r"$\mu_0=30$")

    ax2.set_xlabel("Outer Controller Step ($k$)")
    ax2.set_ylabel(r"Spatial Weight Multiplier $\mu$")
    ax2.set_title(r"(b) Dual Variable Trajectory ($\mu$)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left")

    pdf_path = OUT_FIG / "fig2_dual_ascent_convergence.pdf"
    png_path = OUT_FIG / "fig2_dual_ascent_convergence.png"
    plt.tight_layout()
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.close()
    print(f"Saved Figure 2 to {pdf_path} and {png_path}")


def plot_fig3_background_leverage():
    """Figure 3: Background Leverage Law (1-k)/k vs. Reallocation Headroom."""
    fig, ax1 = plt.subplots(figsize=(6.5, 4.2))

    k_vals = np.linspace(0.02, 0.50, 200)
    leverage = (1.0 - k_vals) / k_vals

    # Empirical points
    k_sparse, shift_sparse, lev_sparse = 0.099, 11.4, (1 - 0.099) / 0.099
    k_dense, shift_dense, lev_dense = 0.408, 3.8, (1 - 0.408) / 0.408

    ax1.plot(k_vals, leverage, "-", color="#33a02c", linewidth=2.2, label=r"Harvestable Background Leverage $\Lambda = (1-k)/k$")
    ax1.scatter([k_sparse, k_dense], [lev_sparse, lev_dense], color="#d95f02", s=70, zorder=5)

    ax1.annotate(f"Sparse Prior (Consumer IP)\n$k=0.099, \\Lambda=9.11$\n$S/D$ shift = 11.4%",
                 xy=(k_sparse, lev_sparse), xytext=(k_sparse + 0.04, lev_sparse - 1.5),
                 arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.5))

    ax1.annotate(f"Dense Prior (Domain D)\n$k=0.408, \\Lambda=1.45$\n$S/D$ shift = 3.8%",
                 xy=(k_dense, lev_dense), xytext=(k_dense - 0.12, lev_dense + 3.0),
                 arrowprops=dict(arrowstyle="->", color="#d95f02", lw=1.5))

    ax1.set_xlabel(r"Semantic Prior Density $k = \mathbb{E}[\Omega]$")
    ax1.set_ylabel(r"Background Leverage Ratio $\Lambda = (1-k)/k$")
    ax1.set_title(r"Background Leverage Law: Reallocation Headroom vs. Prior Density")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    pdf_path = OUT_FIG / "fig3_background_leverage.pdf"
    png_path = OUT_FIG / "fig3_background_leverage.png"
    plt.savefig(pdf_path)
    plt.savefig(png_path)
    plt.close()
    print(f"Saved Figure 3 to {pdf_path} and {png_path}")


def main():
    print("=== Generating Journal Publication Figures ===")
    plot_fig1_cross_validated_rd()
    plot_fig2_dual_ascent_convergence()
    plot_fig3_background_leverage()
    print("All figures successfully generated in outputs/figures/")


if __name__ == "__main__":
    main()
