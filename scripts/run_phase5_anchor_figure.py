"""Anchor RD curves, and the PSNR/utility divergence they expose."""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srdo import plots as P  # noqa: E402

CSV = ROOT / "outputs" / "phase5_anchor_rd.csv"
OUT = ROOT / "outputs" / "figures" / "phase5_anchor_rd"


def main() -> None:
    P.apply_style()
    d = pd.read_csv(CSV)
    agg = d.groupby("crf").agg(bpp=("bpp", "mean"), psnr=("psnr_db", "mean"),
                               u=("soft_u", "mean"), n=("clip", "size")).reset_index()
    sem_u = d.groupby("crf")["soft_u"].sem()

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2))

    ax = axes[0]
    ax.plot(agg["bpp"], agg["psnr"], color=P.PALETTE["codec"], linewidth=2.0,
            marker="D", markersize=5, markeredgecolor="white", markeredgewidth=0.8)
    for _, r in agg.iterrows():
        ax.annotate(f"CRF {r['crf']:.0f}", xy=(r["bpp"], r["psnr"]), xytext=(4, -9),
                    textcoords="offset points", fontsize=7.5, color=P.INK_MUTED)
    ax.set_xscale("log")
    ax.set_xlabel("rate (bpp)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("(a) anchor RD, fidelity axis", loc="left")
    P._tidy(ax)

    ax = axes[1]
    ax.errorbar(agg["bpp"], agg["u"], yerr=sem_u.values, color=P.PALETTE["noise"],
                linewidth=2.0, marker="o", markersize=5, capsize=3,
                markeredgecolor="white", markeredgewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("rate (bpp)")
    ax.set_ylabel(r"soft utility $U$")
    ax.set_title("(b) anchor RD, task axis", loc="left")
    P._tidy(ax)

    # The divergence: at one operating point, across clips, the two axes disagree.
    ax = axes[2]
    for crf, colour in ((24, P.PALETTE["blur"]), (36, P.PALETTE["noise"])):
        g = d[d["crf"] == crf]
        rho = spearmanr(g["psnr_db"], g["soft_u"]).statistic
        ax.scatter(g["psnr_db"], g["soft_u"], s=26, alpha=0.75, color=colour,
                   edgecolor="none", label=rf"CRF {crf}: $\rho={rho:+.2f}$")
        fit = np.polyfit(g["psnr_db"], g["soft_u"], 1)
        xs = np.linspace(g["psnr_db"].min(), g["psnr_db"].max(), 20)
        ax.plot(xs, np.polyval(fit, xs), color=colour, linewidth=1.2, linestyle="--")

    ax.set_xlabel("PSNR (dB) at fixed CRF")
    ax.set_ylabel(r"soft utility $U$")
    ax.set_title("(c) across clips, fidelity anti-predicts utility", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    P._tidy(ax)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT.with_suffix(f".{ext}"))
    plt.close(fig)
    print(f"wrote {OUT}.pdf")
    print(agg.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
