"""Figures for the paper.

Palette is fixed and validated — do not substitute ad-hoc colours. The three
categorical slots below clear all-pairs CVD separation (worst deltaE 13.0
deutan), the normal-vision floor (16.3), and 3:1 contrast on a light surface.
Every series also carries a distinct linestyle and marker, so identity survives
greyscale printing, which colour alone does not.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Validated categorical slots: blue / orange / violet, keyed by distortion
# family. Weight (fg/bg) is carried by linestyle, never by hue — so the
# foreground/background contrast survives greyscale printing and CVD.
PALETTE = {"blur": "#2a78d6", "noise": "#eb6834", "codec": "#4a3aa7"}
MARKERS = {"blur": "o", "noise": "s", "codec": "D"}
LINESTYLES = {"fg": "-", "bg": "--"}
LABELS = {"blur": "Blur", "noise": "Noise", "codec": "Codec (x265)"}
WEIGHT_LABEL = {"fg": r"in $\Omega$", "bg": r"outside $\Omega$ (control)"}

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#d8d7d2"


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.edgecolor": INK_MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,          # recessive: grid must sit behind the marks
        "grid.linewidth": 0.6,
        "text.color": INK,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "legend.frameon": False,
        "figure.facecolor": "white",
    })


def _tidy(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_us_curve(
    probe_df: pd.DataFrame,
    utility_col: str = "soft_u",
    ax=None,
    weights: tuple[str, ...] = ("fg",),
):
    """Figure 1 — the A1 monotonicity curve: utility U against semantic distortion S.

    One axis only. Mean over clips per strength, with a standard-error band.
    Defaults to the foreground arms, which are what A1 is about.
    """
    ok = probe_df[probe_df["error"].fillna("") == ""] if "error" in probe_df else probe_df
    ok = ok[ok[utility_col].notna() & ok["weight"].isin(weights)]
    if ax is None:
        _, ax = plt.subplots(figsize=(6.2, 4.2))

    for (family, weight), g in ok.groupby(["family", "weight"]):
        curve = g.groupby("strength").agg(
            s=("s_semantic", "mean"), u=(utility_col, "mean"),
            sd=(utility_col, "std"), n=(utility_col, "size"),
        ).reset_index().sort_values("s")

        se = curve["sd"] / np.sqrt(curve["n"].clip(lower=1))
        colour = PALETTE.get(family, INK_MUTED)
        label = LABELS.get(family, family)
        if len(weights) > 1:
            label = f"{label} — {WEIGHT_LABEL[weight]}"

        ax.fill_between(curve["s"], curve["u"] - se, curve["u"] + se,
                        color=colour, alpha=0.14, linewidth=0)
        ax.plot(curve["s"], curve["u"], color=colour, linewidth=2.0,
                linestyle=LINESTYLES.get(weight, "-"),
                marker=MARKERS.get(family, "o"), markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, label=label)

    ax.set_xlabel(r"Semantic distortion  $S=\sum_i \Omega_i D_i$")
    ax.set_ylabel(r"Task utility  $U$  (preserved detectability)")
    ax.set_title("A1: utility is non-increasing in semantic distortion", loc="left")
    ax.legend(loc="upper right", fontsize=8.5)
    _tidy(ax)
    return ax


def plot_density(density_df: pd.DataFrame, claim: float = 0.15, ax=None):
    """Figure 2 — distribution of per-clip semantic density k = E[Omega].

    A histogram is the right form: the question is the shape and spread of k
    across clips, not a comparison between named categories. The section 3.9
    claim is drawn as a reference line so the reader can check it by eye.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.6))

    k = density_df["mean_omega"].to_numpy()
    ax.hist(k, bins=30, color=PALETTE["blur"], alpha=0.85, edgecolor="white", linewidth=0.6)

    mean_k = float(k.mean())
    ax.axvline(mean_k, color=INK, linewidth=1.6)
    ax.annotate(f"measured  k = {mean_k:.3f}", xy=(mean_k, ax.get_ylim()[1] * 0.92),
                xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=INK)

    ax.axvline(claim, color=PALETTE["noise"], linewidth=1.6, linestyle="--")
    ax.annotate(f"§3.9 claim  k < {claim}", xy=(claim, ax.get_ylim()[1] * 0.75),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8.5, color=PALETTE["noise"])

    ax.set_xlabel(r"Semantic density  $k=\mathbb{E}[\Omega]$  (per clip)")
    ax.set_ylabel("Clips")
    ax.set_title("Measured semantic density", loc="left")
    _tidy(ax)
    return ax


def plot_selectivity(probe_df: pd.DataFrame, utility_col: str = "soft_u", axes=None):
    """Omega selectivity: within each family, fg vs bg at matched TOTAL distortion.

    This is the plot that answers "you induced the correlation by construction".
    If the two curves coincided, Omega would carry no task-relevant localisation
    and the spatial weight would have nothing to exploit. Separation is the
    evidence that it does. Plotted against total distortion, NOT strength: with
    k ~ 0.09 the background arm perturbs ~10x more area, so matching on strength
    would be meaningless.
    """
    ok = probe_df[probe_df["error"].fillna("") == ""] if "error" in probe_df else probe_df
    ok = ok[ok[utility_col].notna()]
    families = [f for f in ("blur", "noise", "codec") if f in set(ok["family"])]

    if axes is None:
        _, axes = plt.subplots(1, len(families), figsize=(4.0 * len(families), 3.8),
                               sharey=True)
    axes = np.atleast_1d(axes)

    for ax, family in zip(axes, families):
        gf = ok[ok["family"] == family]
        for weight, g in gf.groupby("weight"):
            curve = g.groupby("strength").agg(
                d=("d_total", "mean"), u=(utility_col, "mean"),
            ).reset_index().sort_values("d")
            ax.plot(curve["d"], curve["u"], color=PALETTE.get(family, INK_MUTED),
                    linewidth=2.0, linestyle=LINESTYLES.get(weight, "-"),
                    marker=MARKERS.get(family, "o"), markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.8,
                    label=WEIGHT_LABEL[weight])
        ax.set_title(LABELS.get(family, family), loc="left")
        ax.set_xlabel(r"Total distortion  $\bar{D}$")
        ax.legend(loc="upper right", fontsize=8)
        _tidy(ax)

    axes[0].set_ylabel(r"Task utility  $U$")
    return axes
