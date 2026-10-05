"""Figure: the two-timescale separation failure, from data already on disk.

The fixed-mu sweep trained 1500 steps at each mu and evaluated once at the end,
so its U(mu) is the CONVERGED inner-loop response. The dual runs evaluated on
the fly while mu was still moving, so their U at a given mu is whatever the
codec had managed by then. Plotting both against mu makes the shortfall between
them the direct measurement of eta_mu << eta_theta failing.

Panel B is the part that removes the obvious objection. If the dual run simply
sat at a lower rate, the utility gap would be a rate gap and say nothing about
timescales. It does not: at mu ~ 14 the dual run's bpp matches the sweep's to
within a fraction of a percent, so the comparison is already rate-matched and
the gap is inner-loop convergence alone.

v5 is excluded from both panels. It ran at lambda_0=0.05 against the sweep's
0.01, which is a different operating point on the RD curve, not a different
point on the same curve — its trajectory is not comparable to this sweep.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from srdo import plots as P  # noqa: E402

NB = ROOT / "notebooks" / "phase4-cctv"
OUT = ROOT / "outputs" / "figures"

SWEEP_CSV = NB / "v0" / "phase4b_fixed_mu_sweep.csv"

# (version, eta_mu, eval_every) — all at lambda_0=0.01, tau=0.52, YOUTUBE.
# eta_mu for v2 is not in its notebook config; it is recovered exactly from the
# first update, mu_1 = eta_mu * (tau - U_0), which gives 1.0.
DUAL_RUNS = [
    ("v2", 1.0, 300),
    ("v3", 15.0, 300),
    ("v4", 4.0, 1000),
]

DUAL_COLOUR = {"v2": P.PALETTE["blur"], "v3": P.PALETTE["noise"], "v4": P.INK_MUTED}
DUAL_MARKER = {"v2": "^", "v3": "o", "v4": "v"}
TAU = 0.52


def load() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    sweep = pd.read_csv(SWEEP_CSV).sort_values("mu").reset_index(drop=True)
    duals = {v: pd.read_csv(NB / v / "phase4_dual_trajectory.csv") for v, _, _ in DUAL_RUNS}
    return sweep, duals


def _sweep_at(sweep: pd.DataFrame, col: str, mu: float) -> float:
    """Converged value at an arbitrary mu, linearly interpolated between grid points."""
    return float(np.interp(mu, sweep["mu"], sweep[col]))


def eval_noise(duals: dict[str, pd.DataFrame]) -> float:
    """Per-evaluation noise sigma on soft_U, in utility units.

    Estimated from v4, whose mu moved only 0.43 per outer update across a region
    where the converged curve rises ~0.011 per unit mu — so its nine evaluations
    are close to nine repeated measurements of one operating point, and their
    scatter is dominated by val-set sampling rather than by real change in the
    codec. Single-eval differences smaller than about 2 sigma are not evidence
    of anything, which rules out most of the pilot's run-to-run comparisons.
    """
    return float(duals["v4"]["soft_U"].std(ddof=1))


def gap_table(sweep: pd.DataFrame, duals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-run shortfall at the trajectory's furthest mu.

    Reports the gap against both the instantaneous soft_U and the EMA the
    controller actually feeds back, since the EMA is what sets the residual and
    it lags further still.
    """
    rows = []
    for v, eta, ev in DUAL_RUNS:
        end = duals[v].iloc[-1]
        mu = float(end["mu"])
        u_conv = _sweep_at(sweep, "soft_U", mu)
        bpp_conv = _sweep_at(sweep, "bpp", mu)
        rows.append({
            "run": v, "eta_mu": eta, "eval_every": ev,
            "mu_final": mu,
            "U_dual": float(end["soft_U"]),
            "U_ema": float(end["u_ema"]),
            "U_converged": u_conv,
            "gap_U": u_conv - float(end["soft_U"]),
            "gap_U_ema": u_conv - float(end["u_ema"]),
            "bpp_dual": float(end["bpp"]),
            "bpp_converged": bpp_conv,
            "bpp_rel_err": float(end["bpp"]) / bpp_conv - 1.0,
        })
    df = pd.DataFrame(rows)
    sigma = eval_noise(duals)
    df["sigma"] = sigma
    # A single evaluation carries the full sigma; the endpoint gap is one such
    # evaluation, so this is the honest yardstick for it.
    df["gap_sigmas"] = df["gap_U"] / sigma
    df["significant"] = df["gap_sigmas"].abs() >= 2.0
    return df


def _panel_lag(ax, sweep, duals, gaps) -> None:
    ax.plot(sweep["mu"], sweep["soft_U"], color=P.PALETTE["codec"], linewidth=2.0,
            marker="D", markersize=5, markeredgecolor="white", markeredgewidth=0.8,
            label=r"converged sweep  $U_{\mathrm{sweep}}(\mu)$", zorder=3)

    ax.axhline(TAU, color=P.INK_MUTED, linewidth=0.9, linestyle=(0, (6, 4)), zorder=1)
    ax.annotate(rf"$\tau={TAU}$", xy=(0.99, TAU), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", ha="right", va="bottom",
                fontsize=8.5, color=P.INK_MUTED)

    for v, eta, ev in DUAL_RUNS:
        d = duals[v]
        ax.plot(d["mu"], d["soft_U"], color=DUAL_COLOUR[v], linewidth=1.5,
                linestyle="--", marker=DUAL_MARKER[v], markersize=4, alpha=0.9,
                markeredgecolor="white", markeredgewidth=0.6,
                label=rf"dual {v}: $\eta_\mu={eta:g}$, eval/{ev}", zorder=2)

    # The shortfall, drawn on the run that travelled furthest.
    g = gaps.set_index("run").loc["v3"]
    mu = g["mu_final"]
    ax.plot([mu, mu], [g["U_dual"], g["U_converged"]], color=P.INK,
            linestyle=":", linewidth=1.4, zorder=4)
    ax.annotate(rf"$\Delta U = {g['gap_U']:.3f}$" "\nat matched $\\mu$ and matched rate",
                xy=(mu, 0.5 * (g["U_dual"] + g["U_converged"])),
                xytext=(mu + 6, 0.5 * (g["U_dual"] + g["U_converged"]) - 0.045),
                arrowprops=dict(arrowstyle="->", color=P.INK, linewidth=0.9),
                fontsize=9, va="center")

    # Every dual run died below mu=15, so the full sweep range to mu=60 would
    # compress all three trajectories into the left margin. Panel (b) carries
    # the far end of the curve; this panel is about the ground actually covered.
    ax.set_xlim(-0.6, 20.0)

    ax.set_xlabel(r"semantic weight  $\mu$")
    ax.set_ylabel(r"soft utility  $U$")
    ax.set_title("(a) the dual run reads a utility the codec has not reached yet", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    P._tidy(ax)


def _panel_frontier(ax, sweep, duals) -> None:
    ax.plot(sweep["bpp"], sweep["soft_U"], color=P.PALETTE["codec"], linewidth=2.0,
            marker="D", markersize=5, markeredgecolor="white", markeredgewidth=0.8,
            label="converged sweep", zorder=3)
    for mu, bpp, u in zip(sweep["mu"], sweep["bpp"], sweep["soft_U"]):
        ax.annotate(rf"$\mu={mu:g}$", xy=(bpp, u), xytext=(3, -9),
                    textcoords="offset points", fontsize=7.5, color=P.INK_MUTED)

    for v, _, _ in DUAL_RUNS:
        d = duals[v]
        ax.plot(d["bpp"], d["soft_U"], color=DUAL_COLOUR[v], linewidth=1.2,
                linestyle="--", marker=DUAL_MARKER[v], markersize=4, alpha=0.9,
                markeredgecolor="white", markeredgewidth=0.6, label=f"dual {v}", zorder=2)

    ax.set_xlabel("rate  (bpp)")
    ax.set_ylabel(r"soft utility  $U$")
    ax.set_title("(b) the gap is not a rate gap: dual points sit inside the frontier", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    P._tidy(ax)


def main() -> None:
    P.apply_style()
    sweep, duals = load()
    gaps = gap_table(sweep, duals)

    OUT.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(OUT / "phase4_timescale_gap.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    _panel_lag(axes[0], sweep, duals, gaps)
    _panel_frontier(axes[1], sweep, duals)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"phase4_timescale_separation.{ext}")
    plt.close(fig)

    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(gaps.round(4).to_string(index=False))
    print(f"\nwrote {OUT / 'phase4_timescale_separation.pdf'}")
    print(f"wrote {OUT / 'phase4_timescale_gap.csv'}")


if __name__ == "__main__":
    main()
