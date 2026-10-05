"""Pick tau and eta_mu by simulation on the measured plant, not by guessing.

Every dual pilot chose eta_mu by hand and none converged. The fixed-mu sweep
already measured the thing a controller designer needs — the converged response
U(mu) — and the trajectories measured the two things that make it hard: the
inner loop's lag behind mu, and the noise on a single validation pass. With
those three, the outer loop can be simulated in milliseconds and the settings
chosen before spending three GPU-hours on them.

Plant model, one state per outer update k:

    mu_lag[k+1] = mu_lag[k] + (mu[k] - mu_lag[k]) / T     first-order tracking
    U[k]        = U_sweep(mu_lag[k]) + noise,  noise ~ N(0, sigma)

T is the inner loop's time constant in OUTER UPDATES, so it scales with
eval_every: doubling eval_every halves T and is the cheapest way to buy
separation. It is MEASURED, not fitted — the v6 sweep evaluates six times per
fixed-mu leg, so the step at which a leg first settles to within one sigma of
its final value gives T directly, with no model interposed.

The model is deliberately crude. It is not a claim about codec dynamics; it is
a way to reject settings that cannot possibly settle, which is what the pilots
needed and did not have.
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
from srdo.dual import DualController  # noqa: E402

NB = ROOT / "notebooks" / "phase4-cctv"
OUT = ROOT / "outputs" / "figures"

# The v6 sweep supersedes v0 as the plant: 3000 steps per leg against v0's 1500,
# 86 scored validation clips against roughly 19 effective, and six evaluations
# per leg rather than one, so the curve and its noise are both measured instead
# of assumed. Its utility values sit well below v0's because the validation pool
# changed — null-prior clips are excluded, leaving a set in which every clip
# contains a person — so the two are not comparable point for point and only v6
# is used here.
CURVES = pd.read_csv(NB / "v6" / "phase4b_sweep_curves.csv")
SWEEP = (CURVES.sort_values("step").groupby("mu").tail(1).sort_values("mu")
         .rename(columns={"soft_u": "soft_U"}).reset_index(drop=True))
V3 = pd.read_csv(NB / "v3" / "phase4_dual_trajectory.csv")


def _eval_sigma(curves: pd.DataFrame) -> float:
    """Per-evaluation noise, from residuals of a linear fit within each leg.

    Each leg is six evaluations of one operating point while the codec settles,
    so removing a straight-line trend leaves measurement noise. This supersedes
    the v4-trajectory estimate, which had to assume mu was near-stationary.
    """
    res = []
    for _, g in curves.groupby("mu"):
        u = g.sort_values("step")["soft_u"].to_numpy()
        t = np.arange(u.size)
        res += list(u - np.polyval(np.polyfit(t, u, 1), t))
    return float(np.std(res, ddof=1))


def settling_steps(curves: pd.DataFrame, tol: float = 1.0) -> dict:
    """Step at which each leg first comes within `tol`*sigma of its final value.

    A direct read on how fast the inner loop settles at FIXED mu, which is the
    quantity the two-timescale condition needs. It is not the same as the lag
    fitted from a dual trajectory, where mu is moving and the codec is chasing.
    """
    sigma = _eval_sigma(curves)
    out = {}
    for mu, g in curves.groupby("mu"):
        g = g.sort_values("step")
        final = g["soft_u"].iloc[-1]
        within = g[(g["soft_u"] - final).abs() <= tol * sigma]
        out[float(mu)] = int(within["step"].iloc[0]) if len(within) else None
    return out


SIGMA = _eval_sigma(CURVES)
U_MAX = float(SWEEP["soft_U"].max())
U_MIN = float(SWEEP["soft_U"].min())

# Candidate targets as fractions of the REACHABLE maximum, so the grid follows
# the plant instead of carrying values calibrated against a superseded curve.
TAU_FRACS = (0.75, 0.80, 0.86, 0.91, 0.95)
TAU_GRID = tuple(round(fr * U_MAX, 4) for fr in TAU_FRACS)


def u_conv(mu: np.ndarray | float) -> np.ndarray | float:
    """Converged utility at mu. Flat extrapolation past the swept range."""
    return np.interp(mu, SWEEP["mu"], SWEEP["soft_U"])


def mu_for_u(target: float) -> float:
    """Smallest mu whose converged utility reaches target."""
    return float(np.interp(target, SWEEP["soft_U"], SWEEP["mu"]))


def lag_steps(curves: pd.DataFrame = None) -> int:
    """Inner-loop settling time in INNER STEPS, measured at fixed mu.

    Superseding an earlier estimate, which fitted this constant to the v3 dual
    trajectory against the v0 sweep. That fit is no longer admissible: v3 and
    v0 were scored on the 24-clip validation set and the plant used here is the
    86-clip one, so the two utility scales differ by more than the effect being
    fitted, and the optimizer drives T below 1 — at which point the first-order
    tracker overshoots its target every update and the simulation is fiction.

    The v6 legs measure the quantity directly and better: six evaluations at a
    fixed mu, so the step at which a leg first comes within one sigma of its
    final value IS the settling time, with no model in between. The slowest leg
    governs, since the controller has to be stable at every operating point it
    passes through.
    """
    settle = settling_steps(curves if curves is not None else CURVES)
    return int(max(v for v in settle.values() if v is not None))


def lag_updates(eval_every: int, steps: int = None) -> float:
    """Settling time expressed in outer updates, which is what the loop sees.

    Clamped at 1: a first-order tracker with T < 1 moves further than its own
    target each update, which is a numerical artifact rather than a fast codec.
    Reaching T < 1 means the evaluation interval already exceeds the settling
    time, i.e. the two-timescale condition is comfortably met.
    """
    steps = steps if steps is not None else lag_steps()
    return max(steps / eval_every, 1.0)


def simulate(tau: float, eta: float, T: float, n_updates: int,
             beta: float = 0.8, deadband: float = 0.01,
             mu0: float = 0.0, mu_lag0: float = 0.0,
             seed: int = 0, mu_ref: float = 9.23) -> pd.DataFrame:
    """Run the repo's DualController against the fitted plant."""
    rng = np.random.default_rng(seed)
    ctl = DualController(tau=tau, mu_ref=mu_ref, eta_frac=eta / mu_ref,
                         beta=beta, deadband=deadband)
    ctl.reset(mu0)

    mu_lag, rows = mu_lag0, []
    for k in range(n_updates):
        u = float(u_conv(mu_lag)) + rng.normal(0.0, SIGMA)
        mu = ctl.step(u)
        rows.append({"k": k, "mu": mu, "mu_lag": mu_lag, "u": u,
                     "u_ema": ctl._u_ema, "u_conv": float(u_conv(mu_lag))})
        mu_lag += (mu - mu_lag) / T
    return pd.DataFrame(rows)


def mu_identifiability(tau: float) -> float:
    """Width of the mu interval whose converged U lies within one sigma of tau.

    This is the quantity that decides whether the dual variable means anything.
    Where the response is flat, many mu produce a utility indistinguishable
    from tau at the available measurement precision, so mu* is not identifiable
    and the controller wanders inside the band forever without either settling
    or being wrong. It is a property of tau and the plant alone — no choice of
    eta_mu or eval_every can repair it.
    """
    lo = float(np.interp(tau - SIGMA, SWEEP["soft_U"], SWEEP["mu"]))
    hi = float(np.interp(tau + SIGMA, SWEEP["soft_U"], SWEEP["mu"]))
    return hi - lo


def settled(traj: pd.DataFrame, tau: float, window: int = 6) -> dict:
    """Did it settle, and where? Judged on the last `window` updates."""
    tail = traj.tail(window)
    mu_span = float(tail["mu"].max() - tail["mu"].min())
    mu_end = float(traj["mu"].iloc[-1])
    u_end = float(tail["u_conv"].mean())

    # An overshooting loop gets projected back onto mu >= 0 and sits there with
    # zero span, which the span test alone would score as convergence. Hitting
    # the boundary after the opening transient is bang-bang, not settling.
    hit_floor = bool((traj["mu"].iloc[3:] <= 1e-9).any())
    needs_positive_mu = tau > float(u_conv(0.0)) + SIGMA

    return {
        "mu_end": mu_end,
        "mu_span_tail": mu_span,
        "u_end": u_end,
        "err_end": float(tau - u_end),
        "hit_floor": hit_floor,
        "settled": bool(
            mu_span < 0.10 * max(mu_end, 1.0)
            and abs(tau - u_end) < 0.02
            and not (hit_floor and needs_positive_mu)
        ),
    }


def sweep_settings(t_steps: int, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> pd.DataFrame:
    """Grid over (tau, eval_every, eta_mu), scored across seeds.

    One seed is not enough: with sigma = 0.014 on every reading, whether a
    marginal setting settles is itself a coin flip, and picking a setting that
    worked once is how the pilot got here.
    """
    rows = []
    for tau in TAU_GRID:
        mu_star = mu_for_u(tau)
        width = mu_identifiability(tau)
        for eval_every in (300, 500, 1000):
            # T is in outer updates; longer inner stretches mean fewer updates
            # to cover the same lag, so T shrinks proportionally.
            T = lag_updates(eval_every, t_steps)
            n_updates = 12000 // eval_every
            for eta in (2.0, 5.0, 10.0, 20.0, 40.0):
                runs = [settled(simulate(tau, eta, T, n_updates, seed=s), tau)
                        for s in seeds]
                rows.append({
                    "tau": tau, "mu_star": mu_star, "mu_width_1sigma": width,
                    "eval_every": eval_every, "eta_mu": eta,
                    "n_updates": n_updates, "T": round(T, 2),
                    "settle_rate": float(np.mean([r["settled"] for r in runs])),
                    "mu_end_mean": float(np.mean([r["mu_end"] for r in runs])),
                    "mu_end_std": float(np.std([r["mu_end"] for r in runs])),
                    "err_end_mean": float(np.mean([r["err_end"] for r in runs])),
                    "hit_floor_rate": float(np.mean([r["hit_floor"] for r in runs])),
                })
    return pd.DataFrame(rows)


def _panel_plant(ax, t_steps: int) -> None:
    grid = np.linspace(0, 60, 400)
    ax.plot(grid, u_conv(grid), color=P.PALETTE["codec"], linewidth=2.0,
            label=r"measured plant  $U_{\mathrm{sweep}}(\mu)$")
    ax.fill_between(grid, u_conv(grid) - SIGMA, u_conv(grid) + SIGMA,
                    color=P.PALETTE["codec"], alpha=0.15, linewidth=0,
                    label=rf"$\pm\sigma$ of one eval ({SIGMA:.3f})")

    for tau, style in ((TAU_GRID[-1], "-"), (TAU_GRID[1], "--")):
        mu_star = mu_for_u(tau)
        ax.axhline(tau, color=P.INK_MUTED, linewidth=0.8, linestyle=(0, (5, 4)))
        ax.plot([mu_star], [tau], marker="*", markersize=13, color=P.PALETTE["noise"],
                markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        # Local gain dU/dmu is what sets loop conditioning; it collapses at the
        # top of the curve, which is where tau=0.52 sits.
        gain = float(np.gradient(u_conv(grid), grid)[np.argmin(abs(grid - mu_star))])
        ax.annotate(rf"$\tau={tau}$: $\mu^*\approx{mu_star:.1f}$, "
                    rf"$dU/d\mu\approx{gain:.4f}$",
                    xy=(mu_star, tau), xytext=(6, -14 if tau == 0.52 else 8),
                    textcoords="offset points", fontsize=8.5)

    ax.set_xlabel(r"semantic weight  $\mu$")
    ax.set_ylabel(r"soft utility  $U$")
    ax.set_title(rf"(a) measured plant; inner loop settles in ~{t_steps} steps",
                 loc="left")
    ax.legend(loc="lower right", fontsize=8)
    P._tidy(ax)


def _panel_traj(ax, t_steps: int) -> None:
    """The pilot setting against the recommended one, same plant, same noise."""
    hi, knee = TAU_GRID[-1], TAU_GRID[1]
    cases = [
        (hi, 4.0, 1000, P.PALETTE["blur"],
         rf"ceiling $\tau$={hi:.3f}, $\eta_\mu$=4"),
        (hi, 15.0, 300, P.INK_MUTED,
         rf"ceiling $\tau$={hi:.3f}, $\eta_\mu$=15"),
        (knee, 10.0, 500, P.PALETTE["noise"],
         rf"knee $\tau$={knee:.3f}, $\eta_\mu$=10"),
    ]
    for tau, eta, ev, colour, label in cases:
        T = lag_updates(ev, t_steps)
        traj = simulate(tau, eta, T, 12000 // ev)
        ax.plot(traj["k"] * ev, traj["mu"], color=colour, linewidth=1.8, label=label)
        ax.axhline(mu_for_u(tau), color=colour, linewidth=0.8, linestyle=":")

    ax.set_xlabel("inner steps")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(r"(b) simulated $\mu$; dotted lines are each run's $\mu^*$", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    P._tidy(ax)


def main() -> None:
    P.apply_style()
    t_steps = lag_steps()

    print(f"eval noise sigma        {SIGMA:.4f}")
    print(f"reachable U range       {U_MIN:.3f} .. {U_MAX:.3f}")
    print(f"settling per leg (steps) {settling_steps(CURVES)}")
    print(f"inner-loop settling     {t_steps} steps "
          f"= {lag_updates(500, t_steps):.1f} outer updates at eval/500\n")

    for tau in TAU_GRID:
        mu_star = mu_for_u(tau)
        gain = float(np.gradient(u_conv(np.linspace(0, 60, 400)),
                                 np.linspace(0, 60, 400))[
            int(np.argmin(abs(np.linspace(0, 60, 400) - mu_star)))])
        print(f"tau={tau:.2f}  mu*={mu_star:6.2f}  dU/dmu={gain:.5f}  "
              f"headroom={(U_MAX - tau):.3f}  pct_of_max={100 * tau / U_MAX:.0f}%  "
              f"mu identifiable to +/-{mu_identifiability(tau) / 2:.1f}")

    grid = sweep_settings(t_steps)
    grid.to_csv(OUT / "phase4_controller_grid.csv", index=False)

    ok = grid[grid.settle_rate >= 0.8].sort_values(
        ["tau", "eval_every", "eta_mu"])
    print(f"\n{len(ok)}/{len(grid)} settings settle in >=4 of 5 seeds "
          f"within 12000 inner steps")
    with pd.option_context("display.width", 170, "display.max_columns", 20):
        print(ok[["tau", "mu_star", "eval_every", "eta_mu", "settle_rate",
                  "mu_end_mean", "mu_end_std", "err_end_mean"]]
              .round(3).to_string(index=False))

    # The settling test asks whether the loop stops moving. This asks the harder
    # question: does it stop in the same PLACE regardless of how it was tuned?
    # mu is claimed to be the Lagrange multiplier of the constraint, a property
    # of the problem. If settings that all satisfy tau recover mu* values a
    # factor of two apart, mu is reporting the tuning instead.
    print("\nrecovered mu* across the settings that settle, per tau:")
    for tau, g in ok.groupby("tau"):
        lo, hi = g["mu_end_mean"].min(), g["mu_end_mean"].max()
        print(f"  tau={tau:.2f}  mu* in [{lo:5.1f}, {hi:5.1f}]  "
              f"spread {hi - lo:5.1f} ({hi / max(lo, 1e-9):.1f}x)  "
              f"|  all satisfy tau to within {g['err_end_mean'].abs().max():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    _panel_plant(axes[0], t_steps)
    _panel_traj(axes[1], t_steps)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"phase4_controller_design.{ext}")
    plt.close(fig)
    print(f"\nwrote {OUT / 'phase4_controller_design.pdf'}")


if __name__ == "__main__":
    main()
