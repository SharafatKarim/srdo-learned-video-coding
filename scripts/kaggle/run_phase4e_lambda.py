"""Lambda_0 sweep on domain D: the operating points a reportable BD-BR needs.

WHY THIS RUN. On the neural run's own validation GOPs, re-encoded with x265 for
a like-for-like comparison, SSF reaches U = 0.363 on 0.0745 bpp where x265 needs
0.1140 — a 35% rate saving at matched detection utility. That is the first
clearly positive result in this line, and it currently rests on a single
operating point. A Bjontegaard number over the existing curves is meaningless:
the five mu legs span 0.011 of utility against the anchor's 0.304, a 4% overlap,
and `srdo.bdrate` reports that fraction precisely so the figure cannot be quoted
by accident.

What produces comparable curves is a sweep over lambda_0, which moves the
operating point, rather than over mu, which at fixed lambda_0 does not. The
lambda_0 values below are chosen to bracket the anchor's utility range, roughly
U = 0.17 to 0.48; since raising lambda_0 raises the price of rate, the largest
value sits at the low-rate end.

TWO PASSES, ORDERED SO THE FIRST IS SELF-SUFFICIENT. mu = 0 gives the neural
baseline against the standards anchor, which is the headline RD comparison.
mu = 60 gives the SRDO curve on the same rates. The grid runs all of mu = 0
first and writes after every leg, so a session that runs out of time still
yields a complete, usable curve.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from compressai.zoo import ssf2020
from tqdm.auto import tqdm

_MODEL_BASE = "/kaggle/input/models/msharafatkarim/python-cctv-srdo/pytorch/default"
WORK = pathlib.Path("/kaggle/working")


def _latest_package(base=_MODEL_BASE):
    versions = sorted(int(p.name) for p in pathlib.Path(base).iterdir() if p.name.isdigit())
    if not versions:
        raise RuntimeError(f"no numbered model version attached under {base}")
    return f"{base}/{versions[-1]}"


SRDO_SRC = _latest_package()
print(f"package: {SRDO_SRC}")
sys.path.insert(0, SRDO_SRC)

from srdo import domain_d as dd  # noqa: E402
from srdo.evaluate import evaluate_codec, save_checkpoint, ssf_forward  # noqa: E402
from srdo.loss import srdo_loss  # noqa: E402

DATA_ROOT = dd.find_dataset_root()
print(f"data:    {DATA_ROOT}")

# Raising lambda_0 prices rate higher and therefore spends fewer bits, so the
# largest value is the low-utility end of the curve.
LAMBDAS = [0.002, 0.005, 0.01, 0.03, 0.08]

CFG = dict(
    patch=dd.PATCH,
    lr=1e-4,
    train_steps=2000,        # v8 legs settled at 500-2000 steps at fixed mu
    eval_every=500,
    seed=0,
    normalize_weight=True,
    val_scene="scene_S009",
    val_windows_per_clip=8,
    mus=[0.0, 60.0],
)
TAG = "_lam"
_rng = np.random.default_rng(CFG["seed"])


def train_one(mu, lam, store, tr_clips, va, accelerator):
    codec = ssf2020(quality=4, pretrained=True).train()
    codec.update(force=True)
    opt = torch.optim.Adam((p for p in codec.parameters() if p.requires_grad), lr=CFG["lr"])
    codec, opt = accelerator.prepare(codec, opt)

    def load_val(row):
        return dd.make_gop(store, row["_clip"], int(row["_idx"]))

    label = f"mu{mu:g}_lam{lam:g}"
    curve = []
    pbar = tqdm(range(1, CFG["train_steps"] + 1), desc=label,
                disable=not accelerator.is_local_main_process)
    for step in pbar:
        clip = tr_clips[int(_rng.integers(len(tr_clips)))]
        x, om = dd.make_gop(store, clip, int(_rng.integers(store[clip].shape[0])))
        x, om = x.to(accelerator.device), om.to(accelerator.device)

        x_hat, lik = ssf_forward(codec, x)
        terms = srdo_loss(x_hat, x, lik, om, mu=mu, lambda_0=lam,
                          num_pixels=CFG["patch"] ** 2,
                          normalize_weight=CFG["normalize_weight"])
        opt.zero_grad()
        accelerator.backward(terms.total)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(codec.parameters(), 1.0)
        opt.step()
        if accelerator.is_local_main_process:
            pbar.set_postfix(bpp=f"{float(terms.bpp):.4f}")

        if step % CFG["eval_every"] == 0:
            ev = evaluate_codec(accelerator.unwrap_model(codec),
                                (r for _, r in va.iterrows()), load_val,
                                device=accelerator.device, num_pixels=CFG["patch"] ** 2)
            if accelerator.is_local_main_process:
                row = {"mu": mu, "lambda_0": lam, "step": step, **ev.summary(tau=0.45)}
                curve.append(row)
                print(f"\n  {label} step={step} U={row['soft_u']:.4f} "
                      f"bpp={row['bpp']:.4f} n={row['n_scored']}")
                ev.per_clip.to_csv(WORK / f"perclip_{label}_step{step}{TAG}.csv", index=False)

    if accelerator.is_local_main_process:
        save_checkpoint(WORK / f"codec_{label}{TAG}.pt", accelerator.unwrap_model(codec),
                        mu=mu, step=CFG["train_steps"], lambda_0=lam)
    return curve


def main():
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    store = dd.preload(DATA_ROOT)
    tr_clips, val_clips = dd.split_by_scene(store, CFG["val_scene"])
    va = dd.build_val(store, val_clips, CFG["val_windows_per_clip"])
    print(f"train {len(tr_clips)} clips; val {len(val_clips)} clips -> {len(va)} GOPs "
          f"from {CFG['val_scene']}\n")

    # mu = 0 first, so an interrupted session still leaves a complete curve.
    grid = [(mu, lam) for mu in CFG["mus"] for lam in LAMBDAS]
    print(f"{len(grid)} legs x {CFG['train_steps']} steps\n")

    rows = []
    for mu, lam in grid:
        rows += train_one(mu, lam, store, tr_clips, va, accelerator)
        if accelerator.is_local_main_process:
            df = pd.DataFrame(rows)
            df.to_csv(WORK / f"phase4e_curves{TAG}.csv", index=False)
            final = df.sort_values("step").groupby(["mu", "lambda_0"]).tail(1)
            final.to_csv(WORK / f"phase4e_operating_points{TAG}.csv", index=False)
            print("\n" + final[["mu", "lambda_0", "soft_u", "soft_u_sem", "bpp",
                                "hard_u"]].round(4).to_string(index=False) + "\n")
        torch.cuda.empty_cache()

    if accelerator.is_local_main_process:
        _report(pd.DataFrame(rows))


def _report(df):
    """BD-BR against the matched x265 anchor, with the overlap fraction shown."""
    from srdo.bdrate import RDCurve, bd_rate, bd_quality, overlap_fraction

    anchor_csv = pathlib.Path(SRDO_SRC) / "outputs" / "phase5_anchor_matched.csv"
    if not anchor_csv.exists():
        print(f"\nanchor not in package ({anchor_csv}); compute BD-BR locally")
        return
    a = pd.read_csv(anchor_csv).groupby("crf").agg(
        bpp=("bpp", "mean"), U=("soft_u", "mean"), psnr=("psnr_db", "mean")).sort_values("bpp")
    final = df.sort_values("step").groupby(["mu", "lambda_0"]).tail(1)

    for mu, g in final.groupby("mu"):
        g = g.sort_values("bpp")
        if len(g) < 4:
            continue
        try:
            anc = RDCurve("x265", a.bpp.values, a.U.values)
            test = RDCurve(f"SSF mu={mu:g}", g.bpp.values, g.soft_u.values)
            ov = overlap_fraction(anc, test)
            print(f"\nmu={mu:g}: BD-Rate(U) {bd_rate(anc, test):+.1f}%  "
                  f"BD-Utility {bd_quality(anc, test):+.4f}  overlap {ov:.2f}"
                  f"{'' if ov >= 0.5 else '   <-- too little overlap to quote'}")
        except Exception as exc:
            print(f"\nmu={mu:g}: BD not computable: {exc}")


if __name__ == "__main__":
    main()
