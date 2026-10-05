"""Fixed-mu sweep, instrumented. Kaggle payload for `%%writefile`.

Differences from the pilot sweep, each one closing a specific hole:

  TEST THE REALLOCATION CLAIM DIRECTLY. This is the point of the run. The
  unnormalised sweep raised rate 2.36x across mu and IMPROVED plain distortion
  2.10x, which is what a global rate increase produces and not what moving bits
  toward Omega produces. The cause is in the objective: the weight has mean
  1 + mu*E[Omega], so at mu=60 with E[Omega]~0.088 the whole distortion term is
  scaled 6.28x, arithmetically the same as dividing lambda_0 by 6.28. Dividing
  the weight by its own mean pins the operating point, leaving mu to move only
  the spatial split. If U(mu) still rises under normalisation, reallocation is
  demonstrated at constant rate; if it flattens, the earlier gain was rate all
  along. Either outcome settles the question the unnormalised sweep could not.

  TRAIN LONGER, AND MEASURE WHETHER THAT WAS ENOUGH. 3000 steps per mu with an
  evaluation every 500, so the convergence curve is recorded rather than
  assumed. The v6 legs settled within 500-2000 steps at fixed mu, so 3000 has
  margin; keeping the schedule identical also keeps the two sweeps comparable.

  BIGGER, CLEANER VAL SET. 24 clips gave a per-evaluation sigma of 0.0139 on
  soft utility, which is the same order as the differences the pilot was
  steering on. 96 clips cuts that ~2x. Clips whose semantic prior is
  identically zero are excluded up front: they have no pseudo-GT, so they were
  already being silently dropped mid-evaluation, which made the effective val
  size smaller and noisier than the configured one.

  PER-CLIP RECORDS. The mean alone cannot express the constraint, which is
  written per clip. `srdo.evaluate` keeps the full vector and the val-set S.

  CHECKPOINTS. The pilot deleted each codec after evaluating it, so no dual run
  could ever be warm-started from a converged operating point.

  TEMPORAL SAMPLING. The pilot's loader read frames 0-7 of every clip, every
  time. 627 training clips therefore supplied 627 fixed GOPs and no temporal
  variation whatsoever. GOP starts are now drawn across each clip.

Set SRDO_SRC to a package build that contains src/srdo/evaluate.py and
outputs/phase4_omega_crop.csv.
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
DATA_ROOT = "/kaggle/input/datasets/jonathannield/cctv-action-recognition-dataset/Videos/Videos"
WORK = pathlib.Path("/kaggle/working")


def _latest_package(base=_MODEL_BASE):
    """Highest-numbered attached model version.

    Kaggle model versions are immutable, so every re-upload mints a new number
    and a hardcoded path silently keeps running the previous code — which is
    exactly how a fix gets tested against the bug it was meant to remove.
    """
    versions = sorted(int(p.name) for p in pathlib.Path(base).iterdir() if p.name.isdigit())
    if not versions:
        raise RuntimeError(f"no numbered model version attached under {base}")
    return f"{base}/{versions[-1]}"


SRDO_SRC = _latest_package()
print(f"package: {SRDO_SRC}")
sys.path.insert(0, SRDO_SRC)

from srdo.detection import detect, masks_from  # noqa: E402
from srdo.evaluate import evaluate_codec, save_checkpoint, ssf_forward  # noqa: E402
from srdo.loss import srdo_loss  # noqa: E402
from srdo.omega import compute_omega  # noqa: E402
from srdo.video import Crop, read_frames  # noqa: E402

CFG = dict(
    domain="YOUTUBE",
    lambda_0=0.01,          # multiplies bpp: RAISING it spends FEWER bits
    patch=256,
    lr=1e-4,
    gop=8,
    mu_grid=[0.0, 5.0, 15.0, 30.0, 60.0],   # unchanged, so v6 stays comparable
    train_steps=3000,       # v6 legs settled by 500-2000 steps; keeps margin
    eval_every=500,         # within-leg convergence curve
    val_clips=96,
    seed=0,
    drop_empty_omega=True,
    # Divide the weight by its own mean so mu moves only the SPATIAL split and
    # not the operating point. The unnormalised sweep raised rate 2.36x and
    # improved plain distortion 2.10x, which is what a global rate increase
    # produces; the separable spatial effect was an 18% shift in S/D. This run
    # tests the reallocation claim directly instead of inferring it.
    normalize_weight=True,
)

if not pathlib.Path(DATA_ROOT).exists():
    raise SystemExit(
        f"video corpus not found at {DATA_ROOT}\n"
        f"attach the consumer-IP dataset to this notebook, or run "
        f"run_phase4d_density.py if you meant the domain D sweep")

man = pd.read_csv(pathlib.Path(SRDO_SRC) / "outputs" / "phase0_manifest.csv")
pool = man[(man.usable_codec) & (man.source == CFG["domain"])].reset_index(drop=True)

if CFG["drop_empty_omega"]:
    dens = pd.read_csv(pathlib.Path(SRDO_SRC) / "outputs" / "phase4_omega_crop.csv")
    live = set(dens.loc[~dens["crop_is_empty"], "clip"])
    before = len(pool)
    # pool["clip"], never pool.clip: attribute access resolves to DataFrame.clip,
    # the numeric clamping method, and fails with an unrelated AttributeError.
    pool = pool[pool["clip"].isin(live)].reset_index(drop=True)
    print(f"dropped {before - len(pool)}/{before} clips with Omega == 0 on the crop")

tr = pool.sample(frac=0.8, random_state=CFG["seed"])
va = pool.drop(tr.index).head(CFG["val_clips"])
print(f"{len(pool)} clips; {len(tr)} train, {len(va)} val")

TAG = "_norm" if CFG["normalize_weight"] else ""

_rng = np.random.default_rng(CFG["seed"])


def scale_height_for(crop_w, crop_h):
    """Decode height that leaves BOTH dimensions at least `patch` after scaling.

    ffmpeg's scale=-2:patch fixes the height and lets width follow the aspect
    ratio, so a portrait clip ends up narrower than the patch and the centre
    crop below silently returns fewer columns than it asked for. SSF then feeds
    a non-multiple-of-64 width through four stride-2 convolutions while its
    hyper-decoder upsamples z by a clean factor of 16, and the two disagree:
    a 272x480 source scales to 146 wide, giving a latent of width 10 against a
    prior of width 16. Scaling the SHORT side to `patch` avoids the whole class.
    """
    s = CFG["patch"]
    if crop_w >= crop_h:
        return s
    # Width is the short side: pick the height that carries width up to `patch`.
    # +2 of margin absorbs ffmpeg's rounding of the derived dimension to even.
    return int(np.ceil(s * crop_h / crop_w / 2)) * 2 + 2


def load_gop(row, n=None, start=None):
    """One GOP as (x, omega). `start` defaults to a random position in the clip."""
    n = n or CFG["gop"]
    path = str(pathlib.Path(DATA_ROOT) / str(row["action"]) / str(row["clip"]))
    cw, ch = int(row["crop_w"]), int(row["crop_h"])
    crop = Crop(cw, ch, int(row["crop_x"]), int(row["crop_y"]))

    if start is None:
        # Leave a margin so the requested window always exists; short clips fall
        # back to the head rather than erroring.
        span = max(int(row.get("nb_frames", 0)) - n - 1, 0)
        start = int(_rng.integers(0, span)) if span > 0 else 0

    frames = read_frames(path, crop=crop, height=scale_height_for(cw, ch),
                         max_frames=n, start=start)
    omega = compute_omega(frames, masks=masks_from(detect(frames), frames.shape[:3]))

    x = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div(255.0)
    o = torch.from_numpy(omega).unsqueeze(1).float()

    s = CFG["patch"]
    _, _, H, W = x.shape
    top = (H - s) // 2 if H >= s else 0
    left = (W - s) // 2 if W >= s else 0
    x, o = x[:, :, top:top + s, left:left + s], o[:, :, top:top + s, left:left + s]

    # A wrong shape here costs hours of GPU time to discover inside the codec,
    # where it surfaces as an unrelated-looking tensor mismatch. Fail loudly.
    if x.shape[-2:] != (s, s):
        raise ValueError(f"{row['clip']}: got {tuple(x.shape[-2:])}, expected ({s},{s}) "
                         f"from source crop {cw}x{ch}")
    return x, o


def load_gop_fixed(row):
    """Validation loader: start=0 always, so the val set is identical across mu."""
    return load_gop(row, start=0)


def train_one_mu(mu, accelerator):
    codec = ssf2020(quality=4, pretrained=True).train()
    codec.update(force=True)
    opt = torch.optim.Adam((p for p in codec.parameters() if p.requires_grad),
                           lr=CFG["lr"])
    codec, opt = accelerator.prepare(codec, opt)

    curve, skipped = [], {}
    pbar = tqdm(range(1, CFG["train_steps"] + 1), desc=f"mu={mu:g}",
                disable=not accelerator.is_local_main_process)

    for step in pbar:
        # Retry on a fresh clip rather than aborting the run. Each rank draws
        # independently and still performs exactly one forward/backward per
        # iteration, so a retry here cannot desynchronise the gradient reduction.
        x = om = None
        for attempt in range(8):
            row = tr.sample(1, random_state=int(_rng.integers(1 << 31))).iloc[0]
            try:
                x, om = load_gop(row)
                break
            except Exception as exc:
                skipped[str(row["clip"])] = f"{type(exc).__name__}: {exc}"
        if x is None:
            # Surface the actual reason. Failing at step 1 usually means the
            # video dataset is not attached, not that the clips are bad, and a
            # bare count sends you looking in the wrong place.
            why = list(skipped.items())[-3:]
            raise RuntimeError(
                f"8 consecutive clips failed to load at step {step}. "
                f"DATA_ROOT={DATA_ROOT} exists={pathlib.Path(DATA_ROOT).exists()}. "
                f"Last errors: {why}")

        x = x.to(accelerator.device)
        om = om.to(accelerator.device)

        x_hat, lik = ssf_forward(codec, x)
        terms = srdo_loss(x_hat, x, lik, om, mu=mu, lambda_0=CFG["lambda_0"],
                          num_pixels=CFG["patch"] ** 2,
                          normalize_weight=CFG["normalize_weight"])

        opt.zero_grad()
        accelerator.backward(terms.total)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(codec.parameters(), 1.0)
        opt.step()

        if accelerator.is_local_main_process:
            pbar.set_postfix(bpp=f"{float(terms.bpp):.4f}",
                             S=f"{float(terms.semantic_dist):.2e}")

        if step % CFG["eval_every"] == 0:
            unwrapped = accelerator.unwrap_model(codec)
            ev = evaluate_codec(unwrapped, (r for _, r in va.iterrows()),
                                load_gop_fixed, device=accelerator.device,
                                num_pixels=CFG["patch"] ** 2)
            if accelerator.is_local_main_process:
                row = {"mu": mu, "step": step, **ev.summary(tau=0.45)}
                curve.append(row)
                print(f"\n  mu={mu:g} step={step} U={row['soft_u']:.4f}"
                      f" +/-{row['soft_u_sem']:.4f}  bpp={row['bpp']:.4f}"
                      f"  S={row['s_semantic']:.3e}  sat={row['satisfaction_rate']:.2f}"
                      f"  n={row['n_scored']}")
                pd.DataFrame(curve).to_csv(WORK / f"sweep_curve_mu{mu:g}{TAG}.csv", index=False)
                ev.per_clip.to_csv(WORK / f"sweep_perclip_mu{mu:g}_step{step}{TAG}.csv",
                                   index=False)

    if accelerator.is_local_main_process:
        save_checkpoint(WORK / f"codec_mu{mu:g}{TAG}.pt", accelerator.unwrap_model(codec),
                        mu=mu, step=CFG["train_steps"], lambda_0=CFG["lambda_0"])
        # Silent skips would misrepresent the training pool, so name them.
        if skipped:
            print(f"\n  mu={mu:g}: {len(skipped)} distinct clips failed to load")
            for clip, why in list(skipped.items())[:10]:
                print(f"    {clip}: {why}")
    return curve


def main():
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp])

    all_curves = []
    for mu in CFG["mu_grid"]:
        all_curves += train_one_mu(mu, accelerator)
        if accelerator.is_local_main_process:
            df = pd.DataFrame(all_curves)
            df.to_csv(WORK / f"phase4b_sweep_curves{TAG}.csv", index=False)

            # The headline table is the last evaluation of each mu leg; the
            # earlier ones are what show whether that last one had settled.
            final = df.sort_values("step").groupby("mu").tail(1)
            print("\n" + final[["mu", "step", "soft_u", "soft_u_sem", "bpp",
                                "s_semantic", "satisfaction_rate"]]
                  .round(4).to_string(index=False))
            final.to_csv(WORK / f"phase4b_fixed_mu_sweep{TAG}.csv", index=False)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
