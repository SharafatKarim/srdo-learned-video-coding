"""Rate-controlled fixed-mu sweep on the multi-camera 1080p corpus (domain D).

THE QUESTION. Under rate control the consumer-IP sweep produced a usable
utility range of 3.5 sigma, across which mu* is unidentifiable everywhere, so
utility-targeted control is infeasible there. The obvious explanation is that
E[Omega] ~ 0.088 leaves only 9% of the frame available to improve: no
allocation policy can do better than perfect allocation over 9% of the pixels.
Domain D at NATIVE resolution is far denser. Measured on the GOPs this script
actually keeps, k = 0.41 against the consumer-IP crop's 0.088, a factor of 4.6,
with 4% of GOPs empty against 19%. If the constant-rate headroom scales with
density, the finding becomes a usability criterion — task-constrained control
needs a prior dense enough to move the metric — and if it does not, the ceiling
belongs to the backbone and should be reported as such.

(An earlier centre-crop estimate put domain D at k = 0.208. That understates
what is available here: a centre crop is not where the subjects are, and the
per-GOP window below roughly doubles the density again.)

Protocol is held identical to the consumer-IP run wherever it can be: same mu
grid, 3000 steps per leg, evaluation every 500, lambda_0 = 0.01, patch 256,
8-frame GOPs, and normalize_weight on so rate is matched by construction. What
differs is forced by the corpus and is listed below.

FRAMES ARE PRELOADED, WHICH IS NOT AN OPTIMISATION. The decoder reads from the
head of a file and slices, because seeking in MPEG-2 lands on the nearest
keyframe and would silently shift every sampled window. Domain D clips run
357-651 frames, so a random GOP start would decode an average of ~230 frames of
1080p per training step — hours of the run spent decoding the same footage
repeatedly. Each clip's crop window is therefore decoded once into RAM
and sampled from there; see preload() for what survives the pass.

SPLIT IS BY SCENE, NOT BY CLIP. Four cameras record the same scene at the same
instant, so a clip and its three siblings are near-duplicate content from
different angles. Splitting by clip would put a scene's other views in training
and report memorisation as generalisation.

THE CROP WINDOW IS CHOSEN FROM THE CLEAN VIDEO. A 256x256 window is 1.7% of a
1080p frame and a fixed centre crop lands on empty floor for most surveillance
framings. The window is placed at the median detection centroid of the
uncompressed sequence, which depends on the source alone and never on any codec
or value of mu, and is then held fixed for every leg.
"""

import pathlib
import subprocess
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


def _find_dataset_root():
    """Locate the uploaded domain D videos without hardcoding the dataset slug."""
    base = pathlib.Path("/kaggle/input")
    for cand in sorted(base.rglob("scene_S*")):
        if cand.is_dir() and any(cand.glob("*.mpeg")):
            return cand.parent
    hits = sorted(base.rglob("*.mpeg"))
    if hits:
        return hits[0].parent.parent
    raise RuntimeError("domain D not found under /kaggle/input; upload it as a dataset")


SRDO_SRC = _latest_package()
DATA_ROOT = _find_dataset_root()
print(f"package: {SRDO_SRC}\ndata:    {DATA_ROOT}")
sys.path.insert(0, SRDO_SRC)

from srdo.detection import detect, masks_from  # noqa: E402
from srdo.evaluate import evaluate_codec, save_checkpoint, ssf_forward  # noqa: E402
from srdo.loss import srdo_loss  # noqa: E402
from srdo.omega import compute_omega  # noqa: E402

CFG = dict(
    lambda_0=0.01,
    patch=256,
    lr=1e-4,
    gop=8,
    mu_grid=[0.0, 5.0, 15.0, 30.0, 60.0],
    train_steps=3000,
    eval_every=500,
    seed=0,
    normalize_weight=True,
    val_scene="scene_S009",      # held out entirely; all four of its cameras
    val_windows_per_clip=8,      # 4 clips x 8 GOPs = 32 validation GOPs
    probe_stride=8,              # frames sampled when choosing the crop window
)
TAG = "_d"
_rng = np.random.default_rng(CFG["seed"])


def _probe_size(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True).stdout.strip()
    w, h = out.split("x")[:2]
    return int(w), int(h)


def _decode(path, box=None, max_frames=None):
    """Decode to (T,H,W,3) uint8, optionally cropped to (w,h,x,y)."""
    w0, h0 = _probe_size(path)
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if box is not None:
        w, h, x, y = box
        cmd += ["-vf", f"crop={w}:{h}:{x}:{y}"]
    else:
        w, h = w0, h0
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode()[-300:])
    buf = np.frombuffer(res.stdout, dtype=np.uint8)
    fb = h * w * 3
    return buf[: (buf.size // fb) * fb].reshape(-1, h, w, 3)


def gop_windows(full, stride_probe=None):
    """One window per GOP, centred on that GOP's own detections.

    A single window per clip does not work on this corpus. Subjects walk
    through a fixed camera's field of view, so a window placed at the median
    detection centroid over the whole clip is empty for most of it: measured
    across four clips, 31% to 85% of GOPs contained no detection at all, which
    would train and evaluate largely on empty floor. Choosing the window per
    GOP keeps the subject inside the patch.

    Detection runs on the FULL 1080p frame, not on the candidate crop. A person
    at native scale is frequently larger than a 256-pixel window, and detecting
    on the crop misses anyone straddling its edge — which is precisely the case
    a window-placement routine has to get right.

    Selection depends only on the uncompressed source, so it is fixed before
    any codec runs and is identical for every mu.
    """
    s = CFG["patch"]
    stride_probe = stride_probe or CFG["gop"]
    t, h0, w0 = full.shape[:3]
    probe_idx = list(range(stride_probe // 2, t, stride_probe))
    det = detect(full[probe_idx])

    out = {}
    for j, i in enumerate(probe_idx):
        boxes = det.boxes[j]
        if len(boxes) == 0:
            continue                      # no subject in this GOP: excluded
        cx = float(np.median([(b[0] + b[2]) / 2 for b in boxes]))
        cy = float(np.median([(b[1] + b[3]) / 2 for b in boxes]))
        x = int(np.clip(cx - s / 2, 0, w0 - s)) & ~1
        y = int(np.clip(cy - s / 2, 0, h0 - s)) & ~1
        start = max(min(i - stride_probe // 2, t - CFG["gop"]), 0)
        out[start] = (x, y)
    return out


def preload():
    """Decode each clip once, keep only the per-GOP crops that contain a subject.

    The full 1080p clip is held transiently (about 2.9 GB) and released; what
    survives is one 256x256x8 crop per usable GOP, roughly 1 GB across the
    corpus. Empty GOPs are dropped here rather than mid-training, mirroring the
    null-prior exclusion applied to the consumer-IP pool.
    """
    clips = sorted(pathlib.Path(DATA_ROOT).rglob("*.mpeg"))
    if not clips:
        raise RuntimeError(f"no .mpeg under {DATA_ROOT}")

    store, kept, seen = {}, 0, 0
    for p in clips:
        full = _decode(p)
        if full.shape[0] < CFG["gop"]:
            print(f"  skip {p.name}: {full.shape[0]} frames")
            continue
        wins = gop_windows(full)
        n_possible = full.shape[0] // CFG["gop"]
        gops = []
        for start, (x, y) in sorted(wins.items()):
            g = full[start:start + CFG["gop"], y:y + CFG["patch"], x:x + CFG["patch"]]
            if g.shape[:3] == (CFG["gop"], CFG["patch"], CFG["patch"]):
                gops.append(g.copy())
        del full
        if not gops:
            print(f"  skip {p.name}: no GOP contains a detection")
            continue
        store[p.name] = np.stack(gops)
        kept += len(gops)
        seen += n_possible
        print(f"  {p.name:26s} scene={p.parent.name} "
              f"kept {len(gops)}/{n_possible} GOPs")

    total = sum(v.nbytes for v in store.values())
    print(f"preloaded {len(store)} clips, {kept}/{seen} GOPs "
          f"({100 * kept / max(seen, 1):.0f}% contain a subject), {total / 1e9:.2f} GB\n")
    return store


def scene_of(clip: str) -> str:
    return clip.rsplit("_cam", 1)[0]


def make_gop(store, clip, idx):
    """(x, omega) for one preloaded GOP."""
    frames = store[clip][idx]
    omega = compute_omega(frames, masks=masks_from(detect(frames), frames.shape[:3]))
    x = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div(255.0)
    o = torch.from_numpy(omega).unsqueeze(1).float()
    return x, o


def build_val(store, val_clips):
    """Fixed (clip, start) pairs, evenly spaced, identical across every leg."""
    rows = []
    for c in val_clips:
        n = store[c].shape[0]
        k = min(CFG["val_windows_per_clip"], n)
        for i in np.linspace(0, n - 1, k).round().astype(int):
            rows.append({"clip": f"{c}#{i}", "action": scene_of(c),
                         "source": "D", "_clip": c, "_idx": int(i)})
    return pd.DataFrame(rows)


def train_one_mu(mu, store, tr_clips, va, accelerator):
    codec = ssf2020(quality=4, pretrained=True).train()
    codec.update(force=True)
    opt = torch.optim.Adam((p for p in codec.parameters() if p.requires_grad), lr=CFG["lr"])
    codec, opt = accelerator.prepare(codec, opt)

    def load_val(row):
        return make_gop(store, row["_clip"], int(row["_idx"]))

    curve = []
    pbar = tqdm(range(1, CFG["train_steps"] + 1), desc=f"mu={mu:g}",
                disable=not accelerator.is_local_main_process)
    for step in pbar:
        clip = tr_clips[int(_rng.integers(len(tr_clips)))]
        x, om = make_gop(store, clip, int(_rng.integers(store[clip].shape[0])))
        x, om = x.to(accelerator.device), om.to(accelerator.device)

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
            pbar.set_postfix(bpp=f"{float(terms.bpp):.4f}", S=f"{float(terms.semantic_dist):.2e}")

        if step % CFG["eval_every"] == 0:
            ev = evaluate_codec(accelerator.unwrap_model(codec),
                                (r for _, r in va.iterrows()), load_val,
                                device=accelerator.device, num_pixels=CFG["patch"] ** 2)
            if accelerator.is_local_main_process:
                row = {"mu": mu, "step": step, **ev.summary(tau=0.45)}
                curve.append(row)
                print(f"\n  mu={mu:g} step={step} U={row['soft_u']:.4f}"
                      f" +/-{row['soft_u_sem']:.4f} bpp={row['bpp']:.4f}"
                      f" S={row['s_semantic']:.3e} n={row['n_scored']}")
                pd.DataFrame(curve).to_csv(WORK / f"sweep_curve_mu{mu:g}{TAG}.csv", index=False)
                ev.per_clip.to_csv(WORK / f"sweep_perclip_mu{mu:g}_step{step}{TAG}.csv",
                                   index=False)

    if accelerator.is_local_main_process:
        save_checkpoint(WORK / f"codec_mu{mu:g}{TAG}.pt", accelerator.unwrap_model(codec),
                        mu=mu, step=CFG["train_steps"], lambda_0=CFG["lambda_0"])
    return curve


def main():
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])

    store = preload()
    val_clips = [c for c in store if scene_of(c) == CFG["val_scene"]]
    tr_clips = [c for c in store if scene_of(c) != CFG["val_scene"]]
    if not val_clips:
        raise RuntimeError(f"val scene {CFG['val_scene']} not present: "
                           f"{sorted({scene_of(c) for c in store})}")
    va = build_val(store, val_clips)
    print(f"train {len(tr_clips)} clips from "
          f"{len(({scene_of(c) for c in tr_clips}))} scenes; "
          f"val {len(val_clips)} clips -> {len(va)} GOPs from {CFG['val_scene']}\n")

    all_curves = []
    for mu in CFG["mu_grid"]:
        all_curves += train_one_mu(mu, store, tr_clips, va, accelerator)
        if accelerator.is_local_main_process:
            df = pd.DataFrame(all_curves)
            df.to_csv(WORK / f"phase4d_sweep_curves{TAG}.csv", index=False)
            final = df.sort_values("step").groupby("mu").tail(1)
            final.to_csv(WORK / f"phase4d_fixed_mu_sweep{TAG}.csv", index=False)
            print("\n" + final[["mu", "step", "soft_u", "soft_u_sem", "bpp",
                                "s_semantic", "d_plain"]].round(5).to_string(index=False))
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
