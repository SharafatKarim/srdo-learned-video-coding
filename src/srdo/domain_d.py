"""Data machinery for the multi-camera 1080p corpus (domain D).

Lifted out of the first domain D runner so that later experiments share one
implementation rather than a copy. The original script is left untouched, since
it is the record of what produced its own results.

Two decisions here are load-bearing and were arrived at by measurement.

PER-GOP WINDOWS. A 256x256 patch is 1.7% of a 1080p frame. A fixed centre crop
lands on empty floor for most surveillance framings, and a single window per
clip at the median detection centroid is worse in practice, because subjects
walk through a fixed camera's field of view: measured across four clips, 31% to
85% of GOPs then contained no detection at all. Choosing the window per GOP
keeps roughly 78% of GOPs usable.

DETECTION ON THE FULL FRAME. Window placement detects on the full 1080p frame,
not on the candidate crop. A person at native scale is frequently larger than a
256-pixel window, and detecting on the crop misses anyone straddling its edge —
exactly the case a placement routine has to get right.

Selection depends only on the uncompressed source, never on a codec or on any
parameter under study, so the same GOPs serve every run and remain comparable.
"""

from __future__ import annotations

import pathlib
import subprocess

import numpy as np
import pandas as pd

PATCH = 256
GOP = 8


def find_dataset_root(base="/kaggle/input"):
    """Locate the videos without hardcoding a dataset slug."""
    base = pathlib.Path(base)
    for cand in sorted(base.rglob("scene_S*")):
        if cand.is_dir() and any(cand.glob("*.mpeg")):
            return cand.parent
    hits = sorted(base.rglob("*.mpeg"))
    if hits:
        return hits[0].parent.parent
    raise RuntimeError(f"domain D not found under {base}; upload it as a dataset")


def probe_size(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True).stdout.strip()
    a = out.split("x")
    return int(a[0]), int(a[1])


def decode(path, box=None, max_frames=None):
    """(T,H,W,3) uint8, optionally cropped to (w,h,x,y).

    Decodes from the head rather than seeking: MPEG-2 seeks land on the nearest
    keyframe and would silently shift every window relative to the frame index
    the caller asked for.
    """
    w0, h0 = probe_size(path)
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


def gop_windows(full, stride_probe=GOP):
    """{gop_start: (x, y)} for every GOP whose probe frame contains a detection."""
    from .detection import detect

    t, h0, w0 = full.shape[:3]
    probe_idx = list(range(stride_probe // 2, t, stride_probe))
    det = detect(full[probe_idx])

    out = {}
    for j, i in enumerate(probe_idx):
        boxes = det.boxes[j]
        if len(boxes) == 0:
            continue
        cx = float(np.median([(b[0] + b[2]) / 2 for b in boxes]))
        cy = float(np.median([(b[1] + b[3]) / 2 for b in boxes]))
        x = int(np.clip(cx - PATCH / 2, 0, w0 - PATCH)) & ~1
        y = int(np.clip(cy - PATCH / 2, 0, h0 - PATCH)) & ~1
        out[max(min(i - stride_probe // 2, t - GOP), 0)] = (x, y)
    return out


def preload(data_root, verbose=True):
    """{clip: (n_gops, GOP, PATCH, PATCH, 3) uint8} holding only usable GOPs.

    Each 1080p clip is decoded once and released; what survives is the crops,
    roughly 1 GB across the corpus against 56 GB for the full frames.
    """
    clips = sorted(pathlib.Path(data_root).rglob("*.mpeg"))
    if not clips:
        raise RuntimeError(f"no .mpeg under {data_root}")

    store, kept, seen = {}, 0, 0
    for p in clips:
        full = decode(p)
        if full.shape[0] < GOP:
            continue
        gops = []
        for start, (x, y) in sorted(gop_windows(full).items()):
            g = full[start:start + GOP, y:y + PATCH, x:x + PATCH]
            if g.shape[:3] == (GOP, PATCH, PATCH):
                gops.append(g.copy())
        n_possible = full.shape[0] // GOP
        del full
        if not gops:
            continue
        store[p.name] = np.stack(gops)
        kept += len(gops)
        seen += n_possible
        if verbose:
            print(f"  {p.name:26s} scene={p.parent.name} kept {len(gops)}/{n_possible} GOPs")

    if verbose:
        total = sum(v.nbytes for v in store.values())
        print(f"preloaded {len(store)} clips, {kept}/{seen} GOPs "
              f"({100 * kept / max(seen, 1):.0f}% contain a subject), {total / 1e9:.2f} GB\n")
    return store


def scene_of(clip: str) -> str:
    return clip.rsplit("_cam", 1)[0]


def make_gop(store, clip, idx):
    """(x, omega) for one preloaded GOP, as torch tensors."""
    import torch

    from .detection import detect, masks_from
    from .omega import compute_omega

    frames = store[clip][idx]
    omega = compute_omega(frames, masks=masks_from(detect(frames), frames.shape[:3]))
    x = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float().div(255.0)
    o = torch.from_numpy(omega).unsqueeze(1).float()
    return x, o


def split_by_scene(store, val_scene):
    """Scene-level split. Four cameras see the same scene at the same instant,
    so splitting by clip would place near-duplicate views on both sides."""
    val = [c for c in store if scene_of(c) == val_scene]
    train = [c for c in store if scene_of(c) != val_scene]
    if not val:
        raise RuntimeError(f"val scene {val_scene} absent; have "
                           f"{sorted({scene_of(c) for c in store})}")
    return train, val


def build_val(store, val_clips, per_clip=8):
    """Evenly spaced GOP indices, fixed once so every run scores the same set."""
    rows = []
    for c in val_clips:
        n = store[c].shape[0]
        for i in np.linspace(0, n - 1, min(per_clip, n)).round().astype(int):
            rows.append({"clip": f"{c}#{i}", "action": scene_of(c),
                         "source": "D", "_clip": c, "_idx": int(i)})
    return pd.DataFrame(rows)
