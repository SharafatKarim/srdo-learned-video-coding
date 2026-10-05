"""Video IO via ffmpeg subprocess pipes.

Deliberately avoids OpenCV/PyAV: ffmpeg is already a hard dependency for phase 0
(cropdetect), and piping rawvideo keeps decode behaviour identical across every
phase. Mixed-resolution input is the norm in this dataset, so every reader
returns explicit shapes rather than assuming a constant frame size.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class FFmpegError(RuntimeError):
    pass


@dataclass(frozen=True)
class Crop:
    w: int
    h: int
    x: int
    y: int

    @property
    def ffmpeg(self) -> str:
        return f"crop={self.w}:{self.h}:{self.x}:{self.y}"


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    width: int
    height: int
    fps: float
    nb_frames: int
    duration: float
    bit_rate: float
    codec: str

    @property
    def bpp(self) -> float:
        """Bits per pixel of the SOURCE file.

        Low values here mean the source is already heavily compressed and any
        BD-BR computed against it is contaminated. Reported, not corrected.
        """
        px_per_s = self.width * self.height * self.fps
        return self.bit_rate / px_per_s if px_per_s > 0 else float("nan")


def probe(path: Path) -> VideoMeta:
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames",
        "-show_entries", "format=duration,bit_rate",
        "-of", "json", str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}: {res.stderr.strip()}")
    doc = json.loads(res.stdout)
    if not doc.get("streams"):
        raise FFmpegError(f"no video stream in {path}")
    st, fmt = doc["streams"][0], doc.get("format", {})

    num, _, den = st.get("r_frame_rate", "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0

    duration = float(fmt.get("duration") or 0.0)
    nb = int(st.get("nb_frames") or 0)
    if nb == 0 and fps > 0:  # some muxers omit nb_frames
        nb = int(round(duration * fps))

    return VideoMeta(
        path=path,
        width=int(st["width"]),
        height=int(st["height"]),
        fps=fps,
        nb_frames=nb,
        duration=duration,
        bit_rate=float(fmt.get("bit_rate") or 0.0),
        codec=str(st.get("codec_name", "")),
    )


def detect_crop(path: Path, limit: int, round_to: int, probe_frames: int) -> Crop | None:
    """Run ffmpeg cropdetect and return the last reported crop rectangle.

    Pillarboxed UCF-Crime clips carry large constant-black borders. Encoders
    code those at near-zero cost, which inflates compression ratios and
    corrupts any rate measurement. They must be removed before, not after,
    rate is measured.
    """
    cmd = [
        "ffmpeg", "-v", "info", "-i", str(path),
        "-vf", f"cropdetect={limit}:{round_to}:0",
        "-frames:v", str(probe_frames), "-f", "null", "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    last: Crop | None = None
    for line in res.stderr.splitlines():
        idx = line.rfind("crop=")
        if idx == -1:
            continue
        try:
            w, h, x, y = (int(v) for v in line[idx + 5:].strip().split(":"))
        except ValueError:
            continue
        if w > 0 and h > 0:
            last = Crop(w, h, x, y)
    return last


def read_frames(
    path: Path,
    crop: Crop | None = None,
    height: int | None = None,
    max_frames: int | None = None,
    gray: bool = False,
    start: int = 0,
) -> np.ndarray:
    """Decode to a uint8 array.

    Returns (T, H, W, 3) for RGB or (T, H, W) for grayscale. Scaling preserves
    aspect ratio and forces even dimensions, which every downstream codec needs.

    `start` skips that many frames from the head. Phase 4 trains on 8-frame
    GOPs, and with start=0 every epoch resamples the same eight opening frames
    of each clip forever — a 784-clip pool supplies 6272 distinct frames and no
    temporal variation at all. Decoding from the head and slicing, rather than
    seeking, is deliberate: an -ss seek lands on the nearest keyframe, which
    would quietly shift the requested window by an unknown amount.
    """
    filters = []
    if crop is not None:
        filters.append(crop.ffmpeg)
    if height is not None:
        filters.append(f"scale=-2:{height}")

    pix_fmt = "gray" if gray else "rgb24"
    channels = 1 if gray else 3

    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if max_frames is not None:
        cmd += ["-frames:v", str(start + max_frames)]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix_fmt, "-"]

    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise FFmpegError(f"decode failed on {path}: {res.stderr.decode()[-500:]}")

    # Ask ffmpeg for the post-filter geometry rather than recomputing it, so the
    # buffer reshape can never silently disagree with what was decoded.
    out_w, out_h = _filtered_size(path, filters)
    frame_bytes = out_w * out_h * channels
    if frame_bytes == 0:
        raise FFmpegError(f"zero-sized output for {path}")

    buf = np.frombuffer(res.stdout, dtype=np.uint8)
    n = buf.size // frame_bytes
    if n == 0:
        raise FFmpegError(f"no frames decoded from {path}")
    buf = buf[: n * frame_bytes]

    shape = (n, out_h, out_w) if gray else (n, out_h, out_w, channels)
    frames = buf.reshape(shape)

    if start:
        frames = frames[start:]
        if frames.shape[0] == 0:
            raise FFmpegError(
                f"start={start} is past the end of {path} ({n} frames decoded)")
    return frames


def _filtered_size(path: Path, filters: list[str]) -> tuple[int, int]:
    """Decode a single frame through the same filter chain to learn its size."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0:
        raise FFmpegError(f"probe decode failed on {path}: {res.stderr.decode()[-300:]}")

    meta = probe(path)
    src_w, src_h = meta.width, meta.height
    for f in filters:
        if f.startswith("crop="):
            w, h, _, _ = (int(v) for v in f[5:].split(":"))
            src_w, src_h = w, h
        elif f.startswith("scale=-2:"):
            new_h = int(f.split(":")[1])
            src_w = int(round(src_w * new_h / src_h / 2)) * 2
            src_h = new_h
    return src_w, src_h


def encode_decode(frames: np.ndarray, crf: int, preset: str = "veryfast",
                  codec: str = "libx265") -> np.ndarray:
    """Round-trip frames through a real video encoder at a given CRF.

    Blur and additive noise bracket the smoothing and high-frequency failure
    modes, but neither IS codec distortion — which is blocky, rate-dependent,
    and concentrated where the encoder decides detail is cheap to discard.
    Since A1 must hold in the regime the codec actually occupies, we probe it
    with the genuine article rather than a proxy.

    frames: (T, H, W, 3) uint8 RGB -> same shape, same dtype.
    """
    t, h, w, _ = frames.shape
    # x265 needs even dimensions; crop rather than pad so no synthetic edge is
    # introduced at the boundary.
    eh, ew = h - (h % 2), w - (w % 2)
    src = np.ascontiguousarray(frames[:, :eh, :ew])

    enc = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{ew}x{eh}", "-r", "30", "-i", "-",
         "-c:v", codec, "-crf", str(crf), "-preset", preset,
         "-pix_fmt", "yuv420p", "-f", "matroska", "-"],
        input=src.tobytes(), capture_output=True,
    )
    if enc.returncode != 0 or not enc.stdout:
        raise FFmpegError(f"encode failed (crf={crf}): {enc.stderr.decode()[-300:]}")

    dec = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "matroska", "-i", "-",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        input=enc.stdout, capture_output=True,
    )
    if dec.returncode != 0:
        raise FFmpegError(f"decode failed (crf={crf}): {dec.stderr.decode()[-300:]}")

    frame_bytes = ew * eh * 3
    buf = np.frombuffer(dec.stdout, dtype=np.uint8)
    n = buf.size // frame_bytes
    if n == 0:
        raise FFmpegError(f"encoder returned no frames (crf={crf})")
    out = buf[: n * frame_bytes].reshape(n, eh, ew, 3)

    # Restore original geometry so distortion maps stay aligned with Omega.
    result = frames.copy()
    result[:n, :eh, :ew] = out
    if n < t:  # encoder dropped trailing frames; hold the last decoded one
        result[n:, :eh, :ew] = out[-1]
    return result


def sample_clip(
    path: Path,
    n_frames: int,
    crop: Crop | None = None,
    height: int | None = None,
) -> np.ndarray:
    """Uniformly sample exactly n_frames, looping short clips.

    Clips here run 1-9 s, so some are shorter than the classifier's temporal
    window; looping is preferable to zero-padding, which would inject a hard
    scene cut the action model has never seen.
    """
    frames = read_frames(path, crop=crop, height=height)
    t = frames.shape[0]
    idx = np.linspace(0, t - 1, n_frames).round().astype(int) if t >= n_frames \
        else np.arange(n_frames) % t
    return frames[idx]
