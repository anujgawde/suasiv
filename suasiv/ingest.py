from __future__ import annotations

import json
import platform
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from suasiv.config import SuasivConfig
from suasiv.schema import MediaContext, Tile

LAYOUT_SAMPLES = 12


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") and shutil.which("ffprobe"):
        return

    system = platform.system()
    if system == "Darwin":
        hint = "brew install ffmpeg"
    elif system == "Linux":
        hint = "sudo apt install ffmpeg"
    elif system == "Windows":
        hint = "winget install ffmpeg"
    else:
        hint = "install ffmpeg from https://ffmpeg.org"

    raise RuntimeError(f"ffmpeg not found. Install it: {hint}")


def probe_metadata(video: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def extract_audio(video: Path, output: Path) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-i", str(video),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return output


def sample_frames(video: Path, output_dir: Path, fps: float) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-i", str(video),
            "-vf", f"fps={fps}",
            "-q:v", "2",
            "-y",
            str(output_dir / "frame_%06d.png"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return output_dir


def _content_bands(
    profile: np.ndarray, gutter_intensity: float, min_span: int
) -> list[tuple[int, int]]:
    """Runs along an intensity profile that sit above the gutter threshold."""
    lit = profile >= gutter_intensity

    bands: list[tuple[int, int]] = []
    start: int | None = None

    for i, is_content in enumerate(lit):
        if is_content and start is None:
            start = i
        elif not is_content and start is not None:
            if i - start >= min_span:
                bands.append((start, i - 1))
            start = None

    if start is not None and len(lit) - start >= min_span:
        bands.append((start, len(lit) - 1))

    return bands


def detect_tiles(
    frame_path: Path, min_tile_area: int, gutter_intensity: float = 30.0
) -> list[Tile]:
    """Split a gallery grid on its gutters, row bands first then columns.

    Column gutters only resolve within a row: a short final row leaves its
    tiles unaligned with the rows above, so a full-height column profile
    averages the gutters away.
    """
    img = cv2.imread(str(frame_path))
    if img is None:
        return []

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    min_span = max(16, min(h, w) // 20)

    tiles: list[Tile] = []

    for top, bottom in _content_bands(gray.mean(axis=1), gutter_intensity, min_span):
        strip = gray[top : bottom + 1, :]
        height = bottom - top + 1

        for left, right in _content_bands(
            strip.mean(axis=0), gutter_intensity, min_span
        ):
            width = right - left + 1

            if width * height < min_tile_area:
                continue
            if not 0.4 < width / height < 3.0:
                continue

            tiles.append(Tile(
                x=left, y=top, w=width, h=height, participant_id=len(tiles),
            ))

    if not tiles:
        tiles = [Tile(x=0, y=0, w=w, h=h, participant_id=0)]

    return tiles


def detect_layout(
    frames_dir: Path,
    min_tile_area: int,
    gutter_intensity: float,
    samples: int = LAYOUT_SAMPLES,
) -> list[Tile]:
    """The tile layout that holds across the recording.

    Recordings fade in and rearrange as people join, so a single frame — the
    first one most of all — can catch a transitional grid. Take the layout
    seen most often instead.
    """
    frames = sorted(frames_dir.glob("*.png"))
    if not frames:
        return []

    step = max(1, len(frames) // samples)
    layouts: dict[int, list[list[Tile]]] = defaultdict(list)

    for frame in frames[::step][:samples]:
        tiles = detect_tiles(frame, min_tile_area, gutter_intensity)
        if tiles:
            layouts[len(tiles)].append(tiles)

    if not layouts:
        return []

    # Most frequent layout; on a tie the busier grid, since a fade-in reads as
    # fewer tiles than the settled view.
    _, candidates = max(layouts.items(), key=lambda kv: (len(kv[1]), kv[0]))
    return candidates[0]


def ingest(ctx: MediaContext, config: SuasivConfig) -> MediaContext:
    check_ffmpeg()

    probe = probe_metadata(ctx.video_path)

    ctx.duration = float(probe["format"].get("duration", 0))

    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            ctx.width = int(stream.get("width", 0))
            ctx.height = int(stream.get("height", 0))
            r_frame_rate = stream.get("r_frame_rate", "0/1")
            num, den = r_frame_rate.split("/")
            ctx.fps = float(num) / float(den) if float(den) else 0.0
            break

    audio_path = ctx.workspace / "audio.wav"
    extract_audio(ctx.video_path, audio_path)
    ctx.audio_path = audio_path

    frames_dir = ctx.workspace / "frames"
    sample_frames(ctx.video_path, frames_dir, config.ingest.fps)
    ctx.frames_dir = frames_dir

    if config.ingest.tile_detection:
        ctx.tiles = detect_layout(
            frames_dir,
            config.ingest.min_tile_area,
            config.ingest.gutter_intensity,
        )

    if not ctx.tiles:
        ctx.tiles = [Tile(x=0, y=0, w=ctx.width, h=ctx.height, participant_id=0)]

    return ctx
