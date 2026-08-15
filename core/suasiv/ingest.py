from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from suasiv.context import MediaContext, TileInfo


def _check_ffmpeg() -> None:
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")
    if not shutil.which("ffprobe"):
        missing.append("ffprobe")
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} not found. Install:\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt install ffmpeg\n"
            "  Windows: choco install ffmpeg"
        )


def _probe(video: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", "-show_streams", str(video),
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _extract_metadata(probe: dict[str, Any]) -> tuple[float, tuple[int, int], float]:
    duration = float(probe["format"]["duration"])
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])
    num, den = vs["r_frame_rate"].split("/")
    fps = int(num) / int(den)
    return duration, (w, h), fps


def _has_audio(probe: dict[str, Any]) -> bool:
    return any(s["codec_type"] == "audio" for s in probe["streams"])


def _extract_audio(video: Path, out: Path, sample_rate: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-i", str(video),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", str(sample_rate), "-ac", "1",
            "-y", str(out),
        ],
        capture_output=True, check=True,
    )


def _sample_frames(video: Path, frames_dir: Path, fps: int) -> int:
    subprocess.run(
        [
            "ffmpeg", "-i", str(video),
            "-vf", f"fps={fps}",
            "-y", str(frames_dir / "frame_%05d.png"),
        ],
        capture_output=True, check=True,
    )
    return len(list(frames_dir.glob("frame_*.png")))


def _find_border_positions(profile, threshold: float, min_gap: int) -> list[int]:
    positions: list[int] = []
    i = 0
    n = len(profile)
    while i < n:
        if profile[i] < threshold:
            start = i
            while i < n and profile[i] < threshold:
                i += 1
            center = (start + i) // 2
            if not positions or (center - positions[-1]) > min_gap:
                positions.append(center)
        else:
            i += 1
    return positions


def _detect_tiles(frames_dir: Path, resolution: tuple[int, int]) -> list[TileInfo]:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return [TileInfo(index=0, x=0, y=0, width=resolution[0], height=resolution[1])]

    frames = sorted(frames_dir.glob("frame_*.png"))
    if not frames:
        return [TileInfo(index=0, x=0, y=0, width=resolution[0], height=resolution[1])]

    sample_idx = min(max(1, len(frames) // 10), len(frames) - 1)
    frame = cv2.imread(str(frames[sample_idx]))
    if frame is None:
        return [TileInfo(index=0, x=0, y=0, width=resolution[0], height=resolution[1])]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    row_profile = np.mean(gray, axis=1)
    col_profile = np.mean(gray, axis=0)

    dark_thresh = max(float(np.median(gray)) * 0.15, 10.0)

    h_borders = _find_border_positions(row_profile, dark_thresh, min_gap=h // 8)
    v_borders = _find_border_positions(col_profile, dark_thresh, min_gap=w // 8)

    rows = [0] + h_borders + [h]
    cols = [0] + v_borders + [w]

    tiles: list[TileInfo] = []
    idx = 0
    for r in range(len(rows) - 1):
        for c in range(len(cols) - 1):
            y, x = rows[r], cols[c]
            tw, th = cols[c + 1] - x, rows[r + 1] - y
            if tw > w * 0.08 and th > h * 0.08:
                tiles.append(TileInfo(index=idx, x=x, y=y, width=tw, height=th))
                idx += 1

    if not tiles:
        return [TileInfo(index=0, x=0, y=0, width=w, height=h)]

    return tiles


def _crop_tiles(frames_dir: Path, tiles: list[TileInfo], workspace: Path) -> Path:
    import cv2

    tiles_dir = workspace / "tiles"
    tiles_dir.mkdir(exist_ok=True)

    for tile in tiles:
        (tiles_dir / f"tile_{tile.index:02d}").mkdir(exist_ok=True)

    for frame_path in sorted(frames_dir.glob("frame_*.png")):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        for tile in tiles:
            crop = frame[tile.y : tile.y + tile.height, tile.x : tile.x + tile.width]
            out_path = tiles_dir / f"tile_{tile.index:02d}" / frame_path.name
            cv2.imwrite(str(out_path), crop)

    return tiles_dir


def ingest(ctx: MediaContext) -> None:
    _check_ffmpeg()

    if not ctx.video_path.exists():
        raise FileNotFoundError(f"Video not found: {ctx.video_path}")

    ctx.setup_workspace()

    probe = _probe(ctx.video_path)
    ctx.duration, ctx.resolution, ctx.fps = _extract_metadata(probe)

    ctx.audio_path = ctx.workspace / "audio.wav"
    if _has_audio(probe):
        _extract_audio(ctx.video_path, ctx.audio_path, ctx.config.ingest.audio_sample_rate)
    else:
        ctx.audio_path = None

    ctx.frame_count = _sample_frames(
        ctx.video_path, ctx.frames_dir, ctx.config.ingest.frame_fps
    )

    ctx.tiles = _detect_tiles(ctx.frames_dir, ctx.resolution)

    if len(ctx.tiles) > 1:
        ctx.tiles_dir = _crop_tiles(ctx.frames_dir, ctx.tiles, ctx.workspace)
