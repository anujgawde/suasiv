from __future__ import annotations

import json
import platform
import shutil
import subprocess
from pathlib import Path

from suasiv.config import SuasivConfig
from suasiv.schema import MediaContext


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

    return ctx
