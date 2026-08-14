from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from suasiv.config import SuasivConfig


@dataclass
class TileInfo:
    index: int
    x: int
    y: int
    width: int
    height: int
    is_speaker: bool = False


@dataclass
class MediaContext:
    video_path: Path
    workspace: Path
    config: SuasivConfig

    audio_path: Path | None = None
    frames_dir: Path | None = None
    tiles: list[TileInfo] = field(default_factory=list)
    speaker_tile_index: int | None = None

    transcript: list[dict] | None = None
    speaker_labels: dict[str, str] | None = None
    primary_speaker: str | None = None

    duration: float = 0.0
    resolution: tuple[int, int] | None = None
    fps: float = 0.0

    def setup_workspace(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)
        frames = self.workspace / "frames"
        frames.mkdir(exist_ok=True)
        self.frames_dir = frames
