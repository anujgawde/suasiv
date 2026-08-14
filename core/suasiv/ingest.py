from __future__ import annotations

from suasiv.context import MediaContext, TileInfo


def ingest(ctx: MediaContext) -> None:
    ctx.setup_workspace()
    ctx.audio_path = ctx.workspace / "audio.wav"
    ctx.duration = 16.0
    ctx.resolution = (1920, 1080)
    ctx.fps = 30.0

    ctx.tiles = [
        TileInfo(index=0, x=0, y=0, width=640, height=360, is_speaker=True),
        TileInfo(index=1, x=640, y=0, width=640, height=360),
        TileInfo(index=2, x=0, y=360, width=640, height=360),
        TileInfo(index=3, x=640, y=360, width=640, height=360),
    ]
    ctx.speaker_tile_index = 0
