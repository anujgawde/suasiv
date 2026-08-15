from __future__ import annotations

import json

import numpy as np
from rich.console import Console

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal

console = Console()

_AU_THRESHOLD = 0.5

_NOD_PITCH_CHANGE = 3.0
_NOD_WINDOW_FRAMES = 6
_NOD_MIN_REVERSALS = 2


class AudienceReactionAnalyzer(Analyzer):
    name = "audience_reaction"
    requires = {"frames", "diarization"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        try:
            from feat import Detector  # noqa: F841
        except ImportError:
            raise RuntimeError(
                "py-feat is required for audience reaction analysis.\n"
                "Install: pip install 'suasiv[full]' or pip install py-feat"
            )

        audience_tiles = _get_audience_tile_dirs(ctx)
        if not audience_tiles:
            return AnalyzerResult(
                analyzer=self.name,
                signals=[],
                summary={"error": "no audience tiles detected"},
            )

        fps = ctx.config.ingest.frame_fps or 3
        settings = ctx.config.analyzer_settings(self.name)
        analyzer_fps = settings.fps or fps
        frame_step = max(1, round(fps / analyzer_fps))

        console.print(
            f"    Loading py-feat detector for {len(audience_tiles)} audience tile(s)..."
        )
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
        except ImportError:
            pass

        detector = Detector(device=device)

        tile_reactions: dict[int, list[dict]] = {}

        for tile_idx, tile_dir in audience_tiles:
            frames = sorted(tile_dir.glob("frame_*.png"))
            tile_data: list[dict] = []
            pitches: list[float] = []

            console.print(f"    Processing tile {tile_idx} ({len(frames)} frames)...")

            for fi, frame_path in enumerate(frames):
                if fi % frame_step != 0:
                    continue

                frame_num = int(frame_path.stem.split("_")[1])
                timestamp = (frame_num - 1) / fps

                try:
                    result = detector.detect_image(str(frame_path))
                except Exception:
                    tile_data.append(
                        {"time": round(timestamp, 3), "reaction": "neutral", "pitch": None}
                    )
                    continue

                if result is None or len(result) == 0:
                    tile_data.append(
                        {"time": round(timestamp, 3), "reaction": "neutral", "pitch": None}
                    )
                    continue

                row = result.iloc[0]
                reaction = _classify_reaction(row)
                pitch_val = float(row.get("Pitch", 0))
                pitches.append(pitch_val)

                tile_data.append(
                    {
                        "time": round(timestamp, 3),
                        "reaction": reaction,
                        "pitch": round(pitch_val, 2),
                    }
                )

            nods = _detect_nods(tile_data)
            for nod_start, nod_end in nods:
                for d in tile_data:
                    if nod_start <= d["time"] <= nod_end and d["reaction"] == "neutral":
                        d["reaction"] = "nod"

            tile_reactions[tile_idx] = tile_data

        signals: list[Signal] = []
        positive_count = 0
        negative_count = 0
        total_frames = 0
        notable_moments = 0

        ref = next(iter(tile_reactions.values())) if tile_reactions else []
        for i in range(len(ref)):
            t = ref[i]["time"]
            reactions_at_t: list[str] = []

            for tile_data in tile_reactions.values():
                if i < len(tile_data):
                    reactions_at_t.append(tile_data[i]["reaction"])

            total_frames += 1
            pos = sum(1 for r in reactions_at_t if r in ("smile", "nod"))
            neg = sum(1 for r in reactions_at_t if r in ("frown", "confusion"))
            positive_count += pos
            negative_count += neg

            for r in reactions_at_t:
                if r != "neutral":
                    signals.append(
                        Signal(
                            analyzer=self.name,
                            type=f"audience_{r}",
                            start=t,
                            end=t,
                            confidence=0.7,
                        )
                    )

            if pos >= 2:
                signals.append(
                    Signal(
                        analyzer=self.name,
                        type="simultaneous_positive_reaction",
                        start=t,
                        end=t,
                        value={"count": pos, "total": len(reactions_at_t)},
                        confidence=0.8,
                    )
                )
                notable_moments += 1
            if neg >= 2:
                signals.append(
                    Signal(
                        analyzer=self.name,
                        type="simultaneous_negative_reaction",
                        start=t,
                        end=t,
                        value={"count": neg, "total": len(reactions_at_t)},
                        confidence=0.8,
                    )
                )
                notable_moments += 1

        total_reaction_slots = total_frames * len(tile_reactions) if tile_reactions else 1
        positive_pct = positive_count / total_reaction_slots
        negative_pct = negative_count / total_reaction_slots

        reaction_path = ctx.workspace / "audience_reaction.json"
        with open(reaction_path, "w") as f:
            json.dump(
                {
                    "positive_reaction_pct": round(positive_pct, 3),
                    "negative_reaction_pct": round(negative_pct, 3),
                    "notable_moments": notable_moments,
                    "tiles_analyzed": len(audience_tiles),
                },
                f,
                indent=2,
            )

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "positive_reaction_pct": round(positive_pct, 3),
                "negative_reaction_pct": round(negative_pct, 3),
                "notable_moments": notable_moments,
            },
        )


def _get_audience_tile_dirs(ctx: MediaContext) -> list[tuple[int, object]]:
    if not ctx.tiles_dir or len(ctx.tiles) <= 1:
        return []
    dirs: list[tuple[int, object]] = []
    for tile in ctx.tiles:
        if tile.index == ctx.speaker_tile_index:
            continue
        tile_dir = ctx.tiles_dir / f"tile_{tile.index:02d}"
        if tile_dir.exists() and any(tile_dir.glob("frame_*.png")):
            dirs.append((tile.index, tile_dir))
    return dirs


def _classify_reaction(row) -> str:
    def active(au: str) -> bool:
        return float(row.get(au, 0)) > _AU_THRESHOLD

    if active("AU06") and active("AU12"):
        return "smile"
    if active("AU04") and active("AU07"):
        return "confusion"
    if active("AU01") and active("AU02") and active("AU05"):
        return "surprise"
    if active("AU04"):
        return "frown"
    return "neutral"


def _detect_nods(tile_data: list[dict]) -> list[tuple[float, float]]:
    pitches_with_time = [
        (d["time"], d["pitch"])
        for d in tile_data
        if d["pitch"] is not None
    ]
    if len(pitches_with_time) < 4:
        return []

    times = [p[0] for p in pitches_with_time]
    pitches = [p[1] for p in pitches_with_time]

    diffs = [pitches[i + 1] - pitches[i] for i in range(len(pitches) - 1)]

    nods: list[tuple[float, float]] = []
    i = 0
    while i < len(diffs) - 1:
        reversals = 0
        start_idx = i
        j = i + 1
        while j < len(diffs):
            if abs(diffs[j]) > _NOD_PITCH_CHANGE and abs(diffs[j - 1]) > _NOD_PITCH_CHANGE:
                if diffs[j] * diffs[j - 1] < 0:
                    reversals += 1
            if times[j] - times[start_idx] > 2.0:
                break
            j += 1

        if reversals >= _NOD_MIN_REVERSALS:
            nods.append((times[start_idx], times[min(j, len(times) - 1)]))
            i = j
        else:
            i += 1

    return nods
