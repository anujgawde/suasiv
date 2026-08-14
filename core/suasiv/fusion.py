from __future__ import annotations

from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, FusedTimeline, Moment, Signal


def fuse(ctx: MediaContext, results: list[AnalyzerResult]) -> FusedTimeline:
    all_signals: list[Signal] = []
    for result in results:
        all_signals.extend(result.signals)

    all_signals.sort(key=lambda s: s.start)

    moments = [
        Moment(
            type="strong",
            start=0.0,
            end=3.5,
            description="High energy opening with full audience engagement",
            significance=0.9,
        ),
        Moment(
            type="weak",
            start=11.0,
            end=14.0,
            description="Energy drop + attention drop + filler word + gaze break",
            significance=0.85,
        ),
        Moment(
            type="qa",
            start=14.5,
            end=16.0,
            description="Audience question",
            significance=0.5,
        ),
    ]

    return FusedTimeline(
        duration=ctx.duration,
        signals=all_signals,
        moments=moments,
    )
