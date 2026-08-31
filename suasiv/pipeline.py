from __future__ import annotations

from pathlib import Path

from rich.console import Console

from suasiv.config import SuasivConfig
from suasiv.schema import (
    CoachingReport,
    Dimension,
    DimensionScore,
    MediaContext,
    Moment,
    MomentType,
    Signal,
    SignalType,
)

console = Console()


def run(video: Path, config: SuasivConfig) -> CoachingReport:
    workspace = config.workspace_path / video.stem
    workspace.mkdir(parents=True, exist_ok=True)

    ctx = MediaContext(video_path=video, workspace=workspace)

    console.print("[dim]ingest[/dim]")
    ctx = _ingest(ctx, config)

    console.print("[dim]analyze[/dim]")
    signals = _analyze(ctx, config)

    console.print("[dim]fuse[/dim]")
    moments = _fuse(signals, config)

    console.print("[dim]score[/dim]")
    scores = _score(signals, moments, config)

    console.print("[dim]narrate[/dim]")
    narrative = _narrate(scores, moments, config)

    console.print("[dim]render[/dim]")
    report = CoachingReport(
        scores=scores,
        moments=moments,
        narrative=narrative,
        duration=ctx.duration,
    )
    _render(report, ctx, config)

    return report


def _ingest(ctx: MediaContext, config: SuasivConfig) -> MediaContext:
    ctx.duration = 120.0
    return ctx


def _analyze(ctx: MediaContext, config: SuasivConfig) -> list[Signal]:
    return [
        Signal(type=SignalType.FILLER_WORD, source="transcription", start=15.2, end=15.5, value="um"),
        Signal(type=SignalType.CONFIDENT_DELIVERY, source="emotion", start=0.0, end=30.0, value=0.72),
        Signal(type=SignalType.EYE_CONTACT, source="speaker_visual", start=0.0, end=60.0, value=0.81),
    ]


def _fuse(signals: list[Signal], config: SuasivConfig) -> list[Moment]:
    return [
        Moment(
            type=MomentType.STRONG,
            start=10.0,
            end=25.0,
            significance=0.8,
            signals=signals[:2],
            description="Strong opening with confident delivery",
        ),
    ]


def _score(
    signals: list[Signal], moments: list[Moment], config: SuasivConfig
) -> list[DimensionScore]:
    return [
        DimensionScore(dimension=Dimension.DELIVERY_CONFIDENCE, score=0.72),
        DimensionScore(dimension=Dimension.VOCAL_VARIETY, score=0.55),
        DimensionScore(dimension=Dimension.PACING_FLUENCY, score=0.64),
        DimensionScore(dimension=Dimension.CONTENT_CLARITY, score=0.60),
        DimensionScore(dimension=Dimension.ENGAGEMENT, score=0.70),
        DimensionScore(dimension=Dimension.OVERALL_IMPACT, score=0.65),
    ]


def _narrate(
    scores: list[DimensionScore], moments: list[Moment], config: SuasivConfig
) -> str:
    return "Stub narrative — LLM coaching text will go here."


def _render(report: CoachingReport, ctx: MediaContext, config: SuasivConfig) -> None:
    out = ctx.workspace / "report.md"
    out.write_text(f"# Coaching Report\n\nStub report for {ctx.video_path.name}\n")
    console.print(f"[green]report written to {out}[/green]")
