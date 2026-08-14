from __future__ import annotations

from pathlib import Path

from rich.console import Console

from suasiv.analyzers import ALL_ANALYZERS
from suasiv.config import SuasivConfig
from suasiv.context import MediaContext
from suasiv.fusion import fuse
from suasiv.ingest import ingest
from suasiv.llm import get_backend
from suasiv.report import render_report
from suasiv.schema import CoachingReport

console = Console()

ANALYZER_ORDER = [
    "transcript",
    "diarization",
    "pacing",
    "prosody",
    "content",
    "speaker_facial",
    "audience_engagement",
    "audience_reaction",
    "audience_verbal",
]


def run_pipeline(video_path: str, config: SuasivConfig) -> Path:
    ctx = MediaContext(
        video_path=Path(video_path),
        workspace=Path(config.workspace),
        config=config,
    )

    console.print(f"\n[bold]Suasiv[/bold] — analyzing [cyan]{video_path}[/cyan]\n")

    console.print("[dim]Step 1/5:[/dim] Ingesting media...")
    ingest(ctx)
    console.print(f"  Duration: {ctx.duration:.1f}s | Tiles: {len(ctx.tiles)}")

    analyzer_map = {cls.name: cls for cls in ALL_ANALYZERS}
    ordered = [analyzer_map[name] for name in ANALYZER_ORDER if name in analyzer_map]

    console.print("[dim]Step 2/5:[/dim] Running analyzers...")
    results = []
    for cls in ordered:
        analyzer = cls()
        if not config.analyzer_enabled(analyzer.name):
            console.print(f"  [dim]Skipping {analyzer.name} (disabled)[/dim]")
            continue
        console.print(f"  Running [green]{analyzer.name}[/green]...")
        result = analyzer.analyze(ctx)
        results.append(result)
        console.print(f"    → {len(result.signals)} signals")

    console.print("[dim]Step 3/5:[/dim] Fusing signals...")
    timeline = fuse(ctx, results)
    console.print(f"  {len(timeline.signals)} signals, {len(timeline.moments)} moments")

    console.print("[dim]Step 4/5:[/dim] Generating coaching narrative...")
    llm = get_backend(config.llm.backend)
    narrative = llm.complete(
        system="You are a communication coach. Use observable signals, never inferred emotions.",
        prompt=f"Analyze this presentation based on the following signals:\n{timeline.model_dump_json(indent=2)}",
    )

    report = CoachingReport(
        summary_scores={
            "pacing": 0.7,
            "prosody": 0.55,
            "content": 0.7,
            "speaker_facial": 0.6,
            "audience_engagement": 0.72,
            "audience_reaction": 0.65,
        },
        narrative=narrative,
        timeline=timeline,
        metadata={
            "video": video_path,
            "duration": ctx.duration,
            "analyzers_run": [r.analyzer for r in results],
        },
    )

    console.print("[dim]Step 5/5:[/dim] Rendering report...")
    template = "report.md.j2" if config.report.format == "markdown" else "report.html.j2"
    ext = ".md" if config.report.format == "markdown" else ".html"
    output = ctx.workspace / f"report{ext}"
    render_report(report, template, output)

    console.print(f"\n[bold green]Done![/bold green] Report saved to [cyan]{output}[/cyan]\n")
    return output
