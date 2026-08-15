from __future__ import annotations

import shutil
from pathlib import Path

from rich.console import Console

from suasiv.analyzers import ALL_ANALYZERS
from suasiv.config import SuasivConfig
from suasiv.context import MediaContext
from suasiv.fusion import fuse, stitch_chunk_results
from suasiv.ingest import ingest
from suasiv.llm import get_backend
from suasiv.report import render_report
from suasiv.schema import AnalyzerResult, CoachingReport

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

_CHUNKABLE = {"speaker_facial", "audience_engagement", "audience_reaction"}


def run_pipeline(video_path: str, config: SuasivConfig) -> Path:
    ctx = MediaContext(
        video_path=Path(video_path),
        workspace=Path(config.workspace),
        config=config,
    )

    console.print(f"\n[bold]Suasiv[/bold] — analyzing [cyan]{video_path}[/cyan]\n")

    console.print("[dim]Step 1/5:[/dim] Ingesting media...")
    ingest(ctx)
    console.print(
        f"  Duration: {ctx.duration:.1f}s | {ctx.frame_count} frames | {len(ctx.tiles)} tile(s)"
    )

    analyzer_map = {cls.name: cls for cls in ALL_ANALYZERS}
    ordered = [analyzer_map[name] for name in ANALYZER_ORDER if name in analyzer_map]

    full_run = [cls for cls in ordered if cls.name not in _CHUNKABLE]
    chunkable = [cls for cls in ordered if cls.name in _CHUNKABLE]

    console.print("[dim]Step 2/5:[/dim] Running analyzers...")
    results: list[AnalyzerResult] = []

    for cls in full_run:
        analyzer = cls()
        if not config.analyzer_enabled(analyzer.name):
            console.print(f"  [dim]Skipping {analyzer.name} (disabled)[/dim]")
            continue
        console.print(f"  Running [green]{analyzer.name}[/green]...")
        result = analyzer.analyze(ctx)
        results.append(result)
        console.print(f"    → {len(result.signals)} signals")

    enabled_chunkable = [cls for cls in chunkable if config.analyzer_enabled(cls.name)]
    needs_chunking = (
        ctx.duration > config.fusion.chunk_seconds
        and enabled_chunkable
        and ctx.frames_dir
    )

    if needs_chunking:
        chunk_results = _run_chunked(ctx, enabled_chunkable, config)
        results.extend(chunk_results)
    else:
        for cls in chunkable:
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

    console.print(
        f"\n[bold green]Done![/bold green] Report saved to [cyan]{output}[/cyan]\n"
    )
    return output


def _compute_chunks(
    duration: float, chunk_sec: int, overlap: int
) -> list[tuple[float, float]]:
    chunks: list[tuple[float, float]] = []
    t = 0.0
    while t < duration:
        end = min(t + chunk_sec, duration)
        chunks.append((t, end))
        if end >= duration:
            break
        t += chunk_sec - overlap
    return chunks


def _run_chunked(
    ctx: MediaContext,
    analyzer_classes: list[type],
    config: SuasivConfig,
) -> list[AnalyzerResult]:
    chunk_sec = config.fusion.chunk_seconds
    overlap = config.fusion.chunk_overlap
    chunks = _compute_chunks(ctx.duration, chunk_sec, overlap)

    if len(chunks) <= 1:
        results: list[AnalyzerResult] = []
        for cls in analyzer_classes:
            analyzer = cls()
            console.print(f"  Running [green]{analyzer.name}[/green]...")
            result = analyzer.analyze(ctx)
            results.append(result)
            console.print(f"    → {len(result.signals)} signals")
        return results

    console.print(
        f"  Chunking: {len(chunks)} chunks ({chunk_sec}s, {overlap}s overlap)"
    )

    all_chunk_results: list[list[AnalyzerResult]] = []

    for ci, (c_start, c_end) in enumerate(chunks):
        console.print(
            f"  [dim]Chunk {ci + 1}/{len(chunks)} ({c_start:.0f}–{c_end:.0f}s)[/dim]"
        )
        chunk_ctx = _create_chunk_context(ctx, c_start, c_end)
        chunk_results: list[AnalyzerResult] = []

        for cls in analyzer_classes:
            analyzer = cls()
            console.print(f"    Running [green]{analyzer.name}[/green]...")
            result = analyzer.analyze(chunk_ctx)
            chunk_results.append(result)
            console.print(f"      → {len(result.signals)} signals")

        all_chunk_results.append(chunk_results)
        shutil.rmtree(chunk_ctx.workspace, ignore_errors=True)

    stitched = stitch_chunk_results(all_chunk_results, chunks, overlap)

    total_before = sum(
        len(r.signals) for chunk in all_chunk_results for r in chunk
    )
    total_after = sum(len(r.signals) for r in stitched)
    if total_before != total_after:
        console.print(
            f"  Stitched {len(chunks)} chunks → {total_after} signals "
            f"(deduped {total_before - total_after})"
        )

    return stitched


def _create_chunk_context(
    ctx: MediaContext, chunk_start: float, chunk_end: float
) -> MediaContext:
    fps = ctx.config.ingest.frame_fps or 3
    start_frame = int(chunk_start * fps) + 1
    end_frame = int(chunk_end * fps) + 1

    chunk_ws = ctx.workspace / f"_chunk_{int(chunk_start)}_{int(chunk_end)}"
    chunk_ws.mkdir(parents=True, exist_ok=True)

    chunk_frames = None
    if ctx.frames_dir and ctx.frames_dir.exists():
        chunk_frames = chunk_ws / "frames"
        chunk_frames.mkdir(exist_ok=True)
        for frame in ctx.frames_dir.glob("frame_*.png"):
            num = int(frame.stem.split("_")[1])
            if start_frame <= num <= end_frame:
                link = chunk_frames / frame.name
                if not link.exists():
                    link.symlink_to(frame.resolve())

    chunk_tiles = None
    if ctx.tiles_dir and ctx.tiles_dir.exists():
        chunk_tiles = chunk_ws / "tiles"
        chunk_tiles.mkdir(exist_ok=True)
        for tile in ctx.tiles:
            src_dir = ctx.tiles_dir / f"tile_{tile.index:02d}"
            if not src_dir.exists():
                continue
            dst_dir = chunk_tiles / f"tile_{tile.index:02d}"
            dst_dir.mkdir(exist_ok=True)
            for frame in src_dir.glob("frame_*.png"):
                num = int(frame.stem.split("_")[1])
                if start_frame <= num <= end_frame:
                    link = dst_dir / frame.name
                    if not link.exists():
                        link.symlink_to(frame.resolve())

    return MediaContext(
        video_path=ctx.video_path,
        workspace=chunk_ws,
        config=ctx.config,
        audio_path=ctx.audio_path,
        frames_dir=chunk_frames,
        tiles_dir=chunk_tiles,
        tiles=ctx.tiles,
        speaker_tile_index=ctx.speaker_tile_index,
        frame_count=end_frame - start_frame,
        transcript=ctx.transcript,
        speaker_labels=ctx.speaker_labels,
        primary_speaker=ctx.primary_speaker,
        duration=ctx.duration,
        resolution=ctx.resolution,
        fps=ctx.fps,
    )
