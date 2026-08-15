from __future__ import annotations

import shutil
from pathlib import Path

import yaml
from rich.console import Console

from suasiv.analyzers import ALL_ANALYZERS
from suasiv.config import SuasivConfig
from suasiv.context import MediaContext
from suasiv.fusion import fuse, stitch_chunk_results
from suasiv.ingest import ingest
from suasiv.llm import get_backend
from suasiv.report import render_report
from suasiv.schema import AnalyzerResult, CoachingReport, FusedTimeline

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
    rubric = _load_rubric()
    scores = _compute_scores(results, rubric)
    system_prompt = _build_system_prompt(rubric)
    user_prompt = _build_user_prompt(ctx, results, timeline, scores)

    llm = get_backend(config.llm.backend, config.llm)
    narrative = llm.complete(system_prompt, user_prompt)

    report = CoachingReport(
        summary_scores=scores,
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


# ---------------------------------------------------------------------------
# Rubric + scoring
# ---------------------------------------------------------------------------


def _load_rubric() -> dict:
    for candidate in [
        Path("rubric.yaml"),
        Path(__file__).resolve().parent.parent.parent / "rubric.yaml",
    ]:
        if candidate.exists():
            with open(candidate) as f:
                return yaml.safe_load(f) or {}
    return {}


def _compute_scores(results: list[AnalyzerResult], rubric: dict) -> dict[str, float]:
    summaries = {r.analyzer: r.summary for r in results}
    dims = rubric.get("dimensions", {})
    scores: dict[str, float] = {}

    if "pacing" in summaries and not summaries["pacing"].get("error"):
        s = summaries["pacing"]
        t = dims.get("pacing", {}).get("thresholds", {})
        wpm = s.get("overall_wpm", 130)
        wpm_lo = t.get("wpm_low", 100)
        wpm_hi = t.get("wpm_high", 180)
        wpm_score = 1.0
        if wpm < wpm_lo:
            wpm_score = max(0.2, wpm / wpm_lo)
        elif wpm > wpm_hi:
            wpm_score = max(0.2, wpm_hi / wpm)

        filler_rate = s.get("filler_rate", 0)
        bad = t.get("filler_rate_bad", 0.06)
        conc = t.get("filler_rate_concerning", 0.03)
        if filler_rate >= bad:
            filler_score = 0.2
        elif filler_rate >= conc:
            filler_score = 1.0 - 0.8 * (filler_rate - conc) / (bad - conc)
        else:
            filler_score = 1.0

        scores["pacing"] = round(wpm_score * 0.5 + filler_score * 0.5, 2)

    if "prosody" in summaries and not summaries["prosody"].get("error"):
        s = summaries["prosody"]
        t = dims.get("prosody", {}).get("thresholds", {})
        variety = s.get("vocal_variety_score", 0.3)
        variety_low = t.get("vocal_variety_low", 0.3)
        variety_score = min(1.0, variety / variety_low) if variety_low > 0 else 0.5
        monotone = s.get("monotone_segments", 0)
        monotone_score = max(0.0, 1.0 - monotone * 0.15)
        scores["prosody"] = round(variety_score * 0.6 + monotone_score * 0.4, 2)

    if "content" in summaries and not summaries["content"].get("error"):
        scores["content"] = round(summaries["content"].get("structure_score", 0.5), 2)

    if "speaker_facial" in summaries and not summaries["speaker_facial"].get("error"):
        s = summaries["speaker_facial"]
        eye = s.get("eye_contact_pct", 0.5)
        facing = s.get("facing_audience_pct", 0.5)
        expr = s.get("expressiveness_score", 0.5)
        scores["speaker_facial"] = round(eye * 0.4 + facing * 0.3 + expr * 0.3, 2)

    if "audience_engagement" in summaries and not summaries["audience_engagement"].get(
        "error"
    ):
        s = summaries["audience_engagement"]
        attention = s.get("overall_attention_pct", 0.5)
        drops = s.get("attention_drops", 0)
        drop_penalty = max(0.0, 1.0 - drops * 0.1)
        scores["audience_engagement"] = round(
            attention * 0.7 + drop_penalty * 0.3, 2
        )

    if "audience_reaction" in summaries and not summaries["audience_reaction"].get(
        "error"
    ):
        s = summaries["audience_reaction"]
        positive = s.get("positive_reaction_pct", 0)
        negative = s.get("negative_reaction_pct", 0)
        t = dims.get("audience_reaction", {}).get("thresholds", {})
        pos_low = t.get("positive_ratio_low", 0.3)
        neg_high = t.get("negative_ratio_high", 0.3)
        pos_score = min(1.0, positive / pos_low) if pos_low > 0 else 0.5
        neg_score = max(0.0, 1.0 - negative / neg_high) if neg_high > 0 else 1.0
        scores["audience_reaction"] = round(pos_score * 0.6 + neg_score * 0.4, 2)

    return scores


# ---------------------------------------------------------------------------
# Prompt engineering
# ---------------------------------------------------------------------------


def _build_system_prompt(rubric: dict) -> str:
    coaching = rubric.get("coaching", {})
    framing = coaching.get(
        "framing",
        "Reference observable signals, never inferred emotions. Be specific and actionable.",
    )

    dim_lines: list[str] = []
    for name, dim in rubric.get("dimensions", {}).items():
        dim_lines.append(
            f"- {name.replace('_', ' ').title()}: "
            f"Good = {dim.get('good', 'N/A')}; "
            f"Concerning = {dim.get('concerning', 'N/A')}; "
            f"Needs work = {dim.get('bad', 'N/A')}"
        )

    return (
        "You are a communication coach analyzing a recorded presentation.\n\n"
        f"RULES:\n{framing}\n\n"
        "SCORING CRITERIA:\n"
        f"{chr(10).join(dim_lines)}\n\n"
        "Produce your analysis with these sections:\n"
        "## Overall Assessment\n"
        "2-3 sentences summarizing strengths and areas for improvement.\n\n"
        "## What Worked\n"
        "Bullet points with timestamps and signal evidence. "
        "Reference audience response where available.\n\n"
        "## What to Improve\n"
        "Bullet points with timestamps and signal evidence. "
        "Be constructive.\n\n"
        "## Key Moments\n"
        "Numbered list of significant moments with timestamps, "
        "what the speaker did, and how the audience responded.\n\n"
        "## Audience Reception\n"
        "Attention patterns, reaction trends, verbal interaction summary.\n\n"
        "## Next Steps\n"
        "3-5 specific, actionable recommendations.\n\n"
        "Format timestamps as M:SS. Reference actual signal data."
    )


def _fmt(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _build_user_prompt(
    ctx: MediaContext,
    results: list[AnalyzerResult],
    timeline: FusedTimeline,
    scores: dict[str, float],
) -> str:
    score_lines = [
        f"- {n.replace('_', ' ').title()}: {v * 100:.0f}%"
        for n, v in scores.items()
    ]

    summary_lines: list[str] = []
    for r in results:
        if r.summary and not r.summary.get("error"):
            items = ", ".join(f"{k}={v}" for k, v in r.summary.items())
            summary_lines.append(f"- {r.analyzer}: {items}")

    moment_lines: list[str] = []
    for i, m in enumerate(timeline.moments[:15]):
        moment_lines.append(
            f"{i + 1}. [{_fmt(m.start)}-{_fmt(m.end)}] "
            f"{m.type.upper()} (sig={m.significance}): {m.description}"
        )

    signal_lines: list[str] = []
    for s in timeline.signals[:100]:
        val = f" ({s.value})" if s.value is not None else ""
        signal_lines.append(f"  {_fmt(s.start)} {s.analyzer}/{s.type}{val}")

    return (
        f"Analyze this presentation ({_fmt(ctx.duration)} duration).\n\n"
        f"DIMENSION SCORES:\n{chr(10).join(score_lines)}\n\n"
        f"ANALYZER SUMMARIES:\n{chr(10).join(summary_lines)}\n\n"
        "KEY MOMENTS (ranked by significance):\n"
        f"{chr(10).join(moment_lines) or 'None detected.'}\n\n"
        f"SIGNAL TIMELINE ({len(timeline.signals)} total, "
        f"showing first {min(100, len(timeline.signals))}):\n"
        f"{chr(10).join(signal_lines) or 'No signals.'}"
    )


# ---------------------------------------------------------------------------
# Long-video chunking
# ---------------------------------------------------------------------------


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
            f"  [dim]Chunk {ci + 1}/{len(chunks)} "
            f"({_fmt(c_start)}–{_fmt(c_end)})[/dim]"
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
