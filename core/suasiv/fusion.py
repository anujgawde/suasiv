from __future__ import annotations

import json

from rich.console import Console

from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, FusedTimeline, Moment, Signal

console = Console()

_POSITIVE_SPEAKER = {
    "gaze_on_camera",
    "high_expressiveness",
    "high_energy_segment",
    "pitch_spike",
}

_NEGATIVE_SPEAKER = {
    "filler_word",
    "long_pause",
    "rush",
    "drag",
    "energy_drop",
    "pitch_drop",
    "monotone_segment",
    "gaze_off_camera",
    "head_turn_away",
    "low_expressiveness",
}

_POSITIVE_AUDIENCE = {
    "audience_engaged",
    "audience_smile",
    "audience_nod",
    "simultaneous_positive_reaction",
    "verbal_agreement",
    "attention_recovery",
}

_NEGATIVE_AUDIENCE = {
    "attention_drop",
    "audience_frown",
    "audience_confusion",
    "simultaneous_negative_reaction",
    "verbal_disagreement",
    "interruption",
}

_QA_SIGNALS = {"audience_question"}

_TURNING_SIGNALS = {"attention_drop", "attention_recovery"}

_SIGNAL_LABELS = {
    "gaze_on_camera": "eye contact",
    "high_expressiveness": "expressive delivery",
    "high_energy_segment": "high energy",
    "pitch_spike": "vocal emphasis",
    "filler_word": "filler words",
    "long_pause": "long pause",
    "rush": "rushed pace",
    "drag": "slow pace",
    "energy_drop": "energy drop",
    "energy_spike": "energy spike",
    "pitch_drop": "pitch drop",
    "monotone_segment": "monotone delivery",
    "gaze_off_camera": "gaze break",
    "head_turn_away": "looking away",
    "low_expressiveness": "flat expression",
    "audience_engaged": "audience attentive",
    "audience_smile": "audience smiling",
    "audience_nod": "audience nodding",
    "simultaneous_positive_reaction": "group positive reaction",
    "verbal_agreement": "verbal agreement",
    "attention_recovery": "attention recovered",
    "attention_drop": "attention dropped",
    "audience_frown": "audience frowning",
    "audience_confusion": "audience confused",
    "audience_surprise": "audience surprised",
    "simultaneous_negative_reaction": "group negative reaction",
    "verbal_disagreement": "verbal disagreement",
    "interruption": "interruption",
    "audience_question": "audience question",
}


def fuse(ctx: MediaContext, results: list[AnalyzerResult]) -> FusedTimeline:
    all_signals: list[Signal] = []
    for r in results:
        all_signals.extend(r.signals)
    all_signals.sort(key=lambda s: s.start)

    if not all_signals:
        _save_artifact(ctx, all_signals, [])
        return FusedTimeline(duration=ctx.duration)

    window = ctx.config.fusion.window_seconds or 5
    step = window / 2.0

    raw_moments: list[Moment] = []
    t = 0.0
    while t < ctx.duration:
        w_end = min(t + window, ctx.duration)
        w_sigs = [s for s in all_signals if s.start < w_end and s.end >= t]
        if w_sigs:
            m = _classify_window(t, w_end, w_sigs)
            if m is not None:
                raw_moments.append(m)
        t += step

    moments = _deduplicate_moments(raw_moments)
    moments.sort(key=lambda m: (-m.significance, m.start))

    _save_artifact(ctx, all_signals, moments)

    return FusedTimeline(duration=ctx.duration, signals=all_signals, moments=moments)


def stitch_chunk_results(
    chunk_results: list[list[AnalyzerResult]],
    chunk_ranges: list[tuple[float, float]],
    overlap_sec: float,
) -> list[AnalyzerResult]:
    if len(chunk_results) <= 1:
        return chunk_results[0] if chunk_results else []

    by_analyzer: dict[str, list[AnalyzerResult]] = {}
    for chunk in chunk_results:
        for result in chunk:
            by_analyzer.setdefault(result.analyzer, []).append(result)

    merged: list[AnalyzerResult] = []
    for analyzer_name, results in by_analyzer.items():
        all_sigs: list[Signal] = []
        for r in results:
            all_sigs.extend(r.signals)

        all_sigs.sort(key=lambda s: s.start)
        deduped: list[Signal] = []
        seen: set[tuple] = set()
        for s in all_sigs:
            key = (s.analyzer, s.type, round(s.start, 1), round(s.end, 1))
            if key not in seen:
                deduped.append(s)
                seen.add(key)

        summary: dict = {}
        for r in results:
            summary.update(r.summary)

        merged.append(
            AnalyzerResult(analyzer=analyzer_name, signals=deduped, summary=summary)
        )

    return merged


def _classify_window(
    start: float, end: float, signals: list[Signal]
) -> Moment | None:
    types = {s.type for s in signals}

    pos_spk = types & _POSITIVE_SPEAKER
    neg_spk = types & _NEGATIVE_SPEAKER
    pos_aud = types & _POSITIVE_AUDIENCE
    neg_aud = types & _NEGATIVE_AUDIENCE
    qa = types & _QA_SIGNALS
    turning = types & _TURNING_SIGNALS

    if qa:
        parts = ["audience question"]
        if pos_spk:
            parts.append(_label_set(pos_spk))
        return Moment(
            type="qa",
            start=round(start, 3),
            end=round(end, 3),
            description=", ".join(parts),
            signals=signals,
            significance=_score(signals),
        )

    if pos_spk and pos_aud and not neg_aud:
        return Moment(
            type="strong",
            start=round(start, 3),
            end=round(end, 3),
            description=_build_desc(pos_spk, pos_aud),
            signals=signals,
            significance=_score(signals),
        )

    if neg_spk and neg_aud:
        return Moment(
            type="weak",
            start=round(start, 3),
            end=round(end, 3),
            description=_build_desc(neg_spk, neg_aud),
            signals=signals,
            significance=_score(signals),
        )

    if turning:
        if "attention_drop" in types:
            desc = "attention dropped"
            if neg_spk:
                desc += f" during {_label_set(neg_spk)}"
        else:
            desc = "audience re-engaged"
            if pos_spk:
                desc += f" during {_label_set(pos_spk)}"
        return Moment(
            type="turning_point",
            start=round(start, 3),
            end=round(end, 3),
            description=desc,
            signals=signals,
            significance=_score(signals),
        )

    if pos_spk and len(pos_spk) >= 2 and not neg_aud:
        return Moment(
            type="strong",
            start=round(start, 3),
            end=round(end, 3),
            description=_build_desc(pos_spk, pos_aud),
            signals=signals,
            significance=round(_score(signals) * 0.7, 3),
        )

    if neg_spk and len(neg_spk) >= 2 and not pos_aud:
        return Moment(
            type="weak",
            start=round(start, 3),
            end=round(end, 3),
            description=_build_desc(neg_spk, neg_aud),
            signals=signals,
            significance=round(_score(signals) * 0.7, 3),
        )

    return None


def _score(signals: list[Signal]) -> float:
    unique_types = {s.type for s in signals}
    diversity = min(len(unique_types) / 6.0, 1.0)

    unique_analyzers = {s.analyzer for s in signals}
    breadth = min(len(unique_analyzers) / 4.0, 1.0)

    high_weight = {
        "simultaneous_positive_reaction",
        "simultaneous_negative_reaction",
        "attention_drop",
        "attention_recovery",
        "audience_question",
        "interruption",
    }
    boost = 0.15 if unique_types & high_weight else 0.0

    confidences = [s.confidence for s in signals]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.5

    raw = diversity * 0.35 + breadth * 0.35 + avg_conf * 0.15 + boost
    return round(min(raw, 1.0), 3)


def _build_desc(set_a: set[str], set_b: set[str]) -> str:
    parts = []
    if set_a:
        parts.append(_label_set(set_a))
    if set_b:
        parts.append(_label_set(set_b))
    return " + ".join(parts) if parts else "signal correlation"


def _label_set(types: set[str]) -> str:
    labels = [_SIGNAL_LABELS.get(t, t.replace("_", " ")) for t in sorted(types)]
    return ", ".join(labels[:3])


def _deduplicate_moments(moments: list[Moment]) -> list[Moment]:
    if not moments:
        return []

    moments.sort(key=lambda m: (m.type, m.start))
    merged: list[Moment] = []

    for m in moments:
        if merged and merged[-1].type == m.type and m.start <= merged[-1].end:
            prev = merged[-1]
            new_end = max(prev.end, m.end)
            new_sig = max(prev.significance, m.significance)

            seen = {(s.analyzer, s.type, s.start) for s in prev.signals}
            combined = list(prev.signals)
            for s in m.signals:
                key = (s.analyzer, s.type, s.start)
                if key not in seen:
                    combined.append(s)
                    seen.add(key)

            desc = (
                prev.description
                if prev.significance >= m.significance
                else m.description
            )

            merged[-1] = Moment(
                type=prev.type,
                start=prev.start,
                end=new_end,
                description=desc,
                signals=combined,
                significance=new_sig,
            )
        else:
            merged.append(m)

    return merged


def _save_artifact(
    ctx: MediaContext, signals: list[Signal], moments: list[Moment]
) -> None:
    path = ctx.workspace / "fusion.json"
    with open(path, "w") as f:
        json.dump(
            {
                "total_signals": len(signals),
                "moments_detected": len(moments),
                "duration": ctx.duration,
                "analyzers_contributing": len({s.analyzer for s in signals}),
                "moments": [
                    {
                        "type": m.type,
                        "start": m.start,
                        "end": m.end,
                        "description": m.description,
                        "significance": m.significance,
                        "signal_count": len(m.signals),
                    }
                    for m in moments
                ],
            },
            f,
            indent=2,
        )
