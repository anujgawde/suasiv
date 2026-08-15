from __future__ import annotations

import json

from rich.console import Console

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal

console = Console()

_FILLER_SINGLES = {"um", "uh", "ah", "er", "like", "basically", "right", "actually"}
_FILLER_BIGRAMS = [("you", "know"), ("sort", "of"), ("i", "mean")]

_PAUSE_BREATH = 0.3
_PAUSE_AWKWARD = 0.8
_PAUSE_DRAMATIC = 2.0


class PacingAnalyzer(Analyzer):
    name = "pacing"
    requires = {"transcript"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        if not ctx.transcript:
            return AnalyzerResult(
                analyzer=self.name, signals=[], summary={"error": "no transcript"}
            )

        settings = ctx.config.analyzer_settings(self.name)
        window_sec = settings.window_seconds or 30

        words = _collect_speaker_words(ctx.transcript, ctx.primary_speaker)
        if not words:
            return AnalyzerResult(
                analyzer=self.name, signals=[], summary={"error": "no speaker words"}
            )

        console.print(f"    Analyzing pacing ({len(words)} words, {window_sec}s windows)")

        signals: list[Signal] = []

        fillers = _detect_fillers(words)
        for f in fillers:
            signals.append(
                Signal(
                    analyzer=self.name,
                    type="filler_word",
                    start=f["start"],
                    end=f["end"],
                    value=f["text"],
                    speaker=ctx.primary_speaker,
                )
            )

        pauses = _detect_pauses(words)
        for p in pauses:
            signals.append(
                Signal(
                    analyzer=self.name,
                    type="long_pause",
                    start=p["start"],
                    end=p["end"],
                    value={"duration": p["duration"], "category": p["category"]},
                    speaker=ctx.primary_speaker,
                )
            )

        duration = words[-1]["end"] - words[0]["start"]
        overall_wpm = len(words) / (duration / 60) if duration > 0 else 0

        wpm_windows = _compute_wpm_windows(words, window_sec, overall_wpm)
        for w in wpm_windows:
            if w["type"] in ("rush", "drag"):
                signals.append(
                    Signal(
                        analyzer=self.name,
                        type=w["type"],
                        start=w["start"],
                        end=w["end"],
                        value=w["wpm"],
                        speaker=ctx.primary_speaker,
                    )
                )

        filler_rate = len(fillers) / len(words)
        longest_pause = max((p["duration"] for p in pauses), default=0.0)

        pacing_path = ctx.workspace / "pacing.json"
        with open(pacing_path, "w") as f:
            json.dump(
                {
                    "overall_wpm": round(overall_wpm, 1),
                    "filler_count": len(fillers),
                    "filler_rate": round(filler_rate, 4),
                    "pause_count": len(pauses),
                    "longest_pause": round(longest_pause, 2),
                    "fillers": fillers,
                    "pauses": pauses,
                    "wpm_windows": wpm_windows,
                },
                f,
                indent=2,
            )

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "overall_wpm": round(overall_wpm, 1),
                "filler_count": len(fillers),
                "filler_rate": round(filler_rate, 4),
                "pause_count": len(pauses),
                "longest_pause": round(longest_pause, 2),
            },
        )


def _collect_speaker_words(
    transcript: list[dict], primary_speaker: str | None
) -> list[dict]:
    words = []
    for seg in transcript:
        if primary_speaker and seg.get("speaker") and seg["speaker"] != primary_speaker:
            continue
        for w in seg.get("words", []):
            words.append(w)
    words.sort(key=lambda w: w["start"])
    return words


def _normalize(word: str) -> str:
    return word.strip().lower().rstrip(".,!?;:'\"")


def _detect_fillers(words: list[dict]) -> list[dict]:
    fillers: list[dict] = []
    used: set[int] = set()

    for i in range(len(words) - 1):
        w1 = _normalize(words[i]["word"])
        w2 = _normalize(words[i + 1]["word"])
        for b1, b2 in _FILLER_BIGRAMS:
            if w1 == b1 and w2 == b2:
                fillers.append(
                    {
                        "text": f"{b1} {b2}",
                        "start": round(words[i]["start"], 3),
                        "end": round(words[i + 1]["end"], 3),
                    }
                )
                used.add(i)
                used.add(i + 1)
                break

    for i, w in enumerate(words):
        if i in used:
            continue
        normalized = _normalize(w["word"])
        if normalized in _FILLER_SINGLES:
            fillers.append(
                {
                    "text": normalized,
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                }
            )

    fillers.sort(key=lambda f: f["start"])
    return fillers


def _detect_pauses(words: list[dict]) -> list[dict]:
    pauses: list[dict] = []
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap < _PAUSE_BREATH:
            continue
        if gap >= _PAUSE_DRAMATIC:
            category = "dramatic"
        elif gap >= _PAUSE_AWKWARD:
            category = "awkward_silence"
        else:
            category = "breath"
        pauses.append(
            {
                "start": round(words[i - 1]["end"], 3),
                "end": round(words[i]["start"], 3),
                "duration": round(gap, 3),
                "category": category,
            }
        )
    return pauses


def _compute_wpm_windows(
    words: list[dict], window_sec: int, overall_wpm: float
) -> list[dict]:
    if not words:
        return []

    start_time = words[0]["start"]
    end_time = words[-1]["end"]
    step = window_sec / 2

    windows: list[dict] = []
    t = start_time
    while t < end_time:
        w_end = t + window_sec
        count = sum(1 for w in words if t <= w["start"] < w_end)
        actual_dur = min(w_end, end_time) - t
        wpm = count / (actual_dur / 60) if actual_dur > 0 else 0

        w_type = "normal"
        if overall_wpm > 0:
            ratio = wpm / overall_wpm
            if ratio > 1.3:
                w_type = "rush"
            elif ratio < 0.7:
                w_type = "drag"

        windows.append(
            {
                "start": round(t, 3),
                "end": round(min(w_end, end_time), 3),
                "wpm": round(wpm, 1),
                "type": w_type,
            }
        )
        t += step

    return windows
