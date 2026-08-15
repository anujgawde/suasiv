from __future__ import annotations

import json

from rich.console import Console

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal

console = Console()

_QUESTION_STARTS = {
    "who", "what", "where", "when", "why", "how",
    "can", "could", "would", "should", "will",
    "is", "are", "do", "does", "did", "have", "has",
}

_AGREEMENT_WORDS = {
    "yes", "yeah", "yep", "right", "exactly", "absolutely",
    "agreed", "correct", "true", "sure", "okay", "ok",
    "definitely", "certainly",
}

_DISAGREEMENT_WORDS = {"no", "nope", "disagree", "but", "however"}
_DISAGREEMENT_PHRASES = ["not really", "i don't think", "i disagree"]

_INTERRUPTION_GAP = 0.3


class AudienceVerbalAnalyzer(Analyzer):
    name = "audience_verbal"
    requires = {"transcript", "diarization"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        if not ctx.transcript:
            return AnalyzerResult(
                analyzer=self.name, signals=[], summary={"error": "no transcript"}
            )
        if not ctx.primary_speaker:
            return AnalyzerResult(
                analyzer=self.name,
                signals=[],
                summary={"error": "no diarization — cannot identify audience"},
            )

        audience_segs = [
            s
            for s in ctx.transcript
            if s.get("speaker") and s["speaker"] != ctx.primary_speaker
        ]

        if not audience_segs:
            return AnalyzerResult(
                analyzer=self.name,
                signals=[],
                summary={
                    "questions_count": 0,
                    "interruptions_count": 0,
                    "agreements_count": 0,
                    "disagreements_count": 0,
                },
            )

        console.print(
            f"    Analyzing {len(audience_segs)} audience verbal segments..."
        )

        signals: list[Signal] = []

        questions = _detect_questions(audience_segs)
        for q in questions:
            signals.append(
                Signal(
                    analyzer=self.name,
                    type="audience_question",
                    start=q["start"],
                    end=q["end"],
                    value=q["text"],
                    speaker=q["speaker"],
                )
            )

        interruptions = _detect_interruptions(ctx.transcript, ctx.primary_speaker)
        for intr in interruptions:
            signals.append(
                Signal(
                    analyzer=self.name,
                    type="interruption",
                    start=intr["start"],
                    end=intr["end"],
                    value=intr["text"],
                    speaker=intr["speaker"],
                )
            )

        agreements, disagreements = _detect_verbal_reactions(audience_segs)
        for a in agreements:
            signals.append(
                Signal(
                    analyzer=self.name,
                    type="verbal_agreement",
                    start=a["start"],
                    end=a["end"],
                    value=a["text"],
                    speaker=a["speaker"],
                    confidence=0.7,
                )
            )
        for d in disagreements:
            signals.append(
                Signal(
                    analyzer=self.name,
                    type="verbal_disagreement",
                    start=d["start"],
                    end=d["end"],
                    value=d["text"],
                    speaker=d["speaker"],
                    confidence=0.7,
                )
            )

        verbal_path = ctx.workspace / "audience_verbal.json"
        with open(verbal_path, "w") as f:
            json.dump(
                {
                    "questions_count": len(questions),
                    "interruptions_count": len(interruptions),
                    "agreements_count": len(agreements),
                    "disagreements_count": len(disagreements),
                    "audience_segments": len(audience_segs),
                    "questions": questions,
                    "interruptions": interruptions,
                },
                f,
                indent=2,
            )

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "questions_count": len(questions),
                "interruptions_count": len(interruptions),
                "agreements_count": len(agreements),
                "disagreements_count": len(disagreements),
            },
        )


def _detect_questions(segments: list[dict]) -> list[dict]:
    questions: list[dict] = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        is_q = "?" in text
        if not is_q:
            first_word = text.split()[0].lower().rstrip(".,!?") if text.split() else ""
            is_q = first_word in _QUESTION_STARTS
        if is_q:
            questions.append(
                {
                    "text": text,
                    "start": round(seg["start"], 3),
                    "end": round(seg["end"], 3),
                    "speaker": seg.get("speaker"),
                }
            )
    return questions


def _detect_interruptions(
    transcript: list[dict], primary_speaker: str
) -> list[dict]:
    interruptions: list[dict] = []

    for i in range(1, len(transcript)):
        curr = transcript[i]
        prev = transcript[i - 1]

        if not curr.get("speaker") or curr["speaker"] == primary_speaker:
            continue
        if prev.get("speaker") != primary_speaker:
            continue

        gap = curr["start"] - prev["end"]

        if gap < _INTERRUPTION_GAP:
            speaker_continues = False
            for j in range(i + 1, min(i + 3, len(transcript))):
                if transcript[j].get("speaker") == primary_speaker:
                    speaker_continues = True
                    break

            if speaker_continues:
                interruptions.append(
                    {
                        "text": curr["text"].strip(),
                        "start": round(curr["start"], 3),
                        "end": round(curr["end"], 3),
                        "speaker": curr["speaker"],
                        "gap": round(gap, 3),
                    }
                )

    return interruptions


def _detect_verbal_reactions(
    segments: list[dict],
) -> tuple[list[dict], list[dict]]:
    agreements: list[dict] = []
    disagreements: list[dict] = []

    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        lower = text.lower()

        words = {w.strip(".,!?;:'\"").lower() for w in text.split()}

        is_agreement = bool(words & _AGREEMENT_WORDS)
        is_disagreement = bool(words & _DISAGREEMENT_WORDS) or any(
            p in lower for p in _DISAGREEMENT_PHRASES
        )

        entry = {
            "text": text,
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "speaker": seg.get("speaker"),
        }

        if is_agreement and not is_disagreement:
            agreements.append(entry)
        elif is_disagreement:
            disagreements.append(entry)

    return agreements, disagreements
