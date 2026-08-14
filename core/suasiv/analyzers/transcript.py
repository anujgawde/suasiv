from __future__ import annotations

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal


class TranscriptAnalyzer(Analyzer):
    name = "transcript"
    requires = {"audio"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        segments = [
            {"text": "Welcome everyone to today's presentation.", "start": 0.0, "end": 3.5},
            {"text": "I'd like to discuss our quarterly results.", "start": 3.8, "end": 7.2},
            {"text": "Revenue grew by fifteen percent year over year.", "start": 7.5, "end": 11.0},
            {"text": "Um, let me show you the breakdown.", "start": 11.5, "end": 14.0},
            {"text": "Any questions so far?", "start": 14.5, "end": 16.0},
        ]

        ctx.transcript = segments

        signals = [
            Signal(
                analyzer=self.name,
                type="transcript_segment",
                start=seg["start"],
                end=seg["end"],
                value=seg["text"],
            )
            for seg in segments
        ]

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={"segment_count": len(segments), "total_words": 28},
        )
