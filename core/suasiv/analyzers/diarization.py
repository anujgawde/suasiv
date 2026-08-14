from __future__ import annotations

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal


class DiarizationAnalyzer(Analyzer):
    name = "diarization"
    requires = {"audio", "transcript"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        ctx.speaker_labels = {
            "SPEAKER_0": "primary",
            "SPEAKER_1": "audience",
        }
        ctx.primary_speaker = "SPEAKER_0"

        signals = [
            Signal(
                analyzer=self.name,
                type="speaker_segment",
                start=0.0,
                end=14.0,
                value="SPEAKER_0",
                speaker="SPEAKER_0",
            ),
            Signal(
                analyzer=self.name,
                type="speaker_change",
                start=14.5,
                end=16.0,
                value="SPEAKER_1",
                speaker="SPEAKER_1",
            ),
        ]

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "speaker_count": 2,
                "primary_speaker": "SPEAKER_0",
                "primary_talk_ratio": 0.85,
            },
        )
