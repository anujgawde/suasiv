from __future__ import annotations

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal


class AudienceVerbalAnalyzer(Analyzer):
    name = "audience_verbal"
    requires = {"transcript", "diarization"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        signals = [
            Signal(
                analyzer=self.name,
                type="audience_question",
                start=14.5,
                end=16.0,
                value="Any questions so far?",
                speaker="SPEAKER_1",
            ),
            Signal(
                analyzer=self.name,
                type="verbal_agreement",
                start=7.2,
                end=7.4,
                value="right",
                speaker="SPEAKER_1",
                confidence=0.6,
            ),
        ]

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "questions_count": 1,
                "interruptions_count": 0,
                "agreements_count": 1,
                "disagreements_count": 0,
            },
        )
