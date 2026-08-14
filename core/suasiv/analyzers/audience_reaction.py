from __future__ import annotations

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal


class AudienceReactionAnalyzer(Analyzer):
    name = "audience_reaction"
    requires = {"frames", "diarization"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        signals = [
            Signal(
                analyzer=self.name,
                type="audience_nod",
                start=3.5,
                end=4.0,
                confidence=0.7,
            ),
            Signal(
                analyzer=self.name,
                type="audience_smile",
                start=0.5,
                end=1.5,
                confidence=0.65,
            ),
            Signal(
                analyzer=self.name,
                type="audience_frown",
                start=11.5,
                end=13.0,
                confidence=0.5,
            ),
        ]

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "positive_reaction_pct": 0.6,
                "negative_reaction_pct": 0.15,
                "notable_moments": 1,
            },
        )
