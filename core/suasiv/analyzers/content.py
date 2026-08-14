from __future__ import annotations

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal


class ContentAnalyzer(Analyzer):
    name = "content"
    requires = {"transcript"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        signals = [
            Signal(
                analyzer=self.name,
                type="strong_transition",
                start=3.5,
                end=3.8,
                value="Clear topic shift to quarterly results",
            ),
            Signal(
                analyzer=self.name,
                type="hedging_language",
                start=11.5,
                end=14.0,
                value="let me show you",
                confidence=0.5,
            ),
        ]

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "hedging_count": 1,
                "hedging_rate": 0.036,
                "structure_score": 0.7,
                "transitions_count": 1,
            },
        )
