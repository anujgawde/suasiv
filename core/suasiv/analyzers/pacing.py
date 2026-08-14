from __future__ import annotations

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal


class PacingAnalyzer(Analyzer):
    name = "pacing"
    requires = {"transcript"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        signals = [
            Signal(analyzer=self.name, type="filler_word", start=11.5, end=11.8, value="um"),
            Signal(analyzer=self.name, type="rate_change", start=7.5, end=11.0, value="rush"),
            Signal(
                analyzer=self.name,
                type="long_pause",
                start=14.0,
                end=14.5,
                value={"duration": 0.5},
            ),
        ]

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "overall_wpm": 145,
                "filler_count": 1,
                "filler_rate": 0.036,
                "pause_count": 1,
                "longest_pause": 0.5,
            },
        )
