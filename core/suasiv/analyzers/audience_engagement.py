from __future__ import annotations

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal


class AudienceEngagementAnalyzer(Analyzer):
    name = "audience_engagement"
    requires = {"frames", "diarization"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        signals = [
            Signal(
                analyzer=self.name,
                type="attention_drop",
                start=11.0,
                end=14.0,
                value={"attention_pct": 0.4},
                confidence=0.7,
            ),
            Signal(
                analyzer=self.name,
                type="audience_engaged",
                start=0.0,
                end=7.0,
                value={"attention_pct": 0.9},
                confidence=0.85,
            ),
        ]

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "overall_attention_pct": 0.72,
                "attention_drops": 1,
                "lowest_attention_moment": 12.0,
            },
        )
