from __future__ import annotations

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal


class SpeakerFacialAnalyzer(Analyzer):
    name = "speaker_facial"
    requires = {"frames"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        signals = [
            Signal(
                analyzer=self.name,
                type="gaze_off_camera",
                start=11.0,
                end=14.0,
                confidence=0.7,
            ),
            Signal(
                analyzer=self.name,
                type="gaze_on_camera",
                start=0.0,
                end=7.0,
                confidence=0.85,
            ),
            Signal(
                analyzer=self.name,
                type="low_expressiveness",
                start=7.5,
                end=11.0,
                confidence=0.6,
            ),
        ]

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "eye_contact_pct": 0.65,
                "facing_audience_pct": 0.75,
                "expressiveness_score": 0.5,
            },
        )
