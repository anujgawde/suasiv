from __future__ import annotations

from pydantic import BaseModel, Field


class Signal(BaseModel):
    analyzer: str
    type: str
    start: float
    end: float
    value: str | float | dict | None = None
    confidence: float = 1.0
    speaker: str | None = None


class AnalyzerResult(BaseModel):
    analyzer: str
    signals: list[Signal] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)


class Moment(BaseModel):
    type: str  # strong | weak | turning_point | qa
    start: float
    end: float
    description: str = ""
    signals: list[Signal] = Field(default_factory=list)
    significance: float = 0.0


class FusedTimeline(BaseModel):
    duration: float = 0.0
    signals: list[Signal] = Field(default_factory=list)
    moments: list[Moment] = Field(default_factory=list)


class CoachingReport(BaseModel):
    summary_scores: dict[str, float] = Field(default_factory=dict)
    narrative: str = ""
    timeline: FusedTimeline = Field(default_factory=FusedTimeline)
    metadata: dict = Field(default_factory=dict)
