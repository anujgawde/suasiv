from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SignalType(str, Enum):
    # Transcription
    FILLER_WORD = "filler_word"
    STUTTER = "stutter"
    FALSE_START = "false_start"

    # Diarization
    SPEAKER_CHANGE = "speaker_change"

    # Vocal emotion
    CONFIDENT_DELIVERY = "confident_delivery"
    LOW_CONFIDENCE = "low_confidence"
    HIGH_ENERGY = "high_energy"
    ENERGY_DROP = "energy_drop"
    POSITIVE_TONE = "positive_tone"
    NEGATIVE_TONE = "negative_tone"

    # Speaker visual
    EYE_CONTACT = "eye_contact"
    GAZE_BREAK = "gaze_break"
    HEAD_TURN = "head_turn"
    EXPRESSIVE = "expressive"
    FLAT_EXPRESSION = "flat_expression"

    # Audience visual
    AUDIENCE_ATTENTIVE = "audience_attentive"
    ATTENTION_DROP = "attention_drop"
    ATTENTION_RECOVERY = "attention_recovery"
    AUDIENCE_SMILE = "audience_smile"
    AUDIENCE_CONFUSION = "audience_confusion"
    AUDIENCE_NOD = "audience_nod"

    # Content
    HEDGING_LANGUAGE = "hedging_language"
    CLEAR_TRANSITION = "clear_transition"
    RHETORICAL_QUESTION = "rhetorical_question"
    TOPIC_SHIFT = "topic_shift"

    # Pacing (derived from transcript)
    RUSH = "rush"
    DRAG = "drag"
    LONG_PAUSE = "long_pause"


class MomentType(str, Enum):
    STRONG = "strong"
    WEAK = "weak"
    TURNING_POINT = "turning_point"
    QA = "qa"


class Dimension(str, Enum):
    DELIVERY_CONFIDENCE = "delivery_confidence"
    VOCAL_VARIETY = "vocal_variety"
    PACING_FLUENCY = "pacing_fluency"
    CONTENT_CLARITY = "content_clarity"
    ENGAGEMENT = "engagement"
    OVERALL_IMPACT = "overall_impact"


class WordTag(str, Enum):
    NORMAL = "normal"
    FILLER = "filler"
    STUTTER = "stutter"
    FALSE_START = "false_start"


@dataclass
class TranscriptWord:
    text: str
    start: float
    end: float
    tag: WordTag = WordTag.NORMAL
    confidence: float = 1.0


@dataclass
class TranscriptSegment:
    words: list[TranscriptWord]
    speaker: str | None = None
    start: float = 0.0
    end: float = 0.0

    def __post_init__(self):
        if self.words and self.start == 0.0 and self.end == 0.0:
            self.start = self.words[0].start
            self.end = self.words[-1].end

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def filler_count(self) -> int:
        return sum(1 for w in self.words if w.tag == WordTag.FILLER)


@dataclass
class Transcript:
    segments: list[TranscriptSegment]
    language: str = "en"

    @property
    def full_text(self) -> str:
        return " ".join(seg.text for seg in self.segments)

    @property
    def duration(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[-1].end - self.segments[0].start

    @property
    def word_count(self) -> int:
        return sum(len(seg.words) for seg in self.segments)

    @property
    def total_fillers(self) -> int:
        return sum(seg.filler_count for seg in self.segments)


@dataclass
class Signal:
    type: SignalType
    source: str
    start: float
    end: float
    value: float | str | None = None
    confidence: float = 1.0
    speaker: str | None = None
    metadata: dict | None = None


@dataclass
class Moment:
    type: MomentType
    start: float
    end: float
    significance: float
    signals: list[Signal] = field(default_factory=list)
    description: str = ""


@dataclass
class DimensionScore:
    dimension: Dimension
    score: float
    level: str = ""
    source: str = ""

    def __post_init__(self):
        if not self.level:
            self.level = score_to_level(self.score)


def score_to_level(score: float) -> str:
    if score >= 0.8:
        return "excellent"
    if score >= 0.6:
        return "good"
    if score >= 0.4:
        return "developing"
    if score >= 0.2:
        return "needs_work"
    return "poor"


@dataclass
class Tile:
    x: int
    y: int
    w: int
    h: int
    participant_id: int = 0


@dataclass
class MediaContext:
    video_path: Path
    workspace: Path
    audio_path: Path | None = None
    frames_dir: Path | None = None
    tiles: list[Tile] = field(default_factory=list)
    duration: float = 0.0
    fps: float = 0.0
    width: int = 0
    height: int = 0
    transcript: Transcript | None = None
    primary_speaker: str | None = None
    speaker_labels: list[str] = field(default_factory=list)


@dataclass
class CoachingReport:
    scores: list[DimensionScore]
    moments: list[Moment]
    narrative: str = ""
    summary: str = ""
    duration: float = 0.0
    analyzers_used: list[str] = field(default_factory=list)
    analyzers_skipped: list[str] = field(default_factory=list)
