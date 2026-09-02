from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class IngestConfig(BaseModel):
    fps: float = 3.0
    tile_detection: bool = True
    min_tile_area: int = 5000


class TranscriptionConfig(BaseModel):
    model: str = "small"
    device: str = "auto"
    batch_size: int = 16
    language: str | None = None


class DiarizationConfig(BaseModel):
    model: str = "pyannote/speaker-diarization-3.1"
    hf_token: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


class EmotionConfig(BaseModel):
    model: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    segment_duration: float = 5.0


class FeaturesConfig(BaseModel):
    feature_set: str = "eGeMAPSv02"
    segment_duration: float = 30.0


class PacingConfig(BaseModel):
    window_seconds: float = 15.0
    hop_seconds: float = 5.0
    min_voiced_seconds: float = 4.0
    rush_wpm: float = 190.0
    drag_wpm: float = 105.0
    baseline_swing_wpm: float = 40.0
    min_pause_seconds: float = 1.5
    filler_pause_seconds: float = 0.2
    repeat_gap_seconds: float = 0.6
    false_start_break_seconds: float = 0.25
    false_start_lookahead: int = 5


class QualityConfig(BaseModel):
    model_dir: str = "training/models"


class SpeakerVisualConfig(BaseModel):
    enabled: bool = True
    min_face_confidence: float = 0.5


class AudienceVisualConfig(BaseModel):
    enabled: bool = True
    attention_threshold: float = 0.5
    min_face_confidence: float = 0.5


class ContentConfig(BaseModel):
    enabled: bool = True


class FusionConfig(BaseModel):
    window_seconds: float = 5.0
    max_moments: int = 8
    min_significance: float = 0.3


class LLMConfig(BaseModel):
    backend: str = "ollama"
    model: str = "llama3.1:8b"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 4096


class ReportConfig(BaseModel):
    format: str = "markdown"
    output_dir: str | None = None


class SuasivConfig(BaseModel):
    workspace: str = ".workspace"

    ingest: IngestConfig = Field(default_factory=IngestConfig)
    transcription: TranscriptionConfig = Field(default_factory=TranscriptionConfig)
    diarization: DiarizationConfig = Field(default_factory=DiarizationConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    pacing: PacingConfig = Field(default_factory=PacingConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    speaker_visual: SpeakerVisualConfig = Field(default_factory=SpeakerVisualConfig)
    audience_visual: AudienceVisualConfig = Field(default_factory=AudienceVisualConfig)
    content: ContentConfig = Field(default_factory=ContentConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SuasivConfig:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @property
    def workspace_path(self) -> Path:
        return Path(self.workspace)
