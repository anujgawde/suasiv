from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


PRESETS: dict[str, dict[str, bool]] = {
    "full": {
        "transcript": True,
        "diarization": True,
        "pacing": True,
        "prosody": True,
        "content": True,
        "speaker_facial": True,
        "audience_engagement": True,
        "audience_reaction": True,
        "audience_verbal": True,
    },
    "standard": {
        "transcript": True,
        "diarization": True,
        "pacing": True,
        "prosody": True,
        "content": True,
        "speaker_facial": True,
        "audience_engagement": True,
        "audience_reaction": False,
        "audience_verbal": True,
    },
    "lite": {
        "transcript": True,
        "diarization": False,
        "pacing": True,
        "prosody": False,
        "content": True,
        "speaker_facial": False,
        "audience_engagement": False,
        "audience_reaction": False,
        "audience_verbal": True,
    },
}


class AnalyzerConfig(BaseModel):
    enabled: bool = True
    model_size: str | None = None
    fps: int | None = None
    window_seconds: int | None = None


class LLMConfig(BaseModel):
    backend: str = "ollama"
    model: str = "llama3.1:8b"
    api_key: str | None = None


class ReportConfig(BaseModel):
    format: str = "markdown"


class IngestConfig(BaseModel):
    frame_fps: int = 3
    audio_sample_rate: int = 16000


class FusionConfig(BaseModel):
    window_seconds: int = 5
    chunk_seconds: int = 600
    chunk_overlap: int = 30


class SuasivConfig(BaseModel):
    preset: str | None = None
    workspace: str = ".workspace"
    analyzers: dict[str, AnalyzerConfig] = Field(default_factory=dict)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> SuasivConfig:
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        analyzers = {}
        for name, cfg in raw.get("analyzers", {}).items():
            analyzers[name] = AnalyzerConfig(**(cfg if isinstance(cfg, dict) else {}))
        raw["analyzers"] = analyzers
        return cls(**raw)

    def analyzer_enabled(self, name: str) -> bool:
        cfg = self.analyzers.get(name)
        if cfg is not None:
            return cfg.enabled
        if self.preset and self.preset in PRESETS:
            return PRESETS[self.preset].get(name, True)
        return True

    def analyzer_settings(self, name: str) -> AnalyzerConfig:
        return self.analyzers.get(name, AnalyzerConfig())
