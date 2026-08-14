from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


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


class SuasivConfig(BaseModel):
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
        if cfg is None:
            return True
        return cfg.enabled

    def analyzer_settings(self, name: str) -> AnalyzerConfig:
        return self.analyzers.get(name, AnalyzerConfig())
