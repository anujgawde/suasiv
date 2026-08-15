from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from suasiv.analyzers.base import Analyzer

_ANALYZER_REGISTRY: dict[str, tuple[str, str]] = {
    "transcript": ("suasiv.analyzers.transcript", "TranscriptAnalyzer"),
    "diarization": ("suasiv.analyzers.diarization", "DiarizationAnalyzer"),
    "pacing": ("suasiv.analyzers.pacing", "PacingAnalyzer"),
    "prosody": ("suasiv.analyzers.prosody", "ProsodyAnalyzer"),
    "content": ("suasiv.analyzers.content", "ContentAnalyzer"),
    "speaker_facial": ("suasiv.analyzers.speaker_facial", "SpeakerFacialAnalyzer"),
    "audience_engagement": ("suasiv.analyzers.audience_engagement", "AudienceEngagementAnalyzer"),
    "audience_reaction": ("suasiv.analyzers.audience_reaction", "AudienceReactionAnalyzer"),
    "audience_verbal": ("suasiv.analyzers.audience_verbal", "AudienceVerbalAnalyzer"),
}


def get_analyzer_class(name: str) -> type[Analyzer]:
    if name not in _ANALYZER_REGISTRY:
        raise KeyError(f"Unknown analyzer: {name}")
    module_path, class_name = _ANALYZER_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_all_analyzer_names() -> list[str]:
    return list(_ANALYZER_REGISTRY.keys())
