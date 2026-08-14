from __future__ import annotations

from suasiv.llm.base import LLMBackend


class GeminiBackend(LLMBackend):
    name = "gemini"

    def complete(self, system: str, prompt: str) -> str:
        raise NotImplementedError("Gemini backend not yet implemented — use stub for now")
