from __future__ import annotations

from suasiv.llm.base import LLMBackend


class OllamaBackend(LLMBackend):
    name = "ollama"

    def complete(self, system: str, prompt: str) -> str:
        raise NotImplementedError("Ollama backend not yet implemented — use stub for now")
