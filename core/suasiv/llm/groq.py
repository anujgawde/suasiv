from __future__ import annotations

from suasiv.llm.base import LLMBackend


class GroqBackend(LLMBackend):
    name = "groq"

    def complete(self, system: str, prompt: str) -> str:
        raise NotImplementedError("Groq backend not yet implemented — use stub for now")
