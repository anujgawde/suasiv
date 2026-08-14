from __future__ import annotations

from abc import ABC, abstractmethod


class LLMBackend(ABC):
    name: str = ""

    @abstractmethod
    def complete(self, system: str, prompt: str) -> str:
        ...
