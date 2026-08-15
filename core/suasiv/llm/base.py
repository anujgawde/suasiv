from __future__ import annotations

from abc import ABC, abstractmethod

from suasiv.config import LLMConfig


class LLMBackend(ABC):
    name: str = ""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    @abstractmethod
    def complete(self, system: str, prompt: str) -> str: ...
