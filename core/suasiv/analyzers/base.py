from __future__ import annotations

from abc import ABC, abstractmethod

from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult


class Analyzer(ABC):
    name: str = ""
    requires: set[str] = set()

    @abstractmethod
    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        ...
