from suasiv.config import LLMConfig
from suasiv.llm.base import LLMBackend
from suasiv.llm.gemini import GeminiBackend
from suasiv.llm.groq import GroqBackend
from suasiv.llm.ollama import OllamaBackend
from suasiv.llm.stub import StubBackend

BACKENDS: dict[str, type[LLMBackend]] = {
    "stub": StubBackend,
    "ollama": OllamaBackend,
    "gemini": GeminiBackend,
    "groq": GroqBackend,
}


def get_backend(name: str, config: LLMConfig) -> LLMBackend:
    cls = BACKENDS.get(name)
    if cls is None:
        from rich.console import Console

        Console().print(
            f"[yellow]Warning: unknown LLM backend '{name}', falling back to stub[/yellow]"
        )
        cls = StubBackend
    return cls(config)
