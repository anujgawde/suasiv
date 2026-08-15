from __future__ import annotations

from rich.console import Console

from suasiv.llm.base import LLMBackend

console = Console()


class OllamaBackend(LLMBackend):
    name = "ollama"

    def complete(self, system: str, prompt: str) -> str:
        try:
            import ollama
        except ImportError:
            raise RuntimeError(
                "ollama package required.\n"
                "Install: pip install 'suasiv[full]' or pip install ollama"
            )

        model = self.config.model or "llama3.1:8b"

        try:
            ollama.list()
        except Exception:
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure it's running:\n"
                "  1. Install Ollama: https://ollama.ai\n"
                "  2. Start server: ollama serve\n"
                f"  3. Pull model: ollama pull {model}"
            )

        console.print(f"    Ollama model: [cyan]{model}[/cyan]")
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
