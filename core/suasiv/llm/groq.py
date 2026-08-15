from __future__ import annotations

import os

from rich.console import Console

from suasiv.llm.base import LLMBackend

console = Console()


class GroqBackend(LLMBackend):
    name = "groq"

    def complete(self, system: str, prompt: str) -> str:
        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError(
                "groq package required.\n"
                "Install: pip install 'suasiv[full]' or pip install groq"
            )

        api_key = self.config.api_key or os.environ.get("SUASIV_LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Groq requires an API key.\n"
                "Set api_key in config.yaml or: export SUASIV_LLM_API_KEY=<your-key>"
            )

        model = self.config.model or "llama-3.1-8b-instant"
        console.print(f"    Groq model: [cyan]{model}[/cyan]")

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
