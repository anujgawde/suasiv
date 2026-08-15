from __future__ import annotations

import os

from rich.console import Console

from suasiv.llm.base import LLMBackend

console = Console()


class GeminiBackend(LLMBackend):
    name = "gemini"

    def complete(self, system: str, prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError(
                "google-generativeai package required.\n"
                "Install: pip install 'suasiv[full]' or pip install google-generativeai"
            )

        api_key = self.config.api_key or os.environ.get("SUASIV_LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini requires an API key.\n"
                "Set api_key in config.yaml or: export SUASIV_LLM_API_KEY=<your-key>"
            )

        model_name = self.config.model or "gemini-1.5-flash"
        console.print(f"    Gemini model: [cyan]{model_name}[/cyan]")

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name, system_instruction=system)
        response = model.generate_content(prompt)
        return response.text
