from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from suasiv import pipeline
from suasiv.config import SuasivConfig

app = typer.Typer(name="suasiv", add_completion=False)
console = Console()

DEFAULT_CONFIG = Path("config.yaml")


@app.command()
def analyze(
    video: Path = typer.Argument(..., help="Path to video file", exists=True),
    config: Path = typer.Option(None, "--config", "-c", help="Path to config YAML"),
) -> None:
    """Analyze a video and produce a coaching report."""
    console.print(f"[bold]suasiv[/bold] — {video.name}")

    config_path = config or (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    cfg = SuasivConfig.from_yaml(config_path) if config_path else SuasivConfig()

    pipeline.run(video, cfg)
