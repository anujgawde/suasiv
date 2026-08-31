from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(name="suasiv", add_completion=False)
console = Console()


@app.command()
def analyze(
    video: Path = typer.Argument(..., help="Path to video file", exists=True),
    config: Path = typer.Option(None, "--config", "-c", help="Path to config YAML"),
) -> None:
    """Analyze a video and produce a coaching report."""
    console.print(f"[bold]suasiv[/bold] — {video.name}")
    console.print("not implemented")
    raise typer.Exit()
