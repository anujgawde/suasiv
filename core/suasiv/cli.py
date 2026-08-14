from __future__ import annotations

from pathlib import Path

import typer

from suasiv import __version__
from suasiv.config import SuasivConfig
from suasiv.pipeline import run_pipeline

app = typer.Typer(
    name="suasiv",
    help="Communication coaching — analyzes speaker delivery and audience reception.",
    no_args_is_help=True,
)


@app.command()
def analyze(
    video: Path = typer.Argument(..., help="Path to video file"),
    config: Path = typer.Option("config.yaml", "--config", "-c", help="Path to config YAML"),
) -> None:
    """Analyze a video and generate a coaching report."""
    if not video.exists():
        typer.echo(f"Error: video file not found: {video}", err=True)
        raise typer.Exit(1)

    cfg = SuasivConfig.from_yaml(config)
    run_pipeline(str(video), cfg)


@app.command()
def version() -> None:
    """Print version and exit."""
    typer.echo(f"suasiv {__version__}")
