from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from suasiv import __version__
from suasiv.config import PRESETS, SuasivConfig
from suasiv.pipeline import run_pipeline

app = typer.Typer(
    name="suasiv",
    help="Communication coaching — analyzes speaker delivery and audience reception.",
    no_args_is_help=True,
)

console = Console()

_STATUS_STYLE = {"pass": "green", "warn": "yellow", "fail": "red"}
_STATUS_ICON = {"pass": "✓", "warn": "!", "fail": "✗"}


@app.command()
def analyze(
    video: Path = typer.Argument(..., help="Path to video file"),
    config: Path = typer.Option(
        "config.yaml", "--config", "-c", help="Path to config YAML"
    ),
    preset: str | None = typer.Option(
        None, "--preset", "-p", help="Analyzer preset: full, standard, lite"
    ),
    skip_checks: bool = typer.Option(
        False, "--skip-checks", help="Skip pre-flight checks"
    ),
) -> None:
    """Analyze a video and generate a coaching report."""
    if not video.exists():
        typer.echo(f"Error: video file not found: {video}", err=True)
        raise typer.Exit(1)

    cfg = SuasivConfig.from_yaml(config)

    if preset:
        if preset not in PRESETS:
            typer.echo(
                f"Error: unknown preset '{preset}'. Choose: full, standard, lite",
                err=True,
            )
            raise typer.Exit(1)
        cfg.preset = preset

    if not skip_checks:
        from suasiv.preflight import run_preflight

        checks = run_preflight(cfg)
        _display_checks(checks)
        if any(c.status == "fail" for c in checks):
            console.print(
                "\n[red]Pre-flight checks failed.[/red] "
                "Fix the issues above or use --skip-checks."
            )
            raise typer.Exit(1)
        console.print()

    run_pipeline(str(video), cfg)


def _display_checks(checks) -> None:
    console.print("[bold]Pre-flight checks[/bold]")
    for c in checks:
        style = _STATUS_STYLE.get(c.status, "white")
        icon = _STATUS_ICON.get(c.status, "?")
        console.print(f"  [{style}]{icon}[/{style}] {c.name}: {c.message}")


@app.command()
def version() -> None:
    """Print version and exit."""
    typer.echo(f"suasiv {__version__}")
