from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def build_env() -> Environment:
    env = Environment(
        loader=PackageLoader("suasiv.report", "templates"),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["fmt_time"] = fmt_time
    return env


def render_markdown(
    video_name: str,
    duration: float,
    scores: list,
    moments: list,
    narrative: str,
    output_path: Path,
) -> Path:
    env = build_env()
    template = env.get_template("report.md.j2")
    text = template.render(
        video_name=video_name,
        duration=duration,
        scores=scores,
        moments=moments,
        narrative=narrative,
    )
    output_path.write_text(text)
    return output_path
