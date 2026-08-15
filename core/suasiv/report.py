from __future__ import annotations

import re
from html import escape
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from suasiv.schema import CoachingReport


def render_report(
    report: CoachingReport, template_name: str, output_path: Path
) -> Path:
    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)), autoescape=False)
    env.filters["fmt_time"] = _fmt_time
    env.filters["score_level"] = _score_level
    env.filters["md"] = _md_to_html

    template = env.get_template(template_name)

    rendered = template.render(
        report=report,
        scores=report.summary_scores,
        narrative=report.narrative,
        timeline=report.timeline,
        moments=report.timeline.moments,
        signals=report.timeline.signals,
        metadata=report.metadata,
    )

    output_path.write_text(rendered)
    return output_path


def _fmt_time(seconds: float) -> str:
    s = max(0, int(seconds))
    return f"{s // 60}:{s % 60:02d}"


def _score_level(score: float) -> str:
    if score >= 0.7:
        return "Good"
    if score >= 0.4:
        return "Fair"
    return "Needs work"


def _md_to_html(text: str) -> str:
    lines = text.split("\n")
    result: list[str] = []
    list_type: str | None = None

    def close_list():
        nonlocal list_type
        if list_type:
            result.append(f"</{list_type}>")
            list_type = None

    for line in lines:
        s = line.strip()

        if s.startswith("## "):
            close_list()
            result.append(f"<h3>{_inline(s[3:])}</h3>")
        elif s.startswith("### "):
            close_list()
            result.append(f"<h4>{_inline(s[4:])}</h4>")
        elif s.startswith("- ") or s.startswith("* "):
            if list_type != "ul":
                close_list()
                result.append("<ul>")
                list_type = "ul"
            result.append(f"<li>{_inline(s[2:])}</li>")
        elif re.match(r"^\d+\.\s", s):
            if list_type != "ol":
                close_list()
                result.append("<ol>")
                list_type = "ol"
            content = re.sub(r"^\d+\.\s*", "", s)
            result.append(f"<li>{_inline(content)}</li>")
        elif s == "":
            close_list()
        else:
            close_list()
            result.append(f"<p>{_inline(s)}</p>")

    close_list()
    return "\n".join(result)


def _inline(text: str) -> str:
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\[(\d+:\d+)\s*-\s*(\d+:\d+)\]", r"<span class='ts'>[\1–\2]</span>", text)
    return text
