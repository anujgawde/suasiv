from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from suasiv.schema import CoachingReport


def render_report(report: CoachingReport, template_name: str, output_path: Path) -> Path:
    templates_dir = Path(__file__).parent.parent / "templates"
    env = Environment(loader=FileSystemLoader(str(templates_dir)))
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
