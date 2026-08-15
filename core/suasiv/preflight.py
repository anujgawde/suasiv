from __future__ import annotations

import os
import shutil
from dataclasses import dataclass

from suasiv.analyzers import get_all_analyzer_names
from suasiv.config import SuasivConfig


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "warn" | "fail"
    message: str


_ANALYZER_DEPS: dict[str, list[str]] = {
    "transcript": ["faster_whisper"],
    "diarization": ["pyannote.audio", "torch"],
    "prosody": ["numpy", "parselmouth", "librosa"],
    "speaker_facial": ["numpy", "mediapipe", "cv2"],
    "audience_engagement": ["numpy", "mediapipe", "cv2"],
    "audience_reaction": ["numpy", "feat", "torch"],
    "pacing": [],
    "content": [],
    "audience_verbal": [],
}


def run_preflight(config: SuasivConfig) -> list[CheckResult]:
    checks: list[CheckResult] = []
    checks.append(_check_ffmpeg())
    checks.append(_check_llm_backend(config))
    checks.append(_check_disk_space(config))
    checks.extend(_check_analyzer_deps(config))
    return checks


def _check_ffmpeg() -> CheckResult:
    missing = []
    if not shutil.which("ffmpeg"):
        missing.append("ffmpeg")
    if not shutil.which("ffprobe"):
        missing.append("ffprobe")
    if missing:
        return CheckResult(
            "ffmpeg",
            "fail",
            f"{', '.join(missing)} not found. Install: "
            "brew install ffmpeg (macOS) / sudo apt install ffmpeg (Ubuntu) / "
            "choco install ffmpeg (Windows)",
        )
    return CheckResult("ffmpeg", "pass", "ffmpeg and ffprobe found")


def _check_llm_backend(config: SuasivConfig) -> CheckResult:
    name = config.llm.backend

    if name == "stub":
        return CheckResult(
            "llm", "warn", "Using stub backend — narrative will be placeholder text"
        )

    if name == "ollama":
        try:
            import ollama
        except ImportError:
            return CheckResult(
                "llm", "fail", "ollama package not installed: pip install ollama"
            )
        try:
            ollama.list()
            return CheckResult(
                "llm", "pass", f"Ollama reachable, model: {config.llm.model}"
            )
        except Exception:
            return CheckResult(
                "llm",
                "fail",
                f"Cannot connect to Ollama. Start: ollama serve, then: ollama pull {config.llm.model}",
            )

    if name in ("gemini", "groq"):
        api_key = config.llm.api_key or os.environ.get("SUASIV_LLM_API_KEY")
        if not api_key:
            return CheckResult(
                "llm",
                "fail",
                f"{name} requires API key. Set api_key in config.yaml or: "
                "export SUASIV_LLM_API_KEY=...",
            )
        pkg = "google.generativeai" if name == "gemini" else "groq"
        try:
            __import__(pkg)
            return CheckResult(
                "llm", "pass", f"{name} API key set, package available"
            )
        except ImportError:
            return CheckResult("llm", "fail", f"{pkg} package not installed")

    return CheckResult(
        "llm", "warn", f"Unknown backend '{name}' — will fall back to stub"
    )


def _check_disk_space(config: SuasivConfig, min_gb: float = 2.0) -> CheckResult:
    from pathlib import Path

    ws = Path(config.workspace)
    target = ws if ws.exists() else Path(".")
    try:
        usage = shutil.disk_usage(target)
        free_gb = usage.free / (1024**3)
        if free_gb < min_gb:
            return CheckResult(
                "disk",
                "warn",
                f"Low disk space: {free_gb:.1f}GB free (recommend {min_gb:.0f}GB+)",
            )
        return CheckResult("disk", "pass", f"{free_gb:.1f}GB free")
    except Exception:
        return CheckResult("disk", "warn", "Could not check disk space")


def _check_analyzer_deps(config: SuasivConfig) -> list[CheckResult]:
    results: list[CheckResult] = []
    for name in get_all_analyzer_names():
        if not config.analyzer_enabled(name):
            continue
        deps = _ANALYZER_DEPS.get(name, [])
        if not deps:
            continue
        missing = []
        for dep in deps:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        if missing:
            results.append(
                CheckResult(
                    f"analyzer:{name}",
                    "warn",
                    f"Missing: {', '.join(missing)} — will be skipped at runtime",
                )
            )
    return results
