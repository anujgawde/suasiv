from __future__ import annotations

import json
import os

from rich.console import Console

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal

console = Console()


class DiarizationAnalyzer(Analyzer):
    name = "diarization"
    requires = {"audio", "transcript"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise RuntimeError(
                "pyannote.audio is required for speaker diarization.\n"
                "Install: pip install 'suasiv[full]' or pip install pyannote.audio"
            )

        if ctx.audio_path is None:
            return AnalyzerResult(
                analyzer=self.name, signals=[], summary={"error": "no audio track"}
            )

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "Speaker diarization requires a HuggingFace access token.\n"
                "1. Create an account at https://huggingface.co\n"
                "2. Accept terms: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "3. Accept terms: https://huggingface.co/pyannote/segmentation-3.0\n"
                "4. Create a token: https://huggingface.co/settings/tokens\n"
                "5. export HF_TOKEN=hf_..."
            )

        console.print("    Loading diarization model...")
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

        device = _detect_device()
        if device != "cpu":
            import torch

            pipeline.to(torch.device(device))
        console.print(f"    Device: [cyan]{device}[/cyan]")

        console.print("    Running speaker diarization...")
        diarization = pipeline(str(ctx.audio_path))

        speaker_segments: list[dict] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append(
                {
                    "speaker": speaker,
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                }
            )

        # Assign speakers to transcript segments by maximum overlap
        if ctx.transcript:
            for seg in ctx.transcript:
                seg["speaker"] = _assign_speaker(seg["start"], seg["end"], speaker_segments)
            transcript_path = ctx.workspace / "transcript.json"
            with open(transcript_path, "w") as f:
                json.dump(ctx.transcript, f, indent=2)

        # Primary speaker = most talk time
        talk_times: dict[str, float] = {}
        for s in speaker_segments:
            talk_times[s["speaker"]] = talk_times.get(s["speaker"], 0) + (s["end"] - s["start"])

        primary = max(talk_times, key=talk_times.get) if talk_times else None
        total_time = sum(talk_times.values())

        ctx.speaker_labels = {
            spk: "primary" if spk == primary else "audience" for spk in talk_times
        }
        ctx.primary_speaker = primary

        # Build signals
        signals: list[Signal] = []
        prev_speaker = None
        for s in speaker_segments:
            signals.append(
                Signal(
                    analyzer=self.name,
                    type="speaker_segment",
                    start=s["start"],
                    end=s["end"],
                    value=s["speaker"],
                    speaker=s["speaker"],
                )
            )
            if prev_speaker is not None and s["speaker"] != prev_speaker:
                signals.append(
                    Signal(
                        analyzer=self.name,
                        type="speaker_change",
                        start=s["start"],
                        end=s["start"],
                        value=f"{prev_speaker}->{s['speaker']}",
                        speaker=s["speaker"],
                    )
                )
            prev_speaker = s["speaker"]

        diarization_path = ctx.workspace / "diarization.json"
        with open(diarization_path, "w") as f:
            json.dump(
                {
                    "segments": speaker_segments,
                    "speaker_labels": ctx.speaker_labels,
                    "primary_speaker": primary,
                    "talk_times": {k: round(v, 1) for k, v in talk_times.items()},
                },
                f,
                indent=2,
            )

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "speaker_count": len(talk_times),
                "primary_speaker": primary,
                "primary_talk_ratio": (
                    round(talk_times.get(primary, 0) / total_time, 3) if total_time else 0
                ),
            },
        )


def _assign_speaker(
    start: float, end: float, speaker_segments: list[dict]
) -> str | None:
    best_speaker = None
    best_overlap = 0.0
    for s in speaker_segments:
        overlap = max(0.0, min(end, s["end"]) - max(start, s["start"]))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = s["speaker"]
    return best_speaker


def _detect_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"
