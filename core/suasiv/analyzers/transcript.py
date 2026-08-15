from __future__ import annotations

import json

from rich.console import Console

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal

console = Console()


class TranscriptAnalyzer(Analyzer):
    name = "transcript"
    requires = {"audio"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError(
                "faster-whisper is required for transcription.\n"
                "Install: pip install 'suasiv[full]' or pip install faster-whisper"
            )

        if ctx.audio_path is None:
            return AnalyzerResult(
                analyzer=self.name, signals=[], summary={"error": "no audio track"}
            )

        settings = ctx.config.analyzer_settings(self.name)
        model_size = settings.model_size or "base"
        device, compute_type = _detect_device()

        console.print(
            f"    Loading whisper [cyan]{model_size}[/cyan] ({device}/{compute_type})"
        )
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

        console.print("    Transcribing...")
        segments_iter, info = model.transcribe(
            str(ctx.audio_path),
            word_timestamps=True,
            vad_filter=True,
        )

        transcript: list[dict] = []
        signals: list[Signal] = []

        for segment in segments_iter:
            words = []
            if segment.words:
                words = [
                    {
                        "word": w.word,
                        "start": round(w.start, 3),
                        "end": round(w.end, 3),
                        "probability": round(w.probability, 3),
                    }
                    for w in segment.words
                ]

            transcript.append(
                {
                    "text": segment.text.strip(),
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "words": words,
                    "speaker": None,
                }
            )

            signals.append(
                Signal(
                    analyzer=self.name,
                    type="transcript_segment",
                    start=segment.start,
                    end=segment.end,
                    value=segment.text.strip(),
                )
            )

        ctx.transcript = transcript

        transcript_path = ctx.workspace / "transcript.json"
        with open(transcript_path, "w") as f:
            json.dump(transcript, f, indent=2)

        total_words = sum(len(seg["words"]) for seg in transcript)

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "segment_count": len(transcript),
                "total_words": total_words,
                "language": info.language,
                "language_probability": round(info.language_probability, 3),
                "duration": round(info.duration, 1),
            },
        )


def _detect_device() -> tuple[str, str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"
