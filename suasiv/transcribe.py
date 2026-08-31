from __future__ import annotations

from pathlib import Path

from faster_whisper import WhisperModel

from suasiv.config import TranscriptionConfig
from suasiv.schema import Transcript, TranscriptSegment, TranscriptWord


def _pick_device(preference: str) -> str:
    if preference != "auto":
        return preference

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def transcribe(audio_path: Path, config: TranscriptionConfig) -> Transcript:
    device = _pick_device(config.device)
    model = WhisperModel(config.model, device=device, compute_type="int8")

    raw_segments, _info = model.transcribe(
        str(audio_path),
        language=config.language or "en",
        word_timestamps=True,
    )

    segments = []
    for seg in raw_segments:
        if not seg.words:
            continue

        words = []
        for w in seg.words:
            text = w.word.strip()
            if text:
                words.append(TranscriptWord(
                    text=text,
                    start=w.start,
                    end=w.end,
                    confidence=w.probability,
                ))

        if words:
            segments.append(TranscriptSegment(words=words))

    return Transcript(segments=segments)
