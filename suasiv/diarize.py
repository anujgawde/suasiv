from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

from suasiv.config import DiarizationConfig
from suasiv.schema import MediaContext, Transcript


def _get_hf_token(config: DiarizationConfig) -> str:
    token = config.hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "pyannote requires a HuggingFace token. "
            "Set HF_TOKEN env var or hf_token in config.yaml. "
            "Accept the model terms at https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
    return token


def diarize(ctx: MediaContext, config: DiarizationConfig) -> MediaContext:
    token = _get_hf_token(config)

    from pyannote.audio import Pipeline as DiarizationPipeline

    pipeline = DiarizationPipeline.from_pretrained(config.model, use_auth_token=token)

    params = {}
    if config.min_speakers is not None:
        params["min_speakers"] = config.min_speakers
    if config.max_speakers is not None:
        params["max_speakers"] = config.max_speakers

    diarization = pipeline(str(ctx.audio_path), **params)

    speaker_talk_time: dict[str, float] = defaultdict(float)
    speaker_turns: list[tuple[float, float, str]] = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_turns.append((turn.start, turn.end, speaker))
        speaker_talk_time[speaker] += turn.end - turn.start

    ctx.speaker_labels = sorted(speaker_talk_time.keys())
    ctx.primary_speaker = max(speaker_talk_time, key=speaker_talk_time.get) if speaker_talk_time else None

    if ctx.transcript:
        _assign_speakers(ctx.transcript, speaker_turns)

    return ctx


def _assign_speakers(transcript: Transcript, turns: list[tuple[float, float, str]]) -> None:
    for segment in transcript.segments:
        seg_mid = (segment.start + segment.end) / 2
        best_speaker = None
        best_overlap = 0.0

        for turn_start, turn_end, speaker in turns:
            overlap_start = max(segment.start, turn_start)
            overlap_end = min(segment.end, turn_end)
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker

        segment.speaker = best_speaker
