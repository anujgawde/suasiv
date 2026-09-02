from __future__ import annotations

import numpy as np
import librosa
import opensmile

from suasiv.config import FeaturesConfig
from suasiv.schema import MediaContext

SAMPLE_RATE = 16_000


def _chunk_boundaries(duration: float, window: float) -> list[tuple[float, float]]:
    boundaries = []
    t = 0.0
    while t < duration:
        end = min(t + window, duration)
        if end - t >= 1.0:
            boundaries.append((t, end))
        t += window
    return boundaries


def analyze_features(ctx: MediaContext, config: FeaturesConfig) -> None:
    """Extract eGeMAPSv02 acoustic features per segment, store on ctx."""
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet[config.feature_set],
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    audio, _ = librosa.load(str(ctx.audio_path), sr=SAMPLE_RATE, mono=True)
    duration = len(audio) / SAMPLE_RATE
    boundaries = _chunk_boundaries(duration, config.segment_duration)

    results: list[dict] = []

    for start, end in boundaries:
        start_sample = int(start * SAMPLE_RATE)
        end_sample = min(int(end * SAMPLE_RATE), len(audio))
        chunk = audio[start_sample:end_sample]

        if len(chunk) < SAMPLE_RATE:
            continue

        df = smile.process_signal(chunk, SAMPLE_RATE)

        if df.empty:
            continue

        features = df.iloc[0].to_dict()
        features["start"] = start
        features["end"] = end
        results.append(features)

    ctx.acoustic_features = results
