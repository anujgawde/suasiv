from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model

from suasiv.config import EmotionConfig
from suasiv.schema import MediaContext, Signal, SignalType

SAMPLE_RATE = 16_000

DOMINANCE_HIGH = 0.55
DOMINANCE_LOW = 0.35
AROUSAL_HIGH = 0.55
AROUSAL_LOW = 0.30
VALENCE_HIGH = 0.55
VALENCE_LOW = 0.35


class _RegressionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _EmotionModel(Wav2Vec2Model):
    """audeering wav2vec2 with regression head for arousal/dominance/valence."""

    def __init__(self, config):
        super().__init__(config)
        self.classifier = _RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = super().forward(input_values)
        hidden_states = torch.mean(outputs[0], dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "cpu"  # mps has issues with wav2vec2
    return "cpu"


def _load_audio(path: Path) -> np.ndarray:
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio


def _chunk_boundaries(duration: float, window: float) -> list[tuple[float, float]]:
    """Fixed-duration windows across the full audio."""
    boundaries = []
    t = 0.0
    while t < duration:
        end = min(t + window, duration)
        if end - t >= 0.5:
            boundaries.append((t, end))
        t += window
    return boundaries


def analyze_emotion(ctx: MediaContext, config: EmotionConfig) -> list[Signal]:
    """Run audeering wav2vec2 on audio, emit dominance/arousal/valence signals."""
    device = _pick_device()

    processor = Wav2Vec2Processor.from_pretrained(config.model)
    model = _EmotionModel.from_pretrained(config.model)
    model.to(device)
    model.eval()

    audio = _load_audio(ctx.audio_path)
    duration = len(audio) / SAMPLE_RATE
    boundaries = _chunk_boundaries(duration, config.segment_duration)

    signals: list[Signal] = []

    for start, end in boundaries:
        start_sample = int(start * SAMPLE_RATE)
        end_sample = min(int(end * SAMPLE_RATE), len(audio))
        chunk = audio[start_sample:end_sample]

        inputs = processor(
            chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
        )
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            _, logits = model(input_values)

        scores = logits.squeeze().cpu().numpy()
        arousal = float(scores[0])
        dominance = float(scores[1])
        valence = float(scores[2])

        meta = {"arousal": arousal, "dominance": dominance, "valence": valence}

        if dominance >= DOMINANCE_HIGH:
            signals.append(Signal(
                type=SignalType.CONFIDENT_DELIVERY, source="emotion",
                start=start, end=end, value=dominance, metadata=meta,
            ))
        elif dominance <= DOMINANCE_LOW:
            signals.append(Signal(
                type=SignalType.LOW_CONFIDENCE, source="emotion",
                start=start, end=end, value=dominance, metadata=meta,
            ))

        if arousal >= AROUSAL_HIGH:
            signals.append(Signal(
                type=SignalType.HIGH_ENERGY, source="emotion",
                start=start, end=end, value=arousal, metadata=meta,
            ))
        elif arousal <= AROUSAL_LOW:
            signals.append(Signal(
                type=SignalType.ENERGY_DROP, source="emotion",
                start=start, end=end, value=arousal, metadata=meta,
            ))

        if valence >= VALENCE_HIGH:
            signals.append(Signal(
                type=SignalType.POSITIVE_TONE, source="emotion",
                start=start, end=end, value=valence, metadata=meta,
            ))
        elif valence <= VALENCE_LOW:
            signals.append(Signal(
                type=SignalType.NEGATIVE_TONE, source="emotion",
                start=start, end=end, value=valence, metadata=meta,
            ))

    return signals
