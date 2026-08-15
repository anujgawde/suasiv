from __future__ import annotations

import json

from rich.console import Console

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal

console = Console()


class ProsodyAnalyzer(Analyzer):
    name = "prosody"
    requires = {"audio"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        import numpy as np

        try:
            import parselmouth
        except ImportError:
            raise RuntimeError(
                "parselmouth is required for prosody analysis.\n"
                "Install: pip install 'suasiv[full]' or pip install parselmouth"
            )
        try:
            import librosa
        except ImportError:
            raise RuntimeError(
                "librosa is required for prosody analysis.\n"
                "Install: pip install 'suasiv[full]' or pip install librosa"
            )

        if ctx.audio_path is None:
            return AnalyzerResult(
                analyzer=self.name, signals=[], summary={"error": "no audio track"}
            )

        settings = ctx.config.analyzer_settings(self.name)
        window_sec = settings.window_seconds or 5

        console.print("    Loading audio for prosody analysis...")
        snd = parselmouth.Sound(str(ctx.audio_path))
        y, sr = librosa.load(str(ctx.audio_path), sr=None)

        speaker_segments = _get_speaker_segments(ctx)

        console.print("    Extracting pitch (F0)...")
        pitch_obj = snd.to_pitch(time_step=0.01)
        pitch_times = pitch_obj.xs()
        pitch_values = pitch_obj.selected_array["frequency"].flatten().copy()
        pitch_values[pitch_values == 0] = np.nan

        console.print("    Extracting energy envelope...")
        hop_length = 512
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
        energy_times = librosa.frames_to_time(
            np.arange(len(rms)), sr=sr, hop_length=hop_length
        )

        if speaker_segments:
            pitch_mask = _time_in_segments(pitch_times, speaker_segments)
            energy_mask = _time_in_segments(energy_times, speaker_segments)
            speaker_pitch = np.where(pitch_mask, pitch_values, np.nan)
            speaker_energy = np.where(energy_mask, rms, np.nan)
        else:
            speaker_pitch = pitch_values
            speaker_energy = rms

        valid_pitch = speaker_pitch[~np.isnan(speaker_pitch)]
        valid_energy = speaker_energy[~np.isnan(speaker_energy)]

        mean_pitch = float(np.mean(valid_pitch)) if len(valid_pitch) > 0 else 0.0
        pitch_std = float(np.std(valid_pitch)) if len(valid_pitch) > 0 else 0.0
        pitch_range = float(np.ptp(valid_pitch)) if len(valid_pitch) > 0 else 0.0
        mean_energy = float(np.mean(valid_energy)) if len(valid_energy) > 0 else 0.0
        energy_range = float(np.ptp(valid_energy)) if len(valid_energy) > 0 else 0.0

        vocal_variety = pitch_std / mean_pitch if mean_pitch > 0 else 0.0

        console.print(f"    Analyzing in {window_sec}s windows...")
        signals: list[Signal] = []
        duration = float(snd.duration)
        monotone_count = 0
        step = window_sec / 2

        t = 0.0
        while t < duration:
            w_end = min(t + window_sec, duration)

            w_pitch_mask = (pitch_times >= t) & (pitch_times < w_end)
            w_pitch = speaker_pitch[w_pitch_mask]
            w_valid_pitch = w_pitch[~np.isnan(w_pitch)]

            w_energy_mask = (energy_times >= t) & (energy_times < w_end)
            w_energy = speaker_energy[w_energy_mask]
            w_valid_energy = w_energy[~np.isnan(w_energy)]

            if len(w_valid_pitch) > 5:
                w_pitch_std = float(np.std(w_valid_pitch))
                w_mean_pitch = float(np.mean(w_valid_pitch))
                if w_mean_pitch > 0 and w_pitch_std / w_mean_pitch < 0.05:
                    signals.append(
                        Signal(
                            analyzer=self.name,
                            type="monotone_segment",
                            start=t,
                            end=w_end,
                            value=round(w_pitch_std, 2),
                            confidence=round(
                                min(1.0, 0.05 / (w_pitch_std / w_mean_pitch + 0.001)), 2
                            ),
                            speaker=ctx.primary_speaker,
                        )
                    )
                    monotone_count += 1

                p_ratio = w_mean_pitch / mean_pitch if mean_pitch > 0 else 1.0
                if p_ratio < 0.7:
                    signals.append(
                        Signal(
                            analyzer=self.name,
                            type="pitch_drop",
                            start=t,
                            end=w_end,
                            value=round(p_ratio, 3),
                            confidence=round(min(1.0, 1.0 - p_ratio), 2),
                            speaker=ctx.primary_speaker,
                        )
                    )
                elif p_ratio > 1.3:
                    signals.append(
                        Signal(
                            analyzer=self.name,
                            type="pitch_spike",
                            start=t,
                            end=w_end,
                            value=round(p_ratio, 3),
                            confidence=round(min(1.0, p_ratio - 1.0), 2),
                            speaker=ctx.primary_speaker,
                        )
                    )

            if len(w_valid_energy) > 5 and mean_energy > 0:
                w_mean_e = float(np.mean(w_valid_energy))
                e_ratio = w_mean_e / mean_energy
                if e_ratio < 0.5:
                    signals.append(
                        Signal(
                            analyzer=self.name,
                            type="energy_drop",
                            start=t,
                            end=w_end,
                            value=round(e_ratio, 3),
                            confidence=round(min(1.0, 1.0 - e_ratio), 2),
                            speaker=ctx.primary_speaker,
                        )
                    )
                elif e_ratio > 1.5:
                    signals.append(
                        Signal(
                            analyzer=self.name,
                            type="energy_spike",
                            start=t,
                            end=w_end,
                            value=round(e_ratio, 3),
                            confidence=round(min(1.0, e_ratio - 1.0), 2),
                            speaker=ctx.primary_speaker,
                        )
                    )
                elif e_ratio > 1.3:
                    signals.append(
                        Signal(
                            analyzer=self.name,
                            type="high_energy_segment",
                            start=t,
                            end=w_end,
                            value=round(e_ratio, 3),
                            confidence=0.7,
                            speaker=ctx.primary_speaker,
                        )
                    )

            t += step

        prosody_path = ctx.workspace / "prosody.json"
        with open(prosody_path, "w") as f:
            json.dump(
                {
                    "mean_pitch_hz": round(mean_pitch, 1),
                    "pitch_std_hz": round(pitch_std, 1),
                    "pitch_range_hz": round(pitch_range, 1),
                    "mean_energy": round(mean_energy, 6),
                    "energy_range": round(energy_range, 6),
                    "vocal_variety_score": round(vocal_variety, 3),
                    "monotone_segments": monotone_count,
                },
                f,
                indent=2,
            )

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "mean_pitch_hz": round(mean_pitch, 1),
                "pitch_range_hz": round(pitch_range, 1),
                "energy_range": round(energy_range, 6),
                "vocal_variety_score": round(vocal_variety, 3),
                "monotone_segments": monotone_count,
            },
        )


def _get_speaker_segments(ctx: MediaContext) -> list[tuple[float, float]]:
    if not ctx.primary_speaker or not ctx.transcript:
        return []
    segments = []
    for seg in ctx.transcript:
        if seg.get("speaker") == ctx.primary_speaker:
            segments.append((seg["start"], seg["end"]))
    return segments


def _time_in_segments(times, segments: list[tuple[float, float]]):
    import numpy as np

    mask = np.zeros(len(times), dtype=bool)
    for start, end in segments:
        mask |= (times >= start) & (times <= end)
    return mask
