from __future__ import annotations

import re
import statistics

from suasiv.config import PacingConfig
from suasiv.schema import (
    MediaContext,
    Signal,
    SignalType,
    TranscriptWord,
    WordTag,
)

# faster-whisper emits untagged words, so disfluencies are detected here by
# lexicon and timing rather than read off the transcript.
FILLERS = frozenset({
    "um", "uhm", "uh", "erm", "er", "ah", "eh", "hmm", "hm", "mm", "mhm",
})

MULTIWORD_FILLERS = (
    ("you", "know"),
    ("i", "mean"),
)

# "like" is only a filler when it sits alone between pauses — otherwise it is a
# verb or a comparison and tagging it would poison the filler rate.
PAUSE_BOUNDED_FILLERS = frozenset({"like", "so", "right"})

# Words that plausibly open a clause, used to anchor false-start detection.
CLAUSE_STARTERS = frozenset({
    "i", "we", "you", "they", "he", "she", "it", "the", "a", "an", "and",
    "but", "so", "this", "that", "there", "what", "how", "why", "let",
    "if", "when", "my", "our", "your", "one", "now",
})

_EDGE_PUNCT = re.compile(r"^[^\w']+|[^\w']+$")


def _normalize(text: str) -> str:
    return _EDGE_PUNCT.sub("", text.lower())


def _is_fragment(word: TranscriptWord) -> bool:
    return word.text.rstrip().endswith("-")


def _speaker_words(ctx: MediaContext, speaker: str | None) -> list[TranscriptWord]:
    """Flat, time-ordered words for one speaker (all words if unattributed)."""
    segments = ctx.transcript.segments

    if speaker is not None:
        owned = [seg for seg in segments if seg.speaker == speaker]
        if owned:
            segments = owned

    words = [w for seg in segments for w in seg.words]
    words.sort(key=lambda w: w.start)
    return words


def _tag_fillers(words: list[TranscriptWord], config: PacingConfig) -> None:
    norms = [_normalize(w.text) for w in words]

    i = 0
    while i < len(words):
        if norms[i] in FILLERS:
            words[i].tag = WordTag.FILLER
            i += 1
            continue

        matched = False
        for phrase in MULTIWORD_FILLERS:
            end = i + len(phrase)
            if end <= len(words) and tuple(norms[i:end]) == phrase:
                for w in words[i:end]:
                    w.tag = WordTag.FILLER
                i = end
                matched = True
                break
        if matched:
            continue

        if norms[i] in PAUSE_BOUNDED_FILLERS and _is_pause_bounded(words, i, config):
            words[i].tag = WordTag.FILLER

        i += 1


def _is_pause_bounded(
    words: list[TranscriptWord], i: int, config: PacingConfig
) -> bool:
    gap = config.filler_pause_seconds

    before = words[i].start - words[i - 1].end if i > 0 else gap
    after = words[i + 1].start - words[i].end if i + 1 < len(words) else gap

    return before >= gap and after >= gap


def _tag_stutters(words: list[TranscriptWord], config: PacingConfig) -> None:
    norms = [_normalize(w.text) for w in words]

    for i in range(len(words) - 1):
        if words[i].tag != WordTag.NORMAL:
            continue

        gap = words[i + 1].start - words[i].end
        if gap > config.repeat_gap_seconds:
            continue

        current, following = norms[i], norms[i + 1]
        if not current:
            continue

        repeated = current == following
        fragment = (
            _is_fragment(words[i])
            or (len(current) <= 3 and following.startswith(current) and current != following)
        )

        if repeated or fragment:
            words[i].tag = WordTag.STUTTER


def _tag_false_starts(words: list[TranscriptWord], config: PacingConfig) -> None:
    """An abandoned clause: an opener, then a restart on that same opener."""
    norms = [_normalize(w.text) for w in words]
    lookahead = config.false_start_lookahead

    i = 0
    while i < len(words):
        if words[i].tag != WordTag.NORMAL or norms[i] not in CLAUSE_STARTERS:
            i += 1
            continue

        restart = _find_restart(words, norms, i, lookahead, config)

        if restart is None:
            i += 1
            continue

        for w in words[i:restart]:
            if w.tag == WordTag.NORMAL:
                w.tag = WordTag.FALSE_START
        i = restart


def _find_restart(
    words: list[TranscriptWord],
    norms: list[str],
    i: int,
    lookahead: int,
    config: PacingConfig,
) -> int | None:
    for j in range(i + 2, min(i + lookahead + 1, len(words))):
        # A pause ends the sentence — whatever follows is new material, not a
        # restart of the clause that came before it.
        if words[j].start - words[j - 1].end >= config.min_pause_seconds:
            return None

        if words[j].tag == WordTag.FILLER or norms[j] != norms[i]:
            continue

        # Either the speaker audibly broke off, or they repeated more than the
        # opener — a bare repeat with neither is ordinary grammar ("the ... the").
        broke_off = words[j].start - words[j - 1].end >= config.false_start_break_seconds
        repeats_phrase = (
            j + 1 < len(words) and i + 1 < j and norms[j + 1] == norms[i + 1]
        )

        if broke_off or repeats_phrase:
            return j

    return None


def _disfluency_signals(
    words: list[TranscriptWord], speaker: str | None
) -> list[Signal]:
    signal_types = {
        WordTag.FILLER: SignalType.FILLER_WORD,
        WordTag.STUTTER: SignalType.STUTTER,
        WordTag.FALSE_START: SignalType.FALSE_START,
    }

    signals: list[Signal] = []
    run: list[TranscriptWord] = []

    def flush() -> None:
        if not run:
            return
        text = " ".join(w.text for w in run)
        signals.append(Signal(
            type=signal_types[run[0].tag],
            source="pacing",
            start=run[0].start,
            end=run[-1].end,
            value=text,
            speaker=speaker,
            metadata={"words": len(run)},
        ))
        run.clear()

    for word in words:
        if word.tag == WordTag.NORMAL:
            flush()
            continue
        if run and run[0].tag != word.tag:
            flush()
        run.append(word)

    flush()
    return signals


def _pause_signals(
    words: list[TranscriptWord], speaker: str | None, config: PacingConfig
) -> list[Signal]:
    signals: list[Signal] = []

    for current, following in zip(words, words[1:]):
        gap = following.start - current.end
        if gap >= config.min_pause_seconds:
            signals.append(Signal(
                type=SignalType.LONG_PAUSE,
                source="pacing",
                start=current.end,
                end=following.start,
                value=round(gap, 2),
                speaker=speaker,
                metadata={"before": current.text, "after": following.text},
            ))

    return signals


def _voiced_duration(words: list[TranscriptWord], config: PacingConfig) -> float:
    """Span covered by the words, minus any silence long enough to be a pause."""
    if not words:
        return 0.0

    total = sum(w.end - w.start for w in words)
    for current, following in zip(words, words[1:]):
        gap = following.start - current.end
        if 0 < gap < config.min_pause_seconds:
            total += gap

    return total


def _rate_signals(
    words: list[TranscriptWord], speaker: str | None, config: PacingConfig
) -> list[Signal]:
    if not words:
        return []

    windows = _rate_windows(words, config)
    if not windows:
        return []

    baseline = statistics.median(w["wpm"] for w in windows)
    swing = config.baseline_swing_wpm

    flagged = []
    for window in windows:
        wpm = window["wpm"]

        if wpm >= config.rush_wpm or wpm >= baseline + swing:
            window["type"] = SignalType.RUSH
        elif wpm <= config.drag_wpm or wpm <= baseline - swing:
            window["type"] = SignalType.DRAG
        else:
            continue

        flagged.append(window)

    signals: list[Signal] = []

    for stretch in _merge_windows(flagged):
        inside = [w for w in words if stretch["start"] <= w.start <= stretch["end"]]

        # A stretch spanning a pause is two stretches — reporting the silence as
        # rushed or dragging speech would misstate how long the issue ran.
        for run in _split_at_pauses(inside, config):
            voiced = _voiced_duration(run, config)
            if voiced < config.min_voiced_seconds:
                continue

            content = sum(1 for w in run if w.tag == WordTag.NORMAL)
            signals.append(Signal(
                type=stretch["type"],
                source="pacing",
                start=run[0].start,
                end=run[-1].end,
                value=round(len(run) / (voiced / 60), 1),
                speaker=speaker,
                metadata={
                    "wpm": round(len(run) / (voiced / 60), 1),
                    "content_wpm": round(content / (voiced / 60), 1),
                    "baseline_wpm": round(baseline, 1),
                    "voiced_seconds": round(voiced, 1),
                },
            ))

    return signals


def _rate_windows(words: list[TranscriptWord], config: PacingConfig) -> list[dict]:
    """Speaking rate over sliding windows, measured against voiced time only."""
    windows: list[dict] = []

    start = words[0].start
    last = words[-1].end

    while start < last:
        end = start + config.window_seconds
        inside = [w for w in words if start <= (w.start + w.end) / 2 < end]
        voiced = _voiced_duration(inside, config)

        if voiced >= config.min_voiced_seconds:
            windows.append({
                "start": inside[0].start,
                "end": inside[-1].end,
                "wpm": len(inside) / (voiced / 60),
            })

        start += config.hop_seconds

    return windows


def _split_at_pauses(
    words: list[TranscriptWord], config: PacingConfig
) -> list[list[TranscriptWord]]:
    runs: list[list[TranscriptWord]] = []
    run: list[TranscriptWord] = []

    for word in words:
        if run and word.start - run[-1].end >= config.min_pause_seconds:
            runs.append(run)
            run = []
        run.append(word)

    if run:
        runs.append(run)

    return runs


def _merge_windows(windows: list[dict]) -> list[dict]:
    """Collapse overlapping windows of the same type into contiguous stretches."""
    merged: list[dict] = []

    for window in sorted(windows, key=lambda w: w["start"]):
        previous = merged[-1] if merged else None

        if previous and previous["type"] == window["type"] and window["start"] <= previous["end"]:
            previous["end"] = max(previous["end"], window["end"])
            continue

        merged.append(dict(window))

    return merged


def analyze_pacing(ctx: MediaContext, config: PacingConfig) -> list[Signal]:
    """Tag disfluencies on the transcript, emit pacing and fluency signals."""
    if ctx.transcript is None:
        return []

    speaker = ctx.primary_speaker
    words = _speaker_words(ctx, speaker)

    if not words:
        return []

    _tag_fillers(words, config)
    _tag_stutters(words, config)
    _tag_false_starts(words, config)

    return [
        *_disfluency_signals(words, speaker),
        *_pause_signals(words, speaker, config),
        *_rate_signals(words, speaker, config),
    ]
