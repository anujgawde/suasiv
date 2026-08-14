# Suasiv

Communication coaching platform — analyzes speaker delivery and audience reception from video recordings.

## Quick Start

```bash
cd core
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m suasiv analyze video.mp4 --config ../config.yaml
```

## What It Does

Give Suasiv a video of someone speaking and it analyzes **both sides**:

1. **Speaker**: pacing, prosody, content clarity, facial signals
2. **Audience**: engagement/attention, reactions, verbal interaction

It produces a coaching report tying speaker delivery to audience response — *"When you discussed margins at 3:42, your vocal energy dropped. Two audience members looked away."*

## Status

Step 1 (Skeleton) — all components stubbed with fake data. Pipeline runs end-to-end.
