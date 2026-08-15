# Suasiv

Communication coaching platform — analyzes speaker delivery and audience reception from video recordings.

Give Suasiv a video of someone speaking and it analyzes **both sides**:

1. **Speaker**: pacing, prosody, content clarity, facial signals
2. **Audience**: engagement/attention, reactions, verbal interaction

It produces a coaching report tying speaker delivery to audience response — *"When you discussed margins at 3:42, your vocal energy dropped. Two audience members looked away."*

## Quick Start

```bash
# 1. Install (lite = transcript + pacing, no GPU deps)
cd core
python -m venv .venv && source .venv/bin/activate
pip install -e ".[lite]"

# 2. Install ffmpeg (required)
# macOS:   brew install ffmpeg
# Ubuntu:  sudo apt install ffmpeg
# Windows: choco install ffmpeg

# 3. Start Ollama (default LLM backend)
ollama serve &
ollama pull llama3.1:8b

# 4. Run
suasiv analyze recording.mp4 --preset lite
```

Report is saved to `.workspace/report.md`.

## Installation Tiers

| Tier | Command | What it enables | Disk |
|------|---------|----------------|------|
| **Base** | `pip install -e .` | Pacing, content, audience verbal (on pre-computed transcript) | ~20MB |
| **Lite** | `pip install -e ".[lite]"` | + Transcription (faster-whisper) + Ollama LLM | ~200MB + model weights |
| **Full** | `pip install -e ".[full]"` | All 9 analyzers: prosody, facial, audience engagement/reactions | ~2GB + model weights |

### Model Weight Sizes

| Model | Approximate Size | Location |
|-------|-----------------|----------|
| faster-whisper `base` | ~150MB | `~/.cache/huggingface/` |
| faster-whisper `large` | ~3GB | `~/.cache/huggingface/` |
| pyannote speaker-diarization | ~300MB | `~/.cache/huggingface/` |
| mediapipe face mesh | ~2MB | Downloaded on first use |
| py-feat detector | ~500MB | `~/.cache/torch/` |

Models are downloaded on first run and cached for reuse.

## Presets

Use `--preset` to control which analyzers run:

| Preset | Analyzers | Best for |
|--------|-----------|----------|
| `full` | All 9 | Full analysis with audience video |
| `standard` | All except audience reactions | Faster, no py-feat/torch needed |
| `lite` | Transcript, pacing, content, audience verbal | Quick text-only analysis |

```bash
suasiv analyze video.mp4 --preset standard
suasiv analyze video.mp4 --preset lite
```

You can also enable/disable individual analyzers in `config.yaml`:

```yaml
analyzers:
  speaker_facial:
    enabled: false
```

Explicit config settings override the preset.

## Config Reference

See `config.yaml` for all options. Key settings:

```yaml
# LLM backend: ollama (default), gemini, groq
llm:
  backend: ollama
  model: llama3.1:8b
  # api_key: or set SUASIV_LLM_API_KEY env var

# Report format
report:
  format: markdown  # or html

# Analyzer tuning
analyzers:
  transcript:
    model_size: base  # base | small | medium | large
  pacing:
    window_seconds: 30
  prosody:
    window_seconds: 5
```

## CPU-Only Mode

All analyzers work on CPU — no CUDA required. GPU is auto-detected and used when available for faster processing. No configuration needed.

## CLI Reference

```
suasiv analyze <video> [OPTIONS]

Options:
  -c, --config PATH     Config YAML (default: config.yaml)
  -p, --preset TEXT      Preset: full, standard, lite
  --skip-checks         Skip pre-flight validation
  --help                Show help

suasiv version           Print version
```

## Error Resilience

If an analyzer fails (missing dependency, no faces in video, etc.), the pipeline continues with the remaining analyzers. The report notes which analyzers were skipped or failed and why.

## Troubleshooting

**ffmpeg not found**: Install with `brew install ffmpeg` (macOS), `sudo apt install ffmpeg` (Ubuntu), or `choco install ffmpeg` (Windows).

**Cannot connect to Ollama**: Start the server with `ollama serve`, then pull the model: `ollama pull llama3.1:8b`.

**Missing API key (Gemini/Groq)**: Set `api_key` in config.yaml or `export SUASIV_LLM_API_KEY=your-key`.

**Analyzer skipped due to missing deps**: Install the full tier: `pip install -e ".[full]"`.

## Hardware Requirements

- **Minimum**: 4GB RAM, 2GB free disk (for model weights)
- **Recommended for full analysis**: 8GB+ RAM
- **GPU**: Optional, speeds up transcription and diarization
