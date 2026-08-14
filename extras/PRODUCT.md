# Suasiv — Product Plan

## What It Is

Suasiv is a communication coaching platform. You give it a video of someone speaking — a pitch, a meeting, a town hall, a lecture — and it analyzes **both sides** of the interaction:

1. **The speaker**: how they delivered (voice, pacing, content, facial signals)
2. **The audience**: how they received it (engagement, reactions, verbal interaction)

It produces a coaching report that ties the two together: *"When you discussed margins at 3:42, your vocal energy dropped and you used three filler words. Two audience members looked away and one started typing. When you pivoted to the growth story at 4:10, all heads came back up and there were two nods."*

## Who It's For

Executives and anyone preparing high-stakes spoken communication. The category exists (Yoodli, Poised, etc.) — this is a validated problem shape. The differentiator is audience-side analysis: existing tools only look at the speaker.

## The Product Layers

Suasiv ships in three layers, each building on the last:

### v1 — Processing Pipeline + CLI (current)

The complete analysis engine, shipped as a GitHub repo. Users clone, install, and run:

```
python -m suasiv analyze video.mp4 --config config.yaml
```

Every analyzer ships in v1 — speaker analysis, audience analysis, diarization, fusion, LLM coaching, report generation. Nothing is deferred. The CLI is the product's first public version.

Supports meeting recordings (Zoom/Teams gallery view) as the primary input. Room-camera support (single camera on speaker + audience) is designed for and built where feasible.

### v2 — API Server

FastAPI wrapping the processing pipeline. Endpoints for uploading video, checking processing status, and retrieving reports. WebSocket for live progress during analysis. This is the bridge between the core engine and the web frontend.

### v3 — Hosted Web Application

The full SaaS product:
- Next.js frontend
- User authentication and accounts
- Payment/subscription system (per-video or subscription pricing)
- Video upload and management
- Interactive report viewer: video player synced with analysis timeline, click-to-jump, audience attention heatmaps
- User history: track improvement over time

---

## Analysis Dimensions

### Speaker Analysis (4 dimensions)

| Dimension | What It Measures | Key Signals |
|---|---|---|
| **Pacing** | Speech rhythm and fluency | Words-per-minute, pause length/placement, filler word frequency, rate variation |
| **Prosody** | Vocal delivery quality | Pitch (F0), energy/volume, vocal variety, monotone detection, energy drops/spikes |
| **Content** | What was said and how | Clarity, structure, hedging, whether questions were answered, filler density |
| **Facial** | Visual delivery signals | Expressiveness, head pose, eye contact / gaze direction |

### Audience Analysis (3 dimensions)

| Dimension | What It Measures | Key Signals |
|---|---|---|
| **Engagement** | Attention level | Head pose (facing speaker vs away), gaze direction, aggregate attention % over time |
| **Reactions** | How they received it | Facial Action Units (nodding, smiling, frowning, confusion), aggregate sentiment, simultaneous reactions |
| **Verbal** | Active interaction | Questions asked, interruptions, agreement/disagreement, Q&A quality |

### Cross-Modal Fusion

The fusion layer merges speaker + audience signals onto one timeline and detects correlated moments:
- **Strong moments**: high speaker energy + positive audience reactions + audience attention
- **Weak moments**: speaker energy drops + audience attention drops + filler words
- **Turning points**: where audience engagement shifted notably

---

## Positioning — "Signals, Not Emotions"

Emotion recognition (vocal and facial) is scientifically contested and noisy. Every output is framed as an **observable signal** — "vocal variety," "expressiveness," "energy," "eye contact," "attention" — never an inferred feeling.

The report never says *"the audience was bored"*; it says *"60% of audience members were looking away from the speaker during this segment."*

This is a credibility requirement, not a stylistic preference. The target audience (executives) will notice and distrust overclaiming.

---

## The Rubric

The long-term moat is the **rubric** — what makes a piece of feedback "correct." The ML models are commodity; the scoring criteria and coaching framing are not.

The rubric lives in a versioned `rubric.yaml` file that the LLM reasoning layer loads. Feedback criteria are explicit and tunable without code changes. This makes it possible to:
- Refine coaching quality over time
- Create domain-specific rubrics (investor pitch vs all-hands vs sales demo)
- Let advanced users customize what "good" means for their context

---

## Tech Stack

### v1 — Processing Pipeline

**Core**
| Tool | Purpose |
|---|---|
| Python 3.10+ | Pipeline language — ML ecosystem lives here |
| pyproject.toml + uv | Package management — uv handles heavy ML deps fast, falls back to pip |
| Pydantic | Schema + config validation — reused directly by FastAPI in v2 |
| PyYAML | Config + rubric files |
| Typer | CLI framework |

**Media**
| Tool | Purpose |
|---|---|
| ffmpeg (system) | Audio extraction, frame sampling — industry standard |
| ffmpeg-python | Python wrapper for ffmpeg |
| opencv-python | Frame reading, meeting tile detection |

**Speech**
| Tool | Purpose |
|---|---|
| faster-whisper | Transcription — 4x faster than OpenAI whisper on CPU, word-level timestamps |
| whisperx or pyannote | Speaker diarization — attribute speech to participants |

**Audio Analysis**
| Tool | Purpose |
|---|---|
| librosa | Pitch, energy, audio features |
| parselmouth | Precise F0 via Praat — phonetics gold standard |

**Facial Analysis**
| Tool | Purpose |
|---|---|
| MediaPipe | Face mesh, head pose, gaze — lightweight, CPU-friendly |
| py-feat | Facial Action Units — interpretable reaction signals, heavier |

**LLM**
| Tool | Purpose |
|---|---|
| Ollama | Local inference (default) — free, private, offline |
| google-generativeai | Gemini API (optional) — free tier fallback |
| Groq SDK | Groq API (optional) — fast, generous free tier |

**Report**
| Tool | Purpose |
|---|---|
| Jinja2 | Template rendering to Markdown/HTML |

**Dev**
| Tool | Purpose |
|---|---|
| pytest | Testing |
| ruff | Lint + format |

### v2 — API Server

| Tool | Purpose |
|---|---|
| FastAPI | HTTP API — async, auto-generates OpenAPI docs, native Pydantic support |
| WebSockets (via FastAPI) | Live progress updates during analysis |

### v3 — Web Application

| Tool | Purpose |
|---|---|
| Next.js | Frontend framework |
| Tailwind CSS | Styling |
| Auth (TBD) | Authentication — evaluate at v3 time |
| Payments (TBD) | Stripe or similar — evaluate at v3 time |
| PostgreSQL (likely) | User data, video metadata, report storage |
| Object storage (TBD) | Video file storage — S3 or similar |

---

## Repo Structure

```
suasiv/
├── core/                              # v1: Python processing pipeline
│   ├── suasiv/                        # Python package
│   │   ├── __init__.py
│   │   ├── schema.py                  # Signal, AnalyzerResult, FusedTimeline, CoachingReport
│   │   ├── config.py                  # Config model + loader
│   │   ├── context.py                 # MediaContext
│   │   ├── ingest.py                  # ffmpeg: audio + frames + tile detection
│   │   ├── pipeline.py                # Orchestrates full pipeline
│   │   ├── fusion.py                  # Merge signals, correlate moments
│   │   ├── report.py                  # Jinja2 rendering
│   │   ├── analyzers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Analyzer ABC
│   │   │   ├── transcript.py          # faster-whisper
│   │   │   ├── diarization.py         # Speaker separation
│   │   │   ├── pacing.py              # WPM, pauses, fillers
│   │   │   ├── prosody.py             # Pitch, energy, vocal variety
│   │   │   ├── speaker_facial.py      # Speaker expressiveness, eye contact
│   │   │   ├── audience_engagement.py # Attention tracking
│   │   │   └── audience_reaction.py   # Facial action units
│   │   └── llm/
│   │       ├── __init__.py
│   │       ├── base.py                # LLMBackend ABC
│   │       ├── ollama.py
│   │       ├── gemini.py
│   │       └── groq.py
│   ├── cli.py
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── requirements-lite.txt
│   └── templates/
│       ├── report.md.j2
│       └── report.html.j2
├── api/                               # v2: FastAPI server
│   └── (placeholder)
├── web/                               # v3: Next.js frontend
│   └── (placeholder)
├── config.yaml                        # Default configuration
├── rubric.yaml                        # Scoring criteria
├── PRODUCT.md                         # This file — full product plan
├── V1_BUILD_PLAN.md                   # v1 implementation steps + status
└── README.md
```

---

## Known Hard Parts

1. **ffmpeg is a system install, not pip.** The #1 clone-and-run failure. Check on startup with a clear error. Document per-OS install at the top of README.
2. **PyTorch is multi-GB.** Ship a lite install path (`requirements-lite.txt`) for transcript + pacing only (no torch). Full stack is opt-in.
3. **Model weights download on first run.** Clear progress messages and documented disk requirements, or the first run looks like a hang.
4. **Speaker diarization accuracy.** Diarization is imperfect. Design for graceful handling of mis-attributed segments. Let users correct via config if needed.
5. **Meeting tile detection.** Gallery view layouts vary across Zoom/Teams/Meet versions. Tile detection needs to be robust to different grid sizes, aspect ratios, and UI chrome.
6. **Emotion overclaiming.** Enforced in report templates and LLM system prompts. Observable signals only.
7. **CPU vs GPU.** Everything must work on CPU. Don't hard-require CUDA. Config controls what runs on weak hardware.
8. **Long videos.** Chunking strategy needed for videos over ~30 minutes to avoid OOM. Overlap handling at chunk boundaries so words/sentences at cuts aren't mangled.
