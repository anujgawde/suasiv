# Executive Video Coach — Project Proposal

> An open-source, $0, **locally-runnable** pipeline that ingests a video of someone
> speaking (a pitch, presentation, earnings call, all-hands) and produces a
> coaching-grade report on how they came across — tone, facial expressiveness,
> content, and pacing.

This document is the plan. It covers both the **idea** (what we're building and why)
and the **tech** (how to build it). It is written to be handed to Claude Code for
analysis, scaffolding, and implementation.

---

## 0. Read this first — scope guardrail

This is a **single-machine, local-first, zero-cost tool**. It is shipped as a GitHub
repo that people clone and run on their own hardware. There is **no server, no cloud,
no hosting, no user accounts, no scale infrastructure** in v1.

A hosted, at-scale version of this system exists as a _separate future design_ (queue,
worker fleet, object storage, etc.). **Do not build any of that here.** The relationship
is deliberate: this local pipeline is the _core_ of the hosted system with all the
delivery/scale infrastructure removed. Building the core cleanly is what makes the
hosted version a wrapping exercise later. For now: build only the core.

---

## 1. The idea

### What it is

An AI communication coach. You give it a video; it analyzes how the speaker
communicates and returns a structured, timestamped coaching report — e.g. _"At 3:42
your energy dropped, you broke eye contact, and you used three filler words while
answering the margin question."_

### Who it's for

Executives and anyone preparing high-stakes spoken communication. The category already
exists (Yoodli, Poised, etc.), so this is a known, validated problem shape.

### The four analysis parameters (the product's substance)

1. **Speaking tone** — pitch, energy, vocal variety (monotone vs. dynamic).
2. **Facial expression** — expressiveness, head pose, gaze / eye contact.
3. **Things said** — transcript-level content: clarity, structure, hedging, filler,
   whether questions were actually answered.
4. **Pauses & pacing** — words-per-minute, pause distribution, filler-word frequency.

### Positioning & the honest caveat (important — bake this into the output)

Emotion recognition (vocal and facial) is scientifically contested and noisy. The report
must **never claim to read a speaker's true internal emotions.** Frame every output as an
_observable signal_ — "vocal variety," "expressiveness," "energy," "eye contact" — not an
inferred feeling. This is a credibility requirement, not a stylistic one; the target
audience will notice overclaiming.

The real long-term moat is the **rubric** — what makes a piece of feedback "correct."
The models are commodity; the scoring rubric and framing are not. Keep the rubric
explicit and configurable (see §6) so it can be refined over time.

---

## 2. v1 scope — what it is and isn't

**In scope (v1):**

- Single-speaker videos (one presenter).
- Fully local execution; runs offline with local models.
- CLI-driven: `analyze <video> --config config.yaml` → report on disk.
- Markdown/HTML report output.
- Graceful degradation by hardware (config decides which analyzers run).

**Out of scope (v1), noted as roadmap:**

- Multi-person meetings / speaker diarization.
- Any hosted/server/web-UI component.
- Real-time / streaming analysis.
- Benchmarking a speaker against peers or their own history (SQLite history is a
  stretch goal, not required).

---

## 3. Core design principles

These are the decisions that keep the project modular and make it a _system_, not a script.

1. **Modular by analyzer.** Each of the four parameters is an independent module
   implementing a common interface. Swapping one (e.g. MediaPipe → py-feat) must not
   touch the others.
2. **A single shared data schema.** Every analyzer emits the _same_ timestamped-signal
   shape. This is the contract that holds the system together — get it right first (§5).
3. **Bring-your-own-LLM.** The LLM backend is pluggable behind one interface: local
   Ollama by default, optional free-tier APIs (Gemini/Groq) via config. The pipeline
   never hardcodes a provider.
4. **Config-driven graceful degradation.** A weak machine runs `base` Whisper and skips
   facial analysis; a strong one runs the full stack. Same repo, different `config.yaml`.
5. **Local-first, no external dependency required.** API keys are an optional
   enhancement, never a requirement to run.

---

## 4. Architecture / components

The whole thing is one sequential pipeline. No queue, no workers, no orchestrator —
just stages passing data through a shared schema.

```
video file on disk
  → [ Ingest ]        ffmpeg: extract audio, sample frames (2–5 fps)
  → [ Analyzers ]     transcript · pacing · prosody · facial   (each emits Signals)
  → [ Fusion ]        merge all Signals onto one timeline, correlate cross-modal moments
  → [ Reasoning ]     LLM: rubric-based content scoring + coaching synthesis
  → [ Report ]        render markdown/HTML report next to the video
```

**Component responsibilities (one sentence each):**

- **Ingest** — turn a video file into predictable media: mono audio (normalized sample
  rate) + sampled frames. Owns all ffmpeg interaction.
- **Analyzers** — four peer modules, each takes media in and emits a list of timestamped
  Signals + summary metrics. They know the schema, not each other.
- **Fusion** — merge all analyzers' Signals onto one timeline; detect co-occurring
  moments (e.g. energy drop + gaze break + filler within the same window).
- **Reasoning (LLM)** — take transcript + fused signals, score against the rubric, and
  write the human-readable coaching narrative. Pluggable backend.
- **Report** — render the final artifact. Purely presentation; knows nothing about how
  signals were computed.

Cross-cutting: **Config** (control panel), **Model management** (download/cache weights),
**Workspace** (a scratch dir for intermediate artifacts).

---

## 5. The shared data schema (build this first)

This is the load-bearing decision. Define it before any analyzer. Suggested Pydantic
models:

```python
class Signal(BaseModel):
    analyzer: str            # "prosody", "facial", "pacing", "transcript"
    type: str                # "filler_word", "pitch_drop", "gaze_off_camera", ...
    start: float             # seconds from video start
    end: float               # seconds
    value: float | str | dict | None = None   # the measurement / payload
    confidence: float | None = None
    speaker: str | None = None                # reserved for future multi-speaker

class AnalyzerResult(BaseModel):
    analyzer: str
    signals: list[Signal]
    summary: dict            # analyzer-level aggregates (e.g. {"wpm": 142, "fillers": 11})

class FusedTimeline(BaseModel):
    duration: float
    signals: list[Signal]    # all analyzers merged, time-sorted
    moments: list[dict]      # cross-modal correlated events with timestamps

class Report(BaseModel):
    summary_scores: dict     # small, could persist to SQLite later
    narrative: str           # LLM-written coaching text
    timeline: FusedTimeline
```

If all analyzers emit `AnalyzerResult`, fusion is trivial and new analyzers plug in for free.

---

## 6. Interfaces / contracts

**Analyzer contract** (every analyzer implements this):

```python
class Analyzer(ABC):
    name: str
    requires: set[str]        # {"audio"} or {"frames"} or {"transcript"}
    @abstractmethod
    def analyze(self, ctx: MediaContext) -> AnalyzerResult: ...
```

`MediaContext` carries paths to the extracted audio, sampled frames, and (once available)
the transcript, plus the resolved config. Note the dependency: pacing and content-scoring
**require the transcript**, so transcription runs first and its output is placed in the
context for the others.

**LLM backend contract** (pluggable):

```python
class LLMBackend(ABC):
    @abstractmethod
    def complete(self, system: str, prompt: str) -> str: ...
```

Implementations: `OllamaBackend` (default), `GeminiBackend`, `GroqBackend`. Selected by config.

**Rubric** — keep the scoring rubric in a versioned file (e.g. `rubric.yaml`) that the
reasoning layer loads, so feedback criteria are explicit and tunable without code changes.

---

## 7. Tech stack

| Concern            | Choice                                                   | Notes                                                         |
| ------------------ | -------------------------------------------------------- | ------------------------------------------------------------- |
| Language           | Python 3.10+                                             | The whole ML ecosystem lives here                             |
| Env                | `venv` + `requirements.txt`                              | Mention `uv` in README as optional, don't require             |
| Media              | `ffmpeg` (system dep) + `ffmpeg-python`, `opencv-python` | ffmpeg is NOT pip-installable — see §10                       |
| Transcription      | `faster-whisper` (`base`/`small` default)                | word-level timestamps; `whisperx` if diarization wanted later |
| Pacing             | `numpy` + stdlib                                         | pure math off timestamps; no ML                               |
| Prosody            | `librosa`, `parselmouth`                                 | optional `opensmile` (eGeMAPS)                                |
| Facial             | `mediapipe` (default), `py-feat` (optional, richer)      | py-feat gives interpretable Action Units                      |
| LLM                | `ollama` (local) + optional `google-generativeai` / Groq | pluggable behind `LLMBackend`                                 |
| Schema & config    | `pydantic` + `pyyaml`                                    | one tool for validation + serialization                       |
| CLI                | `typer`                                                  | `argparse` if zero-dep preferred                              |
| Report render      | `jinja2` (→ md/html), optional `matplotlib` charts       |                                                               |
| History (optional) | `sqlite3` (stdlib)                                       | stretch goal only                                             |
| Hygiene            | `pytest`, `ruff`, GitHub Actions CI                      | free for public repos                                         |

**LLM default:** local Ollama with Llama 3.1 8B or Qwen 2.5. Fully free, private, offline.
Free-tier APIs (Gemini/Groq) are opt-in via config for users whose machines can't run a
local model well.

---

## 8. Repo structure

```
exec-video-coach/
├── README.md              # the product's front door — sample report up top, 1-command run
├── PROPOSAL.md            # this file
├── requirements.txt
├── requirements-lite.txt  # no-torch install: transcript + pacing only (see §10)
├── config.yaml            # which analyzers, model sizes, LLM backend, frame rate
├── rubric.yaml            # scoring criteria for the reasoning layer
├── src/
│   ├── ingest.py          # ffmpeg: audio extraction + frame sampling
│   ├── context.py         # MediaContext
│   ├── schema.py          # Signal / AnalyzerResult / FusedTimeline / Report (§5)
│   ├── analyzers/
│   │   ├── base.py        # Analyzer ABC (§6)
│   │   ├── transcript.py
│   │   ├── pacing.py
│   │   ├── prosody.py
│   │   └── facial.py
│   ├── llm/
│   │   ├── base.py        # LLMBackend ABC
│   │   ├── ollama.py
│   │   └── gemini.py
│   ├── fusion.py
│   ├── report.py          # jinja2 rendering
│   └── pipeline.py        # orchestrates ingest → analyzers → fusion → llm → report
├── examples/
│   └── sample_report.md   # so people see output before installing
├── tests/
└── cli.py                 # `python cli.py analyze video.mp4 --config config.yaml`
```

---

## 9. Configuration (example)

```yaml
analyzers:
  transcript: { enabled: true, model: "base" } # base | small | medium | large
  pacing: { enabled: true }
  prosody: { enabled: true }
  facial: { enabled: false, backend: "mediapipe", fps: 3 } # off by default (heavy)
llm:
  backend: "ollama" # ollama | gemini | groq
  model: "llama3.1:8b"
  api_key_env: "GEMINI_API_KEY" # only read for api backends
report:
  format: "markdown" # markdown | html
workspace: "./.workspace"
```

Config is the graceful-degradation lever: a laptop runs the top three analyzers on `base`
Whisper; a workstation flips `facial.enabled: true` and bumps the Whisper model.

---

## 10. Known hard parts & gotchas (address these explicitly)

1. **ffmpeg is a system install, not pip.** The #1 clone-and-run failure. Document per-OS
   install (`apt install ffmpeg`, `brew install ffmpeg`) at the very top of the README, and
   have the code check for it on startup with a clear error.
2. **PyTorch is heavy (~GBs) and pulled in by Whisper/py-feat.** Ship a **lite install
   path** (`requirements-lite.txt`) that runs transcript + pacing with no torch, so a
   curious user can try the MVP without a multi-GB download. Full stack is opt-in.
3. **Model weights download on first run** (hundreds of MB to GBs). Handle with a clear
   progress message and documented disk requirement, or the first run looks like a hang.
4. **Emotion overclaiming.** Enforce the "signals, not emotions" framing in the report
   templates and LLM system prompt (§1).
5. **CPU vs GPU.** Everything must _work_ on CPU (just slower). Don't hard-require CUDA.

---

## 11. Long-video handling (the one piece of scale-thinking to keep)

Even locally, a multi-hour video may not fit in memory and is painfully slow serially.
Keep a **chunking** path: split long audio/video into overlapping segments, analyze each,
then stitch — align on the transcript and dedupe the overlap so words/sentences straddling
a cut aren't mangled. Numeric signals (WPM, pauses, energy) aggregate across chunks; the
LLM synthesis runs once over the combined result (feedback must reason about the whole
talk, not per-chunk). This is optional for the MVP but should be designed for, not bolted
on later.

---

## 12. Build sequence (milestones with acceptance criteria)

Build in this order — cheap/reliable/high-value first, flashiest/shakiest last.

**M0 — Skeleton.** Schema (§5), `Analyzer`/`LLMBackend` ABCs, config loading, CLI, and
`pipeline.py` running end-to-end with **stub analyzers**.
_Done when:_ `python cli.py analyze sample.mp4` runs the full pipeline and emits a report
from stubbed signals.

**M1 — Transcript + Pacing.** Real transcription (faster-whisper, word timestamps) and
pacing (WPM, pauses, fillers) off those timestamps.
_Done when:_ a real video produces a real transcript and accurate pacing metrics. This
alone is already a usable coaching tool — good point to make the repo public.

**M2 — LLM reasoning + report.** Rubric-based content scoring and coaching narrative via
the pluggable backend; polished markdown/HTML report.
_Done when:_ the report reads like genuine coaching feedback, and swapping Ollama↔Gemini
is a one-line config change.

**M3 — Prosody.** Pitch/energy/variation; monotone and energy-drop detection.
_Done when:_ prosody signals appear on the fused timeline and in the report.

**M4 — Facial (optional, gated by config).** Expressiveness, head pose, gaze/eye-contact,
framed as signals.
_Done when:_ enabling `facial` in config adds visual signals without touching other stages.

**M5 — Fusion polish + long-video chunking.** Cross-modal moment correlation on the
timeline; chunking for long inputs.
_Done when:_ the report surfaces correlated moments ("energy drop + gaze break + filler at
3:42") and a 1-hour+ video processes without OOM.

---

## 13. Future roadmap (explicitly not v1)

- Multi-speaker meetings via diarization (whisperx/pyannote); the `speaker` field on
  `Signal` already reserves space for this.
- Speaker history/benchmarking via SQLite.
- The **hosted, at-scale** version: wrap this pipeline in a worker behind a queue, add
  object storage + a web tier. The core pipeline is designed to be surrounded, not rewritten.

---

## 14. What I'd like Claude Code to do

1. Sanity-check this plan — call out anything unsound, over- or under-scoped, or missing.
2. Confirm the dependency/versioning picture is realistic and flag install-friction risks.
3. Scaffold the repo per §8, starting with **M0** (schema + interfaces + stubbed pipeline
   that runs end-to-end) so the architecture is proven before filling in analyzers.
4. Propose the concrete `Signal.type` vocabulary per analyzer and the initial `rubric.yaml`.
