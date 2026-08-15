# Suasiv v1 — Build Plan

> v1 is the complete processing pipeline + CLI. Every analyzer, every signal, speaker + audience. Shipped as a GitHub repo users clone and run.

See [PRODUCT.md](PRODUCT.md) for the full product vision.

---

## Status

| Step | Name | Status |
|------|------|--------|
| 1 | Skeleton | done |
| 2 | Ingest | done |
| 3 | Transcription + Diarization | done |
| 4 | Speaker Analysis | done |
| 5 | Audience Analysis | not started |
| 6 | Fusion | not started |
| 7 | LLM Reasoning + Report | not started |
| 8 | Config + Polish | not started |

**Currently working on:** Step 5

---

## Step 1 — Skeleton

**Goal:** Prove the architecture. Every component exists as a stub, the pipeline runs end-to-end, a report comes out. Zero ML code.

**What to build:**
- `schema.py` — Pydantic models:
  - `Signal(analyzer, type, start, end, value, confidence, speaker)`
  - `AnalyzerResult(analyzer, signals, summary)`
  - `FusedTimeline(duration, signals, moments)`
  - `CoachingReport(summary_scores, narrative, timeline)`
- `config.py` — Pydantic config model + YAML loader. Covers:
  - Which analyzers are enabled, with per-analyzer settings (model size, fps, etc.)
  - LLM backend selection (ollama/gemini/groq) + model name
  - Report format (markdown/html)
  - Workspace directory path
- `context.py` — `MediaContext` dataclass carrying:
  - Paths to extracted audio, sampled frames, transcript
  - Participant/tile information
  - Resolved config
- `analyzers/base.py` — `Analyzer` ABC:
  - `name: str`
  - `requires: set[str]` (e.g., `{"audio"}`, `{"frames"}`, `{"transcript"}`)
  - `analyze(ctx: MediaContext) -> AnalyzerResult`
- Stub analyzers for all 7: transcript, diarization, pacing, prosody, speaker_facial, audience_engagement, audience_reaction — each returns hardcoded fake signals
- `llm/base.py` — `LLMBackend` ABC:
  - `complete(system: str, prompt: str) -> str`
- Stub LLM backend returning canned coaching text
- `pipeline.py` — orchestrator:
  - Calls ingest (stub) → runs analyzers in dependency order → calls fusion (stub) → calls LLM (stub) → renders report
  - Handles analyzer dependency: transcript runs first, its output goes into context for pacing/prosody/content. Diarization depends on transcript. Audience analyzers depend on frames + diarization.
- `fusion.py` — stub that merges all signals by timestamp
- `report.py` — Jinja2 renderer loading from `templates/`
- `cli.py` — Typer CLI: `python -m suasiv analyze <video> --config config.yaml`
- `templates/report.md.j2` — basic Markdown report template with sections for speaker analysis, audience analysis, key moments
- `config.yaml` — default config with all analyzers enabled
- `rubric.yaml` — initial scoring rubric structure
- `pyproject.toml` — package setup, dependencies, entry point

**Dependencies:** None (first step)

**Verify:**
```bash
python -m suasiv analyze sample.mp4 --config config.yaml
```
- Runs without errors
- Creates `.workspace/` with intermediate artifacts
- Produces `report.md` with stub signals rendered through the template
- All 7 analyzer stubs are called and contribute signals to the report

---

## Step 2 — Ingest

**Goal:** Turn a video file into clean, predictable media artifacts that analyzers can consume.

**What to build:**
- `ingest.py` — real implementation:
  - Audio extraction: ffmpeg → mono WAV, normalized sample rate (16kHz for Whisper)
  - Frame sampling: ffmpeg → frames at configurable FPS (default 3), output as numbered PNGs to workspace
  - Meeting tile detection: OpenCV-based grid layout segmentation
    - Detect individual participant rectangles in gallery-view recordings
    - Output: tile map (list of bounding boxes, one per participant)
    - Crop individual participant frames for per-person analysis
  - ffmpeg presence check on startup with clear error message
  - Video metadata extraction: duration, resolution, frame rate
- Update `MediaContext` with real paths to extracted media + tile map

**Dependencies:** Step 1 (skeleton)

**Verify:**
- Given a Zoom gallery-view recording:
  - Audio extracted as mono 16kHz WAV
  - Frames sampled at configured FPS
  - Individual participant tiles detected and cropped
  - `MediaContext` populated with correct paths
- Given a non-gallery video (single speaker): still works, tile map has one entry

---

## Step 3 — Transcription + Diarization

**Goal:** Know what was said and who said it.

**What to build:**
- `analyzers/transcript.py` — real implementation:
  - faster-whisper with configurable model size (base/small/medium/large)
  - Word-level timestamps
  - Outputs: full transcript, per-word timing, sentence segmentation
  - Emits signals: one Signal per sentence/segment with text as value
- `analyzers/diarization.py` — real implementation:
  - Speaker diarization via whisperx or pyannote (evaluate both, pick one)
  - Assign speaker labels to transcript segments
  - Speaker role classification: identify primary speaker vs audience by talk-time ratio
  - Emits signals: speaker-change events, per-segment speaker attribution
- Update `MediaContext` to carry the completed transcript + speaker labels for downstream analyzers
- Model weight download with progress messages

**Dependencies:** Step 2 (needs extracted audio)

**Verify:**
- A meeting recording produces a transcript where each segment is attributed to the correct speaker
- Primary speaker is identified
- Word-level timestamps are accurate (spot-check against manual review)
- Model download shows clear progress, not a silent hang

---

## Step 4 — Speaker Analysis

**Goal:** Analyze how the primary speaker delivered. Three independent sub-analyzers.

### 4a — Pacing

**What to build:**
- `analyzers/pacing.py` — real implementation:
  - Words-per-minute over sliding windows (configurable window size)
  - Pause detection: location, duration, categorization (natural breath vs awkward silence vs dramatic pause)
  - Filler word detection: "um", "uh", "like", "you know", "basically", "sort of", "right", "I mean", "actually"
  - Speaking rate variation: consistent vs rushes vs drags
  - Summary metrics: overall WPM, filler count, filler rate, pause count, longest pause
  - Emits signals: `filler_word`, `long_pause`, `rate_change`, `rush`, `drag`

**Dependencies:** Step 3 (needs transcript with word timestamps)

### 4b — Prosody

**What to build:**
- `analyzers/prosody.py` — real implementation:
  - Pitch (F0) tracking via parselmouth over time windows
  - Energy envelope via librosa
  - Vocal variety score: standard deviation of pitch / energy as proxy for monotone detection
  - Energy drop detection: significant dips in volume/energy at specific timestamps
  - Energy spike detection: sudden increases
  - Volume consistency measurement
  - Summary metrics: mean pitch, pitch range, energy range, vocal variety score, monotone segments count
  - Emits signals: `pitch_drop`, `pitch_spike`, `energy_drop`, `energy_spike`, `monotone_segment`, `high_energy_segment`

**Dependencies:** Step 2 (needs extracted audio)

### 4c — Speaker Facial

**What to build:**
- `analyzers/speaker_facial.py` — real implementation:
  - MediaPipe face mesh on speaker frames (speaker tile or full frame if single-speaker)
  - Head pose estimation: facing audience vs looking down/away
  - Eye contact / gaze direction: looking at camera vs off-camera
  - Expressiveness scoring: facial movement variation over time
  - Summary metrics: eye contact %, time facing audience %, expressiveness score
  - Emits signals: `gaze_off_camera`, `gaze_on_camera`, `head_turn_away`, `low_expressiveness`, `high_expressiveness`

**Dependencies:** Step 2 (needs sampled frames) + Step 3 (needs speaker identification to pick the right tile)

**Verify (all of Step 4):**
- A real video produces:
  - Accurate WPM (~120-180 for normal speech)
  - Filler words detected match manual count (within ~80% accuracy)
  - Pitch/energy curves that visually correspond to the audio
  - Eye contact % that roughly matches what a human would judge
- Each analyzer's signals have correct timestamps relative to the video

---

## Step 5 — Audience Analysis

**Goal:** Analyze how the audience received the speech. Three independent sub-analyzers.

### 5a — Audience Engagement (Attention)

**What to build:**
- `analyzers/audience_engagement.py` — real implementation:
  - For each audience member tile (from ingest tile map):
    - MediaPipe head pose estimation: facing speaker/camera vs looking away vs looking down
    - Gaze direction classification: at speaker, at phone/notes, away, undetermined
  - Per-audience-member attention timeline
  - Aggregate attention score over time: % of audience paying attention at each moment
  - Attention drop detection: moments where aggregate attention falls below threshold
  - Attention recovery detection: when audience re-engages
  - Summary metrics: overall attention %, attention drops count, lowest attention moment
  - Emits signals: `attention_drop`, `attention_recovery`, `audience_disengaged`, `audience_engaged`

**Dependencies:** Step 2 (needs frames + tile map) + Step 3 (needs diarization to identify audience members)

### 5b — Audience Reactions (Expressions)

**What to build:**
- `analyzers/audience_reaction.py` — real implementation:
  - For each audience member tile:
    - py-feat Facial Action Unit detection
    - Reaction classification from AUs: nodding (head movement pattern), smiling (AU6+AU12), frowning (AU4), confusion (AU4+AU7), surprise (AU1+AU2+AU5), neutral
  - Per-audience-member reaction timeline
  - Aggregate audience sentiment over time
  - Notable reaction moments: when multiple audience members react simultaneously
  - Summary metrics: positive reaction %, negative reaction %, notable moments count
  - Emits signals: `audience_smile`, `audience_nod`, `audience_frown`, `audience_confusion`, `simultaneous_positive_reaction`, `simultaneous_negative_reaction`

**Dependencies:** Step 2 (needs frames + tile map) + Step 3 (needs diarization to identify audience members)

### 5c — Audience Verbal Interaction

**What to build:**
- Analysis within `diarization.py` or a separate verbal interaction module:
  - From diarized transcript, extract audience speech segments
  - Question detection: identify when audience members ask questions (question marks, rising intonation, interrogative patterns)
  - Interruption detection: when an audience member speaks while the speaker is mid-sentence
  - Agreement/disagreement signals: "yes", "right", "exactly", "no", "I disagree", "but"
  - Q&A quality: did the speaker actually answer the question? (passed to LLM for assessment)
  - Summary metrics: questions count, interruptions count, Q&A success rate
  - Emits signals: `audience_question`, `interruption`, `verbal_agreement`, `verbal_disagreement`

**Dependencies:** Step 3 (needs diarized transcript with speaker labels)

**Verify (all of Step 5):**
- A meeting recording with multiple participants produces:
  - Per-person attention timelines
  - Aggregate attention % that drops during "boring" sections and rises during engaging ones
  - Reaction events that correspond to visible facial movements
  - Questions and interruptions correctly identified from transcript
- Signals have correct timestamps and speaker attribution

---

## Step 6 — Fusion

**Goal:** Merge all signals from all analyzers onto one timeline and find the moments that matter.

**What to build:**
- `fusion.py` — real implementation:
  - Time-sort all signals from all 7+ analyzers into a single `FusedTimeline`
  - Sliding window correlation (configurable window, default ~5 seconds):
    - Detect co-occurring speaker + audience signals within the same window
    - Classify moments:
      - **Strong moment**: high speaker energy + positive audience reactions + high attention
      - **Weak moment**: energy drop + attention drop + fillers
      - **Turning point**: significant shift in audience engagement (positive or negative)
      - **Q&A moment**: audience question + speaker response
    - Assign a combined significance score to each moment
  - Moment deduplication: merge overlapping windows into single moments
  - Long video chunking:
    - Split audio/video into overlapping segments (configurable chunk size, e.g., 10 minutes with 30-second overlap)
    - Analyze each chunk independently through the pipeline
    - Stitch results: align on transcript, deduplicate signals in overlap regions
    - Numeric signals (WPM, attention %) aggregate across chunks
  - Output: `FusedTimeline` with all signals + classified moments

**Dependencies:** Steps 4 + 5 (needs all analyzer signals)

**Verify:**
- The fused timeline contains signals from all analyzers, time-sorted
- Correlated moments are detected: a deliberate "boring section" in test video produces a weak moment signal
- A 1-hour video processes without OOM using chunking
- Chunk boundaries don't produce duplicate signals or mangled transcript segments

---

## Step 7 — LLM Reasoning + Report

**Goal:** Turn signals into coaching. The report reads like a real communication coach analyzed both the speaker and the audience.

**What to build:**
- `rubric.yaml` — real scoring criteria:
  - Dimensions: pacing, prosody, content_clarity, audience_engagement, audience_reception, overall_delivery
  - Per-dimension: what's good, what's concerning, what's bad, with thresholds
  - Coaching tone guidance: constructive, specific, actionable, "signals not emotions"
- `llm/ollama.py` — real Ollama backend:
  - Connects to local Ollama instance
  - Default model: Llama 3.1 8B or Qwen 2.5
  - Handles model availability check + clear error if Ollama isn't running
- `llm/gemini.py` — real Gemini backend:
  - google-generativeai SDK
  - API key from env var (configured in config.yaml)
- `llm/groq.py` — real Groq backend:
  - Groq SDK
  - API key from env var
- LLM prompt engineering:
  - System prompt enforcing "signals not emotions" framing
  - Input: transcript + fused timeline + classified moments + rubric
  - Output: per-dimension scores + coaching narrative
  - Prompt structured to produce:
    - Overall assessment
    - What worked (strongest moments, supported by audience response)
    - What to improve (weakest moments, supported by signals)
    - Timestamped play-by-play of key moments
    - Audience reception summary
    - Specific, actionable next steps
- `report.py` — real Jinja2 rendering:
  - `report.md.j2` — Markdown template with full report structure
  - `report.html.j2` — HTML template with basic styling
  - Report sections:
    - Summary scores (radar/bar visualization in HTML)
    - Executive summary (2-3 sentences)
    - What worked
    - What to improve
    - Key moments timeline
    - Audience reception analysis
    - Detailed metrics appendix
- Backend selection is a one-line config change: `llm.backend: "ollama"` → `llm.backend: "gemini"`

**Dependencies:** Step 6 (needs fused timeline + moments)

**Verify:**
- Report reads like genuine coaching feedback referencing both speaker and audience
- Swapping `ollama` ↔ `gemini` ↔ `groq` in config works without code changes
- "Signals not emotions" framing is consistent (no "the audience was bored")
- All timestamps in the report correspond to actual moments in the video
- HTML report renders cleanly in a browser

---

## Step 8 — Config + Graceful Degradation + Polish

**Goal:** Make it robust for real users on real hardware.

**What to build:**
- Config-driven analyzer toggling:
  - Any analyzer can be disabled in config.yaml
  - Pipeline skips disabled analyzers, report notes what was skipped
  - Recommended presets: `full` (everything), `standard` (no facial), `lite` (transcript + pacing only)
- `requirements-lite.txt`:
  - Transcript + pacing only, no torch dependency
  - Enables curious users to try the MVP without multi-GB downloads
- Startup checks:
  - ffmpeg installed? Clear per-OS install instructions in error message
  - Ollama running? (if ollama backend selected) Clear instructions
  - Sufficient disk space for model weights?
- Model weight management:
  - Download progress bars (not silent hangs)
  - Document disk requirements per model size in README
  - Cache weights so they're downloaded once
- CPU-only mode:
  - Everything works without CUDA, just slower
  - No hard GPU requirement anywhere
- Error resilience:
  - If one analyzer fails (e.g., facial on a video with no faces), the rest still run
  - Report notes which analyzers succeeded/failed and why
- README:
  - Quick start (3 commands to first report)
  - Per-OS ffmpeg install
  - Full vs lite install
  - Config reference
  - Sample report output
  - Hardware requirements

**Dependencies:** Steps 1-7 (polish layer on top of everything)

**Verify:**
- `requirements-lite.txt` installs and runs transcript + pacing without torch
- Disabling facial analysis in config still produces a valid report
- Running on CPU-only machine works end-to-end
- Clear error messages for missing ffmpeg, missing Ollama, missing API keys
- A new user can go from `git clone` to first report in under 5 minutes (excluding model downloads)
