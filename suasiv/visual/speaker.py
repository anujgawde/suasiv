from __future__ import annotations

import math
import re
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from rich.console import Console

from suasiv.config import SpeakerVisualConfig
from suasiv.ingest import detect_tiles
from suasiv.schema import MediaContext, Signal, SignalType, Tile

console = Console()

_BaseOptions = mp.tasks.BaseOptions
_FaceLandmarker = mp.tasks.vision.FaceLandmarker
_FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
_RunningMode = mp.tasks.vision.RunningMode

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_CACHE_DIR = Path.home() / ".cache" / "suasiv" / "models"

_LEFT_IRIS = 468
_RIGHT_IRIS = 473
_LEFT_EYE_OUTER = 33
_LEFT_EYE_INNER = 133
_RIGHT_EYE_OUTER = 263
_RIGHT_EYE_INNER = 362

_YAW_EYE_CONTACT = 15.0
_YAW_HEAD_TURN = 25.0
_PITCH_EYE_CONTACT = 15.0
_GAZE_CENTER = 0.12
_GAZE_BREAK = 0.15
_EYE_CONTACT_FRAC = 0.6
_GAZE_BREAK_FRAC = 0.4
_UPSCALE_MIN_PX = 80
_UPSCALE_FACTOR = 3


def _ensure_model() -> Path:
    path = _CACHE_DIR / "face_landmarker.task"
    if path.exists():
        return path
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    console.print("[dim]downloading face_landmarker.task (~4 MB)...[/dim]")
    try:
        urllib.request.urlretrieve(_MODEL_URL, str(path))
    except Exception as exc:
        if path.exists():
            path.unlink()
        raise RuntimeError(
            f"failed to download face_landmarker model: {exc}\n"
            f"download manually: {_MODEL_URL} → {path}"
        ) from exc
    return path


def _create_landmarker(confidence: float) -> _FaceLandmarker:
    model_path = _ensure_model()
    return _FaceLandmarker.create_from_options(_FaceLandmarkerOptions(
        base_options=_BaseOptions(model_asset_path=str(model_path)),
        running_mode=_RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=confidence,
        min_face_presence_confidence=confidence,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    ))


def _euler_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """Yaw, pitch, roll in degrees from a 3×3 rotation matrix."""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[2, 1], R[2, 2])
    else:
        yaw = 0.0
        pitch = math.atan2(-R[2, 0], sy)
        roll = math.atan2(-R[1, 2], R[1, 1])
    return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)


def _iris_ratio(landmarks, outer: int, inner: int, iris: int) -> float:
    """Iris position between eye corners: 0=outer, 1=inner, 0.5=centered."""
    o = np.array([landmarks[outer].x, landmarks[outer].y])
    i = np.array([landmarks[inner].x, landmarks[inner].y])
    c = np.array([landmarks[iris].x, landmarks[iris].y])
    v = i - o
    d = np.dot(v, v)
    if d < 1e-10:
        return 0.5
    return float(np.dot(c - o, v) / d)


def _gaze_offset(landmarks) -> float:
    """Average iris deviation from center. 0=camera, larger=away."""
    left = abs(
        _iris_ratio(landmarks, _LEFT_EYE_OUTER, _LEFT_EYE_INNER, _LEFT_IRIS) - 0.5
    )
    right = abs(
        _iris_ratio(landmarks, _RIGHT_EYE_OUTER, _RIGHT_EYE_INNER, _RIGHT_IRIS) - 0.5
    )
    return (left + right) / 2


def _crop_tile(frame: np.ndarray, tile: Tile) -> np.ndarray:
    h, w = frame.shape[:2]
    crop = frame[
        max(0, tile.y) : min(h, tile.y + tile.h),
        max(0, tile.x) : min(w, tile.x + tile.w),
    ]
    if min(crop.shape[:2]) < _UPSCALE_MIN_PX:
        crop = cv2.resize(
            crop,
            (crop.shape[1] * _UPSCALE_FACTOR, crop.shape[0] * _UPSCALE_FACTOR),
            interpolation=cv2.INTER_CUBIC,
        )
    return crop


def _find_reference_photo(ref_dir: str) -> Path | None:
    d = Path(ref_dir)
    if not d.is_dir():
        return None
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        photos = sorted(d.glob(pattern))
        if photos:
            return photos[0]
    return None


def _load_identifier(ref_dir: str):
    """Load insightface and compute reference embedding.

    Returns (embedding, app) or (None, None) if unavailable.
    """
    photo = _find_reference_photo(ref_dir)
    if photo is None:
        return None, None

    try:
        from insightface.app import FaceAnalysis
    except ImportError:
        console.print(
            "[yellow]insightface not installed — skipping speaker identification[/yellow]"
        )
        return None, None

    console.print("[dim]loading speaker identification model...[/dim]")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(str(photo))
    if img is None:
        return None, None

    faces = app.get(img)
    if not faces:
        console.print(f"[yellow]no face in reference photo {photo.name}[/yellow]")
        return None, None

    best = max(
        faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
    )
    return best.embedding, app


def _identify_speaker(
    frame: np.ndarray, tiles: list[Tile], ref_emb: np.ndarray, app
) -> int:
    """Return the tile index whose face best matches the reference embedding."""
    best_idx, best_sim = 0, -1.0
    for i, tile in enumerate(tiles):
        crop = _crop_tile(frame, tile)
        faces = app.get(crop)
        if not faces:
            continue
        face = max(faces, key=lambda f: f.det_score)
        sim = float(
            np.dot(ref_emb, face.embedding)
            / (np.linalg.norm(ref_emb) * np.linalg.norm(face.embedding))
        )
        if sim > best_sim:
            best_sim = sim
            best_idx = i
    return best_idx


def _frame_number(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 0


def analyze_speaker_visual(
    ctx: MediaContext, config: SpeakerVisualConfig
) -> list[Signal]:
    """Gaze and head-pose analysis on the speaker across sampled frames."""
    if not config.enabled or not ctx.frames_dir or not ctx.frames_dir.exists():
        return []

    frames = sorted(ctx.frames_dir.glob("*.png"))
    if not frames:
        return []

    sample_fps = len(frames) / ctx.duration if ctx.duration > 0 else 3.0
    landmarker = _create_landmarker(config.min_face_confidence)

    gallery = len(ctx.tiles) > 1
    ref_emb, face_app = None, None
    if gallery:
        ref_emb, face_app = _load_identifier(config.reference_dir)
        if ref_emb is None:
            gallery = False

    speaker_idx = 0
    speaker_tile = ctx.tiles[0] if ctx.tiles else Tile(
        x=0, y=0, w=ctx.width, h=ctx.height,
    )
    prev_count = len(ctx.tiles)
    min_area = min((t.w * t.h for t in ctx.tiles), default=5000) // 2

    measurements: list[tuple[float, dict | None]] = []

    for frame_path in frames:
        t = (_frame_number(frame_path) - 1) / sample_fps
        frame = cv2.imread(str(frame_path))
        if frame is None:
            measurements.append((t, None))
            continue

        if gallery:
            tiles = detect_tiles(frame_path, min_area, gutter_intensity=30.0)
            if not tiles:
                measurements.append((t, None))
                continue
            if len(tiles) != prev_count:
                speaker_idx = _identify_speaker(frame, tiles, ref_emb, face_app)
                prev_count = len(tiles)
            tile = tiles[min(speaker_idx, len(tiles) - 1)]
        else:
            tile = speaker_tile

        crop = _crop_tile(frame, tile)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.face_landmarks:
            measurements.append((t, None))
            continue

        landmarks = result.face_landmarks[0]

        yaw, pitch, roll = 0.0, 0.0, 0.0
        if result.facial_transformation_matrixes:
            R = np.array(result.facial_transformation_matrixes[0])[:3, :3]
            yaw, pitch, roll = _euler_from_rotation(R)

        gaze = _gaze_offset(landmarks)

        measurements.append((t, {
            "yaw": yaw, "pitch": pitch, "roll": roll,
            "gaze_offset": gaze,
            "eye_contact": (
                abs(yaw) < _YAW_EYE_CONTACT
                and abs(pitch) < _PITCH_EYE_CONTACT
                and gaze < _GAZE_CENTER
            ),
            "gaze_break": gaze >= _GAZE_BREAK or abs(yaw) >= _YAW_EYE_CONTACT,
            "head_turn": abs(yaw) >= _YAW_HEAD_TURN,
        }))

    landmarker.close()
    return _aggregate(measurements, config.window_seconds)


def _aggregate(
    measurements: list[tuple[float, dict | None]], window: float
) -> list[Signal]:
    """Collapse per-frame measurements into windowed signals."""
    if not measurements:
        return []

    signals: list[Signal] = []
    t_start = measurements[0][0]
    t_end = measurements[-1][0]
    t = t_start

    while t < t_end:
        w_end = t + window
        in_w = [m for ts, m in measurements if t <= ts < w_end and m is not None]
        if not in_w:
            t += window
            continue

        n = len(in_w)
        ec = sum(1 for m in in_w if m["eye_contact"]) / n
        gb = sum(1 for m in in_w if m["gaze_break"]) / n
        ht = sum(1 for m in in_w if m["head_turn"]) / n
        avg_yaw = sum(abs(m["yaw"]) for m in in_w) / n
        avg_gaze = sum(m["gaze_offset"] for m in in_w) / n

        meta = {
            "eye_contact_frac": round(ec, 3),
            "gaze_break_frac": round(gb, 3),
            "avg_yaw_deg": round(avg_yaw, 1),
            "avg_gaze_offset": round(avg_gaze, 3),
            "frames": n,
        }

        if ec >= _EYE_CONTACT_FRAC:
            signals.append(Signal(
                type=SignalType.EYE_CONTACT, source="speaker_visual",
                start=t, end=w_end, value=ec, metadata=meta,
            ))
        elif gb >= _GAZE_BREAK_FRAC:
            signals.append(Signal(
                type=SignalType.GAZE_BREAK, source="speaker_visual",
                start=t, end=w_end, value=gb, metadata=meta,
            ))

        if ht > 0:
            signals.append(Signal(
                type=SignalType.HEAD_TURN, source="speaker_visual",
                start=t, end=w_end, value=ht, metadata=meta,
            ))

        t += window

    return signals
