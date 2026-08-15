from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from suasiv.analyzers.base import Analyzer
from suasiv.context import MediaContext
from suasiv.schema import AnalyzerResult, Signal

console = Console()

_MODEL_POINTS_RAW = (
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
)

_POSE_LANDMARKS = [1, 199, 33, 263, 61, 291]

_LEFT_IRIS = [473, 474, 475, 476, 477]
_RIGHT_IRIS = [468, 469, 470, 471, 472]
_LEFT_EYE_INNER = 133
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263

_EXPRESSION_LANDMARKS = [
    13, 14,
    70, 63, 105, 66, 107,
    300, 293, 334, 296, 336,
]


class SpeakerFacialAnalyzer(Analyzer):
    name = "speaker_facial"
    requires = {"frames"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        import numpy as np

        try:
            import mediapipe as mp  # noqa: F841
        except ImportError:
            raise RuntimeError(
                "mediapipe is required for facial analysis.\n"
                "Install: pip install 'suasiv[full]' or pip install mediapipe"
            )
        try:
            import cv2  # noqa: F841
        except ImportError:
            raise RuntimeError(
                "opencv-python is required for facial analysis.\n"
                "Install: pip install 'suasiv[full]' or pip install opencv-python"
            )

        frames_dir = _get_speaker_frames_dir(ctx)
        if frames_dir is None:
            return AnalyzerResult(
                analyzer=self.name, signals=[], summary={"error": "no frames"}
            )

        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            return AnalyzerResult(
                analyzer=self.name, signals=[], summary={"error": "no frame files"}
            )

        fps = ctx.config.ingest.frame_fps or 3
        console.print(f"    Analyzing {len(frames)} frames for facial signals...")

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )

        frame_data: list[dict] = []
        prev_landmarks: np.ndarray | None = None

        for frame_path in frames:
            frame_idx = int(frame_path.stem.split("_")[1])
            timestamp = (frame_idx - 1) / fps

            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                frame_data.append(
                    {
                        "time": round(timestamp, 3),
                        "face_detected": False,
                        "yaw": None,
                        "pitch": None,
                        "facing_audience": False,
                        "gaze_on_camera": False,
                        "expressiveness": 0.0,
                    }
                )
                prev_landmarks = None
                continue

            landmarks = results.multi_face_landmarks[0].landmark

            yaw, pitch_angle = _estimate_head_pose(landmarks, w, h)
            gaze_on_camera = _estimate_gaze(landmarks)

            current_key = _extract_expression_coords(landmarks)
            expressiveness = 0.0
            if prev_landmarks is not None:
                expressiveness = float(np.mean(np.abs(current_key - prev_landmarks)))
            prev_landmarks = current_key

            facing = abs(yaw) < 15 and abs(pitch_angle) < 15

            frame_data.append(
                {
                    "time": round(timestamp, 3),
                    "face_detected": True,
                    "yaw": round(yaw, 1),
                    "pitch": round(pitch_angle, 1),
                    "facing_audience": facing,
                    "gaze_on_camera": gaze_on_camera,
                    "expressiveness": round(expressiveness, 4),
                }
            )

        face_mesh.close()

        detected = [f for f in frame_data if f["face_detected"]]
        if not detected:
            return AnalyzerResult(
                analyzer=self.name, signals=[], summary={"error": "no faces detected"}
            )

        eye_contact_frames = sum(1 for f in detected if f["gaze_on_camera"])
        facing_frames = sum(1 for f in detected if f["facing_audience"])
        eye_contact_pct = eye_contact_frames / len(detected)
        facing_pct = facing_frames / len(detected)

        expr_values = [f["expressiveness"] for f in detected if f["expressiveness"] > 0]
        mean_expr = float(np.mean(expr_values)) if expr_values else 0.0
        expr_score = min(1.0, mean_expr / 0.02)

        signals = _generate_signals(frame_data, fps, ctx.primary_speaker)

        facial_path = ctx.workspace / "speaker_facial.json"
        with open(facial_path, "w") as f:
            json.dump(
                {
                    "eye_contact_pct": round(eye_contact_pct, 3),
                    "facing_audience_pct": round(facing_pct, 3),
                    "expressiveness_score": round(expr_score, 3),
                    "frames_analyzed": len(frames),
                    "faces_detected": len(detected),
                },
                f,
                indent=2,
            )

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "eye_contact_pct": round(eye_contact_pct, 3),
                "facing_audience_pct": round(facing_pct, 3),
                "expressiveness_score": round(expr_score, 3),
            },
        )


def _get_speaker_frames_dir(ctx: MediaContext) -> Path | None:
    if ctx.tiles_dir and ctx.speaker_tile_index is not None:
        tile_dir = ctx.tiles_dir / f"tile_{ctx.speaker_tile_index:02d}"
        if tile_dir.exists():
            return tile_dir
    return ctx.frames_dir


def _estimate_head_pose(landmarks, img_w: int, img_h: int) -> tuple[float, float]:
    import cv2
    import numpy as np

    model_points = np.array(_MODEL_POINTS_RAW, dtype=np.float64)
    image_points = np.array(
        [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in _POSE_LANDMARKS],
        dtype=np.float64,
    )

    focal_length = float(img_w)
    center = (img_w / 2.0, img_h / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    sy = np.sqrt(rotation_mat[0, 0] ** 2 + rotation_mat[1, 0] ** 2)
    yaw = float(np.degrees(np.arctan2(-rotation_mat[2, 0], sy)))
    pitch = float(np.degrees(np.arctan2(rotation_mat[2, 1], rotation_mat[2, 2])))

    return yaw, pitch


def _estimate_gaze(landmarks) -> bool:
    import numpy as np

    def iris_ratio(iris_indices, inner_idx, outer_idx):
        iris_x = np.mean([landmarks[i].x for i in iris_indices])
        inner_x = landmarks[inner_idx].x
        outer_x = landmarks[outer_idx].x
        eye_width = abs(outer_x - inner_x)
        if eye_width < 0.001:
            return 0.5
        return abs(iris_x - outer_x) / eye_width

    left_ratio = iris_ratio(_LEFT_IRIS, _LEFT_EYE_INNER, _LEFT_EYE_OUTER)
    right_ratio = iris_ratio(_RIGHT_IRIS, _RIGHT_EYE_INNER, _RIGHT_EYE_OUTER)
    avg = (left_ratio + right_ratio) / 2
    return 0.3 <= avg <= 0.7


def _extract_expression_coords(landmarks):
    import numpy as np

    coords: list[float] = []
    for i in _EXPRESSION_LANDMARKS:
        coords.extend([landmarks[i].x, landmarks[i].y])
    return np.array(coords)


def _generate_signals(
    frame_data: list[dict], fps: int, primary_speaker: str | None
) -> list[Signal]:
    import numpy as np

    signals: list[Signal] = []
    window = max(3, fps)

    for i in range(0, len(frame_data), window):
        chunk = frame_data[i : i + window]
        detected = [f for f in chunk if f["face_detected"]]
        if not detected:
            continue

        start = chunk[0]["time"]
        end = chunk[-1]["time"]

        on_camera = sum(1 for f in detected if f["gaze_on_camera"])
        gaze_ratio = on_camera / len(detected)

        if gaze_ratio >= 0.7:
            signals.append(
                Signal(
                    analyzer="speaker_facial",
                    type="gaze_on_camera",
                    start=start,
                    end=end,
                    confidence=round(gaze_ratio, 2),
                    speaker=primary_speaker,
                )
            )
        elif gaze_ratio <= 0.3:
            signals.append(
                Signal(
                    analyzer="speaker_facial",
                    type="gaze_off_camera",
                    start=start,
                    end=end,
                    confidence=round(1.0 - gaze_ratio, 2),
                    speaker=primary_speaker,
                )
            )

        facing = sum(1 for f in detected if f["facing_audience"])
        facing_ratio = facing / len(detected)
        if facing_ratio <= 0.3:
            signals.append(
                Signal(
                    analyzer="speaker_facial",
                    type="head_turn_away",
                    start=start,
                    end=end,
                    confidence=round(1.0 - facing_ratio, 2),
                    speaker=primary_speaker,
                )
            )

        expr_vals = [f["expressiveness"] for f in detected if f["expressiveness"] > 0]
        if expr_vals:
            avg_expr = float(np.mean(expr_vals))
            if avg_expr < 0.003:
                signals.append(
                    Signal(
                        analyzer="speaker_facial",
                        type="low_expressiveness",
                        start=start,
                        end=end,
                        value=round(avg_expr, 4),
                        confidence=0.7,
                        speaker=primary_speaker,
                    )
                )
            elif avg_expr > 0.015:
                signals.append(
                    Signal(
                        analyzer="speaker_facial",
                        type="high_expressiveness",
                        start=start,
                        end=end,
                        value=round(avg_expr, 4),
                        confidence=0.8,
                        speaker=primary_speaker,
                    )
                )

    return signals
