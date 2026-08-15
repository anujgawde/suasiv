from __future__ import annotations

import json

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

_ATTENTION_YAW = 20
_ATTENTION_PITCH = 20
_DROP_THRESH = 0.5
_RECOVERY_THRESH = 0.7


class AudienceEngagementAnalyzer(Analyzer):
    name = "audience_engagement"
    requires = {"frames", "diarization"}

    def analyze(self, ctx: MediaContext) -> AnalyzerResult:
        import numpy as np  # noqa: F841

        try:
            import mediapipe as mp  # noqa: F841
        except ImportError:
            raise RuntimeError(
                "mediapipe is required for audience engagement analysis.\n"
                "Install: pip install 'suasiv[full]' or pip install mediapipe"
            )
        try:
            import cv2  # noqa: F841
        except ImportError:
            raise RuntimeError(
                "opencv-python is required for audience engagement analysis.\n"
                "Install: pip install 'suasiv[full]' or pip install opencv-python"
            )

        audience_tiles = _get_audience_tile_dirs(ctx)
        if not audience_tiles:
            return AnalyzerResult(
                analyzer=self.name,
                signals=[],
                summary={"error": "no audience tiles detected"},
            )

        fps = ctx.config.ingest.frame_fps or 3
        settings = ctx.config.analyzer_settings(self.name)
        analyzer_fps = settings.fps or fps
        frame_step = max(1, round(fps / analyzer_fps))

        console.print(
            f"    Analyzing audience attention across {len(audience_tiles)} tile(s)..."
        )

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )

        tile_attention: dict[int, list[dict]] = {}

        for tile_idx, tile_dir in audience_tiles:
            frames = sorted(tile_dir.glob("frame_*.png"))
            tile_data: list[dict] = []

            for fi, frame_path in enumerate(frames):
                if fi % frame_step != 0:
                    continue

                frame_num = int(frame_path.stem.split("_")[1])
                timestamp = (frame_num - 1) / fps

                img = cv2.imread(str(frame_path))
                if img is None:
                    continue

                h, w = img.shape[:2]
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                attentive = False
                gaze = "undetermined"

                if results.multi_face_landmarks:
                    landmarks = results.multi_face_landmarks[0].landmark
                    yaw, pitch = _estimate_head_pose(landmarks, w, h)

                    if abs(yaw) < 15 and abs(pitch) < 15:
                        attentive = True
                        gaze = "at_speaker"
                    elif abs(yaw) < _ATTENTION_YAW and abs(pitch) > 20:
                        gaze = "looking_down"
                    elif abs(yaw) > 30:
                        gaze = "looking_away"
                    else:
                        attentive = abs(yaw) < _ATTENTION_YAW and abs(pitch) < _ATTENTION_PITCH

                tile_data.append(
                    {
                        "time": round(timestamp, 3),
                        "attentive": attentive,
                        "gaze": gaze,
                    }
                )

            tile_attention[tile_idx] = tile_data

        face_mesh.close()

        attention_timeline = _build_attention_timeline(tile_attention)
        if not attention_timeline:
            return AnalyzerResult(
                analyzer=self.name, signals=[], summary={"error": "no attention data"}
            )

        signals = _detect_attention_signals(attention_timeline, self.name)

        pcts = [a["attention_pct"] for a in attention_timeline]
        overall_attention = float(np.mean(pcts)) if pcts else 0.0
        drops = sum(1 for s in signals if s.type == "attention_drop")
        lowest_idx = int(np.argmin(pcts)) if pcts else 0
        lowest_time = attention_timeline[lowest_idx]["time"] if attention_timeline else 0.0

        engagement_path = ctx.workspace / "audience_engagement.json"
        with open(engagement_path, "w") as f:
            json.dump(
                {
                    "overall_attention_pct": round(overall_attention, 3),
                    "attention_drops": drops,
                    "lowest_attention_moment": round(lowest_time, 3),
                    "tiles_analyzed": len(audience_tiles),
                    "timeline_points": len(attention_timeline),
                },
                f,
                indent=2,
            )

        return AnalyzerResult(
            analyzer=self.name,
            signals=signals,
            summary={
                "overall_attention_pct": round(overall_attention, 3),
                "attention_drops": drops,
                "lowest_attention_moment": round(lowest_time, 3),
            },
        )


def _get_audience_tile_dirs(ctx: MediaContext) -> list[tuple[int, object]]:
    if not ctx.tiles_dir or len(ctx.tiles) <= 1:
        return []
    dirs: list[tuple[int, object]] = []
    for tile in ctx.tiles:
        if tile.index == ctx.speaker_tile_index:
            continue
        tile_dir = ctx.tiles_dir / f"tile_{tile.index:02d}"
        if tile_dir.exists() and any(tile_dir.glob("frame_*.png")):
            dirs.append((tile.index, tile_dir))
    return dirs


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


def _build_attention_timeline(
    tile_attention: dict[int, list[dict]],
) -> list[dict]:
    if not tile_attention:
        return []

    ref = next(iter(tile_attention.values()))
    timeline: list[dict] = []

    for i in range(len(ref)):
        t = ref[i]["time"]
        attentive = 0
        total = 0
        for tile_data in tile_attention.values():
            if i < len(tile_data):
                total += 1
                if tile_data[i]["attentive"]:
                    attentive += 1
        pct = attentive / total if total > 0 else 0.0
        timeline.append({"time": t, "attention_pct": round(pct, 3)})

    return timeline


def _detect_attention_signals(
    timeline: list[dict], analyzer_name: str
) -> list[Signal]:
    import numpy as np

    signals: list[Signal] = []
    in_drop = False
    drop_start = 0.0

    for i, point in enumerate(timeline):
        pct = point["attention_pct"]

        if not in_drop and pct < _DROP_THRESH:
            in_drop = True
            drop_start = point["time"]
        elif in_drop and pct >= _RECOVERY_THRESH:
            signals.append(
                Signal(
                    analyzer=analyzer_name,
                    type="attention_drop",
                    start=drop_start,
                    end=point["time"],
                    value={"lowest_pct": min(
                        p["attention_pct"]
                        for p in timeline
                        if drop_start <= p["time"] <= point["time"]
                    )},
                )
            )
            signals.append(
                Signal(
                    analyzer=analyzer_name,
                    type="attention_recovery",
                    start=point["time"],
                    end=point["time"],
                    value={"recovered_to": pct},
                )
            )
            in_drop = False

    if in_drop:
        signals.append(
            Signal(
                analyzer=analyzer_name,
                type="attention_drop",
                start=drop_start,
                end=timeline[-1]["time"],
                value={"lowest_pct": min(
                    p["attention_pct"]
                    for p in timeline
                    if p["time"] >= drop_start
                )},
            )
        )

    # Sustained engagement segments (>= 80% for 5+ consecutive points)
    run_start = 0
    for i, point in enumerate(timeline):
        if point["attention_pct"] >= 0.8:
            if run_start == 0:
                run_start = i
        else:
            if i - run_start >= 5:
                signals.append(
                    Signal(
                        analyzer=analyzer_name,
                        type="audience_engaged",
                        start=timeline[run_start]["time"],
                        end=timeline[i - 1]["time"],
                        value={"attention_pct": round(
                            np.mean([
                                timeline[j]["attention_pct"]
                                for j in range(run_start, i)
                            ]),
                            3,
                        )},
                    )
                )
            run_start = i + 1

    if len(timeline) - run_start >= 5:
        signals.append(
            Signal(
                analyzer=analyzer_name,
                type="audience_engaged",
                start=timeline[run_start]["time"],
                end=timeline[-1]["time"],
                value={"attention_pct": round(
                    np.mean([
                        timeline[j]["attention_pct"]
                        for j in range(run_start, len(timeline))
                    ]),
                    3,
                )},
            )
        )

    return signals
