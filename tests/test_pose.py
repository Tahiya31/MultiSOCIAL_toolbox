from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest


def _install_fake_pose_deps():
    cv2 = types.ModuleType("cv2")

    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 4
    cv2.CAP_PROP_FRAME_COUNT = 7
    cv2.CAP_PROP_FPS = 5
    cv2.CAP_PROP_POS_FRAMES = 1
    cv2.CAP_PROP_POS_AVI_RATIO = 2
    cv2.COLOR_BGR2RGB = 0
    cv2.COLOR_GRAY2BGR = 0
    cv2.COLOR_BGRA2BGR = 0
    cv2.INTER_AREA = 0
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.LINE_AA = 0
    cv2.VideoWriter_fourcc = lambda *a: 0

    class FakeVideoCapture:
        _instances = []

        def __init__(self, path):
            self.path = path
            self._index = 0
            self._max = 20
            FakeVideoCapture._instances.append(self)

        def isOpened(self):
            return self._index < self._max

        def read(self):
            if self._index >= self._max:
                return False, None
            frame = np.zeros((48, 64, 3), dtype=np.uint8)
            self._index += 1
            return True, frame

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return self._max
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 64
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 48
            if prop == cv2.CAP_PROP_FPS:
                return 29.97
            return 0

        def set(self, *args, **kwargs):
            return True

        def release(self):
            pass

    class FakeVideoWriter:
        instances = []

        def __init__(self, *args, **kwargs):
            self.opened = True
            self.args = args
            FakeVideoWriter.instances.append(self)

        def isOpened(self):
            return self.opened

        def write(self, frame):
            return None

        def release(self):
            self.opened = False

    cv2.VideoCapture = FakeVideoCapture
    cv2.VideoWriter = FakeVideoWriter
    cv2.cvtColor = lambda img, code: img
    cv2.resize = lambda img, size, **kw: img
    cv2.circle = lambda *a, **k: None
    cv2.line = lambda *a, **k: None
    cv2.rectangle = lambda *a, **k: None
    cv2.putText = lambda *a, **k: None

    mediapipe = types.ModuleType("mediapipe")
    solutions = types.ModuleType("mediapipe.solutions")
    pose_mod = types.ModuleType("mediapipe.solutions.pose")
    drawing_mod = types.ModuleType("mediapipe.solutions.drawing_utils")

    class _Landmarks:
        landmark = []

    class _Result:
        pose_landmarks = None

    class _Pose:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, image_rgb):
            return _Result()

    class _DrawingSpec:
        def __init__(self, *args, **kwargs):
            pass

    pose_mod.Pose = _Pose
    pose_mod.POSE_CONNECTIONS = [(0, 1)]
    drawing_mod.draw_landmarks = lambda *a, **k: None
    drawing_mod.DrawingSpec = _DrawingSpec
    solutions.pose = pose_mod
    solutions.drawing_utils = drawing_mod
    mediapipe.solutions = solutions

    yolov5 = types.ModuleType("yolov5")

    class _YOLO:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, *args, **kwargs):
            return types.SimpleNamespace(xyxy=[[]])

    yolov5.YOLOv5 = _YOLO

    sys.modules["cv2"] = cv2
    sys.modules["mediapipe"] = mediapipe
    sys.modules["mediapipe.solutions"] = solutions
    sys.modules["mediapipe.solutions.pose"] = pose_mod
    sys.modules["mediapipe.solutions.drawing_utils"] = drawing_mod
    sys.modules["yolov5"] = yolov5


@pytest.fixture
def import_pose():
    for name in ("pose", "cv2", "mediapipe", "mediapipe.solutions", "mediapipe.solutions.pose", "mediapipe.solutions.drawing_utils", "yolov5"):
        sys.modules.pop(name, None)
    _install_fake_pose_deps()
    import importlib

    return importlib.import_module("pose")


def _write_min_pose_csv(path, *, frame=0, person_id=0, x=0.5, y=0.5, confidence=1.0):
    names = [
        'Nose', 'Left_eye_inner', 'Left_eye', 'Left_eye_outer', 'Right_eye_inner',
        'Right_eye', 'Right_eye_outer', 'Left_ear', 'Right_ear', 'Mouth_left',
        'Mouth_right', 'Left_shoulder', 'Right_shoulder', 'Left_elbow', 'Right_elbow',
        'Left_wrist', 'Right_wrist', 'Left_pinky', 'Right_pinky', 'Left_index',
        'Right_index', 'Left_thumb', 'Right_thumb', 'Left_hip', 'Right_hip',
        'Left_knee', 'Right_knee', 'Left_ankle', 'Right_ankle', 'Left_heel',
        'Right_heel', 'Left_foot_index', 'Right_foot_index',
    ]
    columns = ['frame', 'person_id']
    row = [frame, person_id]
    for _ in names:
        columns.extend([f"{_}_x", f"{_}_y", f"{_}_z", f"{_}_confidence"])
        row.extend([x, y, 0.0, confidence])
    pd.DataFrame([row], columns=columns).to_csv(path, index=False)


def test_extract_pose_features_cancel_check_stops_early(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    processor = pose.PoseProcessor(str(tmp_path))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    checks = {"n": 0}

    def cancel_check():
        checks["n"] += 1
        return checks["n"] > 2

    result = processor.extract_pose_features(str(video), cancel_check=cancel_check)
    assert result is False
    assert checks["n"] <= 4

def test_extract_pose_features_stride_progress_reaches_100(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    processor = pose.PoseProcessor(str(tmp_path), frame_stride=5)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    progress = []

    result = processor.extract_pose_features(str(video), progress_callback=progress.append)

    assert result is True
    assert progress[-1] == 100
    assert max(progress) == 100


def test_extract_pose_features_stride_status_reaches_source_end(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    processor = pose.PoseProcessor(str(tmp_path), frame_stride=5, status_callback=lambda msg: messages.append(msg))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    messages = []

    result = processor.extract_pose_features(str(video))

    assert result is True
    assert messages[-1].endswith("(Source frame 20/20)")


def test_extract_multiperson_no_roi_stride_progress_reaches_100(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    processor = pose.PoseProcessor(str(tmp_path), frame_stride=4)
    processor.set_multi_person_mode(True)
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    progress = []

    result = processor.extract_pose_features(str(video), progress_callback=progress.append)

    assert result is True
    assert progress[-1] == 100
    assert max(progress) == 100


def test_find_pose_csv_paths_filters_by_mode(import_pose, tmp_path):
    pose = import_pose
    _write_min_pose_csv(tmp_path / "clip_ID_0.csv")
    _write_min_pose_csv(tmp_path / "clip_multi_ID_0.csv")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    single = pose.find_pose_csv_paths(str(tmp_path), str(video), multi_person=False)
    multi = pose.find_pose_csv_paths(str(tmp_path), str(video), multi_person=True)

    assert single == [str(tmp_path / "clip_ID_0.csv")]
    assert multi == [str(tmp_path / "clip_multi_ID_0.csv")]


def test_embed_pose_video_cancel_check_stops_early(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_min_pose_csv(csv_dir / "clip_ID_0.csv")
    processor = pose.PoseProcessor(str(csv_dir), output_video_folder=str(out_dir))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    checks = {"n": 0}

    def cancel_check():
        checks["n"] += 1
        return checks["n"] > 2

    result = processor.embed_pose_video(str(video), cancel_check=cancel_check)
    assert result is False

def test_embed_pose_video_preserves_fractional_fps(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_min_pose_csv(csv_dir / "clip_ID_0.csv")
    processor = pose.PoseProcessor(str(csv_dir), output_video_folder=str(out_dir))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    result = processor.embed_pose_video(str(video))

    assert result == str(out_dir / "clip_pose.mp4")
    assert pose.cv2.VideoWriter.instances[0].args[2] == 29.97


def test_embed_pose_video_respects_selected_mode(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_min_pose_csv(csv_dir / "clip_ID_0.csv")
    _write_min_pose_csv(csv_dir / "clip_multi_ID_0.csv")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    processor = pose.PoseProcessor(str(csv_dir), output_video_folder=str(out_dir))
    assert processor.embed_pose_video(str(video)) == str(out_dir / "clip_pose.mp4")

    processor.set_multi_person_mode(True)
    assert processor.embed_pose_video(str(video)) == str(out_dir / "clip_multi_pose.mp4")


def test_embed_pose_video_returns_none_when_selected_mode_csv_missing(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_min_pose_csv(csv_dir / "clip_multi_ID_0.csv")
    processor = pose.PoseProcessor(str(csv_dir), output_video_folder=str(out_dir))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    assert processor.embed_pose_video(str(video)) is None


def test_embed_pose_video_returns_none_without_csv(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    processor = pose.PoseProcessor(str(csv_dir), output_video_folder=str(out_dir))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    result = processor.embed_pose_video(str(video))
    assert result is None


def test_csv_row_to_pts_maps_normalized_coordinates(import_pose):
    pose = import_pose
    row = {"Nose_x": 0.5, "Nose_y": 0.25, "Nose_confidence": 0.8}
    for name in pose.PoseProcessor._landmark_column_names():
        if name != "Nose":
            row[f"{name}_x"] = np.nan
            row[f"{name}_y"] = np.nan
            row[f"{name}_confidence"] = 0.0
        elif name == "Nose":
            row.setdefault("Nose_z", 0.0)

    pts = pose.PoseProcessor._csv_row_to_pts(row, 100, 200)
    assert pts[0] == (50, 50, 0.8)
    assert pts[1] is None


def test_roi_spawn_assigns_monotonic_ids(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    processor = pose.PoseProcessor(str(tmp_path))
    processor._next_pid = 0

    def _make_box(x1, y1, x2, y2):
        values = [x1, y1, x2, y2, 0.9, 0]

        class _Slice:
            def __init__(self, data):
                self._data = data

            def int(self):
                return self

            def tolist(self):
                return list(self._data)

        class _Box:
            def __getitem__(self, key):
                if isinstance(key, slice):
                    return _Slice(values[key])
                return values[key]

        return _Box()

    processor.yolo = types.SimpleNamespace(
        predict=lambda *a, **k: types.SimpleNamespace(
            xyxy=[[_make_box(0, 0, 10, 10), _make_box(20, 20, 30, 30)]]
        )
    )
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    rois = processor._seed_rois_if_needed(image, 40, 40, [], margin_ratio=0.0)
    assert [r["id"] for r in rois] == [0, 1]


def test_dedup_preserves_survivor_ids(import_pose, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    processor = pose.PoseProcessor("/tmp")
    processor.frame_threshold = 99
    processor.yolo = types.SimpleNamespace(predict=lambda *a, **k: types.SimpleNamespace(xyxy=[[]]))

    class _FakePose:
        def process(self, cropped):
            return types.SimpleNamespace(pose_landmarks=None)

        def close(self):
            pass

    locked_rois = [
        {"id": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "lost": 5, "pose": _FakePose(), "overlap_streak": 3},
        {"id": 7, "x1": 1, "y1": 1, "x2": 11, "y2": 11, "lost": 0, "pose": _FakePose(), "overlap_streak": 3},
    ]
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    outputs, remaining = processor._process_multiperson_frame(image, 20, 20, locked_rois)
    assert outputs == []
    assert len(remaining) == 1
    assert remaining[0]["id"] == 7


def test_scaled_color_applies_confidence_floor(import_pose):
    pose = import_pose
    P = pose.PoseProcessor
    # full confidence keeps the base color
    assert P._scaled_color((0, 200, 100), 1.0) == (0, 200, 100)
    # zero confidence clamps to the floor (0.25), not black
    assert P._scaled_color((0, 200, 100), 0.0) == (0, 50, 25)
    # intensity is monotonic in confidence
    low = P._scaled_color((0, 200, 0), 0.0)[1]
    mid = P._scaled_color((0, 200, 0), 0.5)[1]
    high = P._scaled_color((0, 200, 0), 1.0)[1]
    assert low < mid < high


def test_extraction_stride_reads_sidecar(import_pose, tmp_path):
    pose = import_pose
    (tmp_path / "clip_meta.json").write_text('{"frame_stride": 3}')
    processor = pose.PoseProcessor(str(tmp_path), frame_stride=1)
    # sidecar value wins over the embed-time stride
    assert processor._extraction_stride(str(tmp_path / "clip.mp4"), is_multi=False) == 3
    # missing sidecar falls back to the embed-time stride
    assert processor._extraction_stride(str(tmp_path / "other.mp4"), is_multi=False) == 1


def test_mux_audio_replaces_file_with_correct_command(import_pose, monkeypatch):
    pose = import_pose
    monkeypatch.setenv("MULTISOCIAL_FFMPEG_EXE", "/fake/ffmpeg")
    monkeypatch.setattr(pose.os.path, "exists", lambda p: True)

    captured = {}

    class FakeProc:
        returncode = 0

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        return FakeProc()

    def fake_replace(src, dst):
        captured["replace"] = (src, dst)

    monkeypatch.setattr(pose.subprocess, "run", fake_run)
    monkeypatch.setattr(pose.os, "replace", fake_replace)

    pose._mux_audio_into_video("/out/clip_pose.mp4", "/in/clip.mp4")

    cmd = captured["cmd"]
    assert cmd[0] == "/fake/ffmpeg"
    # video copied (no re-encode), audio from the second input, source optional
    assert "-c:v" in cmd and cmd[cmd.index("-c:v") + 1] == "copy"
    assert "1:a:0?" in cmd
    assert "-shortest" not in cmd  # keep every rendered pose frame
    assert captured["replace"] == ("/out/clip_pose.mp4.muxtmp.mp4", "/out/clip_pose.mp4")


def test_mux_audio_noop_without_ffmpeg(import_pose, monkeypatch):
    pose = import_pose
    monkeypatch.delenv("MULTISOCIAL_FFMPEG_EXE", raising=False)
    monkeypatch.setattr(pose.shutil, "which", lambda name: None)

    called = {"run": False}

    def fake_run(*a, **k):
        called["run"] = True

    monkeypatch.setattr(pose.subprocess, "run", fake_run)

    messages = []
    pose._mux_audio_into_video("/out/clip_pose.mp4", "/in/clip.mp4", messages.append)

    assert called["run"] is False
    assert messages and "without audio" in messages[0]
