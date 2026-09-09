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


def _yolo_person_box(x1, y1, x2, y2, confidence=0.9):
    values = [x1, y1, x2, y2, confidence, 0]

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


def test_extract_pose_features_writes_all_single_person_rows(import_pose, tmp_path):
    pose = import_pose
    processor = pose.PoseProcessor(str(tmp_path))
    landmarks = [types.SimpleNamespace(x=0.1, y=0.2, z=0.3, visibility=0.9) for _ in range(33)]
    processor.pose.process = lambda image: types.SimpleNamespace(
        pose_landmarks=types.SimpleNamespace(landmark=landmarks)
    )
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    assert processor.extract_pose_features(str(video)) is True

    rows = pd.read_csv(tmp_path / "clip_ID_0.csv")
    assert len(rows) == 20
    assert rows["frame"].tolist() == list(range(20))

def test_frozen_yolov5_weights_missing_does_not_download(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    runtime_path = tmp_path / "runtime" / "assets" / "yolov5s.pt"
    bundled_path = tmp_path / "bundle" / "assets" / "yolov5s.pt"

    monkeypatch.setattr(pose, "get_yolov5_weights_path", lambda: str(runtime_path))
    monkeypatch.setattr(pose.runtime_services, "is_frozen_runtime", lambda: True)
    monkeypatch.setattr(pose.runtime_services, "resource_path", lambda *parts: str(bundled_path))

    class _Requests:
        @staticmethod
        def get(*args, **kwargs):
            raise AssertionError("frozen builds must not download YOLO weights")

    monkeypatch.setitem(sys.modules, "requests", _Requests)

    with pytest.raises(RuntimeError, match="packaged app"):
        pose.ensure_yolov5_weights()


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
    assert processor._max_lost_frames == int((29.97 / 5) * processor.MAX_LOST_SECONDS)
    assert processor._retired_track_ttl_frames == int(
        (29.97 / 5) * processor.RETIRED_TRACK_TTL_SECONDS
    )


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


def test_roi_spawn_confirms_before_assigning_monotonic_ids(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    processor = pose.PoseProcessor(str(tmp_path))
    processor._next_pid = 0
    model_complexities = []

    class _RoiPose:
        def __init__(self, *args, **kwargs):
            model_complexities.append(kwargs["model_complexity"])

    monkeypatch.setattr(pose.mp.solutions.pose, "Pose", _RoiPose)

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
    assert rois == []
    assert model_complexities == []

    rois = processor._seed_rois_if_needed(image, 40, 40, rois, margin_ratio=0.0)
    assert [r["id"] for r in rois] == [0, 1]
    assert model_complexities == [2, 2]
    assert all(roi["provisional"] for roi in rois)


def test_roi_spawn_discards_one_pass_and_low_confidence_false_detections(import_pose, tmp_path):
    pose = import_pose
    processor = pose.PoseProcessor(str(tmp_path))
    processor._next_pid = 0

    def make_box(x1, y1, x2, y2, confidence):
        values = [x1, y1, x2, y2, confidence, 0]

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

    responses = iter([
        [make_box(0, 0, 10, 10, 0.9)],
        [],
        [make_box(0, 0, 10, 10, 0.39)],
    ])
    processor.yolo = types.SimpleNamespace(
        predict=lambda *a, **k: types.SimpleNamespace(xyxy=[next(responses)])
    )
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    rois = processor._seed_rois_if_needed(image, 40, 40, [], margin_ratio=0.0)
    assert rois == []
    rois = processor._seed_rois_if_needed(image, 40, 40, rois, margin_ratio=0.0)
    assert rois == []
    rois = processor._seed_rois_if_needed(image, 40, 40, rois, margin_ratio=0.0)
    assert rois == []


def test_provisional_roi_requires_two_meaningful_pose_results(import_pose, tmp_path):
    pose = import_pose
    processor = pose.PoseProcessor(str(tmp_path))
    landmarks = [types.SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9) for _ in range(33)]

    class _Pose:
        def process(self, image):
            return types.SimpleNamespace(
                pose_landmarks=types.SimpleNamespace(landmark=landmarks)
            )

    rois = [{
        "id": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10,
        "lost": 0, "pose": _Pose(), "overlap_streak": 0,
        "provisional": True, "pose_hits": 0,
    }]
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    outputs, rois = processor._process_multiperson_frame(image, 20, 20, rois)
    assert outputs == []
    assert rois[0]["provisional"] is True
    outputs, rois = processor._process_multiperson_frame(image, 20, 20, rois)
    assert [person_id for person_id, _ in outputs] == [0]
    assert rois[0]["provisional"] is False


def test_confirmed_roi_keeps_existing_partial_pose_behavior(import_pose, tmp_path):
    pose = import_pose
    processor = pose.PoseProcessor(str(tmp_path))
    landmarks = [types.SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.1) for _ in range(33)]

    class _Pose:
        def process(self, image):
            return types.SimpleNamespace(
                pose_landmarks=types.SimpleNamespace(landmark=landmarks)
            )

    rois = [{
        "id": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10,
        "lost": 0, "pose": _Pose(), "overlap_streak": 0,
        "provisional": False,
    }]
    outputs, rois = processor._process_multiperson_frame(
        np.zeros((20, 20, 3), dtype=np.uint8), 20, 20, rois
    )

    assert [person_id for person_id, _ in outputs] == [0]
    assert rois[0]["lost"] == 0


def test_pending_spawn_accepts_normal_motion_on_next_frame(import_pose, tmp_path):
    pose = import_pose
    processor = pose.PoseProcessor(str(tmp_path))

    assert processor._advance_pending_spawns([(0, 0, 10, 10)]) == []
    confirmed = processor._advance_pending_spawns([(8, 0, 18, 10)])

    assert len(confirmed) == 1
    assert confirmed[0]["box"] == (8, 0, 18, 10)


def test_pending_spawn_forces_immediate_confirmation_with_active_rois(import_pose, tmp_path):
    pose = import_pose
    processor = pose.PoseProcessor(str(tmp_path))
    processor._next_pid = 1
    responses = iter([
        [_yolo_person_box(20, 0, 30, 10)],
        [_yolo_person_box(28, 0, 38, 10)],
    ])
    processor.yolo = types.SimpleNamespace(
        predict=lambda *a, **k: types.SimpleNamespace(xyxy=[next(responses)])
    )
    rois = [{
        "id": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10,
        "lost": 0, "pose": object(), "overlap_streak": 0,
    }]
    image = np.zeros((40, 40, 3), dtype=np.uint8)

    rois = processor._seed_rois_if_needed(
        image, 40, 40, rois, margin_ratio=0.0,
        force_spawn_check=True, frame_index=10,
    )
    assert len(rois) == 1
    rois = processor._seed_rois_if_needed(
        image, 40, 40, rois, margin_ratio=0.0,
        force_spawn_check=False, frame_index=11,
    )

    assert [roi["id"] for roi in rois] == [0, 1]


def test_person_confidence_boundary_is_inclusive_at_minimum(import_pose, tmp_path):
    pose = import_pose
    processor = pose.PoseProcessor(str(tmp_path))
    low = types.SimpleNamespace(xyxy=[[_yolo_person_box(0, 0, 10, 10, 0.39)]])
    at_minimum = types.SimpleNamespace(
        xyxy=[[_yolo_person_box(0, 0, 10, 10, processor.MIN_PERSON_CONFIDENCE)]]
    )

    assert processor._person_boxes(low, 40, 40, margin_ratio=0.0) == []
    assert processor._person_boxes(at_minimum, 40, 40, margin_ratio=0.0) == [(0, 0, 10, 10)]


def test_one_person_multiperson_clip_writes_exactly_one_csv(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    processor = pose.PoseProcessor(str(tmp_path))
    processor.set_multi_person_mode(True)
    processor.yolo = types.SimpleNamespace(
        predict=lambda *a, **k: types.SimpleNamespace(
            xyxy=[[_yolo_person_box(10, 10, 30, 40)]]
        )
    )
    landmarks = [
        types.SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9)
        for _ in range(33)
    ]

    class _RoiPose:
        def __init__(self, *args, **kwargs):
            pass

        def process(self, image):
            return types.SimpleNamespace(
                pose_landmarks=types.SimpleNamespace(landmark=landmarks)
            )

        def close(self):
            pass

    monkeypatch.setattr(pose.mp.solutions.pose, "Pose", _RoiPose)
    video = tmp_path / "one_person.mp4"
    video.write_bytes(b"fake")

    assert processor.extract_pose_features(str(video)) is True

    assert sorted(path.name for path in tmp_path.glob("one_person_multi_ID_*.csv")) == [
        "one_person_multi_ID_0.csv"
    ]


def test_retired_roi_reuses_id_after_long_occlusion(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    processor = pose.PoseProcessor(str(tmp_path))
    processor._max_lost_frames = 2
    processor._retired_track_ttl_frames = 100
    processor._next_pid = 8

    class _LostPose:
        def __init__(self):
            self.closed = False

        def process(self, image):
            return types.SimpleNamespace(pose_landmarks=None)

        def close(self):
            self.closed = True

    lost_pose = _LostPose()
    rois = [{
        "id": 7, "x1": 0, "y1": 0, "x2": 10, "y2": 10,
        "lost": 1, "pose": lost_pose, "overlap_streak": 0,
        "provisional": False,
    }]
    _, rois = processor._process_multiperson_frame(
        np.zeros((20, 20, 3), dtype=np.uint8), 20, 20, rois, frame_index=50
    )
    assert rois == []
    assert lost_pose.closed is True

    created = []

    class _RoiPose:
        def __init__(self, *args, **kwargs):
            created.append(kwargs["model_complexity"])

    monkeypatch.setattr(pose.mp.solutions.pose, "Pose", _RoiPose)
    processor._allocate_confirmed_roi((2, 0, 12, 10), rois, frame_index=55)

    assert [roi["id"] for roi in rois] == [7]
    assert processor._next_pid == 8
    assert created == [2]


def test_unconfirmed_provisional_roi_retires_without_reseed_churn(import_pose, tmp_path):
    pose = import_pose
    messages = []
    processor = pose.PoseProcessor(str(tmp_path), status_callback=messages.append)
    processor._max_lost_frames = 3
    processor._last_spawn_status_t = -100.0
    yolo_calls = {"n": 0}
    processor.yolo = types.SimpleNamespace(
        predict=lambda *a, **k: yolo_calls.__setitem__("n", yolo_calls["n"] + 1)
    )
    landmarks = [
        types.SimpleNamespace(x=0.5, y=0.5, z=0.0, visibility=0.9 if i < 7 else 0.1)
        for i in range(33)
    ]

    class _Pose:
        def __init__(self):
            self.closed = False

        def process(self, image):
            return types.SimpleNamespace(
                pose_landmarks=types.SimpleNamespace(landmark=landmarks)
            )

        def close(self):
            self.closed = True

    roi_pose = _Pose()
    rois = [{
        "id": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10,
        "lost": 0, "pose": roi_pose, "overlap_streak": 0,
        "provisional": True, "pose_hits": 0,
    }]
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    for frame_index in range(3):
        _, rois = processor._process_multiperson_frame(
            image, 20, 20, rois, frame_index=frame_index
        )

    assert rois == []
    assert yolo_calls["n"] == 0
    assert roi_pose.closed is True
    assert any("Could not confirm" in message for message in messages)
    assert len(processor._suppressed_spawns) == 1

    processor.yolo = types.SimpleNamespace(
        predict=lambda *a, **k: types.SimpleNamespace(
            xyxy=[[_yolo_person_box(0, 0, 10, 10)]]
        )
    )
    rois = processor._seed_rois_if_needed(
        image, 20, 20, rois, margin_ratio=0.0,
        force_spawn_check=True, frame_index=3,
    )
    assert rois == []
    assert processor._pending_spawns == []


def test_roi_spawn_never_exceeds_tracking_cap(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    processor = pose.PoseProcessor(str(tmp_path))
    processor._next_pid = 0
    processor._max_tracked_people = 1
    created = []

    class _RoiPose:
        def __init__(self, *args, **kwargs):
            created.append(kwargs["model_complexity"])

    monkeypatch.setattr(pose.mp.solutions.pose, "Pose", _RoiPose)

    def make_box(x1, y1, x2, y2):
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

    boxes = [make_box(0, 0, 10, 10), make_box(20, 20, 30, 30)]
    processor.yolo = types.SimpleNamespace(
        predict=lambda *a, **k: types.SimpleNamespace(xyxy=[boxes])
    )
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    rois = processor._seed_rois_if_needed(image, 40, 40, [], margin_ratio=0.0)
    rois = processor._seed_rois_if_needed(image, 40, 40, rois, margin_ratio=0.0)

    assert [roi["id"] for roi in rois] == [0]
    assert created == [2]


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


def test_multiperson_reseed_is_throttled_and_stale_rois_are_removed(import_pose):
    pose = import_pose
    processor = pose.PoseProcessor("/tmp", frame_threshold=1)
    processor._reseed_period = 10
    processor._max_lost_frames = 3
    yolo_calls = {"n": 0}

    def predict(*args, **kwargs):
        yolo_calls["n"] += 1
        return types.SimpleNamespace(xyxy=[[]])

    processor.yolo = types.SimpleNamespace(predict=predict)

    class _LostPose:
        def __init__(self):
            self.closed = False

        def process(self, cropped):
            return types.SimpleNamespace(pose_landmarks=None)

        def close(self):
            self.closed = True

    roi_pose = _LostPose()
    locked_rois = [{
        "id": 0, "x1": 0, "y1": 0, "x2": 10, "y2": 10,
        "lost": 1, "pose": roi_pose, "overlap_streak": 0,
    }]
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    _, locked_rois = processor._process_multiperson_frame(image, 20, 20, locked_rois)
    _, locked_rois = processor._process_multiperson_frame(image, 20, 20, locked_rois)

    assert yolo_calls["n"] == 1
    assert locked_rois == []
    assert roi_pose.closed is True


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
