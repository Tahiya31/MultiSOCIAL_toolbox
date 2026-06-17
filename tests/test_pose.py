from __future__ import annotations

import sys
import types

import numpy as np
import pytest


def _install_fake_pose_deps():
    cv2 = types.ModuleType("cv2")

    class _CapProp:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FRAME_COUNT = 7
        CAP_PROP_FPS = 5
        CAP_PROP_POS_FRAMES = 1
        CAP_PROP_POS_AVI_RATIO = 2

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
                return 25
            return 0

        def set(self, *args, **kwargs):
            return True

        def release(self):
            pass

    class FakeVideoWriter:
        def __init__(self, *args, **kwargs):
            self.opened = True

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


def test_embed_pose_video_cancel_check_stops_early(import_pose, tmp_path, monkeypatch):
    pose = import_pose
    monkeypatch.setattr(pose, "ensure_yolov5_weights", lambda: None)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    processor = pose.PoseProcessor(str(tmp_path), output_video_folder=str(out_dir))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    checks = {"n": 0}

    def cancel_check():
        checks["n"] += 1
        return checks["n"] > 2

    result = processor.embed_pose_video(str(video), cancel_check=cancel_check)
    assert result is False
