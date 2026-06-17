from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def vpe():
    cv2 = types.ModuleType("cv2")

    def _circle(img, center, radius, color, thickness=-1):
        x, y = center
        img[y, x] = color

    cv2.circle = _circle
    cv2.line = lambda *a, **k: None
    cv2.resize = lambda img, size, **kw: img
    cv2.morphologyEx = lambda mask, op, kernel, **kw: mask
    cv2.connectedComponentsWithStats = lambda mask, **kw: (
        1,
        np.zeros(mask.shape, dtype=np.int32),
        np.zeros((1, 5), dtype=np.int32),
        None,
    )
    cv2.VideoCapture = lambda path: None
    cv2.CAP_PROP_FRAME_WIDTH = 3
    cv2.CAP_PROP_FRAME_HEIGHT = 4
    cv2.INTER_AREA = 0
    sys.modules["cv2"] = cv2
    sys.modules.pop("verify_pose_embedding", None)
    return importlib.import_module("verify_pose_embedding")


def _write_pose_csv(vpe, path, *, frame=0, person_id=0, x=0.5, y=0.5, confidence=1.0):
    columns = ["frame", "person_id"] + vpe._landmark_columns()
    row = [frame, person_id]
    for idx in range(0, len(vpe._landmark_columns()), 4):
        row.extend([x, y, 0.0, confidence])
    df = pd.DataFrame([row], columns=columns)
    df.to_csv(path, index=False)


def test_landmark_columns_contains_expected_count(vpe):
    columns = vpe._landmark_columns()

    assert len(columns) == 33 * 4
    assert columns[:4] == ["Nose_x", "Nose_y", "Nose_z", "Nose_confidence"]


def test_sanity_check_csv_detects_bad_coordinates(vpe, tmp_path):
    csv_path = tmp_path / "pose.csv"
    _write_pose_csv(vpe, csv_path, x=1.5)

    problems = vpe.sanity_check_csv(csv_path)

    assert any("out of [0,1] range" in problem for problem in problems)


def test_draw_from_row_marks_expected_pixel(vpe):
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    row = {"Nose_x": 0.5, "Nose_y": 0.5, "Nose_z": 0.0, "Nose_confidence": 1.0}
    drawn = vpe._draw_from_row(img, row, ["Nose_x", "Nose_y", "Nose_z", "Nose_confidence"], color=(0, 255, 0))

    assert drawn[10, 10, 1] == 255


def test_landmark_hit_detects_palette_overlay(vpe):
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[10, 10] = np.array([0, 255, 0], dtype=np.uint8)

    assert vpe._landmark_hit_colored(frame, 10, 10, window=1, expected_bgr=(0, 255, 0)) is True
    assert vpe._landmark_hit_colored(frame, 0, 0, window=1, expected_bgr=(0, 255, 0)) is False


def test_verify_hit_rate_uses_processed_frames(vpe, monkeypatch, tmp_path):
    csv_path = tmp_path / "clip_ID_0.csv"
    _write_pose_csv(vpe, csv_path)

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[10, 10] = np.array([0, 255, 0], dtype=np.uint8)

    class FakeCapture:
        def __init__(self, frames):
            self.frames = list(frames)
            self.index = 0

        def isOpened(self):
            return self.index <= len(self.frames)

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame_value = self.frames[self.index]
            self.index += 1
            return True, frame_value.copy()

        def get(self, prop):
            return {3: 20, 4: 20}.get(int(prop), 1)

        def release(self):
            return None

    monkeypatch.setattr(vpe.cv2, "VideoCapture", lambda path: FakeCapture([frame]))
    monkeypatch.setattr(vpe.cv2, "CAP_PROP_FRAME_WIDTH", 3, raising=False)
    monkeypatch.setattr(vpe.cv2, "CAP_PROP_FRAME_HEIGHT", 4, raising=False)

    report = vpe.verify(str(tmp_path / "video.mp4"), [str(csv_path)], processed_only=True, metric="hit_rate")

    assert report["frames_compared"] == 1
    assert report["mean_hit_rate"] == 1.0
    assert report["below_threshold"] == 0


def test_verify_stride_maps_raw_frame_to_processed_index(vpe, monkeypatch, tmp_path):
    csv_path = tmp_path / "clip_ID_0.csv"
    _write_pose_csv(vpe, csv_path, frame=1)

    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    frame[10, 10] = np.array([0, 255, 0], dtype=np.uint8)

    class FakeCapture:
        def __init__(self, frames):
            self.frames = list(frames)
            self.index = 0

        def isOpened(self):
            return self.index <= len(self.frames)

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame_value = self.frames[self.index]
            self.index += 1
            return True, frame_value.copy()

        def get(self, prop):
            return {3: 20, 4: 20}.get(int(prop), 1)

        def release(self):
            return None

    blank = np.zeros((20, 20, 3), dtype=np.uint8)
    monkeypatch.setattr(vpe.cv2, "VideoCapture", lambda path: FakeCapture([blank, blank, frame]))
    monkeypatch.setattr(vpe.cv2, "CAP_PROP_FRAME_WIDTH", 3, raising=False)
    monkeypatch.setattr(vpe.cv2, "CAP_PROP_FRAME_HEIGHT", 4, raising=False)

    report = vpe.verify(
        str(tmp_path / "video.mp4"),
        [str(csv_path)],
        processed_only=True,
        metric="hit_rate",
        stride=2,
    )

    assert report["frames_compared"] == 1
    assert report["mean_hit_rate"] == 1.0
