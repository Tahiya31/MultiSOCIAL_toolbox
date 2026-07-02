from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")


def _synthetic_pose_frame():
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    cv2.circle(frame, (40, 30), 3, (0, 255, 0), -1)
    return frame


def test_verify_samples_processed_frames_and_reports_metadata(tmp_path, monkeypatch):
    from verify_pose_embedding import _landmark_columns, verify

    video_path = tmp_path / "clip_pose.mp4"
    video_path.write_bytes(b"fake")

    class FakeCapture:
        def __init__(self, path):
            self.path = path
            self._pos = 0

        def isOpened(self):
            return True

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 20
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 80
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 60
            return 0

        def set(self, prop, value):
            if prop == cv2.CAP_PROP_POS_FRAMES:
                self._pos = int(value)
            return True

        def read(self):
            if self._pos >= 20:
                return False, None
            frame = _synthetic_pose_frame()
            self._pos += 1
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr("verify_pose_embedding.cv2.VideoCapture", FakeCapture)

    lmk_cols = _landmark_columns()
    rows = []
    for frame_idx in range(20):
        row = {"frame": frame_idx, "person_id": 0}
        for i in range(0, len(lmk_cols), 4):
            row[lmk_cols[i]] = 0.5
            row[lmk_cols[i + 1]] = 0.5
            row[lmk_cols[i + 2]] = 0.0
            row[lmk_cols[i + 3]] = 1.0
        rows.append(row)

    csv_path = tmp_path / "clip_ID_0.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    progress = []

    report = verify(
        str(video_path),
        [str(csv_path)],
        max_frames=5,
        progress_callback=progress.append,
        metric="hit_rate",
    )

    assert report["eligible_frames"] == 20
    assert report["frames_compared"] == 5
    assert report["sampled"] is True
    assert report["max_frames"] == 5
    assert report["mean_hit_rate"] == 1.0
    assert progress[-1] == 100
