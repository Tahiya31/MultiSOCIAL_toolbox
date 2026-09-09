"""In-process analysis backend used by macOS and non-Windows source launches."""

from __future__ import annotations

import importlib
import os
from typing import Optional

import runtime_services


class NativeAnalysisBackend:
    def create_audio_processor(self, *args, **kwargs):
        processor = importlib.import_module("audio").AudioProcessor
        return processor(*args, **kwargs)

    def create_pose_processor(self, *args, **kwargs):
        processor = importlib.import_module("pose").PoseProcessor
        return processor(*args, **kwargs)

    def find_pose_csv_paths(
        self,
        output_folder: str,
        video_path: str,
        multi_person: Optional[bool] = None,
    ) -> list[str]:
        resolver = importlib.import_module("pose").find_pose_csv_paths
        return resolver(output_folder, video_path, multi_person=multi_person)

    def validate_import_smoke(self, profile: str, verify_heavy_pose_asset: bool) -> str:
        if verify_heavy_pose_asset:
            model = runtime_services.resource_path(
                "mediapipe", "modules", "pose_landmark", "pose_landmark_heavy.tflite"
            )
            if not os.path.isfile(model):
                raise RuntimeError(f"Missing bundled Heavy pose model: {os.path.basename(model)}")
            print("Bundled Heavy pose model check passed.", flush=True)
        if profile == "complete":
            runtime_services.preload_frozen_windows_diarization_dependencies()
            importlib.import_module("pyannote.audio")
        return f"Import smoke test passed ({profile or 'standard'} profile)."
