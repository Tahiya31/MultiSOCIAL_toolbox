"""Native-free analysis backend contract used by the shared GUI."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol


class PoseProcessorLike(Protocol):
    def set_multi_person_mode(self, enabled: bool) -> None: ...

    def extract_pose_features(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Any: ...

    def embed_pose_video(
        self,
        video_path: str,
        progress_callback: Optional[Callable[[int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Any: ...


class AudioProcessorLike(Protocol):
    enable_speaker_diarization: bool

    def preload_speaker_diarizer(self) -> None: ...

    def extract_audio_features_batch(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def extract_transcripts_batch(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...

    def align_features_batch(self, *args: Any, **kwargs: Any) -> dict[str, Any]: ...


class AnalysisBackend(Protocol):
    def create_audio_processor(self, *args: Any, **kwargs: Any) -> AudioProcessorLike: ...

    def create_pose_processor(self, *args: Any, **kwargs: Any) -> PoseProcessorLike: ...

    def find_pose_csv_paths(
        self,
        output_folder: str,
        video_path: str,
        multi_person: Optional[bool] = None,
    ) -> list[str]: ...

    def validate_import_smoke(self, profile: str, verify_heavy_pose_asset: bool) -> str: ...


_backend: AnalysisBackend | None = None


def configure_backend(backend: AnalysisBackend) -> None:
    global _backend
    _backend = backend


def get_backend() -> AnalysisBackend:
    if _backend is None:
        raise RuntimeError(
            "No analysis backend is configured. Launch through source_entry.py, "
            "app_windows.py, or app_macos.py."
        )
    return _backend
