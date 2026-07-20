from __future__ import annotations

from pathlib import Path


def _method_source(source: str, method_name: str, next_method: str) -> str:
    return source.split(f"def {method_name}", 1)[1].split(f"def {next_method}", 1)[0]


def test_folder_selection_only_configures_output_paths():
    source = (Path(__file__).resolve().parents[1] / "src" / "app.py").read_text(encoding="utf-8")
    on_folder_changed = _method_source(source, "on_folder_changed", "_warn_stem_collisions")
    configure_paths = _method_source(source, "configure_output_paths", "_ensure_output_directory")

    assert "self.configure_output_paths(normalized_path)" in on_folder_changed
    assert "os.makedirs" not in on_folder_changed
    assert "os.makedirs" not in configure_paths


def test_each_processing_action_creates_only_its_own_output_directory():
    source = (Path(__file__).resolve().parents[1] / "src" / "app.py").read_text(encoding="utf-8")

    assert 'self._ensure_output_directory(self.converted_audio_folder, "converted_audio")' in _method_source(
        source, "on_convert", "convert_all_videos_to_wav"
    )
    verify = _method_source(source, "on_verify_consistency", "_verify_consistency_batch")
    assert 'self._ensure_output_directory(verification_dir, "verification")' in verify
    assert 'self._ensure_output_directory(worst_frames_root, "verification/worst_frames")' in verify
    assert 'self._ensure_output_directory(self.extracted_pose_folder, "pose_features")' in _method_source(
        source, "on_extract_features", "extract_pose_features_batch"
    )
    assert 'self._ensure_output_directory(self.embedded_pose_folder, "embedded_pose")' in _method_source(
        source, "on_embed_poses", "embed_pose_batch"
    )
    assert 'self._ensure_output_directory(self.captioned_video_folder, "captioned_video")' in _method_source(
        source, "on_embed_transcript", "embed_transcript_batch"
    )
    assert 'self._ensure_output_directory(self.extracted_audio_folder, "audio_features")' in _method_source(
        source, "on_extract_audio_features", "extract_audio_features_batch"
    )
    assert 'self._ensure_output_directory(self.extracted_transcripts_folder, "transcripts")' in _method_source(
        source, "on_extract_transcripts", "extract_transcripts_batch"
    )
    assert 'self._ensure_output_directory(self.extracted_transcripts_folder, "transcripts")' in _method_source(
        source, "on_align_features", "align_features_batch"
    )
