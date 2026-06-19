from __future__ import annotations

from pathlib import Path


def test_caption_embed_gating_requires_all_matching_srts():
    app_source = (Path(__file__).resolve().parents[1] / "src" / "app.py").read_text(encoding="utf-8")

    assert "_has_all_transcript_srts" in app_source
    assert "_has_any_transcript_srt" not in app_source
    assert "not os.path.exists(os.path.join(transcripts_dir, f\"{base}.srt\"))" in app_source
    assert "_has_all_embedded_pose_videos" in app_source
    assert "captionPoseVideoCheckbox" in app_source
    assert "Run 'Embed Pose Features' first" in app_source


def test_caption_embed_can_target_pose_embedded_video():
    app_source = (Path(__file__).resolve().parents[1] / "src" / "app.py").read_text(encoding="utf-8")

    assert "def _embedded_pose_video_path" in app_source
    assert 'f"{base}_pose.mp4"' in app_source
    assert 'f"{base}_multi_pose.mp4"' in app_source
    assert 'output_suffix = "_pose_captioned.mp4"' in app_source
    assert "input_video = self._embedded_pose_video_path(self.embedded_pose_folder, video_file)" in app_source
    assert "captions.burn_subtitles(" in app_source


def test_pose_embed_gating_requires_all_matching_csvs():
    app_source = (Path(__file__).resolve().parents[1] / "src" / "app.py").read_text(encoding="utf-8")

    assert "_has_all_pose_csvs" in app_source
    assert "_has_any_pose_csv" not in app_source
    assert "not glob.glob(os.path.join(pose_dir, f\"{base}_multi_ID_*.csv\"))" in app_source
