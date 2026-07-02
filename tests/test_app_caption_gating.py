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
    assert "max(existing, key=lambda path: os.path.getmtime(path))" in app_source
    assert "the newest one is used" in app_source
    assert 'output_suffix = "_pose_captioned.mp4"' in app_source
    assert "input_video = self._embedded_pose_video_path(self.embedded_pose_folder, video_file)" in app_source
    assert "captions.burn_subtitles(" in app_source


def test_pose_embed_gating_requires_all_matching_csvs():
    app_source = (Path(__file__).resolve().parents[1] / "src" / "app.py").read_text(encoding="utf-8")

    assert "_has_all_pose_csvs" in app_source
    assert "_has_any_pose_csv" not in app_source
    assert "pose_multi_person" in app_source
    assert 'pattern = f"{base}_multi_ID_*.csv" if multi_person else f"{base}_ID_*.csv"' in app_source
    assert "find_pose_csv_paths(self.extracted_pose_folder, video, multi_person=multi_person)" in app_source

def test_verify_uses_source_base_for_multi_pose_videos():
    app_source = (Path(__file__).resolve().parents[1] / "src" / "app.py").read_text(encoding="utf-8")

    assert "def _pose_source_base_from_embedded_video" in app_source
    assert '"_multi_pose", "_pose"' in app_source
    assert "def _pose_csv_paths_for_embedded_video" in app_source
    assert 'base.replace("_pose", "")' not in app_source

def test_verify_uses_mode_specific_csvs_and_meta_stride():
    app_source = (Path(__file__).resolve().parents[1] / "src" / "app.py").read_text(encoding="utf-8")

    assert "def _embedded_pose_video_is_multi" in app_source
    assert "def _pose_stride_for_embedded_video" in app_source
    assert 'pattern = f"{csv_base}_multi_ID_*.csv" if is_multi else f"{csv_base}_ID_*.csv"' in app_source
    assert 'f"{csv_base}{suffix}_meta.json"' in app_source
    assert "stride=self._pose_stride_for_embedded_video(video)" in app_source
    assert 'f"{csv_base}*_ID_*.csv"' not in app_source
    assert "self.frameStrideInput.GetValue()" not in app_source.split("def _verify_consistency_batch", 1)[1].split("def convert_to_wav", 1)[0]
