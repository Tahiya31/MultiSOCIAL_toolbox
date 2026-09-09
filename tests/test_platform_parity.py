from __future__ import annotations

import ast
import inspect
import sys
import time
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from atomic_outputs import OutputStaging
from native_worker_client import (
    WindowsAudioProcessor,
    WindowsPoseProcessor,
    find_pose_csv_paths as worker_pose_paths,
)


def test_output_staging_promotes_complete_files_and_discards_cancelled_work(tmp_path):
    destination = tmp_path / "outputs"
    transaction = OutputStaging(destination)
    with transaction as staged:
        (staged / "complete.csv").write_text("complete", encoding="utf-8")
        transaction.commit()

    assert (destination / "complete.csv").read_text(encoding="utf-8") == "complete"
    assert not list(destination.glob(".multisocial-stage-*"))

    with OutputStaging(destination) as staged:
        (staged / "cancelled.csv").write_text("partial", encoding="utf-8")

    assert not (destination / "cancelled.csv").exists()
    assert not list(destination.glob(".multisocial-stage-*"))


def test_output_staging_discards_stale_work_and_ignores_committed_residue(tmp_path):
    destination = tmp_path / "outputs"
    stale = destination / ".multisocial-stage-abandoned"
    stale.mkdir(parents=True)
    (stale / "partial.csv").write_text("partial", encoding="utf-8")
    old = time.time() - (25 * 60 * 60)
    stale.touch()
    # Set both timestamps so cleanup is portable across local and CI filesystems.
    import os

    os.utime(stale, (old, old))
    active = destination / ".multisocial-stage-active"
    active.mkdir()

    transaction = OutputStaging(destination)
    with transaction as staged:
        (staged / "complete.csv").write_text("complete", encoding="utf-8")
        transaction.commit()
        # Finder/antivirus metadata created after promotion must not make the
        # context manager fail while it cleans the private directory.
        (staged / ".DS_Store").write_text("metadata", encoding="utf-8")

    assert not stale.exists()
    assert active.exists()
    assert (destination / "complete.csv").is_file()
    assert not (destination / ".DS_Store").exists()
    assert list(destination.glob(".multisocial-stage-*")) == [active]


def test_worker_pose_csv_resolution_is_precise_and_mode_compatible(tmp_path):
    for name in (
        "clip_ID_0.csv",
        "clip_multi_ID_0.csv",
        "clip2_ID_0.csv",
    ):
        (tmp_path / name).write_text("frame,person_id\n", encoding="utf-8")
    video = tmp_path / "clip.mp4"

    assert worker_pose_paths(str(tmp_path), str(video), None) == [
        str(tmp_path / "clip_multi_ID_0.csv")
    ]
    assert worker_pose_paths(str(tmp_path), str(video), False) == [
        str(tmp_path / "clip_ID_0.csv")
    ]
    assert worker_pose_paths(str(tmp_path), str(video), True) == [
        str(tmp_path / "clip_multi_ID_0.csv")
    ]


def test_native_and_windows_processors_have_matching_public_operation_signatures(import_audio):
    audio = import_audio
    native_audio = audio.AudioProcessor

    def names(method):
        return list(inspect.signature(method).parameters)

    # The path parameter names differ only to clarify that the worker accepts
    # an audio file; all optional arguments and their order are identical.
    assert names(native_audio.extract_audio_features) == [
        "self", "filepath", "progress_callback", "cancel_check"
    ]
    assert names(WindowsAudioProcessor.extract_audio_features) == [
        "self", "audio_file", "progress_callback", "cancel_check"
    ]
    assert names(native_audio.extract_audio_features_batch) == names(
        WindowsAudioProcessor.extract_audio_features_batch
    )
    assert names(native_audio.extract_transcripts_batch) == names(
        WindowsAudioProcessor.extract_transcripts_batch
    )
    assert names(native_audio.extract_transcript) == [
        "self", "filepath", "progress_callback", "word_timestamps", "cancel_check"
    ]
    assert names(WindowsAudioProcessor.extract_transcript) == [
        "self", "audio_file", "progress_callback", "word_timestamps", "cancel_check"
    ]
    assert names(native_audio.align_features_batch) == names(
        WindowsAudioProcessor.align_features_batch
    )

    pose_source = (SRC_ROOT / "pose.py").read_text(encoding="utf-8")
    pose_class = next(
        node for node in ast.parse(pose_source).body
        if isinstance(node, ast.ClassDef) and node.name == "PoseProcessor"
    )

    def source_signature(method_name):
        method = next(
            node for node in pose_class.body
            if isinstance(node, ast.FunctionDef) and node.name == method_name
        )
        return [argument.arg for argument in method.args.args]

    assert source_signature("extract_pose_features") == names(
        WindowsPoseProcessor.extract_pose_features
    )
    assert source_signature("embed_pose_video") == names(
        WindowsPoseProcessor.embed_pose_video
    )


def test_frozen_yolov5_version_banner_is_configured_only_on_demand():
    pose_source = (SRC_ROOT / "pose.py").read_text(encoding="utf-8")

    assert "def _configure_frozen_yolov5_file_date" in pose_source
    assert "if not runtime_services.is_frozen_runtime():" in pose_source
    assert "yolov5_torch_utils.file_date = safe_file_date" in pose_source
    assert "_configure_frozen_yolov5_file_date()" in pose_source
