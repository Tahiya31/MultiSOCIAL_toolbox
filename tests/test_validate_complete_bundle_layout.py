from __future__ import annotations

from pathlib import Path

import pytest


def test_validate_complete_bundle_layout_skips_non_complete_profile(
    monkeypatch, capsys, validate_complete_bundle_layout_module
):
    module = validate_complete_bundle_layout_module
    monkeypatch.setenv("MULTISOCIAL_BUILD_PROFILE", "standard")

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 0
    assert "Skipping complete bundle layout validation" in capsys.readouterr().out


def test_validate_complete_bundle_layout_passes_for_macos_bundle(
    monkeypatch, tmp_path, capsys, validate_complete_bundle_layout_module
):
    module = validate_complete_bundle_layout_module
    monkeypatch.setenv("MULTISOCIAL_BUILD_PROFILE", "complete")
    monkeypatch.setenv("RUNNER_OS", "macOS")
    monkeypatch.chdir(tmp_path)

    root = tmp_path / "dist" / "MultiSOCIAL-Complete.app"
    (root / "pkg" / "lightning_fabric").mkdir(parents=True)
    (root / "pkg" / "lightning_fabric" / "version.info").write_text("", encoding="utf-8")
    (root / "meta" / "pyannote.audio-1.dist-info").mkdir(parents=True)
    (root / "meta" / "pyannote.audio-1.dist-info" / "METADATA").write_text("", encoding="utf-8")
    (root / "meta" / "speechbrain-1.dist-info").mkdir(parents=True)
    (root / "meta" / "speechbrain-1.dist-info" / "METADATA").write_text("", encoding="utf-8")
    heavy_model = root / "mediapipe" / "modules" / "pose_landmark" / "pose_landmark_heavy.tflite"
    heavy_model.parent.mkdir(parents=True)
    heavy_model.write_bytes(b"model")

    module.main()

    assert "Complete bundle layout validation passed" in capsys.readouterr().out


def test_validate_complete_bundle_layout_fails_when_metadata_missing(
    monkeypatch, tmp_path, validate_complete_bundle_layout_module
):
    module = validate_complete_bundle_layout_module
    monkeypatch.setenv("MULTISOCIAL_BUILD_PROFILE", "complete")
    monkeypatch.setenv("RUNNER_OS", "Windows")
    monkeypatch.chdir(tmp_path)

    root = tmp_path / "dist" / "MultiSOCIAL-Complete"
    root.mkdir(parents=True)

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 1
