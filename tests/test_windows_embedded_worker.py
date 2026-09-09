from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_embedded_worker_validator_uses_the_physical_site_packages_layout():
    validator = (ROOT / "packaging" / "validate_windows_bundle.py").read_text(encoding="utf-8")
    complete_layout = (
        ROOT / ".github" / "scripts" / "validate_complete_bundle_layout.py"
    ).read_text(encoding="utf-8")

    assert "Lib/site-packages/audresample/core/bin/win_amd64/audresample.dll" in validator
    assert "Lib/site-packages/opensmile/core/bin/win_amd64/SMILEapi.dll" in validator
    assert '"worker/assets/pose_landmark_heavy.tflite"' in complete_layout


def test_embedded_worker_excludes_pyinstaller_build_tools():
    builder = (ROOT / "packaging" / "build_windows_embedded_worker.py").read_text(encoding="utf-8")

    assert "_ignore_build_only_worker_files" in builder
    assert '"_pyinstaller_hooks_contrib"' in builder
    assert '"pyinstaller-"' in builder
    assert "ignore=_ignore_build_only_worker_files" in builder


def test_gui_heavy_pose_smoke_uses_the_embedded_worker_asset_path():
    backend = (ROOT / "src" / "worker_backend.py").read_text(encoding="utf-8")

    assert ' / "worker"\n                    / "assets"\n                    / "pose_landmark_heavy.tflite"' in backend
