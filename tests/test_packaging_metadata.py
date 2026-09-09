from __future__ import annotations

from pathlib import Path


def test_captions_module_is_packaged():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert '"captions",' in text


def test_embedded_worker_copies_the_locked_runtime_and_assets():
    root = Path(__file__).resolve().parents[1]
    builder_text = (root / "packaging" / "build_windows_embedded_worker.py").read_text(encoding="utf-8")
    manifest_text = (root / "packaging" / "windows_hiddenimports.py").read_text(encoding="utf-8")

    assert 'shutil.copytree(\n        site_packages,\n        output / "Lib" / "site-packages",' in builder_text
    assert 'shutil.copytree(ROOT / "assets", output / "assets")' in builder_text
    assert '"yolov5.models.yolo"' in manifest_text


def test_embedded_worker_copies_the_heavy_pose_asset():
    root = Path(__file__).resolve().parents[1]
    builder_text = (root / "packaging" / "build_windows_embedded_worker.py").read_text(encoding="utf-8")

    assert 'shutil.copytree(ROOT / "assets", output / "assets")' in builder_text


def test_macos_bundle_uses_project_version_metadata():
    root = Path(__file__).resolve().parents[1]
    spec_text = (root / "MultiSOCIAL.spec").read_text(encoding="utf-8")

    assert '"CFBundleShortVersionString": APP_VERSION' in spec_text
    assert '"CFBundleVersion": APP_VERSION' in spec_text
