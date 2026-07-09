from __future__ import annotations

from pathlib import Path


def test_captions_module_is_packaged():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert '"captions",' in text


def test_yolov5_runtime_dependencies_are_packaged():
    root = Path(__file__).resolve().parents[1]
    spec_text = (root / "MultiSOCIAL.spec").read_text(encoding="utf-8")
    hook_text = (root / "hooks" / "hook-yolov5.py").read_text(encoding="utf-8")

    assert 'collect_data_files("ultralytics")' in spec_text
    assert 'collect_submodules("ultralytics")' in spec_text
    assert '"yolov5", "ultralytics", "torch", "torchvision"' in spec_text
    assert 'os.path.join(ROOT, "assets", "yolov5s.pt")' in spec_text

    assert 'collect_data_files("ultralytics")' in hook_text
    assert 'collect_submodules("ultralytics")' in hook_text
    assert '"yolov5", "ultralytics", "torch", "torchvision"' in hook_text
