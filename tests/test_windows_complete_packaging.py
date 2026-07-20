from __future__ import annotations

import importlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_runtime_hook_uses_a_stable_allowlist_without_recursive_discovery(tmp_path):
    hook = importlib.import_module("runtime_hook_dlls")
    (tmp_path / "torch" / "lib").mkdir(parents=True)
    (tmp_path / "mediapipe" / "python").mkdir(parents=True)
    (tmp_path / "unrelated" / "nested").mkdir(parents=True)

    directories = hook.bundled_dll_directories(str(tmp_path))

    assert directories[0] == str(tmp_path)
    assert str(tmp_path / "torch" / "lib") in directories
    assert str(tmp_path / "mediapipe" / "python") in directories
    assert str(tmp_path / "unrelated" / "nested") not in directories
    assert "os.walk" not in (ROOT / "src" / "runtime_hook_dlls.py").read_text(encoding="utf-8")


def test_complete_smoke_and_diarization_use_the_windows_native_preload():
    app_source = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
    audio_source = (ROOT / "src" / "audio.py").read_text(encoding="utf-8")

    assert "runtime_services.preload_frozen_windows_diarization_dependencies()" in app_source
    assert "preload_frozen_windows_diarization_dependencies()" in audio_source
