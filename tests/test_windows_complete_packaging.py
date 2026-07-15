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


def test_ci_captures_complete_windows_diagnostics_without_a_blanket_lock():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert 'python -m pip install ".[complete,dev]"' in workflow
    assert "requirements/windows-complete.lock" not in workflow
    assert "MULTISOCIAL_SMOKE_TRACE" in workflow
    assert "windows-complete-smoke-diagnostics" in workflow
    assert "if: always() && matrix.profile == 'complete' && runner.os == 'Windows'" in workflow
    assert "${{ github.workspace }}/.tmp/multisocial-smoke-${{ matrix.os }}-${{ matrix.profile }}" in workflow
    assert "${{ runner.temp }}" not in workflow


def test_smoke_checkpoints_cover_bootstrap_asset_and_native_imports():
    app_source = (ROOT / "src" / "app.py").read_text(encoding="utf-8")

    for stage in (
        '"bootstrap"',
        '"wx"',
        '"heavy-model:before"',
        '"heavy-model:passed"',
        '"torch:before"',
        '"torch:passed"',
        '"torchaudio:passed"',
        '"pyannote.audio:before"',
        '"pyannote.audio:passed"',
    ):
        assert stage in app_source
    assert 'trace_path = os.environ.get("MULTISOCIAL_SMOKE_TRACE")' in app_source
    assert "if not trace_path:\n        return" in app_source
    assert 'print("Bundled Heavy pose model check passed.", flush=True)' in app_source
    assert 'print("Import smoke test passed (complete profile).", flush=True)' in app_source
