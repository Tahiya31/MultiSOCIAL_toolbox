from __future__ import annotations

import io
import json
import types


def test_load_state_defaults_when_missing(tmp_path, monkeypatch, import_runtime_services):
    rs = import_runtime_services
    monkeypatch.setattr(rs, "state_path", lambda: str(tmp_path / "missing.json"))

    assert rs.load_state() == {
        "features": {},
        "secrets": {},
        "install_profile": rs.DEFAULT_BUILD_PROFILE,
    }


def test_load_state_defaults_when_corrupt(tmp_path, monkeypatch, import_runtime_services):
    rs = import_runtime_services
    state_file = tmp_path / "state.json"
    state_file.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(rs, "state_path", lambda: str(state_file))

    assert rs.load_state()["install_profile"] == rs.DEFAULT_BUILD_PROFILE


def test_save_hf_token_normalizes_assignment_form(tmp_path, monkeypatch, import_runtime_services):
    rs = import_runtime_services
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(rs, "state_path", lambda: str(state_file))

    rs.save_hf_token("HF_TOKEN=hf_123")
    state = json.loads(state_file.read_text(encoding="utf-8"))
    assert state["secrets"]["hf_token"] == "hf_123"
    assert rs.load_hf_token() == "hf_123"


def test_get_build_profile_prefers_environment(monkeypatch, import_runtime_services):
    rs = import_runtime_services
    monkeypatch.setenv("MULTISOCIAL_INSTALL_PROFILE", "COMPLETE")

    assert rs.get_build_profile() == "complete"


def test_get_build_profile_uses_state_then_diarization(monkeypatch, import_runtime_services):
    rs = import_runtime_services
    monkeypatch.delenv("MULTISOCIAL_INSTALL_PROFILE", raising=False)
    monkeypatch.delenv("MULTISOCIAL_BUILD_PROFILE", raising=False)
    monkeypatch.setattr(rs, "is_frozen_runtime", lambda: False)
    monkeypatch.setattr(rs, "load_state", lambda: {"install_profile": "", "features": {}, "secrets": {}})
    monkeypatch.setattr(rs, "is_diarization_installed", lambda: True)

    assert rs.get_build_profile() == "complete"


def test_frozen_windows_diarization_preload_uses_the_verified_native_order(monkeypatch, import_runtime_services):
    rs = import_runtime_services
    imported = []
    monkeypatch.setattr(rs.sys, "platform", "win32")
    monkeypatch.setattr(rs.sys, "frozen", True, raising=False)
    monkeypatch.setattr(rs.importlib, "import_module", imported.append)

    rs.preload_frozen_windows_diarization_dependencies()

    assert imported == [
        "torch",
        "torchaudio",
        "regex",
        "sentencepiece",
        "pyarrow",
        "speechbrain",
    ]


def test_diarization_preload_is_inactive_outside_frozen_windows(monkeypatch, import_runtime_services):
    rs = import_runtime_services
    imported = []
    monkeypatch.setattr(rs.sys, "platform", "darwin")
    monkeypatch.setattr(rs.sys, "frozen", True, raising=False)
    monkeypatch.setattr(rs.importlib, "import_module", imported.append)

    rs.preload_frozen_windows_diarization_dependencies()

    assert imported == []


def test_get_diarization_feature_state_resets_stale_enabled_status(monkeypatch, import_runtime_services):
    rs = import_runtime_services
    monkeypatch.setattr(
        rs,
        "load_state",
        lambda: {"features": {rs.DIARIZATION_FEATURE: {"status": "enabled", "requested": True}}, "secrets": {}, "install_profile": "standard"},
    )
    monkeypatch.setattr(rs, "is_diarization_installed", lambda: False)
    monkeypatch.setattr(rs, "can_self_install_optional_features", lambda: True)
    monkeypatch.setattr(rs, "get_build_profile", lambda: "standard")

    state = rs.get_diarization_feature_state()
    assert state["status"] == "not_installed"
    assert state["installed"] is False
    assert state["requested"] is True


def test_install_diarization_support_returns_early_when_installed(monkeypatch, import_runtime_services):
    rs = import_runtime_services
    updates = []
    monkeypatch.setattr(rs, "is_diarization_installed", lambda: True)
    monkeypatch.setattr(rs, "update_feature_state", lambda *args, **kwargs: updates.append((args, kwargs)))

    success, error = rs.install_diarization_support()

    assert success is True
    assert error is None
    assert updates[-1][1]["status"] == "enabled"


def test_install_diarization_support_fails_cleanly_without_self_install(monkeypatch, import_runtime_services):
    rs = import_runtime_services
    updates = []
    monkeypatch.setattr(rs, "is_diarization_installed", lambda: False)
    monkeypatch.setattr(rs, "can_self_install_optional_features", lambda: False)
    monkeypatch.setattr(rs, "update_feature_state", lambda *args, **kwargs: updates.append((args, kwargs)))

    success, error = rs.install_diarization_support()

    assert success is False
    assert "cannot install" in error.lower()
    assert updates[-1][1]["status"] == "failed"


def test_install_diarization_support_reports_success(monkeypatch, import_runtime_services):
    rs = import_runtime_services
    updates = []
    statuses = []
    progress = []

    monkeypatch.setattr(rs, "is_diarization_installed", lambda: False)
    monkeypatch.setattr(rs, "can_self_install_optional_features", lambda: True)
    monkeypatch.setattr(rs, "update_feature_state", lambda *args, **kwargs: updates.append((args, kwargs)))
    monkeypatch.setattr(rs.importlib, "invalidate_caches", lambda: None)

    installed_state = {"value": False}

    def fake_is_installed():
        return installed_state["value"]

    monkeypatch.setattr(rs, "is_diarization_installed", fake_is_installed)

    class FakeProcess:
        def __init__(self):
            self.stdout = io.StringIO("Collecting package\nInstalling package\n")
            self.returncode = 0

        def wait(self):
            installed_state["value"] = True

    monkeypatch.setattr(rs.subprocess, "Popen", lambda *a, **k: FakeProcess())

    success, error = rs.install_diarization_support(
        status_callback=statuses.append,
        progress_callback=progress.append,
    )

    assert success is True
    assert error is None
    assert statuses[0] == "Installing optional diarization support..."
    assert "Collecting package" in statuses
    assert progress[0] == 5
    assert progress[-1] == 100
    assert updates[-1][1]["status"] == "enabled"


def test_get_startup_diagnostics_contains_current_state(monkeypatch, import_runtime_services):
    rs = import_runtime_services
    monkeypatch.setattr(rs, "get_app_version", lambda: "9.9.9")
    monkeypatch.setattr(rs, "is_frozen_runtime", lambda: True)
    monkeypatch.setattr(rs, "get_build_profile", lambda: "complete")
    monkeypatch.setattr(
        rs,
        "get_diarization_feature_state",
        lambda: {"status": "enabled", "installed": True},
    )
    monkeypatch.setenv("MULTISOCIAL_FFMPEG_SOURCE", "bundled")

    diagnostics = rs.get_startup_diagnostics()

    assert diagnostics["version"] == "9.9.9"
    assert diagnostics["frozen"] is True
    assert diagnostics["install_profile"] == "complete"
    assert diagnostics["ffmpeg_source"] == "bundled"
    assert diagnostics["diarization_installed"] is True


def test_frozen_resource_path_finds_pyinstaller_internal_assets(tmp_path, monkeypatch, import_runtime_services):
    rs = import_runtime_services
    executable = tmp_path / "MultiSOCIAL-Standard.app" / "Contents" / "MacOS" / "MultiSOCIAL-Standard"
    asset = executable.parent / "_internal" / "assets" / "yolov5s.pt"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"weights")

    monkeypatch.setattr(rs.sys, "frozen", True, raising=False)
    monkeypatch.setattr(rs.sys, "_MEIPASS", str(tmp_path / "missing"), raising=False)
    monkeypatch.setattr(rs.sys, "executable", str(executable))
    monkeypatch.setattr(rs.sys, "platform", "darwin")

    assert rs.resource_path("assets", "yolov5s.pt") == str(asset)


def test_frozen_resource_path_finds_macos_resources_assets(tmp_path, monkeypatch, import_runtime_services):
    rs = import_runtime_services
    executable = tmp_path / "MultiSOCIAL-Complete.app" / "Contents" / "MacOS" / "MultiSOCIAL-Complete"
    asset = executable.parent.parent / "Resources" / "assets" / "yolov5s.pt"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"weights")

    monkeypatch.setattr(rs.sys, "frozen", True, raising=False)
    monkeypatch.setattr(rs.sys, "_MEIPASS", str(tmp_path / "missing"), raising=False)
    monkeypatch.setattr(rs.sys, "executable", str(executable))
    monkeypatch.setattr(rs.sys, "platform", "darwin")

    assert rs.resource_path("assets", "yolov5s.pt") == str(asset)
