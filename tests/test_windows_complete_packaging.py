from __future__ import annotations

import ast
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

    mediapipe_phase = hook.bundled_dll_directories(str(tmp_path), include_tensor_runtime=False)
    assert str(tmp_path / "mediapipe" / "python") in mediapipe_phase
    assert str(tmp_path / "torch" / "lib") not in mediapipe_phase


def test_complete_smoke_and_diarization_use_the_windows_native_preload():
    app_source = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
    native_backend_source = (ROOT / "src" / "native_backend.py").read_text(encoding="utf-8")
    audio_source = (ROOT / "src" / "audio.py").read_text(encoding="utf-8")

    imported_roots = {
        node.module.split(".", 1)[0]
        for node in ast.walk(ast.parse(app_source))
        if isinstance(node, ast.ImportFrom) and node.module
    }
    imported_roots.update(
        alias.name.split(".", 1)[0]
        for node in ast.walk(ast.parse(app_source))
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert "pyannote" not in imported_roots
    assert "runtime_services.preload_frozen_windows_diarization_dependencies()" in native_backend_source
    assert "preload_frozen_windows_diarization_dependencies()" in audio_source


def test_windows_gui_uses_the_private_worker_without_path_mutation():
    app_source = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
    backend_source = (ROOT / "src" / "worker_backend.py").read_text(encoding="utf-8")
    hook_source = (ROOT / "src" / "runtime_hook_dlls.py").read_text(encoding="utf-8")
    worker_source = (ROOT / "src" / "analysis_worker.py").read_text(encoding="utf-8")

    assert "WindowsAudioProcessor" not in app_source
    assert "WindowsPoseProcessor" not in app_source
    assert "WindowsAudioProcessor" in backend_source
    assert "WindowsPoseProcessor" in backend_source
    assert 'os.environ["PATH"]' not in hook_source
    assert "_configure_worker_native_loader" in worker_source
    assert "_enable_worker_tensor_loader" in worker_source


def test_windows_gui_does_not_statically_import_the_worker_native_graph():
    app_source = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
    entry_source = (ROOT / "src" / "app_windows.py").read_text(encoding="utf-8")
    spec_source = (ROOT / "packaging" / "windows_gui.spec").read_text(encoding="utf-8")

    imported_roots = set()
    for source in (app_source, entry_source):
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_roots.add(node.module.split(".", 1)[0])
    assert not imported_roots & {"audio", "pose", "mediapipe", "torch", "transformers", "pyannote"}
    assert "importlib.import_module" not in app_source
    assert 'str(SRC / "app_windows.py")' in spec_source
    assert "assert_gui_graph(a)" in spec_source


def test_windows_gui_and_embedded_worker_builds_are_independent():
    gui_spec = (ROOT / "packaging" / "windows_gui.spec").read_text(encoding="utf-8")
    workflow_source = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "assert_gui_graph(a)" in gui_spec
    assert "analysis_worker.py" not in gui_spec
    assert "Build native-free Windows GUI" in workflow_source
    assert "Build private embedded Windows worker runtime" in workflow_source
    assert "build_windows_embedded_worker.py" in workflow_source
    assert "assemble_windows.py" in workflow_source
    e2e_source = (
        ROOT / ".github" / "scripts" / "windows_packaged_e2e.py"
    ).read_text(encoding="utf-8")
    assert "for attempt in range(1, 11)" in e2e_source
    request_runner = (
        ROOT / ".github" / "scripts" / "run_windows_packaged_request.py"
    ).read_text(encoding="utf-8")
    assert "MULTISOCIAL_WORKER_SMOKE_REQUEST" in request_runner
    assert "run_windows_complete_e2e" in workflow_source
    assert "MULTISOCIAL_CI_HF_TOKEN" in workflow_source


def test_windows_worker_audit_uses_checked_in_runtime_manifests():
    audit_source = (ROOT / "packaging" / "audit_windows_environment.py").read_text(encoding="utf-8")
    manifest = (ROOT / "packaging" / "windows_hiddenimports.py").read_text(encoding="utf-8")

    assert "assert_worker_manifests_exist" in audit_source
    assert "TORCH_RUNTIME_HIDDEN_IMPORTS" in manifest
    assert "TORCHAUDIO_RUNTIME_HIDDEN_IMPORTS" in manifest
    assert "transformers.models.whisper.modeling_whisper" in manifest
    assert "transformers.integrations.ggml" in manifest
    assert "pyannote.audio.pipelines.speaker_diarization" in manifest
def test_embedded_worker_copies_metadata_with_its_runtime_and_uses_native_launcher():
    builder_source = (ROOT / "packaging" / "build_windows_embedded_worker.py").read_text(encoding="utf-8")
    launcher_source = (ROOT / "packaging" / "windows_worker_launcher.c").read_text(encoding="utf-8")

    assert 'shutil.copytree(\n        site_packages,\n        output / "Lib" / "site-packages",' in builder_source
    assert "SetDllDirectoryW(NULL)" in launcher_source
    assert "reset_pyinstaller_environment();" in launcher_source
    assert "_PYI_ARCHIVE_FILE" in launcher_source
    assert "_PYI_PARENT_PROCESS_LEVEL" in launcher_source
    assert "CreateProcessW" in launcher_source
    assert "CREATE_NO_WINDOW" in launcher_source
    assert "MULTISOCIAL_WORKER_PROTOCOL_HOST" in launcher_source
    assert "socket_protocol ? FALSE : TRUE" in launcher_source
    assert "GetWindowsDirectoryW" in launcher_source
    assert "child_directory" in launcher_source
    assert "GetShortPathNameW" in launcher_source
    assert "DefineDosDeviceW" in launcher_source
    assert "create_ascii_drive_alias" in launcher_source
    assert "MULTISOCIAL_WORKER_RUNTIME_ROOT" in launcher_source
    assert "SetEnvironmentVariableW(WORKER_RUNTIME_ROOT_ENV, runtime_root)" in launcher_source
    assert 'L"\\\\\\\\??\\\\%s"' not in launcher_source
    assert 'L"\\\\??\\\\%s"' in launcher_source
    assert "StringCchPrintfA" in launcher_source
    assert "lstrlenA(message)" in launcher_source
    assert "error = GetLastError();" in launcher_source
    assert "return fail(error);" in launcher_source
    assert "CLEAN_BOOTSTRAP_ARGUMENT" not in launcher_source
    assert "WriteFile(" in launcher_source
    assert "_PrivateWorkerSmile" in (ROOT / "src" / "audio.py").read_text(encoding="utf-8")
    launcher_builder = (ROOT / "packaging" / "build_windows_worker_launcher.py").read_text(encoding="utf-8")
    assert "/MT" in launcher_builder
    assert "def _write_build_batch(output: Path)" in launcher_builder
    assert 'subprocess.run([str(batch_file)], check=True)' in launcher_builder
    assert 'call "{vcvars}" >nul' in launcher_builder
    assert "/DUNICODE" not in launcher_builder
    assert "/D_UNICODE" not in launcher_builder
    complete_layout = (ROOT / ".github" / "scripts" / "validate_complete_bundle_layout.py").read_text(encoding="utf-8")
    assert 'for distribution in ("regex", "requests")' in complete_layout


def test_worker_initializes_native_modules_before_starting_the_operation_thread():
    worker_source = (ROOT / "src" / "analysis_worker.py").read_text(encoding="utf-8")

    assert "_initialize_worker_operation_runtime(operation, payload)" in worker_source
    assert worker_source.index("_initialize_worker_operation_runtime(operation, payload)") < worker_source.index(
        "threading.Thread(target=run_operation, daemon=True).start()"
    )


def test_windows_gui_does_not_force_gpu_configuration_into_the_private_worker():
    app_source = (ROOT / "src" / "app.py").read_text(encoding="utf-8")
    client_source = (ROOT / "src" / "native_worker_client.py").read_text(encoding="utf-8")

    assert 'if not sys.platform.startswith("win"):' in app_source
    assert '"CUDA_VISIBLE_DEVICES"' in client_source
    assert '"PYTORCH_ENABLE_MPS_FALLBACK"' in client_source


def test_packaged_windows_gui_uses_handle_free_authenticated_worker_protocol():
    client_source = (ROOT / "src" / "native_worker_client.py").read_text(encoding="utf-8")
    worker_source = (ROOT / "src" / "analysis_worker.py").read_text(encoding="utf-8")

    assert "WORKER_PROTOCOL_HOST_ENV" in client_source
    assert "socket.create_connection" in worker_source
    assert "event\": \"ready\"" in worker_source
    assert "stdin=subprocess.DEVNULL" in client_source
    assert "stdout=subprocess.DEVNULL" in client_source
