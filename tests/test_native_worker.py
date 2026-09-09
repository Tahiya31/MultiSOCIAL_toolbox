from __future__ import annotations

import os
import io
import json
import queue
import sys
import threading
import time
import types
from pathlib import Path

import pytest


def _worker_script(tmp_path, source: str):
    path = tmp_path / "worker.py"
    path.write_text(source, encoding="utf-8")
    return [sys.executable, str(path)]


def test_worker_client_forwards_protocol_progress_and_result(tmp_path):
    from native_worker_client import NativeWorkerClient

    command = _worker_script(
        tmp_path,
        """import json, sys
request = json.loads(sys.stdin.readline())
base = {\"protocol\": 1, \"id\": request[\"id\"]}
print(json.dumps({**base, \"event\": \"progress\", \"value\": 40}), flush=True)
print(json.dumps({**base, \"event\": \"result\", \"result\": {\"succeeded\": [\"ok\"], \"failed\": [], \"cancelled\": False}}), flush=True)
""",
    )
    updates = []
    result = NativeWorkerClient(command=command).run("probe", {}, progress_callback=updates.append)

    assert updates == [40]
    assert result == {"succeeded": ["ok"], "failed": [], "cancelled": False}


def test_worker_client_redacts_child_token_in_errors(tmp_path):
    from native_worker_client import NativeWorkerClient, WorkerError

    command = _worker_script(
        tmp_path,
        """import json, os, sys
request = json.loads(sys.stdin.readline())
print(json.dumps({\"protocol\": 1, \"id\": request[\"id\"], \"event\": \"error\", \"message\": os.environ[\"MULTISOCIAL_WORKER_HF_TOKEN\"]}), flush=True)
""",
    )

    with pytest.raises(WorkerError, match="REDACTED") as exc:
        NativeWorkerClient(command=command).run("probe", {}, hf_token="hf-secret")
    assert "hf-secret" not in str(exc.value)


def test_worker_client_cancels_unresponsive_child(tmp_path):
    from native_worker_client import NativeWorkerClient

    command = _worker_script(
        tmp_path,
        """import json, sys, time
json.loads(sys.stdin.readline())
time.sleep(30)
""",
    )
    cancelled = threading.Event()
    cancelled.set()
    started = time.monotonic()
    result = NativeWorkerClient(command=command, cancel_grace_seconds=0.05).run(
        "probe", {}, cancel_check=cancelled.is_set
    )

    assert result["cancelled"] is True
    assert time.monotonic() - started < 5


def test_worker_client_timeout_reports_the_last_redacted_diagnostic_stage(tmp_path, monkeypatch):
    import native_worker_client
    from native_worker_client import NativeWorkerClient, WorkerError

    diagnostic = tmp_path / "worker.jsonl"
    monkeypatch.setattr(native_worker_client, "_worker_diagnostic_path", lambda _request_id: diagnostic)
    command = _worker_script(
        tmp_path,
        """import json, os, sys, time
json.loads(sys.stdin.readline())
with open(os.environ["MULTISOCIAL_WORKER_DIAGNOSTIC_PATH"], "w", encoding="utf-8") as output:
    output.write('{"stage":"preload:mediapipe"}\\n')
time.sleep(30)
""",
    )

    with pytest.raises(WorkerError, match="preload:mediapipe"):
        NativeWorkerClient(command=command, timeout_seconds=5).run("probe", {})

    assert diagnostic.is_file()

def test_forced_cancellation_cleans_only_its_request_staging(tmp_path):
    from native_worker_client import NativeWorkerClient

    destination = tmp_path / "outputs"
    destination.mkdir()
    unrelated = destination / ".multisocial-worker-unrelated-keep"
    unrelated.mkdir()
    command = _worker_script(
        tmp_path,
        """import json, pathlib, sys, time
request = json.loads(sys.stdin.readline())
destination = pathlib.Path(request["payload"]["output_audio_features_folder"])
(destination / (".multisocial-worker-" + request["id"] + "-orphan")).mkdir()
time.sleep(30)
""",
    )
    result = NativeWorkerClient(
        command=command,
        cancel_grace_seconds=0.05,
    ).run(
        "extract_audio_features",
        {"output_audio_features_folder": str(destination)},
        cancel_check=lambda: True,
    )

    assert result["cancelled"] is True
    assert unrelated.is_dir()
    assert list(destination.iterdir()) == [unrelated]


def test_windows_worker_launch_uses_deterministic_windowless_creation(monkeypatch):
    import native_worker_client

    monkeypatch.setattr(native_worker_client.sys, "platform", "win32")
    monkeypatch.setattr(native_worker_client.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)

    kwargs = native_worker_client._worker_popen_kwargs()

    assert kwargs == {"creationflags": 0x08000000}


def test_packaged_windows_worker_uses_an_authenticated_socket_not_gui_stdio(tmp_path, monkeypatch):
    import native_worker_client
    from native_worker_client import NativeWorkerClient

    class Reader:
        def __init__(self):
            self.lines = queue.Queue()

        def __iter__(self):
            while (line := self.lines.get()) is not None:
                yield line

    class Writer(io.StringIO):
        def __init__(self, reader):
            super().__init__()
            self.reader = reader

        def write(self, value):
            message = json.loads(value)
            if message.get("type") == "run":
                self.reader.lines.put(
                    json.dumps(
                        {
                            "protocol": 1,
                            "id": message["id"],
                            "event": "result",
                            "result": {"succeeded": ["socket"], "failed": [], "cancelled": False},
                        }
                    )
                    + "\n"
                )
            return super().write(value)

    class Connection:
        def __init__(self):
            self.reader = Reader()
            self.writer = Writer(self.reader)

        def makefile(self, mode, **_kwargs):
            return self.reader if mode == "r" else self.writer

        def close(self):
            self.reader.lines.put(None)

    class Listener:
        def getsockname(self):
            return ("127.0.0.1", 4242)

        def close(self):
            pass

    class Process:
        returncode = 0

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

        def terminate(self):
            pass

    connection = Connection()
    connection.reader.lines.put(
        json.dumps({"protocol": 1, "event": "ready", "token": "expected-token"}) + "\n"
    )
    spawned = []
    monkeypatch.setattr(native_worker_client.sys, "platform", "win32")
    monkeypatch.setattr(native_worker_client.sys, "frozen", True, raising=False)
    monkeypatch.setattr(native_worker_client, "_worker_popen_kwargs", lambda: {})
    monkeypatch.setattr(native_worker_client, "_create_worker_protocol_listener", Listener)
    monkeypatch.setattr(native_worker_client, "_accept_worker_protocol", lambda *_args: connection)
    monkeypatch.setattr(native_worker_client.secrets, "token_urlsafe", lambda _size: "expected-token")
    monkeypatch.setattr(native_worker_client, "_spawn_worker", lambda *args, **kwargs: spawned.append(kwargs) or Process())

    result = NativeWorkerClient(command=["fake-worker"]).run("probe", {})

    assert result["succeeded"] == ["socket"]
    assert spawned[0]["stdin"] is native_worker_client.subprocess.DEVNULL
    assert spawned[0]["stdout"] is native_worker_client.subprocess.DEVNULL
    assert spawned[0]["stderr"] is native_worker_client.subprocess.DEVNULL


def test_worker_socket_protocol_consumes_its_configuration_and_announces_ready(monkeypatch):
    import analysis_worker

    class Connection:
        def __init__(self):
            self.reader = io.StringIO()
            self.writer = io.StringIO()
            self.timeout = None

        def settimeout(self, value):
            self.timeout = value

        def makefile(self, mode, **_kwargs):
            return self.reader if mode == "r" else self.writer

    connection = Connection()
    monkeypatch.setenv("MULTISOCIAL_WORKER_PROTOCOL_HOST", "127.0.0.1")
    monkeypatch.setenv("MULTISOCIAL_WORKER_PROTOCOL_PORT", "4242")
    monkeypatch.setenv("MULTISOCIAL_WORKER_PROTOCOL_TOKEN", "private-nonce")
    monkeypatch.setattr(
        analysis_worker.socket,
        "create_connection",
        lambda endpoint, timeout: connection if endpoint == ("127.0.0.1", 4242) and timeout == 15 else None,
    )

    reader, writer = analysis_worker._open_protocol_stream()

    assert reader is connection.reader
    assert writer is connection.writer
    assert connection.timeout is None
    assert json.loads(connection.writer.getvalue()) == {
        "protocol": 1,
        "event": "ready",
        "token": "private-nonce",
    }
    assert "MULTISOCIAL_WORKER_PROTOCOL_HOST" not in os.environ
    assert "MULTISOCIAL_WORKER_PROTOCOL_PORT" not in os.environ
    assert "MULTISOCIAL_WORKER_PROTOCOL_TOKEN" not in os.environ


def test_windows_forced_cancellation_terminates_the_worker_process_tree(monkeypatch):
    import native_worker_client

    class Process:
        pid = 123

        @staticmethod
        def poll():
            return None

    calls = []
    monkeypatch.setattr(native_worker_client.sys, "platform", "win32")
    monkeypatch.setattr(
        native_worker_client.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    native_worker_client._terminate_worker_tree(Process())

    assert calls[0][0] == ["taskkill", "/PID", "123", "/T", "/F"]


def test_frozen_windows_worker_uses_its_private_runtime_directory(tmp_path, monkeypatch):
    import native_worker_client

    executable = tmp_path / "MultiSOCIAL-Standard.exe"
    worker = tmp_path / "worker" / "MultiSOCIAL-Worker-Launcher.exe"
    worker.parent.mkdir()
    executable.touch()
    worker.touch()
    monkeypatch.setattr(native_worker_client.sys, "frozen", True, raising=False)
    monkeypatch.setattr(native_worker_client.sys, "executable", str(executable))

    assert native_worker_client.worker_command() == [str(worker)]


def test_packaged_windows_worker_environment_removes_gui_state(tmp_path, monkeypatch):
    import native_worker_client

    gui = tmp_path / "app" / "MultiSOCIAL-Standard.exe"
    worker = gui.parent / "worker" / "MultiSOCIAL-Worker-Launcher.exe"
    worker.parent.mkdir(parents=True)
    gui.touch()
    worker.touch()
    monkeypatch.setattr(native_worker_client.sys, "platform", "win32")
    monkeypatch.setattr(native_worker_client.sys, "frozen", True, raising=False)
    monkeypatch.setattr(native_worker_client.sys, "executable", str(gui))
    monkeypatch.setenv("PATH", f"{gui.parent};C:\\Windows\\System32")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setenv("PYTHONHOME", "bad")
    monkeypatch.setenv("PYTHONPATH", "bad")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    monkeypatch.setenv("MULTISOCIAL_FFMPEG_EXE", str(gui.parent / "ffmpeg.exe"))
    for name in native_worker_client._PYINSTALLER_PRIVATE_ENVIRONMENT_NAMES:
        monkeypatch.setenv(name, "gui-private-state")

    env = native_worker_client._packaged_windows_environment([str(worker)], None)

    assert env["PYINSTALLER_RESET_ENVIRONMENT"] == "1"
    assert "MULTISOCIAL_WINDOWS_EMBEDDED_WORKER" not in env
    assert env["PATH"].split(";")[0] == str(worker.parent)
    assert str(gui.parent) not in env["PATH"].split(";")[1:]
    windows_root = native_worker_client.Path(r"C:\Windows")
    assert env["PATH"].split(";")[1:] == [
        str(windows_root / "System32"),
        str(windows_root),
        str(windows_root / "System32" / "Wbem"),
        str(windows_root / "System32" / "WindowsPowerShell" / "v1.0"),
    ]
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    for name in native_worker_client._WINDOWS_GUI_NATIVE_ENVIRONMENT_NAMES:
        assert name not in env
    assert "MULTISOCIAL_FFMPEG_EXE" not in env
    for name in native_worker_client._PYINSTALLER_PRIVATE_ENVIRONMENT_NAMES:
        assert name not in env

def test_worker_payload_paths_are_absolute_before_worker_cwd_changes(tmp_path, monkeypatch):
    import native_worker_client

    monkeypatch.chdir(tmp_path)
    payload = native_worker_client._absolutize_worker_payload(
        {
            "video_files": ["inputs/video.avi"],
            "output_csv_folder": "outputs",
            "alignment_pairs": [["features.csv", "transcript.json", "aligned.csv"]],
        }
    )

    assert payload["video_files"] == [str(tmp_path / "inputs" / "video.avi")]
    assert payload["output_csv_folder"] == str(tmp_path / "outputs")
    assert payload["alignment_pairs"] == [[
        str(tmp_path / "features.csv"),
        str(tmp_path / "transcript.json"),
        str(tmp_path / "aligned.csv"),
    ]]


def test_packaged_windows_spawn_does_not_mutate_the_gui_dll_directory(monkeypatch):
    import native_worker_client

    sentinel = object()
    monkeypatch.setattr(native_worker_client.sys, "platform", "win32")
    monkeypatch.setattr(native_worker_client.sys, "frozen", True, raising=False)
    monkeypatch.setattr(native_worker_client.subprocess, "Popen", lambda *args, **kwargs: sentinel)

    result = native_worker_client._spawn_worker(["MultiSOCIAL-Worker.exe"])

    assert result is sentinel


def test_worker_client_preserves_a_caller_supplied_diagnostic_path(tmp_path, monkeypatch):
    import native_worker_client
    from native_worker_client import NativeWorkerClient

    diagnostic = tmp_path / "caller-diagnostic.jsonl"
    command = _worker_script(
        tmp_path,
        """import json, os, sys
request = json.loads(sys.stdin.readline())
with open(os.environ["MULTISOCIAL_WORKER_DIAGNOSTIC_PATH"], "w", encoding="utf-8") as output:
    output.write('{"stage":"result-emitted"}\\n')
print(json.dumps({"protocol": 1, "id": request["id"], "event": "result", "result": {}}), flush=True)
""",
    )
    monkeypatch.setenv(native_worker_client.WORKER_DIAGNOSTIC_ENV, str(diagnostic))

    NativeWorkerClient(command=command).run("probe", {})

    assert not diagnostic.exists()


def test_worker_client_can_preserve_successful_diagnostics_for_ci(tmp_path, monkeypatch):
    import native_worker_client
    from native_worker_client import NativeWorkerClient

    diagnostic = tmp_path / "caller-diagnostic.jsonl"
    command = _worker_script(
        tmp_path,
        """import json, os, sys
request = json.loads(sys.stdin.readline())
with open(os.environ["MULTISOCIAL_WORKER_DIAGNOSTIC_PATH"], "w", encoding="utf-8") as output:
    output.write('{"stage":"result-emitted","elapsed_ms":123}\\n')
print(json.dumps({"protocol": 1, "id": request["id"], "event": "result", "result": {}}), flush=True)
""",
    )
    monkeypatch.setenv(native_worker_client.WORKER_DIAGNOSTIC_ENV, str(diagnostic))
    monkeypatch.setenv(native_worker_client.WORKER_PRESERVE_DIAGNOSTICS_ENV, "1")

    NativeWorkerClient(command=command).run("probe", {})

    assert diagnostic.read_text(encoding="utf-8") == '{"stage":"result-emitted","elapsed_ms":123}\n'


def test_worker_probe_metadata_reports_the_current_architecture():
    from analysis_worker import _runtime_metadata

    metadata = _runtime_metadata()

    assert metadata["platform"] == sys.platform
    assert metadata["is_64bit"] is (sys.maxsize > 2**32)
    assert "worker_runtime" in metadata


def test_worker_diagnostics_exclude_unapproved_detail_fields(tmp_path, monkeypatch):
    import json
    import analysis_worker

    diagnostic = tmp_path / "worker.jsonl"
    monkeypatch.setenv(analysis_worker.WORKER_DIAGNOSTIC_ENV, str(diagnostic))

    analysis_worker._diagnostic_stage(
        "preload:mediapipe",
        operation="probe",
        secret="hf-private-token",
        path=str(tmp_path),
    )

    record = json.loads(diagnostic.read_text(encoding="utf-8"))
    assert record["stage"] == "preload:mediapipe"
    assert record["operation"] == "probe"
    assert "secret" not in record
    assert "path" not in record


def test_worker_diagnostics_record_only_the_clean_native_environment_boolean(tmp_path, monkeypatch):
    import analysis_worker

    diagnostic = tmp_path / "worker.jsonl"
    monkeypatch.setenv(analysis_worker.WORKER_DIAGNOSTIC_ENV, str(diagnostic))
    analysis_worker._diagnostic_stage(
        "base-loader-configured",
        gui_native_environment_clean=True,
    )

    record = __import__("json").loads(diagnostic.read_text(encoding="utf-8"))
    assert record["gui_native_environment_clean"] is True


def test_worker_runtime_flags_dlls_loaded_from_gui_root(tmp_path, monkeypatch):
    import analysis_worker

    worker_root = tmp_path / "app" / "worker"
    worker_root.mkdir(parents=True)
    monkeypatch.setattr(analysis_worker.sys, "platform", "win32")
    monkeypatch.setattr(analysis_worker.sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        analysis_worker,
        "_loaded_windows_module_paths",
        lambda: [
            worker_root / "torch_cpu.dll",
            worker_root.parent / "wxbase.dll",
        ],
    )

    assert analysis_worker._loaded_module_violations(worker_root) == ["wxbase.dll"]


def test_worker_runtime_flags_package_native_modules_outside_worker(tmp_path, monkeypatch):
    import analysis_worker

    worker_root = tmp_path / "app" / "worker"
    worker_root.mkdir(parents=True)
    monkeypatch.setattr(analysis_worker.sys, "platform", "win32")
    monkeypatch.setattr(analysis_worker.sys, "frozen", True, raising=False)
    monkeypatch.setattr(
        analysis_worker,
        "_loaded_windows_module_paths",
        lambda: [
            worker_root / "torch_cpu.dll",
            tmp_path / "foreign" / "_framework_bindings.pyd",
        ],
    )

    assert analysis_worker._native_module_violations(worker_root) == ["_framework_bindings.pyd"]

def test_worker_runtime_rejects_all_non_system_external_modules(tmp_path, monkeypatch):
    import analysis_worker

    worker_root = tmp_path / "app" / "worker"
    windows_root = tmp_path / "Windows"
    worker_root.mkdir(parents=True)
    (windows_root / "System32").mkdir(parents=True)
    monkeypatch.setattr(analysis_worker.sys, "platform", "win32")
    monkeypatch.setattr(analysis_worker.sys, "frozen", True, raising=False)
    monkeypatch.setenv("SystemRoot", str(windows_root))
    loaded = [
        worker_root / "MultiSOCIAL-Worker.exe",
        windows_root / "System32" / "kernel32.dll",
        tmp_path / "foreign" / "injected.dll",
    ]

    assert analysis_worker._external_module_violations(worker_root, loaded) == ["injected.dll"]


def test_worker_path_containment_resolves_an_alias_before_provenance_checks(tmp_path):
    import analysis_worker

    worker_root = tmp_path / "Unicode worker"
    module = worker_root / "Lib" / "site-packages" / "mediapipe" / "__init__.py"
    module.parent.mkdir(parents=True)
    module.touch()
    alias = tmp_path / "worker-alias"
    alias.symlink_to(worker_root, target_is_directory=True)

    assert analysis_worker._path_is_within(alias / "Lib" / "site-packages" / "mediapipe" / "__init__.py", worker_root)


def test_worker_preloads_pose_runtime_in_mediapipe_then_torch_order(monkeypatch):
    import analysis_worker

    events = []
    monkeypatch.setattr(analysis_worker, "_import_worker_runtime_module", lambda name: events.append(name))
    monkeypatch.setattr(analysis_worker, "_enable_worker_tensor_loader", lambda: events.append("enable-torch"))

    analysis_worker._initialize_worker_operation_runtime("extract_pose", {})

    assert events == ["mediapipe", "cv2", "enable-torch", "pose"]


def test_embedded_worker_skips_pyinstaller_dll_directory_registration(monkeypatch):
    import analysis_worker

    monkeypatch.setattr(analysis_worker.sys, "platform", "win32")
    monkeypatch.setattr(analysis_worker, "_is_embedded_worker_runtime", lambda: True)
    monkeypatch.setattr(analysis_worker.os, "add_dll_directory", lambda _path: None, raising=False)

    analysis_worker._configure_worker_native_loader()
    analysis_worker._enable_worker_tensor_loader()


def test_worker_preloads_complete_probe_runtime_on_the_main_thread(monkeypatch):
    import analysis_worker

    events = []
    monkeypatch.setattr(analysis_worker, "_import_worker_runtime_module", lambda name: events.append(name))
    monkeypatch.setattr(analysis_worker, "_enable_worker_tensor_loader", lambda: events.append("enable-torch"))

    analysis_worker._initialize_worker_operation_runtime("probe", {"profile": "complete"})

    assert events == ["mediapipe", "cv2", "opensmile", "enable-torch", "torch", "torchaudio", "pyannote.audio"]

def test_worker_preloads_diarization_dependencies_before_provenance_check(monkeypatch):
    import analysis_worker

    events = []
    monkeypatch.setattr(
        analysis_worker,
        "_import_worker_runtime_module",
        lambda name: events.append(name),
    )
    monkeypatch.setattr(
        analysis_worker,
        "_enable_worker_tensor_loader",
        lambda: events.append("enable-torch"),
    )

    analysis_worker._initialize_worker_operation_runtime(
        "extract_transcripts",
        {"enable_diarization": True},
    )

    assert events == ["enable-torch", "audio", "torchaudio", "pyannote.audio"]
    assert "pyannote.audio" in analysis_worker._operation_module_names(
        "extract_transcripts",
        {"enable_diarization": True},
    )


def test_worker_alignment_reports_the_same_file_status_as_native_batch(tmp_path, monkeypatch):
    import analysis_worker

    class FakeAudioProcessor:
        def __init__(self, *_args, **_kwargs):
            pass

        def align_features(self, _features, _transcript, output):
            Path(output).write_text("aligned", encoding="utf-8")
            return output

    monkeypatch.setattr(analysis_worker, "_enable_worker_tensor_loader", lambda: None)
    monkeypatch.setitem(
        sys.modules,
        "audio",
        types.SimpleNamespace(AudioProcessor=FakeAudioProcessor),
    )
    output = tmp_path / "aligned.csv"
    statuses = []

    result = analysis_worker._run_alignment(
        {"alignment_pairs": [["features.csv", "words.json", str(output)]]},
        threading.Event(),
        lambda _value: None,
        statuses.append,
    )

    assert result["succeeded"] == [str(output)]
    assert statuses == ["Aligning: aligned.csv"]


def test_staged_output_promotes_only_complete_files(tmp_path):
    from analysis_worker import _StagedOutput

    destination = tmp_path / "destination"
    stage = _StagedOutput(str(destination))
    with stage as staged:
        staged_file = os.path.join(staged, "result.csv")
        with open(staged_file, "w", encoding="utf-8") as handle:
            handle.write("complete")
        stage.commit()

    assert (destination / "result.csv").read_text(encoding="utf-8") == "complete"
    assert not list(destination.glob(".multisocial-worker-*"))


def test_staged_output_removes_cancelled_partial_files(tmp_path):
    from analysis_worker import _StagedOutput

    destination = tmp_path / "destination"
    with _StagedOutput(str(destination)) as staged:
        with open(os.path.join(staged, "partial.csv"), "w", encoding="utf-8") as handle:
            handle.write("partial")

    assert not (destination / "partial.csv").exists()
    assert not list(destination.glob(".multisocial-worker-*"))

def test_staged_output_names_are_scoped_to_request_id(tmp_path):
    from analysis_worker import _StagedOutput

    destination = tmp_path / "destination"
    with _StagedOutput(str(destination), "request-123") as staged:
        assert os.path.basename(staged).startswith(
            ".multisocial-worker-request-123-"
        )


def test_worker_resets_its_dll_directory_before_registering_private_native_paths():
    worker_source = (Path(__file__).resolve().parents[1] / "src" / "analysis_worker.py").read_text(
        encoding="utf-8"
    )

    reset = "ctypes.windll.kernel32.SetDllDirectoryW(None)"
    register = "configure_windows_dll_search_path(bundle_root, include_tensor_runtime=False)"
    assert reset in worker_source
    assert worker_source.index(reset) < worker_source.index(register)

def test_packaged_smoke_request_writes_redacted_structured_failure(tmp_path, monkeypatch):
    import worker_backend

    request = tmp_path / "request.json"
    result = tmp_path / "result.json"
    request.write_text(
        '{"operation":"probe","payload":{"profile":"complete"}}',
        encoding="utf-8",
    )
    token = "hf-private-token"

    class FailingClient:
        def run(self, *_args, **_kwargs):
            raise RuntimeError(f"model rejected {token}")

    monkeypatch.setattr(worker_backend, "NativeWorkerClient", FailingClient)
    monkeypatch.setenv(worker_backend.SMOKE_REQUEST_ENV, str(request))
    monkeypatch.setenv(worker_backend.SMOKE_RESULT_ENV, str(result))
    monkeypatch.setenv(worker_backend.WORKER_TOKEN_ENV, token)

    with pytest.raises(RuntimeError, match="REDACTED") as exc:
        worker_backend._run_packaged_smoke_request()
    assert token not in str(exc.value)

    response = __import__("json").loads(result.read_text(encoding="utf-8"))
    assert response == {
        "ok": False,
        "error": "model rejected [REDACTED]",
    }
    assert worker_backend.WORKER_TOKEN_ENV not in os.environ
