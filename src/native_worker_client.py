"""Windows-only client and lightweight facades for the private analysis worker.

This module intentionally has no native ML imports.  It is safe for the GUI
process to import before wxPython starts on every platform.
"""

from __future__ import annotations

import glob
import ctypes
import hmac
import json
import os
import queue
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional


PROTOCOL_VERSION = 1
WORKER_TOKEN_ENV = "MULTISOCIAL_WORKER_HF_TOKEN"
WORKER_DIAGNOSTIC_ENV = "MULTISOCIAL_WORKER_DIAGNOSTIC_PATH"
# Opt-in CI-only switch. Normal application launches always remove the
# short-lived diagnostic file once a request completes successfully.
WORKER_PRESERVE_DIAGNOSTICS_ENV = "MULTISOCIAL_PRESERVE_WORKER_DIAGNOSTICS"
WORKER_PROTOCOL_HOST_ENV = "MULTISOCIAL_WORKER_PROTOCOL_HOST"
WORKER_PROTOCOL_PORT_ENV = "MULTISOCIAL_WORKER_PROTOCOL_PORT"
WORKER_PROTOCOL_TOKEN_ENV = "MULTISOCIAL_WORKER_PROTOCOL_TOKEN"
WORKER_EXECUTABLE_NAME = "python.exe"
WORKER_LAUNCHER_NAME = "MultiSOCIAL-Worker-Launcher.exe"
_PYINSTALLER_PRIVATE_ENVIRONMENT_NAMES = (
    "_PYI_APPLICATION_HOME_DIR",
    "_PYI_ARCHIVE_FILE",
    "_PYI_PARENT_PROCESS_LEVEL",
    "_PYI_SPLASH_IPC",
    "_MEIPASS2",
)
_WINDOWS_GUI_NATIVE_ENVIRONMENT_NAMES = (
    "CUDA_VISIBLE_DEVICES",
    "PYTORCH_ENABLE_MPS_FALLBACK",
)


class WorkerError(RuntimeError):
    """A worker failed without returning a successful operation result."""


def is_windows_worker_enabled() -> bool:
    """Return whether this launch must isolate native analysis from the GUI."""
    return sys.platform == "win32"


def find_pose_csv_paths(output_csv_folder: str, video_path: str, multi_person: Optional[bool] = None) -> list[str]:
    """Native-import-free equivalent of ``pose.find_pose_csv_paths``."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    if multi_person is True:
        return sorted(glob.glob(os.path.join(output_csv_folder, f"{base}_multi_ID_*.csv")))
    if multi_person is False:
        return sorted(glob.glob(os.path.join(output_csv_folder, f"{base}_ID_*.csv")))
    multi = sorted(glob.glob(os.path.join(output_csv_folder, f"{base}_multi_ID_*.csv")))
    return multi or sorted(glob.glob(os.path.join(output_csv_folder, f"{base}_ID_*.csv")))


def _redact(value: str, token: Optional[str]) -> str:
    text = str(value or "")
    if token:
        text = text.replace(token, "[REDACTED]")
    return text.replace("\r", " ").replace("\n", " ")[-2000:]


def _diagnostic_stage(path: Path | None) -> str | None:
    """Read only the final safe stage name from a worker diagnostic stream."""
    if path is None:
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            stage = json.loads(line).get("stage")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(stage, str) and stage:
            return stage
    return None


def _worker_diagnostic_path(request_id: str) -> Path:
    """Create a private, non-user-facing breadcrumb file for one worker launch."""
    return Path(tempfile.gettempdir()) / f"multisocial-worker-{request_id}.jsonl"


def worker_command() -> list[str]:
    """Locate the bundled console worker, or run it from source for tests."""
    if getattr(sys, "frozen", False):
        candidate = Path(sys.executable).parent / "worker" / WORKER_LAUNCHER_NAME
        if not candidate.is_file():
            raise WorkerError(f"Bundled analysis worker launcher is missing: {candidate.name}")
        return [str(candidate)]
    return [sys.executable, str(Path(__file__).with_name("analysis_worker.py"))]


def _worker_popen_kwargs() -> dict[str, Any]:
    """Keep the private console worker windowless without STARTUPINFO state."""
    if sys.platform != "win32":
        return {}
    return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}


def _uses_socket_worker_protocol() -> bool:
    """Use a handle-free protocol only for packaged Windows GUI launches."""
    return sys.platform == "win32" and bool(getattr(sys, "frozen", False))


def _create_worker_protocol_listener() -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    listener.settimeout(0.1)
    return listener


def _accept_worker_protocol(
    listener: socket.socket,
    process: subprocess.Popen,
    *,
    timeout_seconds: float = 30.0,
) -> socket.socket:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            connection, address = listener.accept()
        except socket.timeout:
            if process.poll() is not None:
                raise WorkerError(f"worker exited with code {process.returncode} before connecting")
            continue
        if address[0] == "127.0.0.1":
            connection.settimeout(None)
            return connection
        connection.close()
    raise WorkerError("worker did not connect to its private protocol endpoint")


def _path_is_within(path: str | Path, root: str | Path) -> bool:
    try:
        Path(os.path.normcase(os.path.abspath(path))).relative_to(
            Path(os.path.normcase(os.path.abspath(root)))
        )
        return True
    except (OSError, ValueError):
        return False


def _worker_directory(command: list[str]) -> Path:
    executable = Path(command[0]).resolve()
    if executable.name.casefold() == WORKER_LAUNCHER_NAME.casefold():
        return executable.parent
    if len(command) > 1:
        return Path(command[1]).resolve().parent
    if executable.name.casefold() == WORKER_EXECUTABLE_NAME.casefold():
        return executable.parent
    return executable.parent


def _windows_directory() -> Path:
    if sys.platform == "win32":
        try:
            get_windows_directory = ctypes.windll.kernel32.GetWindowsDirectoryW
            get_windows_directory.argtypes = [
                ctypes.POINTER(ctypes.c_wchar),
                ctypes.c_uint,
            ]
            get_windows_directory.restype = ctypes.c_uint
            buffer = ctypes.create_unicode_buffer(32768)
            length = get_windows_directory(buffer, len(buffer))
            if 0 < length < len(buffer):
                return Path(buffer.value)
        except (AttributeError, OSError):
            pass
    return Path(os.environ.get("SystemRoot") or os.environ.get("WINDIR") or r"C:\Windows")


def _packaged_windows_environment(command: list[str], token: Optional[str]) -> dict[str, str]:
    env = os.environ.copy()
    if token:
        env[WORKER_TOKEN_ENV] = token
    else:
        env.pop(WORKER_TOKEN_ENV, None)

    if sys.platform != "win32" or not getattr(sys, "frozen", False):
        return env

    app_root = Path(sys.executable).resolve().parent
    worker_dir = _worker_directory(command)
    env["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    # The launcher is not itself a PyInstaller executable. Remove the GUI
    # bootloader's private state before it starts the worker so the worker's
    # bootloader cannot mistake itself for a GUI subprocess.
    for name in _PYINSTALLER_PRIVATE_ENVIRONMENT_NAMES:
        env.pop(name, None)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    for name in _WINDOWS_GUI_NATIVE_ENVIRONMENT_NAMES:
        env.pop(name, None)

    windows_root = _windows_directory()
    env["SystemRoot"] = str(windows_root)
    env["WINDIR"] = str(windows_root)
    system_paths = [
        windows_root / "System32",
        windows_root,
        windows_root / "System32" / "Wbem",
        windows_root / "System32" / "WindowsPowerShell" / "v1.0",
    ]
    env["PATH"] = ";".join(
        dict.fromkeys(str(path) for path in [worker_dir, *system_paths])
    )
    env.pop("MULTISOCIAL_FFMPEG_EXE", None)
    return env


def _staging_destinations(payload: dict[str, Any]) -> set[Path]:
    destinations = {
        Path(value)
        for key in (
            "output_csv_folder",
            "output_video_folder",
            "output_audio_features_folder",
            "output_transcripts_folder",
        )
        if (value := payload.get(key))
    }
    for pair in payload.get("alignment_pairs") or []:
        if isinstance(pair, (list, tuple)) and len(pair) == 3 and pair[2]:
            destinations.add(Path(pair[2]).parent)
    return destinations


def _absolutize_worker_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for key in (
        "output_csv_folder",
        "output_video_folder",
        "output_audio_features_folder",
        "output_transcripts_folder",
    ):
        if value := normalized.get(key):
            normalized[key] = os.path.abspath(os.fspath(value))
    for key in ("video_files", "audio_files"):
        if key in normalized:
            normalized[key] = [
                os.path.abspath(os.fspath(value))
                for value in normalized.get(key) or []
            ]
    if "alignment_pairs" in normalized:
        normalized["alignment_pairs"] = [
            [os.path.abspath(os.fspath(value)) for value in pair]
            for pair in normalized.get("alignment_pairs") or []
        ]
    return normalized


def _cleanup_request_staging(payload: dict[str, Any], request_id: str) -> None:
    prefix = f".multisocial-worker-{request_id}-"
    for destination in _staging_destinations(payload):
        try:
            candidates = destination.iterdir()
        except OSError:
            continue
        for candidate in candidates:
            if candidate.is_dir() and candidate.name.startswith(prefix):
                import shutil

                shutil.rmtree(candidate, ignore_errors=True)


def _spawn_worker(command: list[str], **kwargs: Any) -> subprocess.Popen:
    """Spawn the private launcher without changing the running GUI loader.

    The static launcher owns the DLL-directory reset immediately before it
    creates the worker. This keeps wxPython's process-wide loader state intact.
    """
    return subprocess.Popen(command, **kwargs)


def _terminate_worker_tree(process: subprocess.Popen) -> None:
    """Terminate the launcher and every private worker descendant on Windows."""

    if process.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    process.terminate()


class NativeWorkerClient:
    """Run exactly one native operation in an isolated console child process."""

    def __init__(
        self,
        *,
        command: Optional[list[str]] = None,
        cancel_grace_seconds: float = 15.0,
        timeout_seconds: float | None = None,
    ):
        self.command = command or worker_command()
        self.cancel_grace_seconds = cancel_grace_seconds
        self.timeout_seconds = timeout_seconds

    def run(
        self,
        operation: str,
        payload: dict[str, Any],
        *,
        progress_callback: Optional[Callable[[int], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        hf_token: Optional[str] = None,
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        payload = _absolutize_worker_payload(payload)
        env = _packaged_windows_environment(self.command, hf_token)
        inherited_diagnostic = env.get(WORKER_DIAGNOSTIC_ENV)
        diagnostic_path = (
            Path(inherited_diagnostic)
            if inherited_diagnostic
            else _worker_diagnostic_path(request_id)
        )
        diagnostic_path.unlink(missing_ok=True)
        env[WORKER_DIAGNOSTIC_ENV] = str(diagnostic_path)
        worker_dir = _worker_directory(self.command)
        socket_protocol = _uses_socket_worker_protocol()
        listener: socket.socket | None = None
        connection: socket.socket | None = None
        protocol_writer: Any = None
        if socket_protocol:
            listener = _create_worker_protocol_listener()
            host, port = listener.getsockname()
            protocol_token = secrets.token_urlsafe(32)
            env[WORKER_PROTOCOL_HOST_ENV] = str(host)
            env[WORKER_PROTOCOL_PORT_ENV] = str(port)
            env[WORKER_PROTOCOL_TOKEN_ENV] = protocol_token
            process = _spawn_worker(
                self.command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
                cwd=str(worker_dir),
                **_worker_popen_kwargs(),
            )
        else:
            process = _spawn_worker(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                cwd=str(worker_dir),
                **_worker_popen_kwargs(),
            )
        events: queue.Queue[tuple[str, str]] = queue.Queue()
        stderr_lines: list[str] = []
        if socket_protocol:
            assert listener is not None
            try:
                connection = _accept_worker_protocol(listener, process)
            except Exception:
                listener.close()
                _terminate_worker_tree(process)
                raise
            protocol_input = connection.makefile("r", encoding="utf-8", errors="replace", newline="\n")
            protocol_writer = connection.makefile("w", encoding="utf-8", errors="replace", newline="\n", buffering=1)
        else:
            assert process.stdin is not None and process.stdout is not None and process.stderr is not None
            protocol_input = process.stdout
            protocol_writer = process.stdin

        def read_protocol() -> None:
            for line in protocol_input:
                events.put(("protocol", line))
            events.put(("protocol_eof", ""))

        threading.Thread(target=read_protocol, daemon=True).start()
        if not socket_protocol:
            def read_stderr() -> None:
                assert process.stderr is not None
                for line in process.stderr:
                    cleaned = _redact(line, hf_token)
                    if cleaned:
                        stderr_lines.append(cleaned)
                        del stderr_lines[:-20]

            threading.Thread(target=read_stderr, daemon=True).start()

        cancel_sent = False
        cancel_deadline: Optional[float] = None
        started = time.monotonic()
        completed_successfully = False
        try:
            if socket_protocol:
                ready_deadline = time.monotonic() + 30.0
                while time.monotonic() < ready_deadline:
                    try:
                        _, line = events.get(timeout=0.1)
                    except queue.Empty:
                        if process.poll() is not None:
                            raise WorkerError(f"worker exited with code {process.returncode} before protocol handshake")
                        continue
                    try:
                        ready = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (
                        ready.get("protocol") == PROTOCOL_VERSION
                        and ready.get("event") == "ready"
                        and hmac.compare_digest(str(ready.get("token") or ""), protocol_token)
                    ):
                        break
                    raise WorkerError("worker rejected its private protocol handshake")
                else:
                    raise WorkerError("worker did not complete its private protocol handshake")
            request = {
                "protocol": PROTOCOL_VERSION,
                "id": request_id,
                "type": "run",
                "operation": operation,
                "payload": payload,
            }
            protocol_writer.write(json.dumps(request, separators=(",", ":")) + "\n")
            protocol_writer.flush()

            while True:
                if (
                    self.timeout_seconds is not None
                    and time.monotonic() - started >= self.timeout_seconds
                ):
                    stage = _diagnostic_stage(diagnostic_path) or "no worker stage recorded"
                    raise WorkerError(
                        f"Worker operation timed out after {self.timeout_seconds:g} seconds "
                        f"at diagnostic stage: {stage}"
                    )
                if cancel_check and cancel_check() and not cancel_sent:
                    cancel_sent = True
                    cancel_deadline = time.monotonic() + self.cancel_grace_seconds
                    protocol_writer.write(json.dumps({"protocol": PROTOCOL_VERSION, "id": request_id, "type": "cancel"}) + "\n")
                    protocol_writer.flush()

                if cancel_deadline is not None and time.monotonic() >= cancel_deadline:
                    _terminate_worker_tree(process)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return {"succeeded": [], "failed": [], "cancelled": True}

                try:
                    event_kind, line = events.get(timeout=0.1)
                except queue.Empty:
                    if process.poll() is not None:
                        detail = " | ".join(stderr_lines[-5:]) or f"worker exited with code {process.returncode}"
                        raise WorkerError(_redact(detail, hf_token))
                    continue

                if event_kind != "protocol":
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    stderr_lines.append(_redact(line, hf_token))
                    continue
                if message.get("protocol") != PROTOCOL_VERSION or message.get("id") != request_id:
                    continue
                event = message.get("event")
                if event == "progress" and progress_callback:
                    progress_callback(int(message.get("value", 0)))
                elif event == "status" and status_callback:
                    status_callback(str(message.get("message", "")))
                elif event == "result":
                    result = dict(message.get("result") or {})
                    result.setdefault("succeeded", [])
                    result.setdefault("failed", [])
                    result.setdefault("cancelled", False)
                    completed_successfully = True
                    return result
                elif event == "error":
                    raise WorkerError(_redact(str(message.get("message", "Worker operation failed")), hf_token))
        finally:
            try:
                if protocol_writer is not None:
                    protocol_writer.close()
            except Exception:
                pass
            try:
                if listener is not None:
                    listener.close()
                if connection is not None:
                    connection.close()
            except OSError:
                pass
            if process.poll() is None:
                _terminate_worker_tree(process)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
            _cleanup_request_staging(payload, request_id)
            if (
                completed_successfully
                and env.get(WORKER_PRESERVE_DIAGNOSTICS_ENV) != "1"
            ):
                diagnostic_path.unlink(missing_ok=True)


class WindowsPoseProcessor:
    """PoseProcessor-compatible Windows façade backed by ``MultiSOCIAL-Worker``."""

    def __init__(self, output_csv_folder, output_video_folder=None, status_callback=None, frame_threshold=10, frame_stride=1, downscale_to=None):
        self.output_csv_folder = output_csv_folder
        self.output_video_folder = output_video_folder
        self.status_callback = status_callback
        self.frame_threshold = frame_threshold
        self.frame_stride = frame_stride
        self.downscale_to = downscale_to
        self.enable_multi_person_pose = False
        self._client = NativeWorkerClient()

    def set_multi_person_mode(self, enabled: bool) -> None:
        self.enable_multi_person_pose = bool(enabled)

    def _payload(self, video_path: str) -> dict[str, Any]:
        return {
            "video_files": [video_path],
            "output_csv_folder": self.output_csv_folder,
            "output_video_folder": self.output_video_folder,
            "frame_threshold": self.frame_threshold,
            "frame_stride": self.frame_stride,
            "downscale_to": self.downscale_to,
            "multi_person": self.enable_multi_person_pose,
        }

    def extract_pose_features(self, video_path, progress_callback=None, cancel_check=None):
        result = self._client.run("extract_pose", self._payload(video_path), progress_callback=progress_callback, status_callback=self.status_callback, cancel_check=cancel_check)
        if result["cancelled"]:
            return False
        if result["failed"]:
            raise WorkerError(result["failed"][0][1])
        return True

    def embed_pose_video(self, video_path, progress_callback=None, cancel_check=None):
        result = self._client.run("embed_pose", self._payload(video_path), progress_callback=progress_callback, status_callback=self.status_callback, cancel_check=cancel_check)
        if result["cancelled"]:
            return False
        if result["failed"]:
            raise WorkerError(result["failed"][0][1])
        return result["succeeded"][0] if result["succeeded"] else None


class WindowsAudioProcessor:
    """AudioProcessor-compatible Windows façade backed by the private worker."""

    def __init__(self, output_audio_features_folder, output_transcripts_folder, status_callback=None, enable_speaker_diarization=False, auth_token=None):
        self.output_audio_features_folder = output_audio_features_folder
        self.output_transcripts_folder = output_transcripts_folder
        self.status_callback = status_callback
        self.enable_speaker_diarization = enable_speaker_diarization
        self.auth_token = auth_token
        self._client = NativeWorkerClient()

    def _run(self, operation, payload, progress_callback=None, cancel_check=None):
        return self._client.run(operation, payload, progress_callback=progress_callback, status_callback=self.status_callback, cancel_check=cancel_check, hf_token=self.auth_token if self.enable_speaker_diarization else None)

    def preload_speaker_diarizer(self):
        if not self.enable_speaker_diarization:
            return
        self._run("probe", {"profile": "complete", "validate_diarization": True})

    def extract_audio_features_batch(self, audio_files, progress_callback=None, cancel_check=None):
        return self._run("extract_audio_features", {"audio_files": list(audio_files), "output_audio_features_folder": self.output_audio_features_folder}, progress_callback, cancel_check)

    def extract_audio_features(self, audio_file, progress_callback=None, cancel_check=None):
        result = self.extract_audio_features_batch([audio_file], progress_callback, cancel_check)
        if result["cancelled"]:
            return False
        if result["failed"]:
            raise WorkerError(result["failed"][0][1])
        return result["succeeded"][0]

    def extract_transcripts_batch(self, audio_files, progress_callback=None, word_timestamps=False, cancel_check=None):
        return self._run("extract_transcripts", {"audio_files": list(audio_files), "output_transcripts_folder": self.output_transcripts_folder, "word_timestamps": bool(word_timestamps), "enable_diarization": bool(self.enable_speaker_diarization)}, progress_callback, cancel_check)

    def extract_transcript(self, audio_file, progress_callback=None, word_timestamps=False, cancel_check=None):
        result = self.extract_transcripts_batch([audio_file], progress_callback, word_timestamps, cancel_check)
        if result["cancelled"]:
            return False
        if result["failed"]:
            raise WorkerError(result["failed"][0][1])
        return result["succeeded"][0]

    def align_features_batch(self, alignment_pairs, progress_callback=None):
        return self._run("align_features", {"alignment_pairs": [list(pair) for pair in alignment_pairs]}, progress_callback)
