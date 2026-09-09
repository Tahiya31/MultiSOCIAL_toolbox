"""Private console worker for Windows native analysis.

The worker imports native ML libraries only after the GUI has spawned it.  Its
stdout is reserved for the versioned JSON-lines protocol; all library output is
redirected to redacted stderr diagnostics.
"""

from __future__ import annotations

import contextlib
import ctypes
import hashlib
import importlib
import io
import json
import os
import platform
import queue
import shutil
import socket
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from native_worker_client import (
    PROTOCOL_VERSION,
    WORKER_DIAGNOSTIC_ENV,
    WORKER_TOKEN_ENV,
    _WINDOWS_GUI_NATIVE_ENVIRONMENT_NAMES,
    _windows_directory,
)


_WORKER_STARTED_AT = time.monotonic()
_PROTOCOL_OUTPUT = sys.stdout


def _is_embedded_worker_runtime() -> bool:
    """Identify the private CPython worker by its on-disk runtime layout.

    The GUI must not pass a mode flag that changes loader behavior.  The worker
    itself is authoritative: its executable lives in ``worker/`` beside a
    versioned ``python*._pth`` file written by the packager.
    """

    executable_dir = Path(sys.executable).resolve().parent
    return executable_dir.name.casefold() == "worker" and any(
        executable_dir.glob("python*._pth")
    )


def _uses_private_worker_runtime() -> bool:
    """Return whether this is either supported private-worker packaging form."""

    return bool(getattr(sys, "frozen", False)) or _is_embedded_worker_runtime()


def _diagnostic_stage(stage: str, **details: str | int | bool | None) -> None:
    """Persist a redacted, path-free startup breadcrumb for support and CI.

    The parent selects the file location and receives only the last stage name;
    this process never records input paths, environment values, model names, or
    credentials.  Diagnostics intentionally cannot interfere with analysis.
    """
    destination = os.environ.get(WORKER_DIAGNOSTIC_ENV)
    if not destination:
        return
    allowed_keys = {
        "platform", "architecture", "operation", "profile", "error_type",
        "gui_native_environment_clean",
    }
    safe_details = {
        key: value
        for key, value in details.items()
        if key in allowed_keys and (isinstance(value, (str, int, bool)) or value is None)
    }
    record = {
        "stage": stage,
        "elapsed_ms": int((time.monotonic() - _WORKER_STARTED_AT) * 1000),
        **safe_details,
    }
    try:
        with Path(destination).open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            output.flush()
    except OSError:
        pass


class _RedactingStream(io.TextIOBase):
    def __init__(self, target, token: str | None):
        self.target = target
        self.token = token

    def write(self, value):
        text = str(value)
        if self.token:
            text = text.replace(self.token, "[REDACTED]")
        return self.target.write(text) if self.target is not None else len(text)

    def flush(self):
        return self.target.flush() if self.target is not None else None


def _emit(request_id: str, event: str, **values: Any) -> None:
    message = {"protocol": PROTOCOL_VERSION, "id": request_id, "event": event, **values}
    _PROTOCOL_OUTPUT.write(json.dumps(message, separators=(",", ":")) + "\n")
    _PROTOCOL_OUTPUT.flush()


def _open_protocol_stream() -> tuple[Any, Any]:
    """Use the private loopback protocol when the GUI intentionally has no pipes."""
    host = os.environ.pop("MULTISOCIAL_WORKER_PROTOCOL_HOST", None)
    port = os.environ.pop("MULTISOCIAL_WORKER_PROTOCOL_PORT", None)
    token = os.environ.pop("MULTISOCIAL_WORKER_PROTOCOL_TOKEN", None)
    if not any((host, port, token)):
        return sys.stdin, sys.stdout
    if host != "127.0.0.1" or not port or not token:
        raise RuntimeError("invalid private worker protocol configuration")
    try:
        port_number = int(port)
    except ValueError as exc:
        raise RuntimeError("invalid private worker protocol endpoint") from exc
    if not 1 <= port_number <= 65535:
        raise RuntimeError("invalid private worker protocol endpoint")
    connection = socket.create_connection((host, port_number), timeout=15)
    connection.settimeout(None)
    reader = connection.makefile("r", encoding="utf-8", errors="replace", newline="\n")
    writer = connection.makefile("w", encoding="utf-8", errors="replace", newline="\n", buffering=1)
    writer.write(json.dumps({"protocol": PROTOCOL_VERSION, "event": "ready", "token": token}, separators=(",", ":")) + "\n")
    writer.flush()
    return reader, writer


def _safe_error(error: BaseException, token: str | None) -> str:
    text = str(error).replace("\r", " ").replace("\n", " ")
    if token:
        text = text.replace(token, "[REDACTED]")
    return text[-2000:] or type(error).__name__


class _StagedOutput:
    """Stage generated files beside their destination and promote atomically."""

    def __init__(self, destination: str | None, request_id: str | None = None):
        self.destination = Path(destination) if destination else None
        self.request_id = request_id or "unscoped"
        self.path: Path | None = None

    def __enter__(self) -> str | None:
        if self.destination is None:
            return None
        self.destination.mkdir(parents=True, exist_ok=True)
        self.path = Path(
            tempfile.mkdtemp(
                prefix=f".multisocial-worker-{self.request_id}-",
                dir=self.destination,
            )
        )
        return str(self.path)

    def commit(self) -> None:
        if self.destination is None or self.path is None:
            return
        for source in sorted(self.path.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(self.path)
            target = self.destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.path is not None:
            shutil.rmtree(self.path, ignore_errors=True)


def _batch_outcome() -> dict[str, Any]:
    return {"succeeded": [], "failed": [], "cancelled": False}


def _runtime_metadata() -> dict[str, Any]:
    """Return non-sensitive host facts used by packaged Windows compatibility checks."""
    machine = platform.machine().lower()
    is_64bit = sys.maxsize > 2**32
    metadata: dict[str, Any] = {
        "platform": sys.platform,
        "machine": machine,
        "is_64bit": is_64bit,
        "architecture": "x64" if is_64bit and machine in {"amd64", "x86_64"} else machine,
    }
    if sys.platform == "win32":
        release, version, _, _ = platform.win32_ver()
        metadata["windows_release"] = release
        metadata["windows_version"] = version
        try:
            metadata["windows_major"] = int(version.split(".", 1)[0])
        except (TypeError, ValueError):
            metadata["windows_major"] = None
    metadata["worker_runtime"] = _worker_runtime_diagnostics()
    return metadata


def _worker_runtime_diagnostics() -> dict[str, Any]:
    """Return safe evidence that native imports came from the private worker."""
    executable_dir = Path(sys.executable).resolve().parent
    bundle_root = Path(getattr(sys, "_MEIPASS", executable_dir)).resolve()
    binding_hashes = []
    if bundle_root.is_dir():
        for binding in sorted(bundle_root.rglob("_framework_bindings*.pyd")):
            try:
                digest = hashlib.sha256(binding.read_bytes()).hexdigest()
            except OSError:
                continue
            binding_hashes.append({"name": binding.name, "sha256": digest})
    embedded = _is_embedded_worker_runtime()
    return {
        "private_runtime": executable_dir.name.casefold() == "worker" and (
            bundle_root == executable_dir or embedded
        ),
        "mediapipe_bindings": binding_hashes,
        "loaded_module_violations": _loaded_module_violations(bundle_root),
        "native_module_violations": _native_module_violations(bundle_root),
        "external_module_violations": _external_module_violations(bundle_root),
    }


def _path_is_within(path: str | Path, root: str | Path) -> bool:
    """Compare canonical paths so a private Windows drive alias remains private.

    The native launcher may expose ``worker/`` through a process-local ASCII
    drive alias when the real installation path contains non-ASCII characters.
    Windows resolves that alias to its target for imported module paths.  A
    lexical comparison would therefore reject worker-owned modules solely
    because one spelling uses the alias and the other uses the final path.
    ``Path.resolve`` uses Windows' final-path resolution for existing files,
    while preserving the same strict containment check for genuinely external
    modules.
    """
    try:
        Path(os.path.normcase(os.path.abspath(path))).resolve().relative_to(
            Path(os.path.normcase(os.path.abspath(root))).resolve()
        )
        return True
    except (OSError, ValueError):
        return False


def _loaded_windows_module_paths() -> list[Path]:
    """Enumerate this process' loaded modules without adding a native dependency."""
    if sys.platform != "win32":
        return []
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    process_function = kernel32.GetCurrentProcess
    process_function.argtypes = []
    process_function.restype = wintypes.HANDLE
    process = process_function()
    enum_modules = getattr(kernel32, "K32EnumProcessModules", None)
    module_filename = getattr(kernel32, "K32GetModuleFileNameExW", None)
    if enum_modules is None or module_filename is None:
        enum_modules = ctypes.windll.psapi.EnumProcessModules
        module_filename = ctypes.windll.psapi.GetModuleFileNameExW
    enum_modules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    enum_modules.restype = wintypes.BOOL
    module_filename.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    module_filename.restype = wintypes.DWORD

    capacity = 256
    while True:
        module_array = (wintypes.HMODULE * capacity)()
        needed = wintypes.DWORD()
        if not enum_modules(process, module_array, ctypes.sizeof(module_array), ctypes.byref(needed)):
            raise OSError("Could not enumerate loaded worker modules")
        if needed.value <= ctypes.sizeof(module_array):
            break
        capacity = max(capacity * 2, needed.value // ctypes.sizeof(wintypes.HMODULE) + 16)

    count = needed.value // ctypes.sizeof(wintypes.HMODULE)
    paths: list[Path] = []
    for module in module_array[:count]:
        buffer = ctypes.create_unicode_buffer(32768)
        if module_filename(process, module, buffer, len(buffer)):
            paths.append(Path(buffer.value).resolve())
    return paths


def _loaded_module_violations(
    bundle_root: Path,
    loaded_paths: list[Path] | None = None,
) -> list[str]:
    """Return DLLs loaded from the containing GUI runtime instead of worker/."""
    if sys.platform != "win32" or not _uses_private_worker_runtime():
        return []
    app_root = bundle_root.parent
    violations = []
    paths = loaded_paths if loaded_paths is not None else _loaded_windows_module_paths()
    for path in paths:
        if _path_is_within(path, app_root) and not _path_is_within(path, bundle_root):
            violations.append(path.name)
    return sorted(set(violations), key=str.casefold)


def _native_module_violations(
    bundle_root: Path,
    loaded_paths: list[Path] | None = None,
) -> list[str]:
    """Return package-native extensions or DLLs resolved outside worker/."""
    if sys.platform != "win32" or not _uses_private_worker_runtime():
        return []
    native_markers = (
        "_framework_bindings",
        "audresample",
        "c10",
        "concrt140",
        "fbgemm",
        "libiomp",
        "mediapipe",
        "msvcp140",
        "opencv",
        "smileapi",
        "torch",
        "vcamp140",
        "vccorlib140",
        "vcomp140",
        "vcruntime140",
    )
    violations = []
    paths = loaded_paths if loaded_paths is not None else _loaded_windows_module_paths()
    for path in paths:
        basename = path.name.casefold()
        package_native = path.suffix.casefold() == ".pyd" or any(
            marker in basename for marker in native_markers
        )
        if package_native and not _path_is_within(path, bundle_root):
            violations.append(path.name)
    return sorted(set(violations), key=str.casefold)


def _permitted_external_module_roots() -> list[Path]:
    # DriverStore, System32, SysWOW64, and WinSxS all sit beneath this root.
    return [_windows_directory().resolve()]


def _external_module_violations(
    bundle_root: Path,
    loaded_paths: list[Path] | None = None,
) -> list[str]:
    """Reject non-system DLL injection without returning user-specific paths."""
    if sys.platform != "win32" or not _uses_private_worker_runtime():
        return []
    allowed_roots = [bundle_root, *_permitted_external_module_roots()]
    paths = loaded_paths if loaded_paths is not None else _loaded_windows_module_paths()
    violations = [
        path.name
        for path in paths
        if not any(_path_is_within(path, root) for root in allowed_roots)
    ]
    return sorted(set(violations), key=str.casefold)


def _operation_module_names(operation: str, payload: dict[str, Any]) -> tuple[str, ...]:
    if operation in {"extract_pose", "embed_pose"}:
        return ("mediapipe", "mediapipe.python._framework_bindings", "cv2", "torch", "pose")
    if operation in {"extract_audio_features", "extract_transcripts", "align_features"}:
        names = ["audio", "opensmile", "torch", "transformers"]
        if operation == "extract_transcripts" and payload.get("enable_diarization"):
            names.extend(["torchaudio", "pyannote.audio"])
        return tuple(names)
    if operation == "probe":
        names = ["mediapipe", "mediapipe.python._framework_bindings", "cv2", "opensmile", "torch"]
        if payload.get("profile") == "complete":
            names.extend(["torchaudio", "pyannote.audio"])
        return tuple(names)
    return ()


def _validate_worker_runtime_provenance(operation: str, payload: dict[str, Any]) -> None:
    """Fail closed if a frozen worker resolved native state outside worker/."""
    embedded = _is_embedded_worker_runtime()
    if sys.platform != "win32" or not _uses_private_worker_runtime():
        return

    executable_dir = Path(sys.executable).resolve().parent
    bundle_root = executable_dir if embedded else Path(getattr(sys, "_MEIPASS", executable_dir)).resolve()
    if executable_dir.name.casefold() != "worker" or bundle_root != executable_dir:
        raise RuntimeError("Worker executable and bundle root are not the private worker runtime")

    python_dll = bundle_root / f"python{sys.version_info.major}{sys.version_info.minor}.dll"
    if not python_dll.is_file():
        raise RuntimeError(f"Private worker Python runtime is missing: {python_dll.name}")

    for module_name in _operation_module_names(operation, payload):
        module = sys.modules.get(module_name)
        module_path = getattr(module, "__file__", None)
        if not module_path or not _path_is_within(module_path, bundle_root):
            raise RuntimeError(f"Worker module resolved outside private runtime: {module_name}")

    loaded_paths = _loaded_windows_module_paths()
    loaded_python = [path for path in loaded_paths if path.name.casefold() == python_dll.name.casefold()]
    if not loaded_python or any(not _path_is_within(path, bundle_root) for path in loaded_python):
        raise RuntimeError(f"Loaded Python runtime did not resolve beneath worker/: {python_dll.name}")

    violations = _loaded_module_violations(bundle_root, loaded_paths)
    if violations:
        raise RuntimeError(
            "Worker loaded DLLs from the GUI runtime: " + ", ".join(violations[:10])
        )
    native_violations = _native_module_violations(bundle_root, loaded_paths)
    if native_violations:
        raise RuntimeError(
            "Worker loaded package-native modules outside worker/: "
            + ", ".join(native_violations[:10])
        )
    external_violations = _external_module_violations(bundle_root, loaded_paths)
    if external_violations:
        raise RuntimeError(
            "Worker loaded DLLs outside worker/ and permitted Windows/driver roots: "
            + ", ".join(external_violations[:10])
        )


def _import_worker_runtime_module(module_name: str) -> Any:
    """Import through a replaceable seam so preload order is unit-testable."""
    return importlib.import_module(module_name)


def _preload_worker_runtime_module(module_name: str) -> Any:
    _diagnostic_stage(f"preload:{module_name}")
    return _import_worker_runtime_module(module_name)


def _initialize_worker_operation_runtime(operation: str, payload: dict[str, Any]) -> None:
    """Initialize each native extension on the worker main thread.

    The loader phases deliberately preserve MediaPipe-before-Torch ordering.
    Operation handlers subsequently import only already-cached modules from
    their background thread.
    """
    if operation in {"extract_pose", "embed_pose"}:
        _preload_worker_runtime_module("mediapipe")
        _preload_worker_runtime_module("cv2")
        _diagnostic_stage("tensor-loader-configured")
        _enable_worker_tensor_loader()
        _preload_worker_runtime_module("pose")
        return
    if operation in {"extract_audio_features", "extract_transcripts", "align_features"}:
        _diagnostic_stage("tensor-loader-configured")
        _enable_worker_tensor_loader()
        _preload_worker_runtime_module("audio")
        if operation == "extract_transcripts" and payload.get("enable_diarization"):
            _preload_worker_runtime_module("torchaudio")
            _preload_worker_runtime_module("pyannote.audio")
        return
    if operation == "probe":
        _preload_worker_runtime_module("mediapipe")
        _preload_worker_runtime_module("cv2")
        _preload_worker_runtime_module("opensmile")
        _diagnostic_stage("tensor-loader-configured")
        _enable_worker_tensor_loader()
        _preload_worker_runtime_module("torch")
        if payload.get("profile") == "complete":
            _preload_worker_runtime_module("torchaudio")
            _preload_worker_runtime_module("pyannote.audio")


def _configure_worker_ffmpeg() -> None:
    """Use the worker's ffmpeg copy instead of an inherited GUI executable."""
    if not getattr(sys, "frozen", False):
        return
    import imageio_ffmpeg

    executable = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent)).resolve()
    if not _path_is_within(executable, bundle_root):
        raise RuntimeError("Worker ffmpeg resolved outside the private worker runtime")
    os.environ["MULTISOCIAL_FFMPEG_EXE"] = str(executable)


def _run_pose(payload: dict[str, Any], cancelled: threading.Event, emit_progress, emit_status, *, embed: bool) -> dict[str, Any]:
    from pose import PoseProcessor, find_pose_csv_paths

    videos = list(payload.get("video_files") or [])
    output_key = "output_video_folder" if embed else "output_csv_folder"
    destination = payload.get(output_key)
    outcome = _batch_outcome()
    if not destination:
        raise ValueError(f"{output_key} is required")

    stage_context = _StagedOutput(destination, payload.get("_request_id"))
    with stage_context as staged:
        processor = PoseProcessor(
            output_csv_folder=payload.get("output_csv_folder") if embed else staged,
            output_video_folder=staged if embed else None,
            status_callback=emit_status,
            frame_threshold=payload.get("frame_threshold", 10),
            frame_stride=payload.get("frame_stride", 1),
            downscale_to=tuple(payload["downscale_to"]) if payload.get("downscale_to") else None,
        )
        processor.set_multi_person_mode(bool(payload.get("multi_person")))
        total = max(1, len(videos))
        for index, video in enumerate(videos, start=1):
            if cancelled.is_set():
                outcome["cancelled"] = True
                break
            start = int((index - 1) * 100 / total)
            span = 100 / total
            try:
                operation = processor.embed_pose_video if embed else processor.extract_pose_features
                result = operation(
                    video,
                    progress_callback=lambda value, start=start, span=span: emit_progress(int(start + value * span / 100)),
                    cancel_check=cancelled.is_set,
                )
                if result is False:
                    outcome["cancelled"] = True
                    break
                if embed:
                    if result is None:
                        raise RuntimeError("no pose CSV found")
                    outcome["succeeded"].append(os.path.join(destination, os.path.basename(result)))
                else:
                    paths = find_pose_csv_paths(staged, video, multi_person=bool(payload.get("multi_person")))
                    if not paths:
                        raise RuntimeError("no pose CSV was produced")
                    outcome["succeeded"].append(video)
            except Exception as exc:
                outcome["failed"].append((video, str(exc)))
        if not outcome["cancelled"]:
            stage_context.commit()
    emit_progress(100)
    return outcome


def _run_audio_features(payload: dict[str, Any], cancelled: threading.Event, emit_progress, emit_status) -> dict[str, Any]:
    _enable_worker_tensor_loader()
    from audio import AudioProcessor

    destination = payload.get("output_audio_features_folder")
    if not destination:
        raise ValueError("output_audio_features_folder is required")
    stage_context = _StagedOutput(destination, payload.get("_request_id"))
    with stage_context as stage:
        processor = AudioProcessor(stage, None, status_callback=emit_status)
        outcome = processor.extract_audio_features_batch(payload.get("audio_files") or [], progress_callback=emit_progress, cancel_check=cancelled.is_set)
        if not outcome["cancelled"]:
            for index, path in enumerate(outcome["succeeded"]):
                outcome["succeeded"][index] = os.path.join(destination, os.path.basename(path))
            stage_context.commit()
        return outcome


def _run_transcripts(payload: dict[str, Any], cancelled: threading.Event, emit_progress, emit_status, token: str | None) -> dict[str, Any]:
    _enable_worker_tensor_loader()
    from audio import AudioProcessor

    destination = payload.get("output_transcripts_folder")
    if not destination:
        raise ValueError("output_transcripts_folder is required")
    diarization = bool(payload.get("enable_diarization"))
    stage_context = _StagedOutput(destination, payload.get("_request_id"))
    with stage_context as stage:
        processor = AudioProcessor(None, stage, status_callback=emit_status, enable_speaker_diarization=diarization, auth_token=token if diarization else None)
        outcome = processor.extract_transcripts_batch(payload.get("audio_files") or [], progress_callback=emit_progress, word_timestamps=bool(payload.get("word_timestamps")), cancel_check=cancelled.is_set)
        if not outcome["cancelled"]:
            for index, path in enumerate(outcome["succeeded"]):
                outcome["succeeded"][index] = os.path.join(destination, os.path.basename(path))
            stage_context.commit()
        return outcome


def _run_alignment(payload: dict[str, Any], cancelled: threading.Event, emit_progress, emit_status) -> dict[str, Any]:
    _enable_worker_tensor_loader()
    from audio import AudioProcessor

    pairs = [tuple(item) for item in payload.get("alignment_pairs") or []]
    outcome = _batch_outcome()
    total = max(1, len(pairs))
    for index, (features_csv, transcript_json, output_csv) in enumerate(pairs, start=1):
        if cancelled.is_set():
            outcome["cancelled"] = True
            break
        destination = str(Path(output_csv).parent)
        stage_context = _StagedOutput(destination, payload.get("_request_id"))
        with stage_context as stage:
            staged_output = str(Path(stage) / Path(output_csv).name)
            try:
                emit_status(f"Aligning: {os.path.basename(output_csv)}")
                processor = AudioProcessor(None, None, status_callback=emit_status)
                result = processor.align_features(features_csv, transcript_json, staged_output)
                if result != staged_output or not os.path.isfile(staged_output):
                    raise RuntimeError("Alignment did not write its output CSV")
                stage_context.commit()
                outcome["succeeded"].append(output_csv)
            except Exception as exc:
                outcome["failed"].append((output_csv, str(exc)))
        emit_progress(int(index * 100 / total))
    return outcome


def _run_probe(payload: dict[str, Any], token: str | None) -> dict[str, Any]:
    import mediapipe  # noqa: F401
    import cv2  # noqa: F401
    import opensmile  # noqa: F401

    _enable_worker_tensor_loader()
    import torch  # noqa: F401

    if payload.get("profile") == "complete":
        import torchaudio  # noqa: F401
        import pyannote.audio  # noqa: F401
        if payload.get("validate_diarization"):
            from audio import PyAnnoteSpeakerDiarizer

            diarizer = PyAnnoteSpeakerDiarizer(auth_token=token)
            diarizer._load_diarization_model()
            diarizer.offload_model()
    outcome = _batch_outcome()
    outcome["runtime"] = _runtime_metadata()
    return outcome


def _dispatch(operation: str, payload: dict[str, Any], cancelled: threading.Event, emit_progress, emit_status, token: str | None) -> dict[str, Any]:
    if operation == "probe":
        return _run_probe(payload, token)
    if operation == "extract_pose":
        return _run_pose(payload, cancelled, emit_progress, emit_status, embed=False)
    if operation == "embed_pose":
        return _run_pose(payload, cancelled, emit_progress, emit_status, embed=True)
    if operation == "extract_audio_features":
        return _run_audio_features(payload, cancelled, emit_progress, emit_status)
    if operation == "extract_transcripts":
        return _run_transcripts(payload, cancelled, emit_progress, emit_status, token)
    if operation == "align_features":
        return _run_alignment(payload, cancelled, emit_progress, emit_status)
    raise ValueError(f"Unknown worker operation: {operation}")


def _configure_worker_native_loader() -> None:
    """Install the MediaPipe-safe DLL phase after the worker—not wx—has started."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    # Clear any inherited GUI directory in the process that will actually
    # import MediaPipe. The embedded CPython worker relies on its own app
    # directory and package-local dependency layout; adding every old
    # PyInstaller directory makes dependency lookup order unspecified.
    try:
        ctypes.windll.kernel32.SetDllDirectoryW(None)
    except (AttributeError, OSError):
        pass
    if _is_embedded_worker_runtime():
        return
    from runtime_hook_dlls import configure_windows_dll_search_path

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root and os.path.isdir(bundle_root):
        configure_windows_dll_search_path(bundle_root, include_tensor_runtime=False)


def _has_clean_windows_worker_environment() -> bool:
    """Return only whether GUI-specific native settings were excluded."""
    return not any(name in os.environ for name in _WINDOWS_GUI_NATIVE_ENVIRONMENT_NAMES)


def _enable_worker_tensor_loader() -> None:
    """Expose Torch DLLs only after MediaPipe has completed its initialization."""
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    if _is_embedded_worker_runtime():
        return
    from runtime_hook_dlls import configure_windows_dll_search_path

    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root and os.path.isdir(bundle_root):
        configure_windows_dll_search_path(bundle_root, include_tensor_runtime=True)


def main() -> int:
    global _PROTOCOL_OUTPUT
    _diagnostic_stage(
        "boot",
        platform=sys.platform,
        architecture="x64" if sys.maxsize > 2**32 else "x86",
    )
    try:
        protocol_input, _PROTOCOL_OUTPUT = _open_protocol_stream()
    except Exception as exc:
        _diagnostic_stage("protocol-connect-error", error_type=type(exc).__name__)
        return 2
    messages: queue.Queue[dict[str, Any]] = queue.Queue()

    def reader() -> None:
        for line in protocol_input:
            try:
                messages.put(json.loads(line))
            except json.JSONDecodeError:
                continue

    threading.Thread(target=reader, daemon=True).start()
    try:
        request = messages.get(timeout=30)
    except queue.Empty:
        return 2
    if request.get("protocol") != PROTOCOL_VERSION or request.get("type") != "run":
        _diagnostic_stage("invalid-request")
        return 2

    request_id = str(request.get("id") or "")
    operation = str(request.get("operation") or "")
    payload = dict(request.get("payload") or {})
    payload["_request_id"] = request_id
    token = os.environ.pop(WORKER_TOKEN_ENV, None)
    safe_operation = operation if operation in {
        "probe", "extract_pose", "embed_pose", "extract_audio_features",
        "extract_transcripts", "align_features",
    } else "unknown"
    profile = payload.get("profile")
    safe_profile = profile if profile in {"standard", "complete"} else "unknown"
    _diagnostic_stage("request-received", operation=safe_operation, profile=safe_profile)
    _configure_worker_native_loader()
    _diagnostic_stage(
        "base-loader-configured",
        gui_native_environment_clean=_has_clean_windows_worker_environment(),
    )
    try:
        with contextlib.redirect_stdout(_RedactingStream(sys.stderr, token)), contextlib.redirect_stderr(_RedactingStream(sys.stderr, token)):
            _diagnostic_stage("ffmpeg-configured")
            _configure_worker_ffmpeg()
            _initialize_worker_operation_runtime(operation, payload)
            _diagnostic_stage("provenance-validated")
            _validate_worker_runtime_provenance(operation, payload)
    except Exception as exc:
        _diagnostic_stage("startup-error", error_type=type(exc).__name__)
        traceback.print_exc(file=_RedactingStream(sys.stderr, token))
        _emit(request_id, "error", message=_safe_error(exc, token))
        return 1
    cancelled = threading.Event()
    result_box: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def emit_progress(value: int) -> None:
        _emit(request_id, "progress", value=max(0, min(100, int(value))))

    def emit_status(message: str) -> None:
        _emit(request_id, "status", message=_safe_error(Exception(message), token))

    def run_operation() -> None:
        try:
            with contextlib.redirect_stdout(_RedactingStream(sys.stderr, token)), contextlib.redirect_stderr(_RedactingStream(sys.stderr, token)):
                _diagnostic_stage("operation-thread-started")
                result_box.put(("result", _dispatch(operation, payload, cancelled, emit_progress, emit_status, token)))
        except Exception as exc:
            _diagnostic_stage("operation-error", error_type=type(exc).__name__)
            traceback.print_exc(file=_RedactingStream(sys.stderr, token))
            result_box.put(("error", _safe_error(exc, token)))

    threading.Thread(target=run_operation, daemon=True).start()
    while True:
        try:
            kind, value = result_box.get(timeout=0.1)
            if kind == "result":
                try:
                    _diagnostic_stage("result-provenance-validated")
                    _validate_worker_runtime_provenance(operation, payload)
                except Exception as exc:
                    _diagnostic_stage("result-provenance-error", error_type=type(exc).__name__)
                    traceback.print_exc(file=_RedactingStream(sys.stderr, token))
                    _emit(request_id, "error", message=_safe_error(exc, token))
                    return 1
                _diagnostic_stage("result-emitted")
                _emit(request_id, "result", result=value)
                return 0
            _diagnostic_stage("operation-error")
            _emit(request_id, "error", message=value)
            return 1
        except queue.Empty:
            pass
        while True:
            try:
                control = messages.get_nowait()
            except queue.Empty:
                break
            if control.get("protocol") == PROTOCOL_VERSION and control.get("id") == request_id and control.get("type") == "cancel":
                cancelled.set()


if __name__ == "__main__":
    raise SystemExit(main())
