"""Run one bounded probe through a packaged Windows GUI or private worker."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
import uuid
from pathlib import Path


PROTOCOL_VERSION = 1
DIAGNOSTIC_ENV = "MULTISOCIAL_WORKER_DIAGNOSTIC_PATH"
PRESERVE_DIAGNOSTICS_ENV = "MULTISOCIAL_PRESERVE_WORKER_DIAGNOSTICS"


def _diagnostic_stages(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "no worker stage recorded"
    stages: list[str] = []
    for line in lines:
        try:
            stage = json.loads(line).get("stage")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(stage, str) and stage:
            stages.append(stage)
    return " -> ".join(stages[-12:]) or "no worker stage recorded"


def _diagnostic_timeline(path: Path) -> list[dict[str, int | str]]:
    """Return only redacted stage names and monotonic elapsed times for CI."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    timeline: list[dict[str, int | str]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        stage = event.get("stage")
        elapsed_ms = event.get("elapsed_ms")
        if isinstance(stage, str) and stage and isinstance(elapsed_ms, int):
            timeline.append({"stage": stage, "elapsed_ms": elapsed_ms})
    return timeline[-12:]


def _terminate_tree(process: subprocess.Popen) -> None:
    """Terminate the scoped GUI/worker process tree after a diagnostic timeout."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        process.kill()


def _communicate_bounded(
    process: subprocess.Popen,
    *,
    input_text: str | None,
    timeout: int,
    diagnostic_path: Path,
) -> tuple[str, str]:
    try:
        stdout, stderr = process.communicate(input=input_text, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_tree(process)
        process.communicate()
        raise RuntimeError(
            f"Packaged worker diagnostic timed out after {timeout} seconds "
            f"at stages: {_diagnostic_stages(diagnostic_path)}"
        ) from exc
    return stdout or "", stderr or ""


def _diagnostic_path(result: Path) -> Path:
    return result.with_name(f"{result.stem}-worker-diagnostics.jsonl")


def _private_worker_environment(worker: Path, diagnostic_path: Path) -> dict[str, str]:
    """Match the packaged client’s clean worker environment without a GUI parent."""
    environment = os.environ.copy()
    windows_root = Path(environment.get("SystemRoot") or environment.get("WINDIR") or r"C:\Windows")
    environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    environment.pop("MULTISOCIAL_FFMPEG_EXE", None)
    environment["SystemRoot"] = str(windows_root)
    environment["WINDIR"] = str(windows_root)
    environment["PATH"] = ";".join(
        str(path)
        for path in (
            worker.parent,
            windows_root / "System32",
            windows_root,
            windows_root / "System32" / "Wbem",
            windows_root / "System32" / "WindowsPowerShell" / "v1.0",
        )
    )
    environment[DIAGNOSTIC_ENV] = str(diagnostic_path)
    return environment


def _write_result(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _safe_protocol_error(event: dict | None) -> str:
    """Return a bounded worker error without secrets or absolute user paths."""
    if not event:
        return "no protocol error returned"
    message = str(event.get("message") or type(event).__name__)
    token = os.environ.get("MULTISOCIAL_WORKER_HF_TOKEN")
    if token:
        message = message.replace(token, "[REDACTED]")
    message = re.sub(r"[A-Za-z]:[\\/][^\s'\"]+", "[REDACTED_PATH]", message)
    return message.replace("\r", " ").replace("\n", " ")[-500:]


def _safe_launcher_stderr(value: str) -> str:
    """Return a bounded, redacted launcher diagnostic for a failed direct probe."""
    token = os.environ.get("MULTISOCIAL_WORKER_HF_TOKEN")
    message = value.replace("\x00", "")
    if token:
        message = message.replace(token, "[REDACTED]")
    message = re.sub(r"[A-Za-z]:[\\/][^\s'\"]+", "[REDACTED_PATH]", message)
    message = message.replace("\r", " ").replace("\n", " ").strip()
    return message[-500:] or "no launcher stderr returned"


def run_worker_request(
    worker: Path,
    request: Path,
    result: Path,
    *,
    timeout: int = 60,
    preserve_diagnostics: bool = False,
) -> dict:
    """Probe the worker directly, before introducing GUI launch state."""
    worker = worker.resolve()
    request = request.resolve()
    result = result.resolve()
    if not worker.is_file() or not request.is_file():
        raise FileNotFoundError("The packaged worker and request JSON must both exist")
    result.unlink(missing_ok=True)
    diagnostic_path = _diagnostic_path(result)
    diagnostic_path.unlink(missing_ok=True)
    value = json.loads(request.read_text(encoding="utf-8"))
    request_id = str(uuid.uuid4())
    protocol_request = {
        "protocol": PROTOCOL_VERSION,
        "id": request_id,
        "type": "run",
        "operation": str(value["operation"]),
        "payload": dict(value.get("payload") or {}),
    }
    environment = _private_worker_environment(worker, diagnostic_path)
    if preserve_diagnostics:
        environment[PRESERVE_DIAGNOSTICS_ENV] = "1"
    else:
        environment.pop(PRESERVE_DIAGNOSTICS_ENV, None)
    process = subprocess.Popen(
        [str(worker)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(worker.parent),
        env=environment,
    )
    stdout, stderr = _communicate_bounded(
        process,
        input_text=json.dumps(protocol_request, separators=(",", ":")) + "\n",
        timeout=timeout,
        diagnostic_path=diagnostic_path,
    )
    response = None
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("protocol") == PROTOCOL_VERSION and event.get("id") == request_id:
            response = event
    if process.returncode != 0 or not response or response.get("event") != "result":
        raise RuntimeError(
            f"Packaged direct worker probe failed (exit {process.returncode}) "
            f"at stages: {_diagnostic_stages(diagnostic_path)}: "
            f"{_safe_protocol_error(response)}; "
            f"launcher stderr: {_safe_launcher_stderr(stderr)}"
        )
    output = {"ok": True, "result": dict(response.get("result") or {})}
    if preserve_diagnostics:
        output["worker_diagnostics"] = _diagnostic_timeline(diagnostic_path)
    _write_result(result, output)
    return output


def run_request(
    gui: Path,
    request: Path,
    result: Path,
    *,
    verify_heavy_pose_asset: bool = False,
    timeout: int = 1200,
    preserve_diagnostics: bool = False,
) -> dict:
    gui = gui.resolve()
    request = request.resolve()
    result = result.resolve()
    if not gui.is_file() or not request.is_file():
        raise FileNotFoundError("The packaged GUI and request JSON must both exist")
    result.unlink(missing_ok=True)
    diagnostic_path = _diagnostic_path(result)
    diagnostic_path.unlink(missing_ok=True)

    environment = os.environ.copy()
    environment["MULTISOCIAL_IMPORT_SMOKE_TEST"] = "1"
    environment["MULTISOCIAL_WORKER_SMOKE_REQUEST"] = str(request)
    environment["MULTISOCIAL_WORKER_SMOKE_RESULT"] = str(result)
    environment[DIAGNOSTIC_ENV] = str(diagnostic_path)
    if preserve_diagnostics:
        environment[PRESERVE_DIAGNOSTICS_ENV] = "1"
    else:
        environment.pop(PRESERVE_DIAGNOSTICS_ENV, None)
    if verify_heavy_pose_asset:
        environment["MULTISOCIAL_VERIFY_HEAVY_POSE_ASSET"] = "1"

    process = subprocess.Popen(
        [str(gui)],
        cwd=str(gui.parent),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _communicate_bounded(
        process,
        input_text=None,
        timeout=timeout,
        diagnostic_path=diagnostic_path,
    )
    if not result.is_file():
        raise RuntimeError(
            f"Packaged GUI exited with {process.returncode} without a structured result "
            f"at stages: {_diagnostic_stages(diagnostic_path)}"
        )
    response = json.loads(result.read_text(encoding="utf-8"))
    if process.returncode != 0 or not response.get("ok"):
        raise RuntimeError(
            f"Packaged GUI-to-worker request failed (exit {process.returncode}) "
            f"at stages: {_diagnostic_stages(diagnostic_path)}: "
            f"{response.get('error', response)!s}"
        )
    if preserve_diagnostics:
        output = dict(response)
        output["worker_diagnostics"] = _diagnostic_timeline(diagnostic_path)
        return output
    return response


def main() -> None:
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--gui", type=Path)
    target.add_argument("--worker", type=Path)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--verify-heavy-pose-asset", action="store_true")
    parser.add_argument("--preserve-diagnostics", action="store_true")
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    if args.worker:
        response = run_worker_request(
            args.worker,
            args.request,
            args.result,
            timeout=args.timeout,
            preserve_diagnostics=args.preserve_diagnostics,
        )
    else:
        response = run_request(
            args.gui,
            args.request,
            args.result,
            verify_heavy_pose_asset=args.verify_heavy_pose_asset,
            timeout=args.timeout,
            preserve_diagnostics=args.preserve_diagnostics,
        )
    print(json.dumps(response, ensure_ascii=False))


if __name__ == "__main__":
    main()
