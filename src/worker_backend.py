"""Windows GUI backend that communicates only with the private worker."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import runtime_services
from native_worker_client import (
    NativeWorkerClient,
    WORKER_TOKEN_ENV,
    WindowsAudioProcessor,
    WindowsPoseProcessor,
    find_pose_csv_paths as _find_pose_csv_paths,
)

SMOKE_REQUEST_ENV = "MULTISOCIAL_WORKER_SMOKE_REQUEST"
SMOKE_RESULT_ENV = "MULTISOCIAL_WORKER_SMOKE_RESULT"


def _write_smoke_result(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _run_packaged_smoke_request() -> None:
    request_value = os.environ.get(SMOKE_REQUEST_ENV)
    result_value = os.environ.get(SMOKE_RESULT_ENV)
    if not request_value and not result_value:
        return
    if not request_value or not result_value:
        raise RuntimeError("Both packaged worker smoke request and result paths are required")

    request_path = Path(request_value).resolve()
    result_path = Path(result_value).resolve()
    token = os.environ.pop(WORKER_TOKEN_ENV, None)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        operation = str(request["operation"])
        payload = dict(request.get("payload") or {})
        cancel_after = request.get("cancel_after_seconds")
        timeout_seconds = request.get("timeout_seconds")
        started = time.monotonic()
        client = (
            NativeWorkerClient(timeout_seconds=float(timeout_seconds))
            if timeout_seconds is not None
            else NativeWorkerClient()
        )
        result = client.run(
            operation,
            payload,
            cancel_check=(
                (lambda: time.monotonic() - started >= float(cancel_after))
                if cancel_after is not None
                else None
            ),
            hf_token=token,
        )
        _write_smoke_result(result_path, {"ok": True, "result": result})
    except Exception as exc:
        message = str(exc)
        if token:
            message = message.replace(token, "[REDACTED]")
        _write_smoke_result(
            result_path,
            {"ok": False, "error": message[-2000:] or type(exc).__name__},
        )
        raise RuntimeError(message[-2000:] or type(exc).__name__) from None


class WindowsWorkerBackend:
    def create_audio_processor(self, *args, **kwargs):
        return WindowsAudioProcessor(*args, **kwargs)

    def create_pose_processor(self, *args, **kwargs):
        return WindowsPoseProcessor(*args, **kwargs)

    def find_pose_csv_paths(
        self,
        output_folder: str,
        video_path: str,
        multi_person: Optional[bool] = None,
    ) -> list[str]:
        return _find_pose_csv_paths(output_folder, video_path, multi_person)

    def validate_import_smoke(self, profile: str, verify_heavy_pose_asset: bool) -> str:
        if verify_heavy_pose_asset:
            if getattr(sys, "frozen", False):
                model = (
                    Path(sys.executable).resolve().parent
                    / "worker"
                    / "assets"
                    / "pose_landmark_heavy.tflite"
                )
            else:
                model = Path(runtime_services.resource_path("assets", "pose_landmark_heavy.tflite"))
            if not model.is_file():
                raise RuntimeError(f"Missing bundled Heavy pose model: {os.path.basename(model)}")
            print("Bundled Heavy pose model check passed.", flush=True)
        if os.environ.get("MULTISOCIAL_VERIFY_WORKER_LAUNCH") == "1":
            result = NativeWorkerClient().run("probe", {"profile": profile})
            runtime = result.get("runtime") or {}
            worker_runtime = runtime.get("worker_runtime") or {}
            if (
                not worker_runtime.get("private_runtime")
                or worker_runtime.get("loaded_module_violations")
                or worker_runtime.get("native_module_violations")
                or worker_runtime.get("external_module_violations")
            ):
                raise RuntimeError("Private worker runtime provenance check failed")
            print("GUI-to-worker isolated launch check passed.", flush=True)
        _run_packaged_smoke_request()
        return f"Import smoke test passed ({profile or 'standard'} profile)."
