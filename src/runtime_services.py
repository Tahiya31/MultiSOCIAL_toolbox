"""
Runtime services for versioning, optional feature state, and local app persistence.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Callable, Dict, Optional, Tuple

APP_DISPLAY_NAME = "MultiSOCIAL Toolbox"
PROJECT_DIST_NAME = "multisocial-toolbox"
DEFAULT_BUILD_PROFILE = "standard"
DIARIZATION_FEATURE = "diarization"
DIARIZATION_PIP_SPEC = ("pyannote.audio==3.2.0", "speechbrain==0.5.16", "torchaudio==2.5.1")
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"


def bundle_root() -> str:
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.dirname(__file__))


def _frozen_resource_roots() -> list[str]:
    roots = []
    seen = set()

    def add(path: Optional[str]) -> None:
        if not path:
            return
        normalized = os.path.normcase(os.path.abspath(path))
        if normalized in seen:
            return
        seen.add(normalized)
        roots.append(path)

    meipass = getattr(sys, "_MEIPASS", None)
    executable_dir = os.path.dirname(sys.executable) if getattr(sys, "executable", None) else None

    add(meipass)
    add(os.path.join(meipass, "_internal") if meipass else None)
    add(executable_dir)
    add(os.path.join(executable_dir, "_internal") if executable_dir else None)

    if sys.platform == "darwin" and executable_dir:
        contents_dir = os.path.dirname(executable_dir)
        add(contents_dir)
        add(os.path.join(contents_dir, "Resources"))
        add(os.path.join(contents_dir, "Resources", "_internal"))
        add(os.path.join(contents_dir, "Frameworks"))
        add(os.path.join(contents_dir, "Frameworks", "_internal"))

    return roots


def project_root() -> str:
    return bundle_root()


def resource_path(*parts: str) -> str:
    if getattr(sys, "frozen", False):
        for root in _frozen_resource_roots():
            candidate = os.path.join(root, *parts)
            if os.path.exists(candidate):
                return candidate
        preferred_root = _frozen_resource_roots()[0] if _frozen_resource_roots() else bundle_root()
        return os.path.join(preferred_root, *parts)
    return os.path.join(bundle_root(), *parts)


def _config_root() -> str:
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
        return os.path.join(base, APP_DISPLAY_NAME)
    if sys.platform == "darwin":
        return os.path.join(os.path.expanduser("~/Library/Application Support"), APP_DISPLAY_NAME)
    return os.path.join(os.path.expanduser("~"), ".multisocial_toolbox")


def ensure_config_root() -> str:
    candidates = [
        _config_root(),
        os.path.join(project_root(), ".multisocial_toolbox_runtime"),
        os.path.join(tempfile.gettempdir(), "multisocial_toolbox"),
    ]
    for root in candidates:
        try:
            os.makedirs(root, exist_ok=True)
            return root
        except Exception:
            continue
    raise RuntimeError("Could not create a writable config directory for MultiSOCIAL Toolbox.")


def state_path() -> str:
    return os.path.join(ensure_config_root(), "state.json")


def load_state() -> Dict[str, object]:
    path = state_path()
    if not os.path.exists(path):
        return {"features": {}, "secrets": {}, "install_profile": DEFAULT_BUILD_PROFILE}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return {"features": {}, "secrets": {}, "install_profile": DEFAULT_BUILD_PROFILE}
    data.setdefault("features", {})
    data.setdefault("secrets", {})
    data.setdefault("install_profile", DEFAULT_BUILD_PROFILE)
    return data


def save_state(state: Dict[str, object]) -> None:
    with open(state_path(), "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)


def get_app_version() -> str:
    try:
        return importlib.metadata.version(PROJECT_DIST_NAME)
    except Exception:
        pass

    pyproject_file = os.path.join(project_root(), "pyproject.toml")
    try:
        with open(pyproject_file, "r", encoding="utf-8") as handle:
            match = re.search(r'^version\s*=\s*"([^"]+)"', handle.read(), re.MULTILINE)
            if match:
                return match.group(1)
    except Exception:
        pass

    return "0.0.0-dev"


def get_app_title() -> str:
    return f"{APP_DISPLAY_NAME} v{get_app_version()}"


def is_frozen_runtime() -> bool:
    return bool(getattr(sys, "frozen", False))


def can_self_install_optional_features() -> bool:
    if is_frozen_runtime():
        return False
    return sys.executable is not None and bool(importlib.util.find_spec("pip"))


def is_diarization_installed() -> bool:
    try:
        return importlib.util.find_spec("pyannote.audio") is not None
    except ModuleNotFoundError:
        return False


def get_build_profile() -> str:
    env_profile = os.environ.get("MULTISOCIAL_INSTALL_PROFILE") or os.environ.get(
        "MULTISOCIAL_BUILD_PROFILE"
    )
    if env_profile:
        return env_profile.strip().lower()

    if is_frozen_runtime():
        executable_name = os.path.basename(sys.executable).lower()
        if "complete" in executable_name:
            return "complete"
        if "standard" in executable_name:
            return "standard"

    state = load_state()
    state_profile = str(state.get("install_profile") or "").strip().lower()
    if state_profile and state_profile != DEFAULT_BUILD_PROFILE:
        return state_profile
    if is_diarization_installed():
        return "complete"
    return state_profile or DEFAULT_BUILD_PROFILE


def save_install_profile(profile: str) -> None:
    state = load_state()
    state["install_profile"] = profile.strip().lower()
    save_state(state)


def _normalize_hf_token(value: Optional[str]) -> Optional[str]:
    raw = str(value or "").strip()
    if not raw:
        return None
    if raw.upper().startswith("HF_TOKEN="):
        raw = raw.split("=", 1)[1].strip()
    return raw or None


def load_hf_token() -> Optional[str]:
    env_token = _normalize_hf_token(os.environ.get("HF_TOKEN"))
    if env_token:
        return env_token
    state = load_state()
    token = _normalize_hf_token(state.get("secrets", {}).get("hf_token"))
    return token


def save_hf_token(token: str) -> None:
    state = load_state()
    secrets = dict(state.get("secrets", {}))
    secrets["hf_token"] = _normalize_hf_token(token) or ""
    state["secrets"] = secrets
    save_state(state)


def update_feature_state(
    feature_name: str,
    *,
    status: Optional[str] = None,
    requested: Optional[bool] = None,
    last_error: Optional[str] = None,
) -> None:
    state = load_state()
    features = dict(state.get("features", {}))
    feature_state = dict(features.get(feature_name, {}))
    if status is not None:
        feature_state["status"] = status
    if requested is not None:
        feature_state["requested"] = requested
    if last_error is not None:
        feature_state["last_error"] = last_error
    elif "last_error" in feature_state and status in {"enabled", "not_installed"}:
        feature_state.pop("last_error", None)
    features[feature_name] = feature_state
    state["features"] = features
    save_state(state)


def get_diarization_feature_state() -> Dict[str, object]:
    state = load_state()
    feature_state = dict(state.get("features", {}).get(DIARIZATION_FEATURE, {}))
    installed = is_diarization_installed()
    status = "enabled" if installed else str(feature_state.get("status") or "not_installed")
    if status == "enabled" and not installed:
        status = "not_installed"
    return {
        "installed": installed,
        "status": status,
        "requested": bool(feature_state.get("requested", False)),
        "last_error": feature_state.get("last_error"),
        "can_self_install": can_self_install_optional_features(),
        "build_profile": get_build_profile(),
    }


def install_diarization_support(
    *,
    status_callback: Optional[Callable[[str], None]] = None,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> Tuple[bool, Optional[str]]:
    if is_diarization_installed():
        update_feature_state(DIARIZATION_FEATURE, status="enabled", requested=True, last_error="")
        return True, None

    if not can_self_install_optional_features():
        message = (
            "This build cannot install diarization in-place. Use a Complete build or a source install."
        )
        update_feature_state(DIARIZATION_FEATURE, status="failed", requested=True, last_error=message)
        return False, message

    update_feature_state(DIARIZATION_FEATURE, status="installing", requested=True, last_error="")
    if status_callback:
        status_callback("Installing optional diarization support...")
    if progress_callback:
        progress_callback(5)

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--disable-pip-version-check",
    ] + list(DIARIZATION_PIP_SPEC)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    last_output = []
    checkpoints = [15, 35, 55, 75, 90]
    checkpoint_index = 0

    assert process.stdout is not None
    for line in process.stdout:
        cleaned = line.strip()
        if cleaned:
            last_output.append(cleaned)
            last_output = last_output[-15:]
            if status_callback:
                status_callback(cleaned)
        if progress_callback and checkpoint_index < len(checkpoints):
            progress_callback(checkpoints[checkpoint_index])
            checkpoint_index += 1

    process.wait()
    if process.returncode != 0:
        error_message = "\n".join(last_output[-5:]) or "pip install failed"
        update_feature_state(
            DIARIZATION_FEATURE,
            status="failed",
            requested=True,
            last_error=error_message,
        )
        return False, error_message

    importlib.invalidate_caches()
    if not is_diarization_installed():
        error_message = "Installation completed but pyannote.audio is still unavailable in this session."
        update_feature_state(
            DIARIZATION_FEATURE,
            status="failed",
            requested=True,
            last_error=error_message,
        )
        return False, error_message

    update_feature_state(DIARIZATION_FEATURE, status="enabled", requested=True, last_error="")
    if progress_callback:
        progress_callback(100)
    return True, None


def get_startup_diagnostics() -> Dict[str, object]:
    diarization_state = get_diarization_feature_state()
    return {
        "app": APP_DISPLAY_NAME,
        "version": get_app_version(),
        "python": sys.version.split()[0],
        "frozen": is_frozen_runtime(),
        "install_profile": get_build_profile(),
        "ffmpeg_source": os.environ.get("MULTISOCIAL_FFMPEG_SOURCE", "unknown"),
        "diarization_status": diarization_state["status"],
        "diarization_installed": diarization_state["installed"],
    }
