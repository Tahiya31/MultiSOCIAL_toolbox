"""Audit installed distributions before either Windows PyInstaller process runs."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.machinery
import re
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version
from windows_hiddenimports import (
    COMPLETE_HIDDEN_IMPORTS,
    STANDARD_HIDDEN_IMPORTS,
    TORCHAUDIO_RUNTIME_HIDDEN_IMPORTS,
    TORCH_RUNTIME_HIDDEN_IMPORTS,
    YOLOV5_INFERENCE_HIDDEN_IMPORTS,
)


NATIVE_ANALYSIS_DISTRIBUTIONS = {
    "mediapipe",
    "opensmile",
    "opencv-contrib-python",
    "opencv-python",
    "opencv-python-headless",
    "pyannote-audio",
    "speechbrain",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "yolov5",
}


def canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def versions_match(installed: str, locked: str) -> bool:
    """Compare distribution versions using PEP 440 normalization."""
    try:
        return Version(installed) == Version(locked)
    except InvalidVersion as exc:
        raise RuntimeError(
            f"Invalid distribution version while comparing installed {installed!r} "
            f"with locked {locked!r}"
        ) from exc


def installed_versions() -> dict[str, str]:
    inventory: dict[str, list[str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        inventory.setdefault(canonicalize_name(name.strip()), []).append(distribution.version)
    duplicates = {
        name: versions
        for name, versions in inventory.items()
        if len(versions) != 1
    }
    if duplicates:
        details = ", ".join(
            f"{name}={versions!r}" for name, versions in sorted(duplicates.items())
        )
        raise RuntimeError("Duplicate installed distributions detected: " + details)
    return {name: versions[0] for name, versions in inventory.items()}


def locked_versions(lock_path: Path) -> dict[str, str]:
    locked: dict[str, str] = {}
    requirement = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;\\]+)")
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        match = requirement.match(line)
        if not match:
            continue
        name = canonicalize_name(match.group(1))
        if name in locked:
            raise RuntimeError(f"Duplicate locked requirement: {name}")
        block_end = next(
            (
                following
                for following in range(index + 1, len(lines))
                if requirement.match(lines[following])
            ),
            len(lines),
        )
        if "--hash=sha256:" not in "\n".join(lines[index:block_end]):
            raise RuntimeError(f"Locked requirement is missing a SHA256 hash: {name}")
        locked[name] = match.group(2)
    if not locked:
        raise RuntimeError(f"No pinned requirements found in lock: {lock_path}")
    return locked


def assert_matches_lock(installed: dict[str, str], lock_path: Path) -> None:
    locked = locked_versions(lock_path)
    # pip is created by venv and is intentionally not part of application locks.
    unexpected = sorted(set(installed) - set(locked) - {"pip"})
    missing = sorted(set(locked) - set(installed))
    mismatched = sorted(
        f"{name}=={installed[name]} (locked {locked[name]})"
        for name in set(installed) & set(locked)
        if not versions_match(installed[name], locked[name])
    )
    failures = []
    if unexpected:
        failures.append("unexpected distributions: " + ", ".join(unexpected))
    if missing:
        failures.append("missing distributions: " + ", ".join(missing))
    if mismatched:
        failures.append("version mismatches: " + ", ".join(mismatched))
    if failures:
        raise RuntimeError("Installed environment differs from hashed lock; " + "; ".join(failures))


def missing_manifest_modules(
    module_names: list[str],
    search_roots: list[Path] | None = None,
) -> list[str]:
    """Check manifest paths without importing native package parents."""
    roots = search_roots or [
        Path(__file__).resolve().parent.parent / "src",
        *(Path(value) for value in sys.path if value),
    ]
    suffixes = importlib.machinery.all_suffixes()
    missing = []
    for module_name in module_names:
        relative = Path(*module_name.split("."))
        found = False
        for root in roots:
            candidate = root / relative
            if candidate.is_dir() or any(
                candidate.with_suffix(suffix).is_file()
                for suffix in suffixes
            ):
                found = True
                break
        if not found:
            missing.append(module_name)
    return sorted(set(missing), key=str.casefold)


def assert_worker_manifests_exist(installed_names: set[str]) -> None:
    modules = [
        *STANDARD_HIDDEN_IMPORTS,
        *TORCH_RUNTIME_HIDDEN_IMPORTS,
        *YOLOV5_INFERENCE_HIDDEN_IMPORTS,
    ]
    if "pyannote-audio" in installed_names:
        modules.extend(COMPLETE_HIDDEN_IMPORTS)
        modules.extend(TORCHAUDIO_RUNTIME_HIDDEN_IMPORTS)
    missing = missing_manifest_modules(modules)
    if missing:
        raise RuntimeError(
            "Windows worker manifest references missing modules: "
            + ", ".join(missing[:30])
        )


def audit(kind: str, lock_path: Path | None = None) -> None:
    installed = installed_versions()
    if lock_path is not None:
        assert_matches_lock(installed, lock_path)
    names = set(installed)
    if kind == "gui":
        hits = sorted(names & NATIVE_ANALYSIS_DISTRIBUTIONS)
        if hits:
            raise RuntimeError("Native analysis distributions installed in GUI environment: " + ", ".join(hits))
        if "wxpython" not in names:
            raise RuntimeError("GUI environment is missing wxPython")
        return

    if "wxpython" in names:
        raise RuntimeError("Worker environment contains wxPython")
    providers = sorted(
        name for name in names
        if name in {"opencv-contrib-python", "opencv-python", "opencv-python-headless"}
    )
    if providers != ["opencv-contrib-python"]:
        raise RuntimeError("Worker must have exactly one cv2 provider (opencv-contrib-python): " + repr(providers))
    cv2_owners = sorted(
        canonicalize_name(name)
        for name in importlib.metadata.packages_distributions().get("cv2", [])
    )
    if cv2_owners != ["opencv-contrib-python"]:
        raise RuntimeError("cv2 namespace ownership is ambiguous: " + repr(cv2_owners))
    assert_worker_manifests_exist(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("gui", "worker"), required=True)
    parser.add_argument("--lock", type=Path)
    args = parser.parse_args()
    audit(args.kind, args.lock)
    print(f"Windows {args.kind} environment ownership audit passed.")


if __name__ == "__main__":
    main()
