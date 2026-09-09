"""Post-assembly ownership checks for the two Windows onedir runtimes."""

from __future__ import annotations

import argparse
from pathlib import Path

from windows_build_support import PE_MACHINE_AMD64, VC_RUNTIME_NAMES, _pe_machine

PYTHON_DLL_PREFIX = "python3"
VERSIONED_PYTHON_RUNTIME_DLL = "python310.dll"
STABLE_ABI_PYTHON_RUNTIME_DLL = "python3.dll"
ALLOWED_PYTHON_RUNTIME_DLLS = {
    VERSIONED_PYTHON_RUNTIME_DLL,
    STABLE_ABI_PYTHON_RUNTIME_DLL,
}
GUI_FORBIDDEN_FILE_MARKERS = {
    "_framework_bindings",
    "audresample",
    "c10",
    "fbgemm",
    "libiomp",
    "mediapipe",
    "opencv",
    "smileapi",
    "torch",
}
WORKER_FORBIDDEN_FILE_MARKERS = {"wxbase", "wxmsw", "wxpython"}


def _runtime_files(root: Path, excluded: Path | None = None) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if excluded is not None and (path == excluded or excluded in path.parents):
            continue
        files.append(path)
    return files


def _python_runtime_dlls(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_file()
            and path.name.casefold().startswith(PYTHON_DLL_PREFIX)
            and path.suffix.casefold() == ".dll"
        ),
        key=lambda path: path.name.casefold(),
    )


def _assert_runtime_root(root: Path, executable: str, excluded: Path | None = None) -> None:
    if not (root / executable).is_file():
        raise RuntimeError(f"Missing runtime executable: {root / executable}")
    python_dlls = _python_runtime_dlls(root)
    python_dll_names = {path.name.casefold() for path in python_dlls}
    unexpected_python_dlls = sorted(python_dll_names - ALLOWED_PYTHON_RUNTIME_DLLS)
    if VERSIONED_PYTHON_RUNTIME_DLL not in python_dll_names or unexpected_python_dlls:
        found = ", ".join(path.name for path in python_dlls) or "<none>"
        expected = (
            f"{VERSIONED_PYTHON_RUNTIME_DLL}, with optional "
            f"{STABLE_ABI_PYTHON_RUNTIME_DLL}"
        )
        details = (
            f"; unexpected: {', '.join(unexpected_python_dlls)}"
            if unexpected_python_dlls
            else ""
        )
        raise RuntimeError(
            f"Invalid Python runtime DLL set at {root}; expected {expected}; "
            f"found: {found}{details}"
        )
    missing_vc = sorted(name for name in VC_RUNTIME_NAMES if not (root / name).is_file())
    if missing_vc:
        raise RuntimeError(f"Missing VC runtime files at {root}: {', '.join(missing_vc)}")
    architecture_files = [
        root / executable,
        *python_dlls,
        *(root / name for name in sorted(VC_RUNTIME_NAMES)),
    ]
    wrong_architecture = [
        path.name
        for path in architecture_files
        if _pe_machine(path) != PE_MACHINE_AMD64
    ]
    if wrong_architecture:
        raise RuntimeError(
            f"Runtime contains non-AMD64 PE files at {root}: {', '.join(wrong_architecture)}"
        )
    nested_python_dlls = [
        path.relative_to(root)
        for path in _runtime_files(root, excluded)
        if path.name.casefold().startswith(PYTHON_DLL_PREFIX)
        and path.suffix.casefold() == ".dll"
        and path.parent != root
    ]
    if nested_python_dlls:
        raise RuntimeError(
            f"Duplicate nested Python runtime files under {root}: {nested_python_dlls[:10]}"
        )

def _path_has_marker(path: Path, markers: set[str]) -> bool:
    parts = {part.casefold() for part in path.parts}
    basename = path.name.casefold()
    return bool(parts & markers) or any(marker in basename for marker in markers)


def _complete_runtime_packages_in_standard_worker(worker: Path) -> list[str]:
    """Return actually-installed Complete-only packages, not matching filenames.

    ``transformers`` deliberately includes optional-dependency stubs such as
    ``dummy_torchaudio_objects.py``.  PyInstaller's *build-time* hook package
    similarly has files named after packages it can analyze.  Neither is an
    installed worker runtime dependency, so the boundary check must inspect
    concrete package roots and distribution metadata instead of substrings in
    arbitrary filenames.
    """

    site_packages = worker / "Lib" / "site-packages"
    hits: list[str] = []
    for package in ("pyannote", "speechbrain", "torchaudio"):
        if (site_packages / package).is_dir():
            hits.append(str(Path("Lib/site-packages") / package))
        metadata = sorted(site_packages.glob(f"{package}-*.dist-info"))
        hits.extend(str(path.relative_to(worker)) for path in metadata if path.is_dir())
    return hits


def validate(root: Path, profile: str) -> None:
    app_name = "MultiSOCIAL-Complete" if profile == "complete" else "MultiSOCIAL-Standard"
    worker = root / "worker"
    _assert_runtime_root(root, f"{app_name}.exe", excluded=worker)
    _assert_runtime_root(worker, "python.exe")
    launcher = worker / "MultiSOCIAL-Worker-Launcher.exe"
    if not launcher.is_file() or _pe_machine(launcher) != PE_MACHINE_AMD64:
        raise RuntimeError("Worker native launcher is missing or is not AMD64")

    forbidden_gui_parts = {
        "audio.py",
        "cv2",
        "mediapipe",
        "opensmile",
        "pose.py",
        "pyannote",
        "speechbrain",
        "torch",
        "torchaudio",
        "transformers",
        "ultralytics",
        "yolov5",
    }
    gui_hits = [
        str(path.relative_to(root))
        for path in _runtime_files(root, excluded=worker)
        if _path_has_marker(path, forbidden_gui_parts | GUI_FORBIDDEN_FILE_MARKERS)
    ]
    if gui_hits:
        raise RuntimeError("GUI runtime contains worker-owned files: " + ", ".join(gui_hits[:20]))
    if any(
        _path_has_marker(path.relative_to(worker), WORKER_FORBIDDEN_FILE_MARKERS)
        or any(
            part.casefold() == "wx" or part.casefold().startswith("wx.")
            for part in path.relative_to(worker).parts
        )
        for path in worker.rglob("*")
    ):
        raise RuntimeError("Worker runtime contains wxPython")

    required_worker_paths = [
        "Lib/site-packages/audresample/core/bin/win_amd64/audresample.dll",
        "Lib/site-packages/opensmile/core/bin/win_amd64/SMILEapi.dll",
        "assets/pose_landmark_heavy.tflite",
    ]
    missing = [path for path in required_worker_paths if not (worker / path).is_file()]
    if missing:
        raise RuntimeError("Worker-native files missing: " + ", ".join(missing))
    if not list(worker.rglob("_framework_bindings*.pyd")):
        raise RuntimeError("Worker MediaPipe binding is missing")
    if not list(worker.rglob("cv2*.pyd")):
        raise RuntimeError("Worker OpenCV binding is missing")
    if not any(path.name.casefold() in {"torch_cpu.dll", "torch.dll"} for path in worker.rglob("*.dll")):
        raise RuntimeError("Worker Torch runtime is missing")
    if profile == "standard":
        hits = _complete_runtime_packages_in_standard_worker(worker)
        if hits:
            raise RuntimeError("Standard worker contains Complete-only files: " + ", ".join(hits[:20]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--profile", choices=("standard", "complete"), required=True)
    args = parser.parse_args()
    validate(args.root.resolve(), args.profile)
    print("Windows GUI/worker runtime boundary validation passed.")


if __name__ == "__main__":
    main()
