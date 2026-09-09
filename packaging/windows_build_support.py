"""Shared helpers for the two physically independent Windows PyInstaller specs."""

from __future__ import annotations

import os
import struct
import sys
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable


VC_RUNTIME_NAMES = {
    "concrt140.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcamp140.dll",
    "vccorlib140.dll",
    "vcomp140.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vcruntime140_threads.dll",
}
GUI_FORBIDDEN_MODULE_ROOTS = {
    "analysis_worker",
    "audio",
    "audresample",
    "cv2",
    "mediapipe",
    "native_backend",
    "opensmile",
    "pose",
    "pyannote",
    "source_entry",
    "speechbrain",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "ultralytics",
    "yolov5",
}
GUI_FORBIDDEN_PATH_PARTS = GUI_FORBIDDEN_MODULE_ROOTS | {
    "numpy.libs",
    "opencv",
    "openmp",
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
PE_MACHINE_AMD64 = 0x8664


def _pe_machine(path: Path) -> int | None:
    try:
        with path.open("rb") as handle:
            if handle.read(2) != b"MZ":
                return None
            handle.seek(0x3C)
            pe_offset_data = handle.read(4)
            if len(pe_offset_data) != 4:
                return None
            handle.seek(struct.unpack("<I", pe_offset_data)[0])
            if handle.read(4) != b"PE\0\0":
                return None
            machine_data = handle.read(2)
            return struct.unpack("<H", machine_data)[0] if len(machine_data) == 2 else None
    except OSError:
        return None


def _normalized_source(entry) -> str | None:
    if not isinstance(entry, tuple) or not entry:
        return None
    value = entry[1] if len(entry) >= 3 else entry[0]
    return os.path.normcase(os.path.abspath(str(value)))


def collect_vc_runtime_binaries() -> list[tuple[str, str]]:
    """Collect exactly one approved VC runtime set from the build environment."""
    if sys.maxsize <= 2**32:
        raise RuntimeError("Windows packages require a 64-bit Python build environment")
    candidate_directories: list[Path] = []
    spec = find_spec("msvc_runtime")
    if spec is not None and spec.origin:
        origin = Path(spec.origin)
        msvc_root = origin.parent if origin.is_file() else origin
        candidate_directories.extend(
            sorted(
                {
                    path.parent
                    for path in msvc_root.rglob("*.dll")
                    if path.name.casefold() in VC_RUNTIME_NAMES
                },
                key=lambda path: (
                    not any(part.casefold() in {"amd64", "x64"} for part in path.parts),
                    len(path.parts),
                    str(path).casefold(),
                ),
            )
        )
    python_roots = list(dict.fromkeys([Path(sys.executable).resolve().parent, Path(sys.base_prefix)]))
    for root in python_roots:
        for directory in (root, root / "DLLs", root / "Library" / "bin"):
            if directory.is_dir():
                candidate_directories.append(directory)

    seen: set[Path] = set()
    bad_architecture_parts = {"arm", "arm64", "win32", "x86"}
    for directory in candidate_directories:
        resolved = directory.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if {part.casefold() for part in resolved.parts} & bad_architecture_parts:
            continue
        available = {
            path.name.casefold(): path
            for path in resolved.iterdir()
            if path.is_file() and path.name.casefold() in VC_RUNTIME_NAMES
        }
        if set(available) == VC_RUNTIME_NAMES and all(
            _pe_machine(path) == PE_MACHINE_AMD64 for path in available.values()
        ):
            return [(str(available[name]), ".") for name in sorted(available)]
    raise RuntimeError(
        "No complete same-directory AMD64 VC runtime set was found in the isolated build environment"
    )


def filter_vc_runtime_entries(entries: Iterable[tuple], approved_sources: set[str]) -> list[tuple]:
    filtered = []
    for entry in entries:
        if not isinstance(entry, tuple) or not entry:
            filtered.append(entry)
            continue
        destination = str(entry[0]).replace("\\", "/")
        basename = os.path.basename(destination).casefold()
        source = _normalized_source(entry)
        if basename in VC_RUNTIME_NAMES:
            if os.path.dirname(destination) not in ("", ".") or source not in approved_sources:
                continue
        filtered.append(entry)
    return filtered


def approved_vc_sources(entries: Iterable[tuple]) -> set[str]:
    return {
        source
        for source in (_normalized_source(entry) for entry in entries)
        if source
    }


def _module_names(entries: Iterable[tuple]) -> set[str]:
    return {str(entry[0]) for entry in entries if isinstance(entry, tuple) and entry}


def _entry_text(entry: tuple) -> str:
    return " | ".join(str(value).replace("\\", "/").casefold() for value in entry)

def _entry_has_marker(entry: tuple, markers: set[str]) -> bool:
    components = {
        component
        for value in entry
        for component in str(value).replace("\\", "/").casefold().split("/")
    }
    basenames = {
        os.path.basename(str(value).replace("\\", "/")).casefold()
        for value in entry
    }
    return bool(components & markers) or any(
        marker in basename for marker in markers for basename in basenames
    )


def assert_gui_graph(analysis) -> None:
    module_hits = sorted(
        name
        for name in _module_names(analysis.pure)
        if name.split(".", 1)[0].casefold() in GUI_FORBIDDEN_MODULE_ROOTS
    )
    native_hits = []
    for entry in [*analysis.binaries, *analysis.datas]:
        if _entry_has_marker(entry, GUI_FORBIDDEN_PATH_PARTS | GUI_FORBIDDEN_FILE_MARKERS):
            native_hits.append(str(entry[0]))
    if module_hits or native_hits:
        details = [*(f"module:{name}" for name in module_hits), *(f"file:{name}" for name in native_hits)]
        raise RuntimeError("Windows GUI graph crossed the native-analysis boundary: " + ", ".join(details[:30]))


def assert_worker_graph(analysis) -> None:
    wx_hits = sorted(name for name in _module_names(analysis.pure) if name.split(".", 1)[0].casefold() == "wx")
    native_hits = [
        str(entry[0])
        for entry in [*analysis.binaries, *analysis.datas]
        if _entry_has_marker(entry, WORKER_FORBIDDEN_FILE_MARKERS)
        or any(
            component == "wx" or component.startswith("wx.")
            for value in entry
            for component in str(value).replace("\\", "/").casefold().split("/")
        )
    ]
    if wx_hits or native_hits:
        details = [*(f"module:{name}" for name in wx_hits), *(f"file:{name}" for name in native_hits)]
        raise RuntimeError("Windows worker graph contains wxPython: " + ", ".join(details[:20]))
