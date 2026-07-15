# -*- mode: python ; coding: utf-8 -*-

import os
import re
import sys
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)

# PyInstaller can hit Python's default recursion depth on large dependency
# graphs, especially on Windows with the current ML stack.
sys.setrecursionlimit(max(sys.getrecursionlimit() * 5, 5000))


ROOT = os.path.abspath(globals().get("SPECPATH", os.getcwd()))
SRC = os.path.join(ROOT, "src")
HOOKS = os.path.join(ROOT, "hooks")
BUILD_PROFILE = os.environ.get("MULTISOCIAL_BUILD_PROFILE", "standard").strip().lower()
APP_NAME = "MultiSOCIAL-Complete" if BUILD_PROFILE == "complete" else "MultiSOCIAL-Standard"
_pyproject_text = Path(ROOT, "pyproject.toml").read_text(encoding="utf-8")
_version_match = re.search(r'^version\s*=\s*"([^"]+)"', _pyproject_text, re.MULTILINE)
if _version_match is None:
    raise RuntimeError("Could not determine application version from pyproject.toml")
APP_VERSION = _version_match.group(1)
IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"
ICON_ICO = os.path.join(ROOT, "assets", "MultiSOCIAL_logo.ico")
ICON_ICNS = os.path.join(ROOT, "assets", "MultiSOCIAL_logo.icns")


def collect_sidecar_binaries(module_name, patterns):
    spec = find_spec(module_name)
    if spec is None or spec.origin is None:
        return []

    origin_path = Path(spec.origin)
    search_root = origin_path.parent if origin_path.is_file() else origin_path
    collected = []
    for pattern in patterns:
        for path in search_root.glob(pattern):
            if path.is_file():
                collected.append((str(path), "."))
    return collected


def collect_named_runtime_dlls(candidate_dirs, dll_names, recursive=False):
    collected = []
    seen = set()
    normalized_names = {name.lower() for name in dll_names}
    for directory in candidate_dirs:
        if not directory:
            continue
        dir_path = Path(directory)
        if not dir_path.exists():
            continue

        search_dirs = [dir_path]
        if not recursive:
            search_dirs.extend(
                [
                    dir_path / "DLLs",
                    dir_path / "bin",
                    dir_path / "libs",
                    dir_path / "Library" / "bin",
                ]
            )

        if recursive:
            matches = [
                path
                for path in dir_path.rglob("*")
                if path.is_file() and path.name.lower() in normalized_names
            ]
        else:
            matches = []
            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue
                for path in search_dir.iterdir():
                    if path.is_file() and path.name.lower() in normalized_names:
                        matches.append(path)

        for candidate in matches:
            key = os.path.normcase(str(candidate.resolve()))
            if key in seen:
                continue
            seen.add(key)
            collected.append((str(candidate), "."))
    return collected


def normalized_source_path(entry):
    if not isinstance(entry, tuple) or len(entry) == 0:
        return None
    # PyInstaller TOC entries (like a.binaries) use (name, source_path, typecode)
    source_str = str(entry[1]) if len(entry) >= 3 else str(entry[0])
    return os.path.normcase(os.path.abspath(source_str))


def filter_blocked_runtime_dlls(entries, blocked_names, allowed_sources):
    filtered = []
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) == 0:
            filtered.append(entry)
            continue
            
        source_path = normalized_source_path(entry)
        
        if len(entry) >= 3:
            # TOC entry: (dest_name, source_path, typecode)
            dest_name = str(entry[0]).replace("\\", "/")
            basename = os.path.basename(dest_name).lower()
            
            if basename in blocked_names:
                # Block if it's going into a subfolder (like wx/ or sklearn/.libs/)
                if os.path.dirname(dest_name) not in ("", "."):
                    continue
                # Block if it's not from an allowed source
                if source_path not in allowed_sources:
                    continue
        else:
            # Initial binaries list: (source_path, dest_dir)
            if source_path:
                basename = os.path.basename(source_path).lower()
                if basename in blocked_names and source_path not in allowed_sources:
                    continue
                    
        filtered.append(entry)
    return filtered

hiddenimports = [
    "backports",
    "backports.tarfile",
    "pkg_resources",
    "wx.adv",
    "wx.lib.stattext",
    "imageio_ffmpeg",
    "mediapipe.python._framework_bindings",
    "mediapipe.python.solutions",
    "mediapipe.python.solutions.drawing_utils",
    "mediapipe.python.solutions.pose",
    "mediapipe.python.solutions.pose_connections",
    "opensmile",
    "soundfile",
    "transformers",
]

datas = [
    (os.path.join(ROOT, "assets"), "assets"),
    (os.path.join(ROOT, "assets", "yolov5s.pt"), "assets"),
    (
        os.path.join(ROOT, "assets", "pose_landmark_heavy.tflite"),
        os.path.join("mediapipe", "modules", "pose_landmark"),
    ),
    (os.path.join(ROOT, "env.example"), "."),
    (os.path.join(ROOT, "pyproject.toml"), "."),
]
datas += collect_data_files("mediapipe")
datas += collect_data_files("audresample")
datas += collect_data_files("opensmile")
datas += collect_data_files("audinterface")
datas += collect_data_files("imageio_ffmpeg")
datas += collect_data_files("yolov5")
datas += collect_data_files("ultralytics")
datas += collect_data_files("lightning_fabric")

for package in ("yolov5", "ultralytics", "torch", "torchvision"):
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

hiddenimports += collect_submodules("ultralytics")
binaries = collect_dynamic_libs("mediapipe")
binaries += collect_dynamic_libs("audresample")
binaries += collect_dynamic_libs("opensmile")
binaries += collect_dynamic_libs("audinterface")
binaries += collect_dynamic_libs("soundfile")

if IS_WINDOWS:
    _required_runtime_dlls = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]

    msvc_runtime_binaries = collect_sidecar_binaries("msvc_runtime", ["*.dll"])

    msvc_spec = find_spec("msvc_runtime")
    msvc_search_root = None
    if msvc_spec is not None and msvc_spec.origin is not None:
        msvc_origin = Path(msvc_spec.origin)
        msvc_search_root = msvc_origin.parent if msvc_origin.is_file() else msvc_origin

    # GitHub-hosted Windows runners may not provide sidecar DLLs via msvc_runtime,
    # so also pull runtime DLLs from msvc_runtime package tree and Python install paths.
    fallback_runtime_binaries = collect_named_runtime_dlls(
        [msvc_search_root] if msvc_search_root else [],
        _required_runtime_dlls,
        recursive=True,
    )
    fallback_runtime_binaries += collect_named_runtime_dlls(
        [
            sys.base_prefix,
            sys.base_exec_prefix,
            os.path.dirname(sys.executable),
            os.environ.get("WINDIR"),
            os.path.join(os.environ.get("WINDIR", ""), "System32"),
        ],
        _required_runtime_dlls,
    )

    msvc_runtime_binaries += fallback_runtime_binaries

    # Keep first occurrence for each source path to avoid duplicate entries.
    deduped_runtime_binaries = []
    seen_runtime_sources = set()
    for entry in msvc_runtime_binaries:
        source_path = normalized_source_path(entry)
        if source_path and source_path not in seen_runtime_sources:
            deduped_runtime_binaries.append(entry)
            seen_runtime_sources.add(source_path)
    msvc_runtime_binaries = deduped_runtime_binaries
    binaries += msvc_runtime_binaries
    allowed_runtime_sources = {
        normalized_source_path(entry) for entry in msvc_runtime_binaries if normalized_source_path(entry)
    }
    # Prevent bundling VC runtime DLLs from package-local folders (notably wx),
    # which can conflict with the system redist and crash at startup.
    _blocked_runtime_dlls = {"msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"}
    binaries = filter_blocked_runtime_dlls(binaries, _blocked_runtime_dlls, allowed_runtime_sources)

if BUILD_PROFILE == "complete":
    hiddenimports += collect_submodules("pyannote")
    hiddenimports += collect_submodules("speechbrain")
    hiddenimports += collect_submodules("huggingface_hub")
    hiddenimports += collect_submodules("pytorch_lightning")
    hiddenimports += collect_submodules("torchaudio")
    hiddenimports += [
        "pyannote.audio.pipelines.speaker_diarization",
        "pyannote.audio.models.segmentation",
        "pyannote.audio.models.segmentation.debug",
        "pyannote.audio.tasks.segmentation",
        "asteroid_filterbanks",
        "omegaconf",
        "torchmetrics",
    ]
    datas += copy_metadata("pyannote.audio")
    datas += copy_metadata("pyannote.core")
    datas += copy_metadata("huggingface_hub")
    datas += copy_metadata("speechbrain")
    datas += copy_metadata("torch")
    datas += copy_metadata("pytorch_lightning")
    # lightning_fabric is a submodule; PyPI metadata is under distribution name "lightning".
    datas += copy_metadata("lightning")
    datas += copy_metadata("transformers")
    datas += copy_metadata("torchmetrics")
    datas += copy_metadata("torchaudio")
    datas += copy_metadata("torchvision")
    datas += collect_data_files("pyannote")
    datas += collect_data_files("speechbrain")
    datas += collect_data_files("huggingface_hub")
    datas += collect_data_files("pytorch_lightning")
    datas += collect_data_files("torchaudio")

a = Analysis(
    [os.path.join(SRC, "app.py")],
    pathex=[ROOT, SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[HOOKS],
    hooksconfig={
        "matplotlib": {
            "backends": ["Agg"],
        },
    },
    runtime_hooks=[os.path.join(SRC, "runtime_hook_dlls.py")],
    excludes=[],
    noarchive=False,
)
# Also filter VC runtime DLLs from auto-collected binaries (e.g., wx package
# entries) to avoid startup crashes from bundled conflicting runtimes.
if IS_WINDOWS:
    _blocked_runtime_dlls = {"msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"}
    a.binaries = filter_blocked_runtime_dlls(a.binaries, _blocked_runtime_dlls, allowed_runtime_sources)
    a.datas = filter_blocked_runtime_dlls(a.datas, _blocked_runtime_dlls, allowed_runtime_sources)

yolov5_general_spec = find_spec("yolov5.utils.general")
if yolov5_general_spec is not None and yolov5_general_spec.origin:
    a.datas.append(("yolov5/utils/general.pyc", yolov5_general_spec.origin, "DATA"))

pyz = PYZ(a.pure)

if IS_MACOS:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        icon=ICON_ICO if os.path.exists(ICON_ICO) else None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=APP_NAME,
    )
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=ICON_ICNS if os.path.exists(ICON_ICNS) else None,
        bundle_identifier="edu.colby.multisocial",
        info_plist={
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        # UPX corrupts the torch/torchvision native DLLs on Windows, which
        # crashes the packaged app with a segfault when the ML stack is
        # imported. Leave binaries uncompressed there.
        upx=not IS_WINDOWS,
        console=False,
        icon=ICON_ICO if os.path.exists(ICON_ICO) else None,
        contents_directory='.',
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=not IS_WINDOWS,
        upx_exclude=[],
        name=APP_NAME,
    )
