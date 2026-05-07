# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from importlib.util import find_spec
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


ROOT = os.path.abspath(globals().get("SPECPATH", os.getcwd()))
SRC = os.path.join(ROOT, "src")
BUILD_PROFILE = os.environ.get("MULTISOCIAL_BUILD_PROFILE", "standard").strip().lower()
APP_NAME = "MultiSOCIAL-Complete" if BUILD_PROFILE == "complete" else "MultiSOCIAL-Standard"
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


def normalized_source_path(entry):
    if not (isinstance(entry, tuple) and len(entry) >= 1):
        return None
    return os.path.normcase(os.path.abspath(str(entry[0])))


def filter_blocked_runtime_dlls(entries, blocked_names, allowed_sources):
    filtered = []
    for entry in entries:
        source_path = normalized_source_path(entry)
        if source_path is None:
            filtered.append(entry)
            continue

        basename = os.path.basename(source_path).lower()
        if basename not in blocked_names or source_path in allowed_sources:
            filtered.append(entry)
    return filtered

hiddenimports = [
    "pkg_resources",
    "wx.adv",
    "wx.lib.stattext",
    "imageio_ffmpeg",
    "librosa",
    "mediapipe.python._framework_bindings",
    "mediapipe.python.solutions",
    "mediapipe.python.solutions.drawing_utils",
    "mediapipe.python.solutions.pose",
    "mediapipe.python.solutions.pose_connections",
    "opensmile",
    "transformers",
]

datas = [
    (os.path.join(ROOT, "assets"), "assets"),
    (os.path.join(ROOT, "env.example"), "."),
]
datas += collect_data_files("mediapipe")
datas += collect_data_files("audresample")
datas += collect_data_files("opensmile")
datas += collect_data_files("audinterface")
binaries = collect_dynamic_libs("mediapipe")
binaries += collect_dynamic_libs("audresample")
binaries += collect_dynamic_libs("opensmile")
binaries += collect_dynamic_libs("audinterface")

if IS_WINDOWS:
    msvc_runtime_binaries = collect_sidecar_binaries("msvc_runtime", ["*.dll"])
    binaries += msvc_runtime_binaries
    allowed_runtime_sources = {
        normalized_source_path(entry) for entry in msvc_runtime_binaries if normalized_source_path(entry)
    }
    # Prevent bundling VC runtime DLLs from package-local folders (notably wx),
    # which can conflict with the system redist and crash at startup.
    _blocked_runtime_dlls = {"msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"}
    binaries = filter_blocked_runtime_dlls(binaries, _blocked_runtime_dlls, allowed_runtime_sources)

if BUILD_PROFILE == "complete":
    hiddenimports += [
        "pyannote.audio",
        "pyannote.core",
        "pyannote.pipeline",
    ]


a = Analysis(
    [os.path.join(SRC, "app.py")],
    pathex=[ROOT, SRC],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
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
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
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
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=APP_NAME,
    )
