# -*- mode: python ; coding: utf-8 -*-

import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs


ROOT = os.path.abspath(globals().get("SPECPATH", os.getcwd()))
SRC = os.path.join(ROOT, "src")
BUILD_PROFILE = os.environ.get("MULTISOCIAL_BUILD_PROFILE", "standard").strip().lower()
APP_NAME = "MultiSOCIAL-Complete" if BUILD_PROFILE == "complete" else "MultiSOCIAL-Standard"
IS_MACOS = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"
ICON_ICO = os.path.join(ROOT, "assets", "MultiSOCIAL_logo.ico")
ICON_ICNS = os.path.join(ROOT, "assets", "MultiSOCIAL_logo.icns")

hiddenimports = [
    "wx.adv",
    "wx.lib.stattext",
    "imageio_ffmpeg",
    "librosa",
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
binaries = collect_dynamic_libs("mediapipe")

if IS_WINDOWS:
    try:
        binaries += collect_dynamic_libs("msvc_runtime")
    except Exception:
        pass

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
