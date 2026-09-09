# -*- mode: python ; coding: utf-8 -*-
"""macOS-only in-process application spec.

Windows is intentionally built by packaging/windows_gui.spec and
packaging/windows_worker.spec in independent Python processes.
"""

import os
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
)


if sys.platform == "win32":
    raise RuntimeError(
        "MultiSOCIAL.spec is macOS-only. Build the Windows GUI and worker with "
        "their independent specs, then assemble the two onedir outputs."
    )

ROOT = Path(globals().get("SPECPATH", os.getcwd())).resolve()
SRC = ROOT / "src"
HOOKS = ROOT / "hooks"
profile = os.environ.get("MULTISOCIAL_BUILD_PROFILE", "standard").strip().lower()
APP_NAME = "MultiSOCIAL-Complete" if profile == "complete" else "MultiSOCIAL-Standard"

pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
if match is None:
    raise RuntimeError("Could not determine application version")
APP_VERSION = match.group(1)

hiddenimports = [
    "audio",
    "backports",
    "backports.tarfile",
    "imageio_ffmpeg",
    "macos_packaged_e2e",
    "mediapipe.python._framework_bindings",
    "mediapipe.python.solutions",
    "mediapipe.python.solutions.drawing_utils",
    "mediapipe.python.solutions.pose",
    "mediapipe.python.solutions.pose_connections",
    "opensmile",
    "pkg_resources",
    "pose",
    "soundfile",
    "transformers",
    "wx.adv",
    "wx.lib.stattext",
]
hiddenimports += collect_submodules("ultralytics")

datas = [
    (str(ROOT / "assets"), "assets"),
    (
        str(ROOT / "assets" / "pose_landmark_heavy.tflite"),
        os.path.join("mediapipe", "modules", "pose_landmark"),
    ),
    (str(ROOT / "env.example"), "."),
    (str(ROOT / "pyproject.toml"), "."),
]
for package in (
    "audinterface",
    "audresample",
    "imageio_ffmpeg",
    "lightning_fabric",
    "mediapipe",
    "opensmile",
    "ultralytics",
    "yolov5",
):
    datas += collect_data_files(package)
for package in ("torch", "torchvision", "transformers", "ultralytics", "yolov5"):
    try:
        datas += copy_metadata(package)
    except Exception:
        pass

binaries = []
for package in ("audinterface", "audresample", "mediapipe", "opensmile", "soundfile"):
    binaries += collect_dynamic_libs(package)

if profile == "complete":
    for package in ("huggingface_hub", "pyannote", "pytorch_lightning", "speechbrain", "torchaudio"):
        hiddenimports += collect_submodules(package)
        datas += collect_data_files(package)
    hiddenimports += [
        "asteroid_filterbanks",
        "omegaconf",
        "pyannote.audio.models.segmentation",
        "pyannote.audio.models.segmentation.debug",
        "pyannote.audio.pipelines.speaker_diarization",
        "pyannote.audio.tasks.segmentation",
        "torchmetrics",
    ]
    for package in (
        "huggingface_hub",
        "lightning",
        "pyannote.audio",
        "pyannote.core",
        "pytorch_lightning",
        "speechbrain",
        "torch",
        "torchaudio",
        "torchmetrics",
        "torchvision",
        "transformers",
    ):
        try:
            datas += copy_metadata(package)
        except Exception:
            pass

a = Analysis(
    [str(SRC / "app_macos.py")],
    pathex=[str(ROOT), str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(HOOKS)],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    runtime_hooks=[str(SRC / "runtime_hook_dlls.py")],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)
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
    icon=str(ROOT / "assets" / "MultiSOCIAL_logo.ico"),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    name=APP_NAME,
)
app = BUNDLE(
    coll,
    name=f"{APP_NAME}.app",
    icon=str(ROOT / "assets" / "MultiSOCIAL_logo.icns"),
    bundle_identifier="edu.colby.multisocial",
    info_plist={
        "CFBundleShortVersionString": APP_VERSION,
        "CFBundleVersion": APP_VERSION,
    },
)
