# -*- mode: python ; coding: utf-8 -*-

import os
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files


SPEC_DIR = Path(globals().get("SPECPATH", Path.cwd() / "packaging")).resolve()
ROOT = SPEC_DIR.parent if SPEC_DIR.name == "packaging" else SPEC_DIR
SRC = ROOT / "src"
HOOKS = ROOT / "hooks"
sys.path.insert(0, str(ROOT / "packaging"))

from windows_build_support import (  # noqa: E402
    GUI_FORBIDDEN_MODULE_ROOTS,
    approved_vc_sources,
    assert_gui_graph,
    collect_vc_runtime_binaries,
    filter_vc_runtime_entries,
)

profile = os.environ.get("MULTISOCIAL_BUILD_PROFILE", "standard").strip().lower()
APP_NAME = "MultiSOCIAL-Complete" if profile == "complete" else "MultiSOCIAL-Standard"
ICON = ROOT / "assets" / "MultiSOCIAL_logo.ico"

vc_binaries = collect_vc_runtime_binaries()
approved_sources = approved_vc_sources(vc_binaries)
datas = [
    (str(ROOT / "assets" / "MultiSOCIAL_logo.png"), "assets"),
    (str(ROOT / "env.example"), "."),
    (str(ROOT / "pyproject.toml"), "."),
]
datas += collect_data_files("imageio_ffmpeg")

a = Analysis(
    [str(SRC / "app_windows.py")],
    pathex=[str(ROOT), str(SRC)],
    binaries=vc_binaries,
    datas=datas,
    hiddenimports=[
        "backports",
        "backports.tarfile",
        "imageio_ffmpeg",
        "pkg_resources",
        "wx.adv",
        "wx.lib.stattext",
    ],
    hookspath=[str(HOOKS)],
    runtime_hooks=[],
    excludes=sorted(GUI_FORBIDDEN_MODULE_ROOTS),
    noarchive=False,
)
a.binaries = filter_vc_runtime_entries(a.binaries, approved_sources)
a.datas = filter_vc_runtime_entries(a.datas, approved_sources)
assert_gui_graph(a)

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
    upx=False,
    console=False,
    icon=str(ICON) if ICON.exists() else None,
    contents_directory=".",
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name=APP_NAME,
)
