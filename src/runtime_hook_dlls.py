"""
Runtime hook to make bundled native libraries visible on Windows.
"""

from __future__ import annotations

import os
import sys

_DLL_DIR_HANDLES = []


def _iter_candidate_dirs(root: str):
    yielded = set()
    for current_root, _dirs, files in os.walk(root):
        if any(name.lower().endswith((".dll", ".pyd")) for name in files):
            if current_root not in yielded:
                yielded.add(current_root)
                yield current_root


if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root and os.path.isdir(bundle_root):
        search_dirs = []
        for directory in _iter_candidate_dirs(bundle_root):
            try:
                handle = os.add_dll_directory(directory)
                _DLL_DIR_HANDLES.append(handle)
                search_dirs.append(directory)
            except OSError:
                continue
        if search_dirs:
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = os.pathsep.join(search_dirs + [current_path])
