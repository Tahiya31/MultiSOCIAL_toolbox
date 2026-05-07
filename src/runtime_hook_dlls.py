"""
Runtime hook to make bundled native libraries visible on Windows.
"""

from __future__ import annotations

import os
import sys


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
        for directory in _iter_candidate_dirs(bundle_root):
            try:
                os.add_dll_directory(directory)
            except OSError:
                continue
