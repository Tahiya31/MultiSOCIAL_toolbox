"""PyInstaller entry point for the existing in-process macOS application."""

import os

from analysis_backend import configure_backend
from native_backend import NativeAnalysisBackend

configure_backend(NativeAnalysisBackend())

if os.environ.get("MULTISOCIAL_MACOS_PACKAGED_E2E") == "1":
    from macos_packaged_e2e import main  # noqa: E402
else:
    from app import main  # noqa: E402


if __name__ == "__main__":
    main()
