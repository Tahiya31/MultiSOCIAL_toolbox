"""Source launcher that selects the platform backend outside packaged graphs."""

import sys

from analysis_backend import configure_backend

if sys.platform == "win32":
    from worker_backend import WindowsWorkerBackend

    configure_backend(WindowsWorkerBackend())
else:
    from native_backend import NativeAnalysisBackend

    configure_backend(NativeAnalysisBackend())

from app import main  # noqa: E402


if __name__ == "__main__":
    main()
