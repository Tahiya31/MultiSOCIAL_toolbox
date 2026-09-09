"""PyInstaller entry point for the native-free Windows GUI."""

from analysis_backend import configure_backend
from worker_backend import WindowsWorkerBackend

configure_backend(WindowsWorkerBackend())

from app import main  # noqa: E402


if __name__ == "__main__":
    main()
