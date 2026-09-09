"""Small file-output transaction used by native analysis operations."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from pathlib import Path


STALE_OUTPUT_STAGING_SECONDS = 24 * 60 * 60


def discard_stale_output_staging(
    destination: str | os.PathLike[str],
    *,
    older_than_seconds: int = STALE_OUTPUT_STAGING_SECONDS,
) -> None:
    """Remove abandoned output transactions without touching active work.

    A hard process kill can leave a private staging directory behind. Directories
    younger than a day are deliberately retained: another process could still be
    writing them. Final user-visible outputs never match this private prefix.
    """
    output_dir = Path(destination)
    if not output_dir.is_dir():
        return
    cutoff = time.time() - older_than_seconds
    for candidate in output_dir.glob(".multisocial-stage-*"):
        try:
            if candidate.is_dir() and candidate.stat().st_mtime < cutoff:
                shutil.rmtree(candidate, ignore_errors=True)
        except OSError:
            # A concurrently removed or protected abandoned directory should
            # never prevent a user operation from starting.
            continue


class OutputStaging:
    """Stage complete output files beside their destination, then promote them.

    A cancellation or failed operation removes the private staging directory.
    Promotion happens only after all required files have been written, so users
    never see a truncated CSV, transcript sidecar, or video at its final name.
    Each individual final-file replacement is atomic. A multi-file result is
    promoted file-by-file, so it is not a cross-file filesystem transaction.
    """

    def __init__(self, destination: str | os.PathLike[str]):
        self.destination = Path(destination)
        self.path: Path | None = None

    def __enter__(self) -> Path:
        self.destination.mkdir(parents=True, exist_ok=True)
        discard_stale_output_staging(self.destination)
        self.path = Path(
            tempfile.mkdtemp(prefix=".multisocial-stage-", dir=self.destination)
        )
        return self.path

    def commit(self) -> None:
        if self.path is None:
            raise RuntimeError("Output staging has not been opened")
        for staged_path in self.path.iterdir():
            os.replace(staged_path, self.destination / staged_path.name)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.path is not None:
            # commit() moves staged files out individually. Remove any unexpected
            # residue without turning a completed user operation into an error.
            shutil.rmtree(self.path, ignore_errors=True)
