"""Assemble two finished onedir applications without merging their TOCs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

def _assert_safe_output(gui: Path, worker: Path, output: Path) -> None:
    if output in {gui, worker}:
        raise ValueError("Assembly output must be distinct from both PyInstaller inputs")
    if output in gui.parents or output in worker.parents:
        raise ValueError("Assembly output must not contain either PyInstaller input")
    if gui in output.parents or worker in output.parents:
        raise ValueError("Assembly output must not be nested inside either PyInstaller input")


def _copy_contents(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def assemble(gui: Path, worker: Path, launcher: Path, output: Path) -> None:
    gui = gui.resolve()
    worker = worker.resolve()
    launcher = launcher.resolve()
    output = output.resolve()
    _assert_safe_output(gui, worker, output)
    if not gui.is_dir() or not worker.is_dir() or not launcher.is_file():
        raise FileNotFoundError("GUI, worker, and native launcher outputs are required")
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(gui, output)
    _copy_contents(worker, output / "worker")
    shutil.copy2(launcher, output / "worker" / "MultiSOCIAL-Worker-Launcher.exe")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.gui.resolve(), args.worker.resolve(), args.launcher.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
