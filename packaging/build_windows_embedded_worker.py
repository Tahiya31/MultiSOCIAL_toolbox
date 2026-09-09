"""Build the private, non-PyInstaller Windows analysis-worker runtime.

The GUI remains a normal PyInstaller application.  The worker deliberately
uses the CPython runtime installed by ``actions/setup-python`` plus the locked
worker venv's site-packages, so it cannot share PyInstaller bootloader state
with the windowed GUI process.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from windows_build_support import collect_vc_runtime_binaries

ROOT = Path(__file__).resolve().parents[1]

# The worker is an application runtime, not a build environment.  These two
# distributions are needed only to create the GUI bundle; copying their hook
# source into the worker both wastes space and makes package-name based audits
# report optional hook names (for example ``hook-torchaudio.py``) as runtimes.
_BUILD_ONLY_WORKER_DIRECTORIES = {"pyinstaller", "_pyinstaller_hooks_contrib"}
_BUILD_ONLY_WORKER_METADATA_PREFIXES = (
    "pyinstaller-",
    "pyinstaller_hooks_contrib-",
)


def _base_prefix(python: Path) -> Path:
    return Path(
        subprocess.check_output(
            [str(python), "-c", "import sys; print(sys.base_prefix)"],
            text=True,
            encoding="utf-8",
        ).strip()
    ).resolve()


def _copy_required_runtime(base: Path, destination: Path) -> None:
    for name in ("python.exe", "pythonw.exe", "python310.dll", "python3.dll"):
        source = base / name
        if source.is_file():
            shutil.copy2(source, destination / name)
    for source in base.glob("*runtime*.dll"):
        shutil.copy2(source, destination / source.name)
    for source in base.glob("msvcp*.dll"):
        shutil.copy2(source, destination / source.name)
    for name in ("DLLs", "Lib"):
        source = base / name
        if not source.is_dir():
            raise RuntimeError(f"Python runtime is missing {name}: {source}")
        shutil.copytree(source, destination / name, dirs_exist_ok=True)


def _ignore_build_only_worker_files(_directory: str, names: list[str]) -> set[str]:
    """Exclude PyInstaller's build tooling from the shipped worker runtime."""

    ignored = set()
    for name in names:
        normalized = name.casefold()
        if normalized in _BUILD_ONLY_WORKER_DIRECTORIES or normalized.startswith(
            _BUILD_ONLY_WORKER_METADATA_PREFIXES
        ):
            ignored.add(name)
    return ignored


def build(python: Path, output: Path) -> None:
    python = python.resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Worker venv Python is missing: {python}")
    venv = python.parents[1]
    site_packages = venv / "Lib" / "site-packages"
    if not site_packages.is_dir():
        raise RuntimeError(f"Worker venv site-packages is missing: {site_packages}")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    _copy_required_runtime(_base_prefix(python), output)
    # The embeddable interpreter intentionally does not promise every Visual
    # C++ runtime DLL required by the locked native wheels. Copy the curated,
    # complete x64 set used by the GUI and validate it at package assembly.
    for source, _destination in collect_vc_runtime_binaries():
        shutil.copy2(source, output / Path(source).name)
    shutil.copytree(
        site_packages,
        output / "Lib" / "site-packages",
        dirs_exist_ok=True,
        ignore=_ignore_build_only_worker_files,
    )
    shutil.copytree(ROOT / "src", output / "app")
    shutil.copytree(ROOT / "assets", output / "assets")
    (output / "python310._pth").write_text(
        "python310.zip\n.\nDLLs\nLib\nLib/site-packages\napp\nimport site\n",
        encoding="utf-8",
    )
    if not (output / "python.exe").is_file() or not (output / "app" / "analysis_worker.py").is_file():
        raise RuntimeError("Embedded worker runtime was not assembled")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.python, args.output)


if __name__ == "__main__":
    main()
