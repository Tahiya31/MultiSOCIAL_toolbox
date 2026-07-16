"""Runtime hook for the frozen application's Windows-native dependencies."""

from __future__ import annotations

import os
import sys


_DLL_DIR_HANDLES = []

# Keep DLL lookup deterministic.  Adding every directory containing a DLL to
# PATH lets unrelated packages win filename collisions during native imports.
_DLL_DIRECTORY_RELATIVE_PATHS = (
    ".",
    "torch/lib",
    "torchvision",
    "torchaudio",
    "torchaudio/lib",
    "mediapipe",
    "mediapipe/python",
    "opensmile/core/bin/win_amd64",
    "audresample/core/bin/win_amd64",
    "numpy.libs",
    "scipy.libs",
    "pandas.libs",
    "pyarrow.libs",
    "cv2",
)


def bundled_dll_directories(bundle_root: str) -> list[str]:
    """Return existing native-library directories in their required order."""
    directories = []
    seen = set()
    for relative_path in _DLL_DIRECTORY_RELATIVE_PATHS:
        directory = os.path.normcase(os.path.abspath(os.path.join(bundle_root, relative_path)))
        if directory not in seen and os.path.isdir(directory):
            directories.append(directory)
            seen.add(directory)
    return directories


def configure_windows_dll_search_path(bundle_root: str) -> list[str]:
    """Add only the known bundle-native directories to the Windows loader."""
    directories = bundled_dll_directories(bundle_root)
    for directory in directories:
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(directory))
        except OSError:
            pass
    if directories:
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(directories + [current_path])
    return directories


def _preload_torch_before_gui() -> None:
    """Initialize Torch's native libraries before the app imports wx.

    Importing a Qt/wx GUI stack before Torch has made Torch's native
    initialization crash on Windows (pytorch/pytorch#166628), and the app
    imports ``wx`` at module load before any Torch use.  This runtime hook
    runs before ``app.py``, so importing Torch here guarantees the safe
    order regardless of undefined AddDllDirectory search order.  Scoped to
    the frozen Complete build, whose diarization stack is where the crash
    appears; the Standard build's startup is left unchanged.
    """
    import torch  # noqa: F401


if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _bundle_root = getattr(sys, "_MEIPASS", None)
    if _bundle_root and os.path.isdir(_bundle_root):
        configure_windows_dll_search_path(_bundle_root)
        if "complete" in os.path.basename(sys.executable).lower():
            _preload_torch_before_gui()


# SpeechBrain uses os.listdir(os.path.dirname(__file__)) to discover modules
# that PyInstaller keeps in the PYZ archive rather than on disk.
_original_listdir = os.listdir


def _patched_listdir(path="."):
    str_path = str(path).replace("\\", "/")
    try:
        return _original_listdir(path)
    except FileNotFoundError:
        if "speechbrain/utils" in str_path:
            return [
                "Accuracy.py", "DER.py", "EDER.py", "__init__.py", "_workarounds.py",
                "bleu.py", "callchains.py", "checkpoints.py", "data_pipeline.py",
                "data_utils.py", "depgraph.py", "distributed.py", "edit_distance.py",
                "epoch_loop.py", "hparams.py", "hpopt.py", "logger.py", "metric_stats.py",
                "optimizers.py", "parallel.py", "parameter_transfer.py", "profiling.py",
                "superpowers.py", "text_to_sequence.py", "torch_audio_backend.py", "train_logger.py",
            ]
        if "speechbrain/dataio" in str_path:
            return [
                "__init__.py", "batch.py", "dataio.py", "dataloader.py", "dataset.py",
                "encoder.py", "iterators.py", "legacy.py", "preprocess.py", "sampler.py", "wer.py",
            ]
        if "speechbrain/nnet" in str_path:
            return [
                "CNN.py", "RNN.py", "__init__.py", "activations.py", "attention.py",
                "autoencoders.py", "containers.py", "diffusion.py", "dropout.py",
                "embedding.py", "linear.py", "losses.py", "normalization.py",
                "pooling.py", "quantisers.py", "schedulers.py", "unet.py", "utils.py",
            ]
        raise


os.listdir = _patched_listdir
