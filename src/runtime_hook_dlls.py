"""Frozen-runtime compatibility helpers for native analysis dependencies."""

from __future__ import annotations

import os
import sys

# Optional-model discovery must never mutate an installed Windows worker.
if sys.platform == "win32":
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLOv5_AUTOINSTALL"] = "false"

_DLL_DIR_HANDLES = []

# Keep DLL lookup deterministic without changing PATH. Windows workers retain
# these loader handles for their lifetime; macOS also uses this hook for the
# packaged SpeechBrain compatibility shim below.
_BASE_DLL_DIRECTORY_RELATIVE_PATHS = (
    ".",
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

_TENSOR_DLL_DIRECTORY_RELATIVE_PATHS = (
    "torch/lib",
    "torchvision",
    "torchaudio",
    "torchaudio/lib",
)
_DLL_DIRECTORY_RELATIVE_PATHS = (*_BASE_DLL_DIRECTORY_RELATIVE_PATHS, *_TENSOR_DLL_DIRECTORY_RELATIVE_PATHS)
_REGISTERED_DLL_DIRECTORIES = set()


def bundled_dll_directories(bundle_root: str, *, include_tensor_runtime: bool = True) -> list[str]:
    """Return known bundle-native directories in their required import phase."""
    directories = []
    seen = set()
    relative_paths = _BASE_DLL_DIRECTORY_RELATIVE_PATHS
    if include_tensor_runtime:
        relative_paths = (*relative_paths, *_TENSOR_DLL_DIRECTORY_RELATIVE_PATHS)
    for relative_path in relative_paths:
        for prefix in ("", os.path.join("Lib", "site-packages")):
            directory = os.path.normcase(
                os.path.abspath(os.path.join(bundle_root, prefix, relative_path))
            )
            if directory not in seen and os.path.isdir(directory):
                directories.append(directory)
                seen.add(directory)
    return directories


def configure_windows_dll_search_path(bundle_root: str, *, include_tensor_runtime: bool = True) -> list[str]:
    """Add known directories; defer Torch until MediaPipe has initialized."""
    directories = bundled_dll_directories(bundle_root, include_tensor_runtime=include_tensor_runtime)
    for directory in directories:
        if directory in _REGISTERED_DLL_DIRECTORIES:
            continue
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(directory))
            _REGISTERED_DLL_DIRECTORIES.add(directory)
        except OSError:
            pass
    return directories



# SpeechBrain uses os.listdir(os.path.dirname(__file__)) to discover modules
# that PyInstaller keeps in the PYZ archive rather than on disk.
_original_listdir = os.listdir

_WINDOWS_SPEECHBRAIN_FILES = {
    "utils": [
        "__init__.py",
        "callchains.py",
        "checkpoints.py",
        "data_pipeline.py",
        "data_utils.py",
        "depgraph.py",
        "distributed.py",
        "metric_stats.py",
        "parameter_transfer.py",
        "superpowers.py",
        "text_to_sequence.py",
        "torch_audio_backend.py",
    ],
    "dataio": [
        "__init__.py",
        "batch.py",
        "dataio.py",
        "dataloader.py",
        "dataset.py",
        "encoder.py",
        "preprocess.py",
    ],
    "nnet": [
        "CNN.py",
        "RNN.py",
        "__init__.py",
        "activations.py",
        "attention.py",
        "containers.py",
        "embedding.py",
        "linear.py",
        "losses.py",
        "normalization.py",
        "pooling.py",
        "schedulers.py",
    ],
}
_NON_WINDOWS_SPEECHBRAIN_FILES = {
    "utils": [
        "Accuracy.py", "DER.py", "EDER.py", "__init__.py", "_workarounds.py",
        "bleu.py", "callchains.py", "checkpoints.py", "data_pipeline.py",
        "data_utils.py", "depgraph.py", "distributed.py", "edit_distance.py",
        "epoch_loop.py", "hparams.py", "hpopt.py", "logger.py", "metric_stats.py",
        "optimizers.py", "parallel.py", "parameter_transfer.py", "profiling.py",
        "superpowers.py", "text_to_sequence.py", "torch_audio_backend.py", "train_logger.py",
    ],
    "dataio": [
        "__init__.py", "batch.py", "dataio.py", "dataloader.py", "dataset.py",
        "encoder.py", "iterators.py", "legacy.py", "preprocess.py", "sampler.py", "wer.py",
    ],
    "nnet": [
        "CNN.py", "RNN.py", "__init__.py", "activations.py", "attention.py",
        "autoencoders.py", "containers.py", "diffusion.py", "dropout.py",
        "embedding.py", "linear.py", "losses.py", "normalization.py",
        "pooling.py", "quantisers.py", "schedulers.py", "unet.py", "utils.py",
    ],
}


def _patched_listdir(path=".", *args, **kwargs):
    """Use the original listdir except for missing SpeechBrain package paths."""
    str_path = str(path).replace("\\", "/")
    try:
        return _original_listdir(path, *args, **kwargs)
    except FileNotFoundError:
        if args or kwargs:
            raise
        files = (
            _WINDOWS_SPEECHBRAIN_FILES
            if sys.platform == "win32"
            else _NON_WINDOWS_SPEECHBRAIN_FILES
        )
        if "speechbrain/utils" in str_path:
            return list(files["utils"])
        if "speechbrain/dataio" in str_path:
            return list(files["dataio"])
        if "speechbrain/nnet" in str_path:
            return list(files["nnet"])
        raise


class _SpeechBrainListdirProxy:
    """Callable wrapper that cannot become a bound method on ``pathlib``."""

    def __call__(self, path=".", *args, **kwargs):
        return _patched_listdir(path, *args, **kwargs)


# ``pathlib`` copies ``os.listdir`` onto an accessor class. A plain Python
# function would bind ``self`` when that class invokes it, unlike the original
# built-in function. A callable instance keeps the original one-argument ABI.
os.listdir = _SpeechBrainListdirProxy()
