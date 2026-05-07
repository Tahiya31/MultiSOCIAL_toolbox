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


# --- SPEECHBRAIN OS.LISTDIR PATCH ---
# PyInstaller compiles python files into a PYZ archive.
# speechbrain uses os.listdir(os.path.dirname(__file__)) to find its submodules
# in dataio, nnet, and utils. This fails if the directory doesn't exist on disk.
# We patch os.listdir to return the correct files.
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
                "superpowers.py", "text_to_sequence.py", "torch_audio_backend.py", "train_logger.py"
            ]
        elif "speechbrain/dataio" in str_path:
            return [
                "__init__.py", "batch.py", "dataio.py", "dataloader.py", "dataset.py",
                "encoder.py", "iterators.py", "legacy.py", "preprocess.py", "sampler.py", "wer.py"
            ]
        elif "speechbrain/nnet" in str_path:
            return [
                "CNN.py", "RNN.py", "__init__.py", "activations.py", "attention.py",
                "autoencoders.py", "containers.py", "diffusion.py", "dropout.py",
                "embedding.py", "linear.py", "losses.py", "normalization.py",
                "pooling.py", "quantisers.py", "schedulers.py", "unet.py", "utils.py"
            ]
        raise

os.listdir = _patched_listdir
