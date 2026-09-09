"""Narrow Transformers hook for the Whisper-only Windows worker graph."""

from PyInstaller.utils.hooks import copy_metadata

datas = []
for distribution in (
    "filelock",
    "huggingface-hub",
    "numpy",
    "packaging",
    "pyyaml",
    "regex",
    "requests",
    "safetensors",
    "tokenizers",
    "tqdm",
    "transformers",
):
    try:
        datas += copy_metadata(distribution)
    except Exception:
        pass
module_collection_mode = "pyz+py"
excludedimports = [
    "transformers.integrations.awq",
    "transformers.integrations.bitsandbytes",
    "transformers.models.gptq",
]
