"""Metadata-only hook; dynamic diarization imports live in the checked-in manifest."""

from PyInstaller.utils.hooks import collect_data_files, copy_metadata

datas = collect_data_files("pyannote.audio")
datas += copy_metadata("pyannote.audio")
module_collection_mode = "pyz+py"
