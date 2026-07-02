from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

module_collection_mode = "pyz+py"

hiddenimports = collect_submodules("yolov5")
datas = collect_data_files("yolov5")

try:
    datas += copy_metadata("yolov5")
except Exception:
    pass
