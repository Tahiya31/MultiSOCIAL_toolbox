from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

module_collection_mode = "pyz+py"

hiddenimports = collect_submodules("yolov5")
hiddenimports += collect_submodules("ultralytics")

datas = collect_data_files("yolov5")
datas += collect_data_files("ultralytics")

for package in ("yolov5", "ultralytics", "torch", "torchvision"):
    try:
        datas += copy_metadata(package)
    except Exception:
        pass
