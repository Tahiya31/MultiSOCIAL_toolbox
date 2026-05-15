from __future__ import annotations

import os
import types


def test_normalize_path_strips_quotes_and_resolves_home(monkeypatch, import_gui_utils):
    gui_utils = import_gui_utils
    home = os.path.expanduser("~")
    path = gui_utils.normalize_path('"~/example/../folder"')

    assert path == os.path.normpath(os.path.join(home, "folder"))


def test_resolved_dataset_root_handles_converted_audio(import_gui_utils, tmp_path):
    gui_utils = import_gui_utils
    dataset = tmp_path / "dataset"
    converted = dataset / "converted_audio"
    converted.mkdir(parents=True)

    assert gui_utils.resolved_dataset_root(str(converted)) == str(dataset)


def test_get_files_from_folder_filters_dotfiles_and_sorts(import_gui_utils, tmp_path):
    gui_utils = import_gui_utils
    (tmp_path / "b.wav").write_text("", encoding="utf-8")
    (tmp_path / "a.wav").write_text("", encoding="utf-8")
    (tmp_path / ".hidden.wav").write_text("", encoding="utf-8")
    (tmp_path / "note.txt").write_text("", encoding="utf-8")

    files = gui_utils.get_files_from_folder(str(tmp_path), (".wav",))

    assert files == [str(tmp_path / "a.wav"), str(tmp_path / "b.wav")]


def test_get_audio_files_for_processing_deduplicates_root_and_converted(import_gui_utils, tmp_path):
    gui_utils = import_gui_utils
    dataset = tmp_path / "dataset"
    converted = dataset / "converted_audio"
    dataset.mkdir()
    converted.mkdir()
    (dataset / "root.wav").write_text("", encoding="utf-8")
    (converted / "root.wav").write_text("", encoding="utf-8")
    (converted / "extra.flac").write_text("", encoding="utf-8")

    files = gui_utils.get_audio_files_for_processing(str(dataset), (".wav", ".flac"))

    assert files == sorted(
        [
            str(converted / "extra.flac"),
            str(converted / "root.wav"),
            str(dataset / "root.wav"),
        ]
    )


def test_get_ffmpeg_executable_prefers_cached_value(monkeypatch, import_gui_utils, tmp_path):
    gui_utils = import_gui_utils
    exe = tmp_path / "ffmpeg"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setenv("MULTISOCIAL_FFMPEG_EXE", str(exe))
    monkeypatch.setattr(gui_utils, "_register_win32_ffmpeg_dll_directory", lambda path: None)

    assert gui_utils.get_ffmpeg_executable() == str(exe)


def test_get_ffmpeg_executable_uses_system_binary(monkeypatch, import_gui_utils, tmp_path):
    gui_utils = import_gui_utils
    exe = tmp_path / "ffmpeg"
    exe.write_text("", encoding="utf-8")
    monkeypatch.delenv("MULTISOCIAL_FFMPEG_EXE", raising=False)
    monkeypatch.setattr(gui_utils.shutil, "which", lambda name: str(exe))
    monkeypatch.setattr(gui_utils, "_register_win32_ffmpeg_dll_directory", lambda path: None)

    resolved = gui_utils.get_ffmpeg_executable()

    assert resolved == str(exe)
    assert os.environ["MULTISOCIAL_FFMPEG_SOURCE"] == "system"
    assert os.environ["MULTISOCIAL_FFMPEG_EXE"] == str(exe)


def test_get_ffmpeg_executable_falls_back_to_imageio(monkeypatch, import_gui_utils, tmp_path):
    gui_utils = import_gui_utils
    exe = tmp_path / "ffmpeg-imageio"
    exe.write_text("", encoding="utf-8")
    monkeypatch.delenv("MULTISOCIAL_FFMPEG_EXE", raising=False)
    monkeypatch.setattr(gui_utils.shutil, "which", lambda name: None)
    monkeypatch.setattr(gui_utils, "_bundled_ffmpeg_candidates", lambda: [])
    monkeypatch.setattr(gui_utils, "_register_win32_ffmpeg_dll_directory", lambda path: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "imageio_ffmpeg",
        types.SimpleNamespace(get_ffmpeg_exe=lambda: str(exe)),
    )

    assert gui_utils.get_ffmpeg_executable() == str(exe)
    assert os.environ["MULTISOCIAL_FFMPEG_SOURCE"] == "bundled"


def test_ensure_ffmpeg_available_marks_missing(monkeypatch, import_gui_utils):
    gui_utils = import_gui_utils
    monkeypatch.setattr(gui_utils, "get_ffmpeg_executable", lambda: None)

    assert gui_utils.ensure_ffmpeg_available() is False
    assert os.environ["MULTISOCIAL_FFMPEG_SOURCE"] == "missing"
