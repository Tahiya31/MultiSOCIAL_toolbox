from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _purge_modules(*names: str) -> None:
    for name in names:
        sys.modules.pop(name, None)


def _install_fake_wx() -> None:
    wx = types.ModuleType("wx")

    class _DummyBase:
        def __init__(self, *args, **kwargs):
            pass

    class _Colour:
        def __init__(self, *args, **kwargs):
            self._rgb = args[:3] if len(args) >= 3 else (0, 0, 0)

        def Red(self):
            return self._rgb[0]

        def Green(self):
            return self._rgb[1]

        def Blue(self):
            return self._rgb[2]

    class _Font:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    wx.Colour = _Colour
    wx.Font = _Font
    wx.StaticText = _DummyBase
    wx.AutoBufferedPaintDC = _DummyBase
    wx.GCDC = lambda dc: dc
    wx.Brush = _DummyBase
    wx.FONTFAMILY_SWISS = 0
    wx.FONTSTYLE_NORMAL = 0
    wx.FONTWEIGHT_NORMAL = 0
    wx.FONTWEIGHT_BOLD = 1
    wx.FONTFAMILY_DEFAULT = 0
    wx.BG_STYLE_PAINT = 0
    wx.SOUTH = 0
    wx.NORTH = 1
    wx.TRANSPARENT = 0
    wx.NullColour = object()

    wx_lib = types.ModuleType("wx.lib")
    wx_stattext = types.ModuleType("wx.lib.stattext")

    class _GenStaticText(_DummyBase):
        pass

    wx_stattext.GenStaticText = _GenStaticText
    wx_lib.stattext = wx_stattext
    wx.lib = wx_lib

    sys.modules["wx"] = wx
    sys.modules["wx.lib"] = wx_lib
    sys.modules["wx.lib.stattext"] = wx_stattext


def _install_fake_ffmpeg() -> None:
    ffmpeg = types.ModuleType("ffmpeg")

    class _Node:
        def output(self, *args, **kwargs):
            return self

        def overwrite_output(self):
            return self

        def run(self, *args, **kwargs):
            return None

    ffmpeg.input = lambda *args, **kwargs: _Node()
    sys.modules["ffmpeg"] = ffmpeg


def _install_fake_audio_deps() -> None:
    torch = types.ModuleType("torch")
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.mps = types.SimpleNamespace(empty_cache=lambda: None)
    torch.device = lambda value: value
    sys.modules["torch"] = torch

    opensmile = types.ModuleType("opensmile")
    opensmile.FeatureSet = types.SimpleNamespace(ComParE_2016="ComParE_2016")
    opensmile.FeatureLevel = types.SimpleNamespace(LowLevelDescriptors="LowLevelDescriptors")

    class _Smile:
        def __init__(self, *args, **kwargs):
            pass

        def process_signal(self, y, sr):
            return pd.DataFrame({"energy": [1.0, 2.0], "pitch": [3.0, 4.0]})

    opensmile.Smile = _Smile
    sys.modules["opensmile"] = opensmile

    transformers = types.ModuleType("transformers")

    class _DummyAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return types.SimpleNamespace(device=types.SimpleNamespace(type="cpu"), to=lambda *a, **k: None)

    class _DummyProcessor:
        tokenizer = object()
        feature_extractor = object()

        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _DummyProcessor()

    transformers.AutoModelForSpeechSeq2Seq = _DummyAutoModel
    transformers.AutoProcessor = _DummyProcessor
    transformers.pipeline = lambda *args, **kwargs: lambda *a, **k: {"text": "stub", "chunks": []}
    sys.modules["transformers"] = transformers


@pytest.fixture(autouse=True)
def add_src_to_path(monkeypatch):
    monkeypatch.syspath_prepend(str(SRC_ROOT))


@pytest.fixture
def import_runtime_services():
    _purge_modules("runtime_services")
    return importlib.import_module("runtime_services")


@pytest.fixture
def import_gui_utils():
    _purge_modules("gui_utils", "wx", "wx.lib", "wx.lib.stattext", "ffmpeg")
    _install_fake_wx()
    _install_fake_ffmpeg()
    return importlib.import_module("gui_utils")


@pytest.fixture
def import_audio():
    _purge_modules("audio", "torch", "opensmile", "transformers")
    _install_fake_audio_deps()
    return importlib.import_module("audio")


@pytest.fixture
def validate_complete_bundle_layout_module():
    module_name = "validate_complete_bundle_layout"
    _purge_modules(module_name)
    spec = importlib.util.spec_from_file_location(
        module_name,
        REPO_ROOT / ".github" / "scripts" / "validate_complete_bundle_layout.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module
