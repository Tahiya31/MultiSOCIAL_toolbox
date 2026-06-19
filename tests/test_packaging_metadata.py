from __future__ import annotations

from pathlib import Path


def test_captions_module_is_packaged():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")

    assert '"captions",' in text
