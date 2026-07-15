"""Headless regression coverage for custom wx event-flow behavior.

The test environment intentionally uses a minimal wx shim, so these tests
inspect the small widget methods that own paint scheduling and timer lifetime.
"""

from __future__ import annotations

import ast
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def _class_source(path: Path, class_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    return ast.get_source_segment(source, node) or ""


def _method_source(class_source: str, method_name: str) -> str:
    source = textwrap.dedent(class_source)
    tree = ast.parse(source)
    class_node = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    node = next(node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    return ast.get_source_segment(source, node) or ""


def test_checkbox_hover_enter_requests_immediate_repaint():
    checkbox = _class_source(ROOT / "src" / "ui_components.py", "CustomCheckBox")
    assert "self.Refresh()" in _method_source(checkbox, "_on_enter")


def test_logo_cache_retains_normal_and_hover_diameters():
    logo = _class_source(ROOT / "src" / "ui_components.py", "ElevatedLogoPanel")
    create_bitmap = _method_source(logo, "_create_circular_bitmap")

    assert "self._circular_bitmap_cache = {}" in logo
    assert "self._circular_bitmap_cache.get(diameter)" in create_bitmap
    assert "self._circular_bitmap_cache[diameter] = circ_bmp" in create_bitmap
    assert "_cached_diameter" not in logo


def test_gauge_ignores_unchanged_values_and_has_no_unused_pulse_api():
    gauge = _class_source(ROOT / "src" / "ui_components.py", "CustomGauge")
    set_value = _method_source(gauge, "SetValue")

    assert "if clamped_value == self._value:" in set_value
    assert "return" in set_value
    assert "self.Update()" not in set_value
    assert "def Pulse" not in gauge


def test_info_icon_constructs_and_binds_one_reusable_hide_timer():
    icon = _class_source(ROOT / "src" / "ui_components.py", "InfoIcon")
    init = _method_source(icon, "__init__")
    schedule_hide = _method_source(icon, "_schedule_hide")

    assert init.count("wx.Timer(self)") == 1
    assert "self.Bind(wx.EVT_TIMER, self._on_hide_timer, self._hide_timer)" in init
    assert "wx.Timer(" not in schedule_hide
    assert "self._cancel_hide()" in _method_source(icon, "_on_destroy")


def test_status_updates_coalesce_label_layout_and_parent_refresh():
    app = _class_source(ROOT / "src" / "app.py", "VideoToWavConverter")
    set_status = _method_source(app, "set_status_message")
    flush_status = _method_source(app, "_flush_status_message")
    apply_status = _method_source(app, "_apply_status_wrap_and_center")

    assert "final_text in (self._pending_status_text, self._displayed_status_text)" in set_status
    assert "if not self._status_update_pending:" in set_status
    assert "if not self._status_layout_pending:" in flush_status
    assert "status_text == self._status_layout_text" in apply_status
    assert "parent.Refresh()" in apply_status
    assert "self.Refresh()" not in apply_status
    assert "self.Layout()" not in apply_status
