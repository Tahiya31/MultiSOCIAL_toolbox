import sys
import wx
from gui_utils import Theme, _mix_colors, create_transparent_text, composited_background_colour


def _fill_windows_background(window, dc):
    """On Windows, fill the control's client rect with the colour painted behind
    it so owner-drawn rounded corners / shadows don't show background artifacts.
    No-op elsewhere (native BG_STYLE_PAINT transparency already works)."""
    if sys.platform.startswith("win"):
        dc.SetBackground(wx.Brush(composited_background_colour(window)))
        dc.Clear()

class GradientPanel(wx.ScrolledWindow):
    def __init__(self, parent):
        super(GradientPanel, self).__init__(parent, style=wx.VSCROLL)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.SetScrollRate(0, 10) # Vertical scrolling only
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        self.DoPrepareDC(dc)
        width, height = self.GetVirtualSize()
        rect = wx.Rect(0, 0, width, height)
        dc.GradientFillLinear(rect, Theme.COLOR_BG_GRADIENT_START, Theme.COLOR_BG_GRADIENT_END, wx.NORTH)

        gc = wx.GraphicsContext.Create(dc)
        if gc and width > 0 and height > 0:
            glow = Theme.COLOR_BG_GLOW
            cx, cy = width / 2.0, height * 0.12
            radius = max(width, height) * 0.55
            brush = gc.CreateRadialGradientBrush(
                cx, cy, cx, cy, radius,
                wx.Colour(glow[0], glow[1], glow[2], glow[3]),
                wx.Colour(glow[0], glow[1], glow[2], 0),
            )
            gc.SetBrush(brush)
            gc.SetPen(wx.TRANSPARENT_PEN)
            gc.DrawRectangle(0, 0, width, height)


class GlassPanel(wx.Panel):
    """A translucent rounded-rectangle container used to group options.

    Children should be added to its internal BoxSizer via SetSizer as usual.
    """
    def __init__(self, parent, corner_radius=10, fill_rgba=Theme.COLOR_GLASS_FILL, chrome=True):
        super(GlassPanel, self).__init__(parent, style=wx.BORDER_NONE)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.corner_radius = corner_radius
        self.fill_rgba = fill_rgba
        self.chrome = chrome
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def _paint_gradient_background_windows(self, dc, rect):
        """Paint the underlying gradient background for Windows compatibility.
        
        On Windows, wx.BG_STYLE_PAINT doesn't provide true transparency,
        so we need to manually paint the gradient that would show through.
        """
        try:
            # Find the GradientPanel parent to get gradient dimensions
            gradient_parent = self.GetParent()
            if not gradient_parent:
                return
            
            # Get the full virtual size of the gradient panel
            fw, fh = gradient_parent.GetVirtualSize()
            fh = max(fh, 1)  # Avoid division by zero
            
            # Calculate this panel's position relative to the gradient
            self_screen = self.GetScreenPosition()
            parent_screen = gradient_parent.GetScreenPosition()
            
            # Get scroll offset if parent is scrolled
            scroll_y = 0
            if hasattr(gradient_parent, 'CalcUnscrolledPosition'):
                _, scroll_y = gradient_parent.CalcUnscrolledPosition(0, 0)
            
            rel_y_top = (self_screen.y - parent_screen.y) + scroll_y
            rel_y_bot = rel_y_top + rect.height
            
            # Calculate gradient colors at top and bottom of this panel
            c_start = wx.Colour(Theme.COLOR_BG_GRADIENT_START)
            c_end = wx.Colour(Theme.COLOR_BG_GRADIENT_END)
            
            pct_top = max(0.0, min(1.0, rel_y_top / float(fh)))
            pct_bot = max(0.0, min(1.0, rel_y_bot / float(fh)))
            
            color_top = _mix_colors(c_end, c_start, pct_top)
            color_bot = _mix_colors(c_end, c_start, pct_bot)
            
            # Fill with gradient (wx.SOUTH means starting color is at top)
            dc.GradientFillLinear(rect, color_top, color_bot, wx.SOUTH)
        except Exception:
            # Fallback: fill with gradient start color
            dc.SetBackground(wx.Brush(wx.Colour(Theme.COLOR_BG_GRADIENT_START)))
            dc.Clear()

    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        rect = self.GetClientRect()
        if not gc:
            return
        
        if sys.platform.startswith("win"):
            self._paint_gradient_background_windows(dc, rect)

        if not self.chrome:
            return

        inset = self.FromDIP(4)
        x = rect.x + inset
        y = rect.y + inset
        w = max(0, rect.width - inset * 2)
        h = max(0, rect.height - inset * 2)
        r = self.FromDIP(self.corner_radius)
        path = gc.CreatePath()
        path.AddRoundedRectangle(x, y, w, h, r)
        rcol = wx.Colour(*self.fill_rgba)
        gc.SetBrush(wx.Brush(rcol))
        gc.SetPen(wx.Pen(wx.Colour(*Theme.COLOR_GLASS_BORDER), 1))
        gc.DrawPath(path)

        # Subtle full-height top sheen that fades to nothing (no hard mid-card edge).
        sheen = gc.CreatePath()
        sheen.AddRoundedRectangle(x, y, w, h, r)
        gc.SetBrush(gc.CreateLinearGradientBrush(
            x, y, x, y + h,
            wx.Colour(255, 255, 255, 14),
            wx.Colour(255, 255, 255, 0),
        ))
        gc.SetPen(wx.TRANSPARENT_PEN)
        gc.DrawPath(sheen)


class ElevatedLogoPanel(wx.Panel):
    def __init__(self, parent, logo_bitmap):
        super(ElevatedLogoPanel, self).__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.logo_bitmap = logo_bitmap
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_ENTER_WINDOW, self.OnMouseEnter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.OnMouseLeave)
        self.hover = False
        self.scale_factor = 1.0
        # Simple cache to avoid regenerating the circular bitmap too often
        self._cached_diameter = None
        self._cached_circular_bitmap = None

    def _create_circular_bitmap(self, src_bitmap: wx.Bitmap, diameter: int) -> wx.Bitmap:
        """Return a circular-masked bitmap of the requested diameter.
        Alpha outside the circle is set to 0 so corners are transparent.
        """
        if diameter <= 0:
            return src_bitmap

        # Return cached if same diameter
        if self._cached_diameter == diameter and self._cached_circular_bitmap is not None:
            return self._cached_circular_bitmap

        image = src_bitmap.ConvertToImage()
        image = image.Scale(diameter, diameter, wx.IMAGE_QUALITY_HIGH)

        if not image.HasAlpha():
            image.InitAlpha()

        radius = diameter / 2.0
        cx = radius
        cy = radius

        # Apply circular alpha mask
        for y in range(diameter):
            for x in range(diameter):
                dx = x - cx
                dy = y - cy
                if (dx * dx + dy * dy) <= (radius * radius):
                    image.SetAlpha(x, y, 255)
                else:
                    image.SetAlpha(x, y, 0)

        circ_bmp = wx.Bitmap(image)
        self._cached_diameter = diameter
        self._cached_circular_bitmap = circ_bmp
        return circ_bmp
 
    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        rect = self.GetClientRect()
        
        # Clear background to avoid artifacts on Windows
        if sys.platform.startswith("win"):
            # Paint gradient background to match parent
            try:
                parent = self.GetParent()
                if parent:
                    fw, fh = parent.GetVirtualSize()
                    fh = max(fh, 1)
                    self_screen = self.GetScreenPosition()
                    parent_screen = parent.GetScreenPosition()
                    scroll_y = 0
                    if hasattr(parent, 'CalcUnscrolledPosition'):
                        _, scroll_y = parent.CalcUnscrolledPosition(0, 0)
                    rel_y_top = (self_screen.y - parent_screen.y) + scroll_y
                    rel_y_bot = rel_y_top + rect.height
                    c_start = wx.Colour(Theme.COLOR_BG_GRADIENT_START)
                    c_end = wx.Colour(Theme.COLOR_BG_GRADIENT_END)
                    pct_top = max(0.0, min(1.0, rel_y_top / float(fh)))
                    pct_bot = max(0.0, min(1.0, rel_y_bot / float(fh)))
                    color_top = _mix_colors(c_end, c_start, pct_top)
                    color_bot = _mix_colors(c_end, c_start, pct_bot)
                    dc.GradientFillLinear(rect, color_top, color_bot, wx.SOUTH)
            except Exception:
                dc.SetBackground(wx.Brush(wx.Colour(Theme.COLOR_BG_GRADIENT_START)))
                dc.Clear()

        if not self.logo_bitmap:
            return

        # Determine circle diameter and center; add small padding so border doesn't clip
        padding = self.FromDIP(6)
        base_diameter = max(10, min(rect.width, rect.height) - padding * 2)
        diameter = int(base_diameter * self.scale_factor)

        # Center position
        x = rect.x + (rect.width - diameter) // 2
        y = rect.y + (rect.height - diameter) // 2

        # Subtle shadow (fake blur via semi-transparent larger ellipse)
        gc = wx.GraphicsContext.Create(dc)
        if gc:
            shadow_color = wx.Colour(0, 0, 0, 70)
            gc.SetBrush(wx.Brush(shadow_color))
            gc.SetPen(wx.Pen(shadow_color, 1))
            gc.DrawEllipse(x + 2, y + 5, diameter, diameter)

        circ_bmp = self._create_circular_bitmap(self.logo_bitmap, diameter)
        dc.DrawBitmap(circ_bmp, x, y, True)

        if gc:
            gc.SetBrush(wx.TRANSPARENT_BRUSH)
            ring = Theme.colour(Theme.COLOR_ACCENT_GREEN)
            gc.SetPen(wx.Pen(ring, 2))
            gc.DrawEllipse(x, y, diameter, diameter)
            gc.SetPen(wx.Pen(wx.Colour(255, 255, 255, 40), 1))
            gc.DrawEllipse(x + 2, y + 2, diameter - 4, diameter - 4)

    def OnMouseEnter(self, event):
        self.hover = True
        self.scale_factor = 1.08  # pop a bit more on hover
        self.SetCursor(wx.Cursor(wx.CURSOR_HAND))
        self.Refresh()
        event.Skip()
     
    def OnMouseLeave(self, event):
        self.hover = False
        self.scale_factor = 1.0
        self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
        self.Refresh()
        event.Skip()


class CustomTooltip(wx.PopupWindow):
    def __init__(self, parent, text):
        super(CustomTooltip, self).__init__(parent, wx.SIMPLE_BORDER)
        self.text = text
        # Darker, higher-contrast tooltip
        self.SetBackgroundColour(Theme.COLOR_TOOLTIP_BG)
        self.SetForegroundColour(Theme.COLOR_TOOLTIP_FG)
        
        # Create a panel for the tooltip content
        panel = wx.Panel(self)
        panel.SetBackgroundColour(Theme.COLOR_TOOLTIP_BG)
        self.content_panel = panel
        
        self.text_ctrl = wx.StaticText(panel, label=text, style=wx.ALIGN_LEFT)
        self.text_ctrl.SetFont(Theme.get_font(Theme.FONT_CAPTION, bold=False))
        self.text_ctrl.SetForegroundColour(Theme.COLOR_TOOLTIP_FG)
        self.text_ctrl.Wrap(340)
        
        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.text_ctrl, flag=wx.ALL, border=10)
        panel.SetSizer(sizer)
        
        # Size the tooltip
        panel.Fit()
        # Add a subtle border by enlarging slightly and drawing via parent window border
        self.SetSize(panel.GetSize() + wx.Size(6, 6))


class InfoIcon(wx.StaticText):
    def __init__(self, parent, tooltip_text):
        super(InfoIcon, self).__init__(parent, label="?", style=wx.ALIGN_CENTER)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.tooltip_text = tooltip_text
        self.tooltip = None
        self._hide_timer = None
        self._top_level = self.GetTopLevelParent()
        
        self.SetFont(Theme.get_font(Theme.FONT_CAPTION, bold=True))
        self.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_MUTED))
        self.SetMinSize(self.FromDIP(wx.Size(24, 24)))
        
        self.Bind(wx.EVT_ENTER_WINDOW, self.OnMouseEnter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.OnMouseLeave)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        if self._top_level:
            self._top_level.Bind(wx.EVT_MOVE, self._on_parent_move_or_resize)
            self._top_level.Bind(wx.EVT_SIZE, self._on_parent_move_or_resize)
        self.Bind(wx.EVT_WINDOW_DESTROY, self._on_destroy)
    
    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        if not gc:
            return
        rect = self.GetClientRect()
        cx = rect.width / 2.0
        cy = rect.height / 2.0
        radius = min(rect.width, rect.height) / 2.0 - 1

        hovered = getattr(self, "_hover", False)
        if hovered:
            fill = Theme.colour(Theme.COLOR_ACCENT_INFO_SOFT)
            border = Theme.colour(Theme.COLOR_ACCENT_INFO)
            glyph = Theme.colour(Theme.COLOR_ACCENT_INFO)
        else:
            fill = Theme.colour(Theme.COLOR_INFO_ICON_BG)
            border = Theme.colour(Theme.COLOR_INFO_ICON_BORDER)
            glyph = Theme.colour(Theme.COLOR_TEXT_MUTED)

        gc.SetBrush(wx.Brush(fill))
        gc.SetPen(wx.Pen(border, 1))
        gc.DrawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

        gc.SetFont(self.GetFont(), glyph)
        label = "?"
        tw, th = gc.GetTextExtent(label)
        gc.DrawText(label, cx - tw / 2.0, cy - th / 2.0)
    
    def OnMouseEnter(self, event):
        self._hover = True
        self.Refresh()
        self._cancel_hide()
        if not self.tooltip:
            self.tooltip = CustomTooltip(self.GetParent(), self.tooltip_text)
            self._bind_tooltip_hover()
        self.ShowTooltip()
        event.Skip()

    def OnMouseLeave(self, event):
        self._hover = False
        self.Refresh()
        self._schedule_hide()
        event.Skip()
    
    def _bind_tooltip_hover(self):
        if not self.tooltip:
            return
        for win in (self.tooltip, getattr(self.tooltip, "content_panel", None), getattr(self.tooltip, "text_ctrl", None)):
            if win:
                win.Bind(wx.EVT_ENTER_WINDOW, self._on_tooltip_enter)
                win.Bind(wx.EVT_LEAVE_WINDOW, self._on_tooltip_leave)

    def _on_tooltip_enter(self, event):
        self._cancel_hide()
        event.Skip()

    def _on_tooltip_leave(self, event):
        self._schedule_hide()
        event.Skip()

    def _cancel_hide(self):
        if self._hide_timer and self._hide_timer.IsRunning():
            self._hide_timer.Stop()

    def _schedule_hide(self):
        self._cancel_hide()
        self._hide_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_hide_timer, self._hide_timer)
        self._hide_timer.Start(150, oneShot=True)

    def _on_hide_timer(self, event):
        self._destroy_tooltip()

    def _destroy_tooltip(self):
        if self.tooltip:
            try:
                self.tooltip.Destroy()
            except Exception:
                pass
            self.tooltip = None
    
    def ShowTooltip(self):
        if not self.tooltip:
            return
        icon_rect = self.GetRect()
        parent_pos = self.GetParent().GetScreenPosition()
        tooltip_x = parent_pos.x + icon_rect.x
        tooltip_y = parent_pos.y + icon_rect.y + icon_rect.height + self.FromDIP(6)
        try:
            screen_w, screen_h = wx.GetDisplaySize()
            tip_w, tip_h = self.tooltip.GetSize()
            tooltip_x = max(0, min(tooltip_x, screen_w - tip_w - 8))
            tooltip_y = max(0, min(tooltip_y, screen_h - tip_h - 8))
        except Exception:
            pass
        self.tooltip.Position((tooltip_x, tooltip_y), (0, 0))
        self.tooltip.Show()
        self.tooltip.Raise()

    def _on_parent_move_or_resize(self, event):
        self._destroy_tooltip()
        event.Skip()

    def _on_destroy(self, event):
        self._destroy_tooltip()
        self._cancel_hide()
        try:
            if self._top_level:
                self._top_level.Unbind(wx.EVT_MOVE, handler=self._on_parent_move_or_resize)
                self._top_level.Unbind(wx.EVT_SIZE, handler=self._on_parent_move_or_resize)
        except Exception:
            pass
        event.Skip()


class FlatButton(wx.Window):
    """Owner-drawn rounded button (primary / secondary / danger)."""

    VARIANT_PRIMARY = "primary"
    VARIANT_SECONDARY = "secondary"
    VARIANT_DANGER = "danger"

    _VARIANT_COLORS = {
        VARIANT_PRIMARY: (
            Theme.COLOR_PRIMARY,
            Theme.COLOR_PRIMARY_HOVER,
            Theme.COLOR_PRIMARY_PRESSED,
        ),
        VARIANT_SECONDARY: (
            Theme.COLOR_SECONDARY,
            Theme.COLOR_SECONDARY_HOVER,
            Theme.COLOR_SECONDARY_PRESSED,
        ),
        VARIANT_DANGER: (
            Theme.COLOR_DANGER,
            Theme.COLOR_DANGER_HOVER,
            Theme.COLOR_DANGER_PRESSED,
        ),
    }

    def __init__(self, parent, label="", variant=VARIANT_PRIMARY, id=wx.ID_ANY, size=(-1, -1)):
        super(FlatButton, self).__init__(parent, id=id, size=size, style=wx.BORDER_NONE)
        self._label = label
        self._variant = variant if variant in self._VARIANT_COLORS else self.VARIANT_PRIMARY
        self._hover = False
        self._pressed = False
        # "Locked" is a soft-disabled state: the button greys out and ignores
        # clicks (like disabled) but stays wx-enabled so hover/tooltip still work.
        self._locked = False
        self._font = Theme.get_font(Theme.FONT_BUTTON)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_ENTER_WINDOW, self._on_enter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._on_leave)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_down)
        self.Bind(wx.EVT_LEFT_UP, self._on_up)
        self.Bind(wx.EVT_MOUSE_CAPTURE_LOST, self._on_capture_lost)
        self._update_min_size()

    def _update_min_size(self):
        dc = wx.ClientDC(self)
        dc.SetFont(self._font)
        tw, th = dc.GetTextExtent(self._label)
        pad_x = self.FromDIP(Theme.SPACE_LG * 2)
        pad_y = self.FromDIP(Theme.SPACE_SM + 4)
        self.SetMinSize((tw + pad_x, th + pad_y))

    def Enable(self, enable=True):
        result = super(FlatButton, self).Enable(enable)
        self.Refresh()
        return result

    def set_locked(self, locked, reason=""):
        """Soft-disable: grey out and block clicks while keeping the window
        wx-enabled so the hover tooltip still appears. `reason` becomes the
        tooltip shown on hover while locked."""
        self._locked = bool(locked)
        if self._locked and reason:
            self.SetToolTip(reason)
        else:
            self.UnsetToolTip()
        self.Refresh()

    def _is_interactive(self):
        return self.IsEnabled() and not self._locked

    def SetLabel(self, label):
        self._label = label
        self._update_min_size()
        self.Refresh()

    def GetLabel(self):
        return self._label

    def SetFont(self, font):
        self._font = font
        self._update_min_size()
        self.Refresh()

    def _fill_colour(self):
        if not self._is_interactive():
            return Theme.colour(Theme.COLOR_BTN_DISABLED_FILL)
        normal, hover, pressed = self._VARIANT_COLORS[self._variant]
        if self._pressed:
            return Theme.colour(pressed)
        if self._hover:
            return Theme.colour(hover)
        return Theme.colour(normal)

    def _on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        _fill_windows_background(self, dc)
        gc = wx.GraphicsContext.Create(dc)
        if not gc:
            return
        rect = self.GetClientRect()
        radius = self.FromDIP(Theme.RADIUS_BUTTON)
        is_secondary = self._variant == self.VARIANT_SECONDARY
        enabled = self._is_interactive()

        if enabled and not self._pressed:
            shadow = gc.CreatePath()
            shadow.AddRoundedRectangle(rect.x, rect.y + 2, rect.width, rect.height, radius)
            gc.SetBrush(wx.Brush(wx.Colour(0, 0, 0, 55)))
            gc.SetPen(wx.TRANSPARENT_PEN)
            gc.DrawPath(shadow)

        path = gc.CreatePath()
        path.AddRoundedRectangle(rect.x, rect.y, rect.width, rect.height, radius)
        gc.SetBrush(wx.Brush(self._fill_colour()))
        if not enabled:
            gc.SetPen(wx.Pen(Theme.colour(Theme.COLOR_BTN_DISABLED_BORDER), 1))
        elif is_secondary:
            gc.SetPen(wx.Pen(Theme.colour(Theme.COLOR_SECONDARY_BORDER), 1))
        else:
            gc.SetPen(wx.Pen(wx.Colour(0, 0, 0, 0)))
        gc.DrawPath(path)

        # No shine on primary: keep it flat so the CTA green reads identical to the
        # active tab pill (same COLOR_PRIMARY). The drop shadow alone gives depth.

        if not enabled:
            text_colour = Theme.colour(Theme.COLOR_TEXT_SUBTLE)
        else:
            text_colour = Theme.colour(Theme.COLOR_TEXT_ON_DARK)
        gc.SetFont(self._font, text_colour)
        tw, th = gc.GetTextExtent(self._label)
        gc.DrawText(
            self._label,
            rect.x + (rect.width - tw) / 2.0,
            rect.y + (rect.height - th) / 2.0,
        )

    def _on_enter(self, event):
        if self._is_interactive():
            self._hover = True
            self.SetCursor(wx.Cursor(wx.CURSOR_HAND))
            self.Refresh()
        event.Skip()

    def _on_leave(self, event):
        # Leave hover state, but keep _pressed/capture intact while the button is
        # held: _on_up (or capture-lost) is the only place that releases capture,
        # so clearing _pressed here would leak the mouse capture and freeze input.
        self._hover = False
        self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
        self.Refresh()
        event.Skip()

    def _on_down(self, event):
        if self._is_interactive():
            self._pressed = True
            self.CaptureMouse()
            self.Refresh()
        event.Skip()

    def _on_up(self, event):
        was_pressed = self._pressed
        self._pressed = False
        if self.HasCapture():
            self.ReleaseMouse()
        self.Refresh()
        if was_pressed and self._is_interactive() and self.GetClientRect().Contains(event.GetPosition()):
            evt = wx.CommandEvent(wx.EVT_BUTTON.typeId, self.GetId())
            evt.SetEventObject(self)
            self.GetEventHandler().ProcessEvent(evt)
        event.Skip()

    def _on_capture_lost(self, event):
        # Required whenever CaptureMouse() is used: if the capture is stolen
        # (e.g. a modal dialog opens), reset state instead of asserting.
        self._pressed = False
        self.Refresh()


class _TabFontProxy:
    """Allows font-scaling registry to target segmented tab labels."""

    def __init__(self, bar):
        self._bar = bar

    def SetFont(self, font):
        self._bar._tab_font = font
        self._bar.Refresh()


class ToggleTabBar(wx.Panel):
    """Segmented Video / Audio mode control."""

    _SEGMENTS = (("video", "Video"), ("audio", "Audio"))

    def __init__(self, parent):
        super(ToggleTabBar, self).__init__(parent, style=wx.BORDER_NONE)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self._selected = "video"
        self._tabs_enabled = True
        self._on_change = None
        self._tab_font = Theme.get_font(Theme.FONT_BODY, bold=True)
        self._hover_segment = None

        self.video_tab = _TabFontProxy(self)
        self.audio_tab = _TabFontProxy(self)

        tab_h = self.FromDIP(50)
        self.SetMinSize((-1, tab_h))
        self.SetMaxSize((-1, tab_h))
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_click)
        self.Bind(wx.EVT_MOTION, self._on_motion)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._on_leave)

    def set_on_change(self, callback):
        self._on_change = callback

    def set_selected(self, mode):
        self._select(mode, fire=False)

    def get_selected(self):
        return self._selected

    def EnableTabs(self, enable=True):
        self._tabs_enabled = enable
        self.Refresh()

    def _segment_at(self, x):
        rect = self.GetClientRect()
        pad = self.FromDIP(Theme.SPACE_XS)
        track_w = max(1, rect.width - pad * 2)
        local_x = x - pad
        if local_x < track_w / 2.0:
            return "video"
        return "audio"

    def _select(self, mode, fire=False):
        if not self._tabs_enabled and fire:
            return
        if mode not in ("video", "audio"):
            return
        self._selected = mode
        self.Refresh()
        if fire and self._on_change:
            self._on_change(mode)

    def _on_click(self, event):
        if not self._tabs_enabled:
            return
        self._select(self._segment_at(event.GetX()), fire=True)

    def _on_motion(self, event):
        if not self._tabs_enabled:
            self._hover_segment = None
            return
        seg = self._segment_at(event.GetX())
        if seg != self._hover_segment:
            self._hover_segment = seg
            self.SetCursor(wx.Cursor(wx.CURSOR_HAND))
            self.Refresh()

    def _on_leave(self, event):
        self._hover_segment = None
        self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
        self.Refresh()
        event.Skip()

    def _on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        _fill_windows_background(self, dc)
        gc = wx.GraphicsContext.Create(dc)
        if not gc:
            return

        rect = self.GetClientRect()
        pad = self.FromDIP(Theme.SPACE_XS)
        track = wx.Rect(pad, pad, rect.width - pad * 2, rect.height - pad * 2)
        radius = self.FromDIP(Theme.RADIUS_TAB)

        # Opaque track so tabs never disappear under siblings on any platform
        track_path = gc.CreatePath()
        track_path.AddRoundedRectangle(track.x, track.y, track.width, track.height, radius)
        gc.SetBrush(wx.Brush(Theme.colour(Theme.COLOR_TAB_TRACK)))
        gc.SetPen(wx.Pen(wx.Colour(255, 255, 255, 20), 1))
        gc.DrawPath(track_path)

        seg_w = track.width / 2.0
        for idx, (mode, _label) in enumerate(self._SEGMENTS):
            if mode != self._selected:
                continue
            pill_x = track.x + idx * seg_w + self.FromDIP(3)
            pill_y = track.y + self.FromDIP(3)
            pill_w = seg_w - self.FromDIP(6)
            pill_h = track.height - self.FromDIP(6)
            pill_path = gc.CreatePath()
            pill_path.AddRoundedRectangle(pill_x, pill_y, pill_w, pill_h, radius - 2)
            gc.SetBrush(wx.Brush(Theme.colour(Theme.COLOR_TAB_ACTIVE)))
            gc.SetPen(wx.Pen(Theme.colour(Theme.COLOR_TAB_ACTIVE_BORDER), 1))
            gc.DrawPath(pill_path)

        font = self._tab_font
        for idx, (mode, label) in enumerate(self._SEGMENTS):
            if not self._tabs_enabled:
                colour = Theme.colour(Theme.COLOR_DISABLED)
            elif mode == self._selected:
                colour = Theme.colour(Theme.COLOR_TEXT_BLACK)
            elif mode == self._hover_segment:
                colour = Theme.colour(Theme.COLOR_ACCENT_GREEN)
            else:
                colour = Theme.colour(Theme.COLOR_TEXT_MUTED)
            gc.SetFont(font, colour)
            tw, th = gc.GetTextExtent(label)
            cx = track.x + seg_w * idx + seg_w / 2.0
            cy = track.y + track.height / 2.0
            gc.DrawText(label, cx - tw / 2.0, cy - th / 2.0)


class CustomCheckBox(wx.Window):
    """Owner-drawn checkbox with label."""

    def __init__(self, parent, label="", id=wx.ID_ANY):
        super(CustomCheckBox, self).__init__(parent, id=id, style=wx.BORDER_NONE)
        self._label = label
        self._value = False
        self._hover = False
        self._font = Theme.get_font(Theme.FONT_BODY)
        self._text_colour = Theme.colour(Theme.COLOR_TEXT_WHITE)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self._on_paint)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_click)
        self.Bind(wx.EVT_ENTER_WINDOW, self._on_enter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._on_leave)
        self._update_min_size()

    def _box_size(self):
        return self.FromDIP(18)

    def _update_min_size(self):
        dc = wx.ClientDC(self)
        dc.SetFont(self._font)
        tw, th = dc.GetTextExtent(self._label)
        box = self._box_size()
        gap = self.FromDIP(Theme.SPACE_SM)
        self.SetMinSize((box + gap + tw, max(box, th) + self.FromDIP(Theme.SPACE_XS)))

    def GetValue(self):
        return self._value

    def SetValue(self, value):
        self._value = bool(value)
        self.Refresh()

    def SetLabel(self, label):
        self._label = label
        self._update_min_size()
        self.Refresh()

    def GetLabel(self):
        return self._label

    def SetFont(self, font):
        self._font = font
        self._update_min_size()
        self.Refresh()

    def SetForegroundColour(self, colour):
        self._text_colour = colour
        self.Refresh()

    def Enable(self, enable=True):
        result = super(CustomCheckBox, self).Enable(enable)
        self.Refresh()
        return result

    def _on_paint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        _fill_windows_background(self, dc)
        gc = wx.GraphicsContext.Create(dc)
        if not gc:
            return
        rect = self.GetClientRect()
        box = self._box_size()
        by = rect.y + (rect.height - box) / 2.0
        bx = rect.x
        gap = self.FromDIP(Theme.SPACE_SM)

        if self._hover and self.IsEnabled():
            row_path = gc.CreatePath()
            row_path.AddRoundedRectangle(rect.x - 4, rect.y, rect.width + 8, rect.height, self.FromDIP(6))
            gc.SetBrush(wx.Brush(wx.Colour(255, 255, 255, 12)))
            gc.SetPen(wx.TRANSPARENT_PEN)
            gc.DrawPath(row_path)

        enabled = self.IsEnabled()
        radius = self.FromDIP(5)
        path = gc.CreatePath()
        path.AddRoundedRectangle(bx, by, box, box, radius)
        if self._value and enabled:
            gc.SetBrush(wx.Brush(Theme.colour(Theme.COLOR_PRIMARY)))
            gc.SetPen(wx.Pen(Theme.colour(Theme.COLOR_PRIMARY_HOVER), 1))
        else:
            gc.SetBrush(wx.Brush(Theme.colour(Theme.COLOR_BTN_DISABLED_FILL if not enabled else Theme.COLOR_INPUT_BG)))
            gc.SetPen(wx.Pen(Theme.colour(Theme.COLOR_BTN_DISABLED_BORDER if not enabled else Theme.COLOR_SECONDARY_BORDER), 1))
        gc.DrawPath(path)
        if self._value and enabled:
            gc.SetPen(wx.Pen(Theme.colour(Theme.COLOR_TEXT_ON_DARK), 2))
            cx, cy = bx + box / 2.0, by + box / 2.0
            gc.StrokeLine(cx - box * 0.2, cy, cx - box * 0.05, cy + box * 0.18)
            gc.StrokeLine(cx - box * 0.05, cy + box * 0.18, cx + box * 0.22, cy - box * 0.15)
        label_colour = self._text_colour if enabled else Theme.colour(Theme.COLOR_TEXT_SUBTLE)
        gc.SetFont(self._font, label_colour)
        tw, th = gc.GetTextExtent(self._label)
        gc.DrawText(self._label, bx + box + gap, rect.y + (rect.height - th) / 2.0)

    def _on_click(self, event):
        if not self.IsEnabled():
            return
        self._value = not self._value
        self.Refresh()
        evt = wx.CommandEvent(wx.EVT_CHECKBOX.typeId, self.GetId())
        evt.SetInt(int(self._value))
        evt.SetEventObject(self)
        self.GetEventHandler().ProcessEvent(evt)

    def _on_enter(self, event):
        self._hover = True
        self.SetCursor(wx.Cursor(wx.CURSOR_HAND))
        event.Skip()

    def _on_leave(self, event):
        self._hover = False
        self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))
        self.Refresh()
        event.Skip()


class SectionCard(GlassPanel):
    """Glass card with optional heading and standardized padding."""

    def __init__(self, parent, heading=None, corner_radius=Theme.RADIUS_CARD, fill_rgba=Theme.COLOR_SURFACE_ELEVATED):
        super(SectionCard, self).__init__(parent, corner_radius=corner_radius, fill_rgba=fill_rgba)
        outer = wx.BoxSizer(wx.VERTICAL)
        pad = Theme.SPACE_LG
        if heading:
            heading_label = create_transparent_text(self, label=heading.upper(), style=wx.ALIGN_LEFT)
            heading_label.SetFont(Theme.get_font(Theme.FONT_OVERLINE, bold=True))
            heading_label.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_MUTED))
            outer.Add(heading_label, flag=wx.LEFT | wx.RIGHT | wx.TOP, border=pad)
            divider = wx.Panel(self, size=(-1, 1))
            divider.SetBackgroundColour(Theme.colour(Theme.COLOR_GLASS_BORDER))
            outer.Add(divider, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=Theme.SPACE_SM)
            self._heading_label = heading_label
        else:
            self._heading_label = None
        self.content_sizer = wx.BoxSizer(wx.VERTICAL)
        outer.Add(self.content_sizer, proportion=1, flag=wx.EXPAND | wx.ALL, border=pad)
        self.SetSizer(outer)


class TooltipButton:
    """Factory for FlatButton + InfoIcon action rows."""

    @staticmethod
    def create_with_icon(parent, label, tooltip_text, font=None, handler=None, variant=FlatButton.VARIANT_PRIMARY):
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        btn = FlatButton(parent, label=label, variant=variant)
        if font:
            btn.SetFont(font)
        if handler:
            btn.Bind(wx.EVT_BUTTON, handler)
        sizer.Add(btn, proportion=1, flag=wx.EXPAND | wx.ALL, border=Theme.SPACE_SM)
        info_icon = InfoIcon(parent, tooltip_text)
        sizer.Add(info_icon, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=Theme.SPACE_SM)
        return btn, sizer

class CustomGauge(wx.Panel):
    """A custom-drawn gauge that respects height on macOS and Windows."""
    def __init__(self, parent, range=100, size=(-1, 32)):
        super(CustomGauge, self).__init__(parent, size=size, style=wx.NO_BORDER | wx.NO_FULL_REPAINT_ON_RESIZE)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self._range = range
        self._value = 0
        self._bg_color = Theme.colour(Theme.COLOR_PROGRESS_TRACK)
        self._fg_color = Theme.colour(Theme.COLOR_PROGRESS_FILL)

        if wx.Platform == '__WXMSW__':
            self.SetDoubleBuffered(True)
        
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)

    def OnEraseBackground(self, event):
        # Do nothing to prevent flickering; letting the parent background show through
        pass

    def SetValue(self, value):
        self._value = max(0, min(value, self._range))
        self.Refresh()
        self.Update()
        
    def GetValue(self):
        return self._value
        
    def SetRange(self, range):
        self._range = range
        
    def GetRange(self):
        return self._range
    
    def Pulse(self):
        # Indeterminate mode not fully implemented, but method exists for compatibility.
        pass
        
    def SetForegroundColour(self, color):
        self._fg_color = color
        self.Refresh()

    def OnSize(self, event):
        self.Refresh()
        event.Skip()

    def OnPaint(self, event):
        # Use AutoBufferedPaintDC for cross-platform consistency (prevents flicker on Windows)
        dc = wx.AutoBufferedPaintDC(self)
        _fill_windows_background(self, dc)

        gc = wx.GraphicsContext.Create(dc)
        if not gc:
            return

        rect = self.GetClientRect()
        r = min(self.FromDIP(8), rect.height / 2.0, rect.width / 2.0)

        track_path = gc.CreatePath()
        track_path.AddRoundedRectangle(rect.x, rect.y, rect.width, rect.height, r)
        gc.SetBrush(wx.Brush(self._bg_color))
        gc.SetPen(wx.Pen(wx.Colour(255, 255, 255, 15), 1))
        gc.DrawPath(track_path)

        if self._value > 0 and self._range > 0:
            pct = float(self._value) / float(self._range)
            w = max(1.0, rect.width * pct)
            fill_rect = wx.Rect(rect.x, rect.y, int(w), rect.height)
            gc.Clip(fill_rect.x, fill_rect.y, fill_rect.width, fill_rect.height)

            glow = Theme.COLOR_PROGRESS_GLOW
            bar_brush = gc.CreateLinearGradientBrush(
                rect.x, rect.y, rect.x, rect.y + rect.height,
                wx.Colour(46, 212, 122),
                wx.Colour(34, 184, 106),
            )
            gc.SetBrush(bar_brush)
            gc.SetPen(wx.TRANSPARENT_PEN)
            bar_path = gc.CreatePath()
            bar_path.AddRoundedRectangle(rect.x, rect.y, rect.width, rect.height, r)
            gc.DrawPath(bar_path)
            gc.ResetClip()

            shine = gc.CreatePath()
            shine.AddRoundedRectangle(rect.x + 1, rect.y + 1, min(w, rect.width) - 2, rect.height * 0.4, r)
            gc.SetBrush(wx.Brush(wx.Colour(255, 255, 255, 35)))
            gc.DrawPath(shine)
