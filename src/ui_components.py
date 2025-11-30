import wx
from gui_utils import Theme

class GradientPanel(wx.Panel):
    def __init__(self, parent):
        super(GradientPanel, self).__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        rect = self.GetClientRect()
        dc.GradientFillLinear(rect, Theme.COLOR_BG_GRADIENT_START, Theme.COLOR_BG_GRADIENT_END, wx.NORTH)


class GlassPanel(wx.Panel):
    """A translucent rounded-rectangle container used to group options.

    Children should be added to its internal BoxSizer via SetSizer as usual.
    """
    def __init__(self, parent, corner_radius=10, fill_rgba=Theme.COLOR_GLASS_FILL):
        super(GlassPanel, self).__init__(parent, style=wx.BORDER_NONE)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.corner_radius = corner_radius
        self.fill_rgba = fill_rgba
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        gc = wx.GraphicsContext.Create(dc)
        rect = self.GetClientRect()
        if not gc:
            return
        # Inset so the rounded border isn't clipped
        inset = 6
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
            shadow_color = wx.Colour(0, 0, 0, 60)  # low alpha for soft shadow
            gc.SetBrush(wx.Brush(shadow_color))
            gc.SetPen(wx.Pen(shadow_color, 1))
            gc.DrawEllipse(x + 3, y + 4, diameter, diameter)  # slight offset

        # Prepare circular bitmap and draw
        circ_bmp = self._create_circular_bitmap(self.logo_bitmap, diameter)
        dc.DrawBitmap(circ_bmp, x, y, True)

        # Black circular border
        if gc:
            gc.SetBrush(wx.TRANSPARENT_BRUSH)
            gc.SetPen(wx.Pen(wx.Colour(0, 0, 0), 3))
            gc.DrawEllipse(x, y, diameter, diameter)

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
        
        # Create text control for the tooltip with word wrapping
        self.text_ctrl = wx.StaticText(panel, label=text, style=wx.ALIGN_LEFT)
        self.text_ctrl.SetFont(Theme.get_font(11, bold=False))
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
        super(InfoIcon, self).__init__(parent, label="ℹ", style=wx.ALIGN_CENTER)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.tooltip_text = tooltip_text
        self.tooltip = None
        self._top_level = self.GetTopLevelParent()
        
        # Style the info icon
        self.SetFont(Theme.get_font(12, bold=True))
        self.SetForegroundColour(Theme.COLOR_TEXT_WHITE)
        self.SetBackgroundColour(Theme.COLOR_INFO_ICON_BG)
        self.SetMinSize(self.FromDIP(wx.Size(20, 20)))
        
        # Bind mouse events
        self.Bind(wx.EVT_ENTER_WINDOW, self.OnMouseEnter)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.OnMouseLeave)
        self.Bind(wx.EVT_MOTION, self.OnMouseMove)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        # Hide tooltip when the window moves/resizes to avoid misplacement
        if self._top_level:
            self._top_level.Bind(wx.EVT_MOVE, self._on_parent_move_or_resize)
            self._top_level.Bind(wx.EVT_SIZE, self._on_parent_move_or_resize)
        # Cleanup bindings on destroy
        self.Bind(wx.EVT_WINDOW_DESTROY, self._on_destroy)
    
    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        rect = self.GetClientRect()
        
        # Draw circular background
        dc.SetBrush(wx.Brush(Theme.COLOR_INFO_ICON_BG))
        dc.SetPen(wx.Pen(Theme.COLOR_INFO_ICON_BG, 1))
        dc.DrawCircle(rect.width//2, rect.height//2, min(rect.width, rect.height)//2 - 1)
        
        # Draw the "i" text
        dc.SetTextForeground(Theme.COLOR_TEXT_WHITE)
        dc.SetFont(self.GetFont())
        text_width, text_height = dc.GetTextExtent("ℹ")
        dc.DrawText("ℹ", (rect.width - text_width)//2, (rect.height - text_height)//2)
    
    def OnMouseEnter(self, event):
        if not self.tooltip:
            self.tooltip = CustomTooltip(self.GetParent(), self.tooltip_text)
            self.ShowTooltip()
        event.Skip()
    
    def OnMouseLeave(self, event):
        if self.tooltip:
            self.tooltip.Destroy()
            self.tooltip = None
        event.Skip()
    
    def OnMouseMove(self, event):
        if self.tooltip:
            self.ShowTooltip()
        event.Skip()
    
    def ShowTooltip(self):
        if self.tooltip:
            # Get the screen position of the icon
            icon_rect = self.GetRect()
            parent_pos = self.GetParent().GetScreenPosition()
            
            # Calculate tooltip position to the right of the icon
            tooltip_x = parent_pos.x + icon_rect.x + icon_rect.width + 2
            tooltip_y = parent_pos.y + icon_rect.y
            
            # Clamp within visible display bounds to avoid overflow
            try:
                screen_w, screen_h = wx.GetDisplaySize()
                tip_w, tip_h = self.tooltip.GetSize()
                tooltip_x = max(0, min(tooltip_x, screen_w - tip_w - 8))
                tooltip_y = max(0, min(tooltip_y, screen_h - tip_h - 8))
            except Exception:
                pass
            
            self.tooltip.Position((tooltip_x, tooltip_y), (0, 0))
            self.tooltip.Show()

    def _on_parent_move_or_resize(self, event):
        if self.tooltip:
            try:
                self.tooltip.Hide()
            except Exception:
                pass
        event.Skip()

    def _on_destroy(self, event):
        # Unbind move/size handlers when this icon is destroyed
        try:
            if self._top_level:
                self._top_level.Unbind(wx.EVT_MOVE, handler=self._on_parent_move_or_resize)
                self._top_level.Unbind(wx.EVT_SIZE, handler=self._on_parent_move_or_resize)
        except Exception:
            pass
        event.Skip()


class TooltipButton(wx.Button):
    def __init__(self, parent, label, tooltip_text):
        super(TooltipButton, self).__init__(parent, label=label)
        self.tooltip_text = tooltip_text
        self.info_icon = None
        
    def add_info_icon(self, parent):
        """Add info icon to the button's parent container"""
        self.info_icon = InfoIcon(parent, self.tooltip_text)
        return self.info_icon
    
    @staticmethod
    def create_with_icon(parent, label, tooltip_text, font=None, handler=None):
        """Factory method to create a button and its info icon in a horizontal sizer."""
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        btn = TooltipButton(parent, label, tooltip_text)
        if font:
            btn.SetFont(font)
        if handler:
            btn.Bind(wx.EVT_BUTTON, handler)
            
        sizer.Add(btn, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        info_icon = btn.add_info_icon(parent)
        sizer.Add(info_icon, flag=wx.ALIGN_CENTER|wx.LEFT, border=5)
        
        return btn, sizer

class CustomGauge(wx.Panel):
    """A custom-drawn gauge that respects height on macOS."""
    def __init__(self, parent, range=100, size=(-1, 28)):
        super(CustomGauge, self).__init__(parent, size=size)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self._range = range
        self._value = 0
        # Default colors
        self._bg_color = wx.Colour(40, 40, 40, 100)  # Dark semi-transparent background
        self._fg_color = wx.Colour(33, 150, 243)  # Blue
        
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
        
        # Use GraphicsContext for smoother anti-aliased drawing
        gc = wx.GraphicsContext.Create(dc)
        if not gc:
            return

        rect = self.GetClientRect()
        
        # 1. Draw Track
        # We don't clear background, so the parent gradient should show through in empty areas
        # (assuming the windowing system supports it, which macOS usually does)
        
        # Track Color
        gc.SetBrush(wx.Brush(self._bg_color))
        # Transparent pen to avoid border artifacts
        gc.SetPen(wx.Pen(wx.Colour(0,0,0,0), 1)) 
        
        # Draw rounded track
        corner_radius = 4
        # Ensure radius isn't too big for the rect
        r = min(corner_radius, rect.height / 2, rect.width / 2)
        
        path = gc.CreatePath()
        path.AddRoundedRectangle(rect.x, rect.y, rect.width, rect.height, r)
        gc.DrawPath(path)
        
        # 2. Draw Progress
        if self._value > 0 and self._range > 0:
            pct = float(self._value) / float(self._range)
            w = rect.width * pct
            
            # Only draw if width is meaningful
            if w >= 1:
                gc.SetBrush(wx.Brush(self._fg_color))
                gc.SetPen(wx.Pen(wx.Colour(0,0,0,0), 1))
                
                path_bar = gc.CreatePath()
                
                if w >= rect.width:
                    # Full width - regular rounded rect
                    path_bar.AddRoundedRectangle(rect.x, rect.y, rect.width, rect.height, r)
                else:
                    # Partially filled - trickier to do "rounded left, square right"
                    # But for simplicity and aesthetics, a rounded rect clipped or just a rounded rect
                    # often looks okay if it's just the bar inside the track.
                    # Let's try to match the track shape.
                    
                    # We can intersect the track path with a rectangle of width w
                    # But clipping in wx.GraphicsContext:
                    gc.Clip(rect.x, rect.y, w, rect.height)
                    path_bar.AddRoundedRectangle(rect.x, rect.y, rect.width, rect.height, r)
                    gc.DrawPath(path_bar)
                    gc.ResetClip()
                    return # Done
                    
                gc.DrawPath(path_bar)
