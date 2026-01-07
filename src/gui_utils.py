import os
import shutil
import wx
import ffmpeg

# --- Constants & Theme ---
import sys
import ctypes
import wx.lib.stattext as stattext

def setup_high_dpi():
    """Enable High DPI awareness on Windows. No-op on other platforms."""
    if sys.platform.startswith("win"):
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # Process_System_DPI_Aware
        except Exception:
            pass

def _mix_colors(c1, c2, w2):
    """Mix color c1 with c2, where w2 is the weight (0.0-1.0) of c2."""
    w1 = 1.0 - w2
    return wx.Colour(
        int(c1.Red() * w1 + c2.Red() * w2),
        int(c1.Green() * w1 + c2.Green() * w2),
        int(c1.Blue() * w1 + c2.Blue() * w2),
        255 # Assume opaque result
    )

class TransparentStaticText(stattext.GenStaticText):
    """
    A StaticText that manually paints its background to match the parent's gradient/glass
    appearance, solving the 'transparency' issue on Windows.
    """
    def __init__(self, *args, **kwargs):
        super(TransparentStaticText, self).__init__(*args, **kwargs)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda e: None)

    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        # Note: DoPrepareDC is only for ScrolledWindow subclasses, not Controls.
        # Scroll offset is handled manually below via CalcUnscrolledPosition.
        
        # 1. Paint Background to match the parent environment
        # Find the GradientPanel and calculate relative position
        # Start with fallback background (use gradient color on Windows to avoid gray)
        if sys.platform.startswith("win"):
            # Use gradient start color as fallback instead of system gray
            fallback_color = wx.Colour(Theme.COLOR_BG_GRADIENT_START)
            dc.SetBackground(wx.Brush(fallback_color))
        else:
            dc.SetBackground(wx.Brush(self.GetParent().GetBackgroundColour()))
        dc.Clear()
        
        try:
            # Walk up to find GradientPanel
            win = self.GetParent()
            glass_parent = None
            gradient_parent = None
            
            # Simple check since we know the structure: GlassPanel -> GradientPanel
            # Or directly GradientPanel
            if hasattr(win, "fill_rgba"): # Identify GlassPanel by attribute
                glass_parent = win
                win = win.GetParent()
                
            # Check for GradientPanel (ScrolledWindow with OnPaint gradient logic)
            # We assume it's the next parent up
            if win:
                gradient_parent = win

            if gradient_parent:
                # Get dimensions
                # Use VirtualSize because gradient covers whole scroll area
                fw, fh = gradient_parent.GetVirtualSize()
                fh = max(fh, 1) # avoid div zero
                
                # Determine absolute logical Y of this control's top and bottom
                # Map control (0,0) to screen, then screen to parent client, then unscored?
                # Easier: Screen positions.
                # NOTE: Gradient is painted on the Virtual canvas.
                # So we need (ChildScreenPos - GradientScreenPos) + ScrollOffset
                
                child_screen = self.GetScreenPosition()
                grad_screen = gradient_parent.GetScreenPosition()
                
                # CalcUnscrolledPosition(0,0) gives the scroll offset in pixels
                scroll_x, scroll_y = gradient_parent.CalcUnscrolledPosition(0, 0)
                
                rel_y_top = (child_screen.y - grad_screen.y) + scroll_y
                rel_y_bot = rel_y_top + self.GetSize().height
                
                # Colors
                c_start = wx.Colour(Theme.COLOR_BG_GRADIENT_START) # Bottom (if NORTH)
                c_end = wx.Colour(Theme.COLOR_BG_GRADIENT_END)     # Top (if NORTH)
                # wx.NORTH means Gradient starts at bottom?
                # "wx.NORTH: The starting color is at the bottom"
                # So y=0 is END, y=H is START.
                
                # Calculate color at top and bottom of this control
                # Linear interp: val = End * (1-pct) + Start * pct
                pct_top = max(0.0, min(1.0, rel_y_top / float(fh)))
                pct_bot = max(0.0, min(1.0, rel_y_bot / float(fh)))
                
                color_top = _mix_colors(c_end, c_start, pct_top)
                color_bot = _mix_colors(c_end, c_start, pct_bot)
                
                # If inside GlassPanel, blend the glass color on top
                if glass_parent:
                    glass_color = wx.Colour(*Theme.COLOR_GLASS_FILL[:3])
                    alpha = Theme.COLOR_GLASS_FILL[3] / 255.0
                    # Composite: Result = Glass * alpha + Gradient * (1-alpha)
                    color_top = _mix_colors(color_top, glass_color, alpha)
                    color_bot = _mix_colors(color_bot, glass_color, alpha)

                # Fill the background with gradient (top to bottom)
                # wx.SOUTH means starting color is at top
                rect = self.GetClientRect()
                dc.GradientFillLinear(rect, color_top, color_bot, wx.SOUTH)

        except Exception as e:
            # Fallback if math fails
            # print(f"bg error: {e}")
            pass

        # 2. Draw Text
        # Use GCDC for nice antialiasing
        try:
            gc = wx.GCDC(dc)
            dc_to_use = gc
        except Exception:
            dc_to_use = dc

        dc_to_use.SetFont(self.GetFont())
        dc_to_use.SetTextForeground(self.GetForegroundColour())
        dc_to_use.SetBackgroundMode(wx.TRANSPARENT)
        
        label = self.GetLabel()
        rect = self.GetClientRect()
        dc_to_use.DrawLabel(label, rect, self.GetWindowStyleFlag())

def create_transparent_text(parent, *args, **kwargs):
    """
    Creates a static text control that supports low-level transparency on Windows.
    On macOS, returns a native wx.StaticText to preserve exact look-and-feel.
    On Windows, returns a custom TransparentStaticText.
    """
    if sys.platform.startswith("win"):
        return TransparentStaticText(parent, *args, **kwargs)
    else:
        return wx.StaticText(parent, *args, **kwargs)

def style_checkbox_for_glass(checkbox):
    """
    Style a checkbox for use inside a GlassPanel on Windows.
    On macOS, native transparency works fine so no changes needed.
    On Windows, we set a dark background to approximate the glass panel color.
    """
    if sys.platform.startswith("win"):
        # Use a dark color that approximates the glass panel's appearance
        # The glass fill is (20, 20, 20, 100) alpha-blended over gradient
        # A dark gray provides a reasonable approximation
        dark_bg = wx.Colour(35, 50, 45)  # Dark teal-ish to blend with gradient
        checkbox.SetBackgroundColour(dark_bg)

class Theme:
    # Colors
    COLOR_BG_GRADIENT_START = '#00695C'  # Dark Teal
    COLOR_BG_GRADIENT_END = '#2E7D32'    # Medium Forest Green
    COLOR_TEXT_WHITE = '#FFFFFF'
    COLOR_TEXT_BLACK = '#000000'
    COLOR_GLASS_FILL = (20, 20, 20, 100)
    COLOR_GLASS_BORDER = (255, 255, 255, 40)
    COLOR_ACCENT_GREEN = '#A5D6A7'
    COLOR_PROGRESS_GREEN = (70, 180, 130)
    COLOR_TOOLTIP_BG = '#1E1E1E'
    COLOR_TOOLTIP_FG = '#F5F5F5'
    COLOR_INFO_ICON_BG = '#666666'
    
    # Fonts
    FONT_FAMILY = wx.FONTFAMILY_SWISS
    FONT_STYLE_NORMAL = wx.FONTSTYLE_NORMAL
    FONT_WEIGHT_NORMAL = wx.FONTWEIGHT_NORMAL
    FONT_WEIGHT_BOLD = wx.FONTWEIGHT_BOLD
    
    @staticmethod
    def get_font(size, bold=False):
        weight = Theme.FONT_WEIGHT_BOLD if bold else Theme.FONT_WEIGHT_NORMAL
        if sys.platform.startswith("win"):
            # Use Segoe UI for a cleaner, modern Windows look
            return wx.Font(size, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, weight, False, "Segoe UI")
        return wx.Font(size, Theme.FONT_FAMILY, Theme.FONT_STYLE_NORMAL, weight)

# --- Utility Functions ---

def ensure_video_playable(input_path, output_path=None):
    """Re-encode to H.264/yuv420p for broad compatibility (optional helper).

    If output_path is None, creates a sibling file with suffix "_fixed.mp4".
    Returns output path on success, else None.
    """
    try:
        if output_path is None:
            base, _ = os.path.splitext(input_path)
            output_path = base + "_fixed.mp4"
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                vcodec='libx264',
                pix_fmt='yuv420p',
                movflags='+faststart',
                acodec='copy'
            )
            .overwrite_output()
            .run()
        )
        return output_path
    except Exception:
        return None


def ensure_ffmpeg_available():
    """Ensure an ffmpeg executable is available for ffmpeg-python.

    Tries system PATH first, then imageio-ffmpeg fallback. If found via imageio,
    prepend its directory to PATH for this process so ffmpeg-python can launch it.
    Returns True if available, else False.
    """
    # System ffmpeg available?
    if shutil.which("ffmpeg"):
        return True
    # Try bundled binary from imageio-ffmpeg
    try:
        import imageio_ffmpeg  # type: ignore
        exe_path = imageio_ffmpeg.get_ffmpeg_exe()
        if exe_path and os.path.exists(exe_path):
            exe_dir = os.path.dirname(exe_path)
            os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")
            return True
    except Exception:
        pass
    return False

def get_files_from_folder(folder_path, extensions):
    """Return a list of full paths for files in folder_path matching extensions."""
    files = []
    if os.path.exists(folder_path):
        for f in os.listdir(folder_path):
            if f.lower().endswith(extensions):
                files.append(os.path.join(folder_path, f))
    return files
