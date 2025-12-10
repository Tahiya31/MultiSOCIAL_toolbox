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

class TransparentStaticText(stattext.GenStaticText):
    """
    A StaticText that strictly prevents background painting to ensure transparency
    over complex backgrounds like gradients.
    """
    def __init__(self, *args, **kwargs):
        super(TransparentStaticText, self).__init__(*args, **kwargs)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_ERASE_BACKGROUND, lambda e: None)

    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        self.DoPrepareDC(dc)
        
        # Try to create a GCDC for smoother text, fall back to standard DC
        try:
            gc = wx.GCDC(dc)
            dc_to_use = gc
        except Exception:
            dc_to_use = dc

        # Setup font and text color
        dc_to_use.SetFont(self.GetFont())
        dc_to_use.SetTextForeground(self.GetForegroundColour())
        dc_to_use.SetBackgroundMode(wx.TRANSPARENT)
        
        # Draw text centered vertically/horizontally depending on style?
        # GenStaticText has internal logic for alignment, but here we can keep it simple 
        # or reuse some of GenericStaticText's helpers if accessible.
        # For this specific app, most labels are centered or simple. 
        # Using DrawLabel handles alignment flags (wx.ALIGN_CENTER, etc).
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
