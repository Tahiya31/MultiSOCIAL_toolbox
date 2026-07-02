import os
import re
import shutil
import unicodedata
import wx

# --- Constants & Theme ---
import sys
import subprocess
import ctypes
import wx.lib.stattext as stattext

import runtime_services

_registered_win32_ffmpeg_dll_dirs = set()


def _register_win32_ffmpeg_dll_directory(exe_path):
    """Help torchaudio/pyannote find FFmpeg DLLs on Windows (Python 3.8+ safe DLL loading)."""
    if not exe_path or not sys.platform.startswith("win"):
        return
    bin_dir = os.path.normcase(os.path.abspath(os.path.dirname(exe_path)))
    if bin_dir in _registered_win32_ffmpeg_dll_dirs:
        return
    add_fn = getattr(os, "add_dll_directory", None)
    if not callable(add_fn):
        return
    try:
        add_fn(bin_dir)
        _registered_win32_ffmpeg_dll_dirs.add(bin_dir)
    except (OSError, FileNotFoundError, ValueError):
        pass


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
        if sys.platform.startswith("win"):
            # Set initial background to gradient start color to prevent white flash
            self.SetBackgroundColour(wx.Colour(Theme.COLOR_BG_GRADIENT_START))
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
                # Using screen positions for simplicity.
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

class Theme:
    # Background — desaturated deep teal → slate-green (recedes behind content)
    COLOR_BG_GRADIENT_START = '#152826'  # slate-teal (top)
    COLOR_BG_GRADIENT_END = '#1C2B22'    # slate-green (bottom)
    COLOR_BG_GLOW = (46, 212, 122, 22)   # faint emerald ambient

    # Surfaces — neutral charcoal with a faint green tint
    COLOR_SURFACE = (32, 34, 32, 248)
    COLOR_SURFACE_ELEVATED = (40, 42, 40, 252)
    COLOR_GLASS_FILL = COLOR_SURFACE
    COLOR_INPUT_BG = (24, 26, 24)
    COLOR_GLASS_BORDER = (255, 255, 255, 16)
    COLOR_SURFACE_BORDER = COLOR_GLASS_BORDER
    COLOR_BORDER_HIGHLIGHT = (255, 255, 255, 40)

    # Accent — vivid emerald reserved for primary CTA + focus rings
    COLOR_PRIMARY = '#2ED47A'
    COLOR_PRIMARY_HOVER = '#5CE09A'
    COLOR_PRIMARY_PRESSED = '#22B869'
    COLOR_PRIMARY_GLOW = (46, 212, 122, 90)
    COLOR_SECONDARY = (255, 255, 255, 10)
    COLOR_SECONDARY_HOVER = (255, 255, 255, 18)
    COLOR_SECONDARY_PRESSED = (255, 255, 255, 6)
    COLOR_SECONDARY_BORDER = (255, 255, 255, 28)
    COLOR_BTN_DISABLED_FILL = (42, 44, 42)
    COLOR_BTN_DISABLED_BORDER = (255, 255, 255, 12)
    COLOR_DANGER = '#EF4444'
    COLOR_DANGER_HOVER = '#F87171'
    COLOR_DANGER_PRESSED = '#DC2626'

    # Tabs — neutral track, emerald active pill
    COLOR_TAB_TRACK = (18, 20, 18, 220)
    COLOR_TAB_ACTIVE = (46, 212, 122, 255)
    COLOR_TAB_ACTIVE_BORDER = (46, 212, 122, 140)

    # Text
    COLOR_TEXT_WHITE = '#F3F4F6'
    COLOR_TEXT_BLACK = '#000000'
    COLOR_TEXT_ON_DARK = '#FFFFFF'
    COLOR_TEXT_MUTED = '#9CA3AF'
    COLOR_TEXT_SUBTLE = '#6B7280'
    COLOR_DISABLED = (107, 114, 128, 200)
    COLOR_ACCENT_GREEN = '#2ED47A'
    COLOR_FOCUS_RING = (46, 212, 122, 180)
    # Secondary accent — reserved for help/info affordances only (not actions),
    # so the "?" icons read as a distinct, discoverable category, not a CTA.
    COLOR_ACCENT_INFO = '#38BDF8'        # sky/cyan
    COLOR_ACCENT_INFO_SOFT = (56, 189, 248, 40)

    # Progress
    COLOR_PROGRESS_GREEN = (46, 212, 122)
    COLOR_PROGRESS_FILL = COLOR_PROGRESS_GREEN
    COLOR_PROGRESS_TRACK = (0, 0, 0, 70)
    COLOR_PROGRESS_GLOW = (46, 212, 122, 50)

    # Tooltips / chrome
    COLOR_TOOLTIP_BG = '#1F2120'
    COLOR_TOOLTIP_FG = '#E5E7EB'
    COLOR_INFO_ICON_BG = (255, 255, 255, 14)
    COLOR_INFO_ICON_BORDER = (255, 255, 255, 30)

    # Spacing (logical px; use FromDIP in widgets)
    SPACE_XS = 4
    SPACE_SM = 8
    SPACE_MD = 12
    SPACE_LG = 16
    SPACE_XL = 24
    SPACE_XXL = 32

    # Corner radii
    RADIUS_BUTTON = 10
    RADIUS_CARD = 14
    RADIUS_TAB = 12
    RADIUS_TAB_UNDERLINE = 3
    RADIUS_INPUT = 8

    # Font sizes (pt) — deliberate ramp for a clear, consistent hierarchy
    FONT_DISPLAY = 32   # app name
    FONT_TITLE = 17     # section/screen titles
    FONT_BUTTON = 16    # action buttons
    FONT_HEADING = 15   # primary settings labels
    FONT_BODY = 14      # labels, inputs, status
    FONT_SUBTITLE = 13  # header subtitle
    FONT_CAPTION = 12   # secondary/status sublabels
    FONT_OVERLINE = 11  # uppercase card eyebrow headings

    # Legacy font constants
    FONT_FAMILY = wx.FONTFAMILY_SWISS
    FONT_STYLE_NORMAL = wx.FONTSTYLE_NORMAL
    FONT_WEIGHT_NORMAL = wx.FONTWEIGHT_NORMAL
    FONT_WEIGHT_BOLD = wx.FONTWEIGHT_BOLD

    @staticmethod
    def get_font(size, bold=False):
        weight = Theme.FONT_WEIGHT_BOLD if bold else Theme.FONT_WEIGHT_NORMAL
        if sys.platform.startswith("win"):
            return wx.Font(size, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, weight, False, "Segoe UI")
        if sys.platform == "darwin":
            return wx.Font(size, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, weight, False, "SF Pro Text")
        return wx.Font(size, Theme.FONT_FAMILY, Theme.FONT_STYLE_NORMAL, weight)

    @staticmethod
    def colour(hex_or_rgba):
        """Return wx.Colour from '#RRGGBB' or (r, g, b[, a])."""
        if isinstance(hex_or_rgba, str):
            return wx.Colour(hex_or_rgba)
        if len(hex_or_rgba) == 4:
            return wx.Colour(hex_or_rgba[0], hex_or_rgba[1], hex_or_rgba[2], hex_or_rgba[3])
        return wx.Colour(*hex_or_rgba)


def style_native_input(ctrl):
    """Theme native SpinCtrl / DirPicker for dark card surfaces."""
    try:
        ctrl.SetBackgroundColour(Theme.colour(Theme.COLOR_INPUT_BG))
        ctrl.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_WHITE))
    except Exception:
        pass


def composited_background_colour(window):
    """Best-effort opaque colour of whatever is painted *behind* ``window``.

    Owner-drawn controls (FlatButton, CustomCheckBox, ToggleTabBar, CustomGauge)
    draw rounded shapes with translucent effects. On macOS the uncovered corners
    show the parent through ``BG_STYLE_PAINT``; on Windows that style gives no real
    transparency, so the corners/shadow band render against an uninitialised
    background. Filling the client rect with this colour first removes that
    divergence. Mirrors the gradient+glass sampling used by ``TransparentStaticText``.
    """
    try:
        glass_fills = []
        gradient = None
        win = window.GetParent()
        while win is not None:
            # Only count glass panels that actually paint their fill (chrome=True).
            fill = getattr(win, "fill_rgba", None)
            if fill is not None and getattr(win, "chrome", True):
                glass_fills.append(fill)
            if hasattr(win, "CalcUnscrolledPosition") and hasattr(win, "GetVirtualSize"):
                gradient = win
                break
            win = win.GetParent()

        if gradient is None:
            return wx.Colour(Theme.COLOR_BG_GRADIENT_START)

        _fw, fh = gradient.GetVirtualSize()
        fh = max(fh, 1)
        child_screen = window.GetScreenPosition()
        grad_screen = gradient.GetScreenPosition()
        _sx, scroll_y = gradient.CalcUnscrolledPosition(0, 0)
        rel_y = (child_screen.y - grad_screen.y) + scroll_y + window.GetSize().height / 2.0

        # wx.NORTH gradient: END at top, START at bottom (see TransparentStaticText).
        c_start = wx.Colour(Theme.COLOR_BG_GRADIENT_START)
        c_end = wx.Colour(Theme.COLOR_BG_GRADIENT_END)
        pct = max(0.0, min(1.0, rel_y / float(fh)))
        base = _mix_colors(c_end, c_start, pct)

        # Composite painting glass layers (farthest first, nearest last).
        for fill in reversed(glass_fills):
            rgb = wx.Colour(fill[0], fill[1], fill[2])
            alpha = (fill[3] if len(fill) == 4 else 255) / 255.0
            base = _mix_colors(base, rgb, alpha)
        return base
    except Exception:
        return wx.Colour(Theme.COLOR_BG_GRADIENT_START)

# --- Utility Functions ---

def _mark_executable(path):
    try:
        mode = os.stat(path).st_mode
        os.chmod(path, mode | 0o111)
    except Exception:
        pass


def _bundled_ffmpeg_candidates():
    root = runtime_services.bundle_root()
    if not os.path.isdir(root):
        return []

    candidates = []
    for current_root, _dirs, files in os.walk(root):
        for name in files:
            lower = name.lower()
            if lower == "ffmpeg" or lower.startswith("ffmpeg-") or lower == "ffmpeg.exe":
                candidates.append(os.path.join(current_root, name))
    return candidates


def get_ffmpeg_executable():
    """Return the concrete ffmpeg executable path to use, if available."""
    cached = os.environ.get("MULTISOCIAL_FFMPEG_EXE")
    if cached and os.path.exists(cached):
        _register_win32_ffmpeg_dll_directory(cached)
        return cached

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        os.environ["MULTISOCIAL_FFMPEG_SOURCE"] = "system"
        os.environ["MULTISOCIAL_FFMPEG_EXE"] = system_ffmpeg
        _register_win32_ffmpeg_dll_directory(system_ffmpeg)
        return system_ffmpeg

    for candidate in _bundled_ffmpeg_candidates():
        if os.path.exists(candidate):
            _mark_executable(candidate)
            os.environ["PATH"] = os.path.dirname(candidate) + os.pathsep + os.environ.get("PATH", "")
            os.environ["MULTISOCIAL_FFMPEG_SOURCE"] = "bundled"
            os.environ["MULTISOCIAL_FFMPEG_EXE"] = candidate
            _register_win32_ffmpeg_dll_directory(candidate)
            return candidate

    try:
        import imageio_ffmpeg  # type: ignore

        exe_path = imageio_ffmpeg.get_ffmpeg_exe()
        if exe_path and os.path.exists(exe_path):
            _mark_executable(exe_path)
            os.environ["PATH"] = os.path.dirname(exe_path) + os.pathsep + os.environ.get("PATH", "")
            os.environ["MULTISOCIAL_FFMPEG_SOURCE"] = "bundled"
            os.environ["MULTISOCIAL_FFMPEG_EXE"] = exe_path
            _register_win32_ffmpeg_dll_directory(exe_path)
            return exe_path
    except Exception:
        pass

    return None


def ensure_ffmpeg_available():
    """Ensure an ffmpeg executable is available for CLI-based video/audio work.

    Tries system PATH first, then imageio-ffmpeg fallback. If found via imageio,
    prepend its directory to PATH for this process so subprocess-based ffmpeg
    calls and native libraries can find it.
    Returns True if available, else False.
    """
    if get_ffmpeg_executable():
        return True
    os.environ["MULTISOCIAL_FFMPEG_SOURCE"] = "missing"
    return False


def _ffmpeg_has_subtitles_filter(exe):
    """True if the given ffmpeg supports the libass-based ``subtitles`` filter."""
    try:
        out = subprocess.run(
            [exe, "-hide_banner", "-filters"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=20,
        ).stdout or ""
    except Exception:
        return False
    return any(line.split()[1:2] == ["subtitles"] for line in out.splitlines() if line.strip())


def get_subtitle_capable_ffmpeg():
    """Return an ffmpeg executable that supports subtitle burn-in (libass), or None.

    ``get_ffmpeg_executable`` prefers system ffmpeg, which on some machines lacks
    the ``subtitles`` filter. Fall back to the bundled imageio-ffmpeg build, which
    ships with libass.
    """
    cached = os.environ.get("MULTISOCIAL_FFMPEG_SUBS_EXE")
    if cached and os.path.exists(cached):
        return cached

    primary = get_ffmpeg_executable()
    if primary and _ffmpeg_has_subtitles_filter(primary):
        os.environ["MULTISOCIAL_FFMPEG_SUBS_EXE"] = primary
        return primary

    try:
        import imageio_ffmpeg  # type: ignore

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and os.path.exists(bundled) and _ffmpeg_has_subtitles_filter(bundled):
            _mark_executable(bundled)
            os.environ["MULTISOCIAL_FFMPEG_SUBS_EXE"] = bundled
            return bundled
    except Exception:
        pass

    return None


_FFMPEG_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_FFMPEG_TIME_RE = re.compile(r"time=\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _hms_to_seconds(h, m, s):
    return int(h) * 3600 + int(m) * 60 + float(s)


def run_ffmpeg_with_progress(cmd, progress_callback=None, cancel_check=None, cwd=None):
    """Run an ffmpeg command, reporting real progress parsed from its stderr.

    ffmpeg prints the input ``Duration`` once and a running ``time=`` per update;
    the ratio gives true progress (no ffprobe needed). Shared by any ffmpeg step
    that wants a live bar.

    Returns:
        True on success, False if cancelled via ``cancel_check``.
    Raises:
        RuntimeError: if ffmpeg exits non-zero.
    """
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        universal_newlines=True, bufsize=1,
    )
    total_seconds = None
    stderr_tail = []
    last_pct = -1
    try:
        for line in proc.stderr:
            stderr_tail.append(line)
            if len(stderr_tail) > 40:
                stderr_tail.pop(0)

            if cancel_check and cancel_check():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return False

            if total_seconds is None:
                m = _FFMPEG_DURATION_RE.search(line)
                if m:
                    total_seconds = _hms_to_seconds(*m.groups())

            if progress_callback and total_seconds:
                t = _FFMPEG_TIME_RE.search(line)
                if t:
                    elapsed = _hms_to_seconds(*t.groups())
                    pct = max(0, min(99, int((elapsed / total_seconds) * 100)))
                    if pct != last_pct:
                        progress_callback(pct)
                        last_pct = pct
    finally:
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {proc.returncode}).\n" + "".join(stderr_tail).strip()
        )
    if progress_callback:
        progress_callback(100)
    return True


def normalize_path(path):
    """Normalize a filesystem path from the UI or runtime environment."""
    if path is None:
        return ""
    normalized = str(path).strip().strip('"').strip("'")
    if not normalized:
        return ""
    normalized = unicodedata.normalize("NFC", normalized)
    normalized = os.path.expandvars(os.path.expanduser(normalized))
    return os.path.normpath(os.path.abspath(normalized))


def resolved_dataset_root(folder_path):
    """Folder that owns pose/video outputs (parent of ``converted_audio`` when that subfolder is selected)."""
    fp = normalize_path(folder_path)
    if not fp:
        return fp
    base = os.path.basename(fp.rstrip(os.sep))
    if base.lower() == "converted_audio":
        parent = os.path.dirname(fp)
        return parent if parent else fp
    return fp


def resolved_converted_audio_folder(folder_path):
    """Directory where converted WAV files live (always ``…/converted_audio`` for a dataset root)."""
    fp = normalize_path(folder_path)
    if not fp:
        return fp
    if os.path.basename(fp.rstrip(os.sep)).lower() == "converted_audio":
        return fp
    return os.path.join(resolved_dataset_root(fp), "converted_audio")


def transcripts_output_folder(folder_path):
    """Transcript text files live at the dataset root beside ``audio_features``."""
    return os.path.join(resolved_dataset_root(folder_path), "transcripts")


def get_audio_files_for_processing(folder_path, extensions):
    """Audio files from dataset root and ``converted_audio``.

    If both locations contain the same basename, prefer the converted copy so
    generated outputs named by basename cannot overwrite/cross-pair.
    """
    dr = resolved_dataset_root(folder_path)
    cad = resolved_converted_audio_folder(folder_path)
    by_name = {}
    for root in (dr, cad):
        if root and os.path.isdir(root):
            for fpath in get_files_from_folder(root, extensions):
                by_name[os.path.basename(fpath).lower()] = fpath
    return sorted(by_name.values())


def get_files_from_folder(folder_path, extensions):
    """Return a list of full paths for files in folder_path matching extensions."""
    files = []
    folder_path = normalize_path(folder_path)
    if os.path.isdir(folder_path):
        for entry in os.scandir(folder_path):
            if entry.name.startswith('.'):
                continue
            if entry.is_file() and entry.name.lower().endswith(extensions):
                files.append(entry.path)
    files.sort()
    return files
