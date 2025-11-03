'''
This is the main script for multisocial app

'''


# Import necessary system and utility modules
import os
import threading


# Third-party libraries (assumed pre-installed via requirements.txt)
import ffmpeg
import wx
import shutil

# Import the core processing classes
from pose import PoseProcessor
from audio import AudioProcessor

# Set up GPU environment specially for Mediapipe (specific for Saturn Cloud), if you use some other high performance computing platform check compatibility before usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Make sure the system uses the GPU


## All dependencies are expected to be installed ahead of time via requirements.txt

# (Optional) Helper for Windows FFmpeg setup was removed to avoid runtime installs

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


class GradientPanel(wx.Panel):
    def __init__(self, parent):
        super(GradientPanel, self).__init__(parent)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event):
        dc = wx.AutoBufferedPaintDC(self)
        rect = self.GetClientRect()
        dc.GradientFillLinear(rect, '#00695C', '#2E7D32', wx.NORTH)  # Dark Teal to Medium Forest Green


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
        padding = 6
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
        self.SetBackgroundColour('#D3D3D3')  # Light gray to match buttons
        self.SetForegroundColour('#FFFFFF')  # White text to match buttons
        
        # Create a panel for the tooltip content
        panel = wx.Panel(self)
        panel.SetBackgroundColour('#D3D3D3')  # Light gray background
        
        # Create text control for the tooltip with word wrapping
        self.text_ctrl = wx.StaticText(panel, label=text, style=wx.ALIGN_LEFT)
        self.text_ctrl.SetFont(wx.Font(10, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.text_ctrl.SetForegroundColour('#FFFFFF')  # White text
        self.text_ctrl.Wrap(300)  # Wrap text at 300 pixels width
        
        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.text_ctrl, flag=wx.ALL, border=8)
        panel.SetSizer(sizer)
        
        # Size the tooltip
        panel.Fit()
        self.SetSize(panel.GetSize())


class InfoIcon(wx.StaticText):
    def __init__(self, parent, tooltip_text):
        super(InfoIcon, self).__init__(parent, label="ℹ", style=wx.ALIGN_CENTER)
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)
        self.tooltip_text = tooltip_text
        self.tooltip = None
        self._top_level = self.GetTopLevelParent()
        
        # Style the info icon
        self.SetFont(wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        self.SetForegroundColour('#FFFFFF')
        self.SetBackgroundColour('#666666')
        self.SetMinSize((20, 20))
        
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
        dc.SetBrush(wx.Brush('#666666'))
        dc.SetPen(wx.Pen('#666666', 1))
        dc.DrawCircle(rect.width//2, rect.height//2, min(rect.width, rect.height)//2 - 1)
        
        # Draw the "i" text
        dc.SetTextForeground('#FFFFFF')
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


class VideoToWavConverter(wx.Frame):
    def __init__(self, *args, **kw):
        super(VideoToWavConverter, self).__init__(*args, **kw)
        
        # Start at designed size; prevent shrinking below baseline
        self._baseline_size = None  # will be captured on first resize to drive responsive scaling
        
        pnl = GradientPanel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        # (removed top status label to avoid duplication)
        
        # Add extra space above the title
        vbox.Add((0, 30))  # Add a 30-pixel high spacer, adjust as needed
		
		  
        new_width, new_height = wx.GetDisplaySize()
        logo_size = min(int(new_height * 0.08), 150)
        
        # Top layout for logo and title
        top_box = wx.BoxSizer(wx.HORIZONTAL)
        
        # Logo image with elevated presentation
        logo_path = "MultiSOCIAL_logo.png"  # Path to your logo image
        logo_image = wx.Image(logo_path, wx.BITMAP_TYPE_ANY)
        logo_image = logo_image.Scale(logo_size, logo_size, wx.IMAGE_QUALITY_HIGH)
        logo_bmp = wx.Bitmap(logo_image)
        
        # Create elevated logo panel with shadow and rounded corners
        elevated_logo = ElevatedLogoPanel(pnl, logo_bmp)
        elevated_logo.SetMinSize((logo_size + 20, logo_size + 20))  # Add padding around logo
        
        top_box.AddStretchSpacer(1)
        top_box.Add(elevated_logo, flag=wx.ALIGN_CENTER | wx.ALL, border=15)
        top_box.AddStretchSpacer(1)

        vbox.Add(top_box, flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=10)

        # GitHub link
        #github_link = hl.HyperlinkCtrl(pnl, id=wx.ID_ANY, label="GitHub", url="https://github.com")
        #github_icon_path = "github_icon.png"  # Path to your GitHub icon image
        #github_bmp = wx.Bitmap(github_icon_path, wx.BITMAP_TYPE_ANY)
        #github_bitmap = wx.StaticBitmap(pnl, bitmap=github_bmp)

        # Horizontal box sizer to arrange logo and GitHub link
        #hbox = wx.BoxSizer(wx.HORIZONTAL)
        #hbox.AddStretchSpacer(1) 
        #hbox.Add(logo_bitmap, flag=wx.ALL|wx.ALIGN_LEFT, border=5)
        #hbox.AddStretchSpacer(1)  # Add stretchable space between logo and GitHub link
        #vbox.Add(hbox, proportion=1, flag=wx.ALIGN_CENTER|wx.ALL, border=10)
        
        #hbox.Add(github_link, flag=wx.ALL|wx.ALIGN_RIGHT, border=5)
        #hbox.Add(github_bitmap, flag=wx.ALL|wx.ALIGN_RIGHT, border=5)
        
        
        # Title
        self.title = wx.StaticText(pnl, label="Welcome to", style=wx.ALIGN_CENTER)
        title_font = wx.Font(20, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.title.SetFont(title_font)
        self.title.SetForegroundColour('#FFFFFF')
        vbox.Add(self.title, flag=wx.ALIGN_CENTER | wx.BOTTOM, border=10)

        # Logo
        self.logoLabel = wx.StaticText(pnl, label="MultiSOCIAL Toolbox", style=wx.ALIGN_CENTER)
        font = wx.Font(24, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        self.logoLabel.SetFont(font)
        self.logoLabel.SetForegroundColour('#FFFFFF')

        vbox.Add(self.logoLabel, flag=wx.ALIGN_CENTER|wx.TOP, border=5)  # Adjusted the top border

        # File Picker
        #self.filePicker = wx.FilePickerCtrl(pnl, message="Select a video or an audio file", wildcard="Video files (*.mp4;*.avi;*.mov;*.mkv)|*.mp4;*.avi;*.mov;*.mkv|WAV files (*.wav)|*.wav")
        #vbox.Add(self.filePicker, flag=wx.EXPAND|wx.ALL, border=14)
        # Folder Picker
        self.folderPicker = wx.DirPickerCtrl(pnl, message="Select a folder containing media files")
        vbox.Add(self.folderPicker, flag=wx.EXPAND | wx.ALL, border=10,proportion=0)

        # Placeholder above buttons
        self.placeholderVideoLabel = wx.StaticText(pnl, label="If you have a video file:")
        placeholder_above_font = wx.Font(20, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.placeholderVideoLabel.SetFont(placeholder_above_font)
        vbox.Add(self.placeholderVideoLabel, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        # Toggle for multi-person pose (video option)
        self.multiPersonCheckbox = wx.CheckBox(pnl, label="Enable Multi-Person Pose")
        self.multiPersonCheckbox.SetFont(wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.multiPersonCheckbox.SetForegroundColour('#FFFFFF')
        vbox.Add(self.multiPersonCheckbox, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        # Downsampling controls
        ds_box = wx.BoxSizer(wx.HORIZONTAL)
        # Frame stride
        ds_label = wx.StaticText(pnl, label="Process every k-th frame:")
        ds_label.SetFont(wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        ds_label.SetForegroundColour('#FFFFFF')
        ds_box.Add(ds_label, flag=wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border=8)

        self.frameStrideInput = wx.SpinCtrl(pnl, value="1", min=1, max=10)
        self.frameStrideInput.SetFont(wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        ds_box.Add(self.frameStrideInput, flag=wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border=20)

        # Resolution downscale checkbox (fixed 720p)
        self.downscaleCheckbox = wx.CheckBox(pnl, label="Downscale to 720p for processing")
        self.downscaleCheckbox.SetFont(wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        self.downscaleCheckbox.SetForegroundColour('#FFFFFF')
        ds_box.Add(self.downscaleCheckbox, flag=wx.ALIGN_CENTER_VERTICAL)

        vbox.Add(ds_box, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        # Frame threshold input for bounding box recalibration
        frame_threshold_box = wx.BoxSizer(wx.HORIZONTAL)
        frame_threshold_label = wx.StaticText(pnl, label="Frame Threshold for Bounding Box Recalibration:")
        frame_threshold_label.SetFont(wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        frame_threshold_label.SetForegroundColour('#FFFFFF')
        frame_threshold_box.Add(frame_threshold_label, flag=wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border=10)
        
        self.frameThresholdInput = wx.SpinCtrl(pnl, value="10", min=1, max=100)
        self.frameThresholdInput.SetFont(wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        frame_threshold_box.Add(self.frameThresholdInput, flag=wx.ALIGN_CENTER_VERTICAL)
        
        vbox.Add(frame_threshold_box, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        # Button Font
        button_font = wx.Font(16, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        placeholder_font = wx.Font(20, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)

        # Convert Button with Placeholder
        hbox_convert = wx.BoxSizer(wx.HORIZONTAL)
        #placeholder_convert = wx.StaticText(pnl, label="If you have a video file:")
        #placeholder_convert.SetFont(placeholder_font)
        #hbox_convert.Add(placeholder_convert, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        self.convertBtn = TooltipButton(pnl, 'Convert video to audio', 
                                       'Converts video files (.mp4, .avi, .mov, .mkv) to audio files (.wav) for further processing')
        self.convertBtn.SetFont(button_font)
        self.convertBtn.Bind(wx.EVT_BUTTON, self.on_convert)
        hbox_convert.Add(self.convertBtn, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        # Add info icon
        info_icon = self.convertBtn.add_info_icon(pnl)
        hbox_convert.Add(info_icon, flag=wx.ALIGN_CENTER|wx.LEFT, border=5)
        
        vbox.Add(hbox_convert, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        # Extract Pose Features Button with Placeholder
        hbox_extract_pose = wx.BoxSizer(wx.HORIZONTAL)
        #placeholder_pose = wx.StaticText(pnl, label="To extract pose features:")
        #placeholder_pose.SetFont(placeholder_font)
        #hbox_extract_pose.Add(placeholder_pose, flag=wx.ALIGN_CENTER|wx.ALL, border=12)

        self.extractFeaturesBtn = TooltipButton(pnl, 'Extract Pose Features', 
                                               'Extracts human pose landmarks and features from video files using MediaPipe. Supports single and multi-person detection.')
        self.extractFeaturesBtn.SetFont(button_font)
        self.extractFeaturesBtn.Bind(wx.EVT_BUTTON, self.on_extract_features)
        hbox_extract_pose.Add(self.extractFeaturesBtn, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        # Add info icon
        info_icon = self.extractFeaturesBtn.add_info_icon(pnl)
        hbox_extract_pose.Add(info_icon, flag=wx.ALIGN_CENTER|wx.LEFT, border=5)

        vbox.Add(hbox_extract_pose, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        # Embed Pose Features Button with Placeholder
        hbox_embed_pose = wx.BoxSizer(wx.HORIZONTAL)
        #placeholder_pose = wx.StaticText(pnl, label="To embed pose features:")
        #placeholder_pose.SetFont(placeholder_font)
        #hbox_extract_pose.Add(placeholder_pose, flag=wx.ALIGN_CENTER|wx.ALL, border=12)

        self.embedFeaturesBtn = TooltipButton(pnl, 'Embed Pose Features', 
                                             'Creates vector embeddings from extracted pose features for machine learning and analysis purposes.')
        self.embedFeaturesBtn.SetFont(button_font)
        self.embedFeaturesBtn.Bind(wx.EVT_BUTTON, self.on_embed_poses)
        hbox_embed_pose.Add(self.embedFeaturesBtn, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        # Add info icon
        info_icon = self.embedFeaturesBtn.add_info_icon(pnl)
        hbox_embed_pose.Add(info_icon, flag=wx.ALIGN_CENTER|wx.LEFT, border=5)

        vbox.Add(hbox_embed_pose, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        # Verify Consistency Button
        hbox_verify = wx.BoxSizer(wx.HORIZONTAL)
        self.verifyBtn = TooltipButton(pnl, 'Verify Consistency', 'Verifies that embedded videos match extracted pose CSVs and saves a report with worst-frame thumbnails.')
        self.verifyBtn.SetFont(button_font)
        self.verifyBtn.Bind(wx.EVT_BUTTON, self.on_verify_consistency)
        hbox_verify.Add(self.verifyBtn, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        info_icon = self.verifyBtn.add_info_icon(pnl)
        hbox_verify.Add(info_icon, flag=wx.ALIGN_CENTER|wx.LEFT, border=5)
        vbox.Add(hbox_verify, flag=wx.ALIGN_CENTER|wx.ALL, border=12)

        # Placeholder middle
        self.placeholderAudioLabel = wx.StaticText(pnl, label="If you have an audio file:")
        self.placeholderAudioLabel.SetFont(placeholder_font)
        vbox.Add(self.placeholderAudioLabel, flag=wx.ALIGN_CENTER|wx.ALL, border=12)

        # Extract Audio Features Button with Placeholder
        hbox_extract_audio = wx.BoxSizer(wx.HORIZONTAL)
        #placeholder_audio = wx.StaticText(pnl, label="To extract audio features:")
        #placeholder_audio.SetFont(placeholder_font)
        #hbox_extract_audio.Add(placeholder_audio, flag=wx.ALIGN_CENTER|wx.ALL, border=12)

        self.extractAudioFeaturesBtn = TooltipButton(pnl, 'Extract Audio Features', 
                                                    'Extracts acoustic features from audio files (.wav) including MFCC, spectral features, and prosodic characteristics.')
        self.extractAudioFeaturesBtn.SetFont(button_font)
        self.extractAudioFeaturesBtn.Bind(wx.EVT_BUTTON, self.on_extract_audio_features)
        hbox_extract_audio.Add(self.extractAudioFeaturesBtn, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        # Add info icon
        info_icon = self.extractAudioFeaturesBtn.add_info_icon(pnl)
        hbox_extract_audio.Add(info_icon, flag=wx.ALIGN_CENTER|wx.LEFT, border=5)

        vbox.Add(hbox_extract_audio, flag=wx.ALIGN_CENTER|wx.ALL, border=12)

        # Extract Transcripts Button with Placeholder
        hbox_extract_transcripts = wx.BoxSizer(wx.HORIZONTAL)
        #placeholder_transcripts = wx.StaticText(pnl, label="To extract transcripts:")
        #placeholder_transcripts.SetFont(placeholder_font)
        #hbox_extract_transcripts.Add(placeholder_transcripts, flag=wx.ALIGN_CENTER|wx.ALL, border=12)

        self.extractTranscriptsBtn = TooltipButton(pnl, 'Extract Transcripts', 
                                                  'Converts speech in audio files (.wav) to text transcripts using automatic speech recognition (ASR) technology.')
        self.extractTranscriptsBtn.SetFont(button_font)
        self.extractTranscriptsBtn.Bind(wx.EVT_BUTTON, self.on_extract_transcripts)
        hbox_extract_transcripts.Add(self.extractTranscriptsBtn, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        # Add info icon
        info_icon = self.extractTranscriptsBtn.add_info_icon(pnl)
        hbox_extract_transcripts.Add(info_icon, flag=wx.ALIGN_CENTER|wx.LEFT, border=5)

        vbox.Add(hbox_extract_transcripts, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        
        #status update (bottom, red)
        self.statusLabel = wx.StaticText(pnl, label="", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.statusLabel.SetForegroundColour('#800000')
        vbox.Add(self.statusLabel, flag=wx.EXPAND|wx.ALL, border=10)

        # Progress Bar
        self.progress = wx.Gauge(pnl, range=100, style=wx.GA_HORIZONTAL)
        vbox.Add(self.progress, proportion=1, flag=wx.EXPAND|wx.ALL, border=12)
        
        pnl.SetSizer(vbox)
        pnl.Layout()
        
        # Capture a design baseline size from the layout's minimum so we scale relative to intended UI, not the maximized size
        best_min = vbox.CalcMin()
        # Ensure we have a sensible floor similar to initially set size
        self._baseline_size = (max(400, best_min.width), max(800, best_min.height))
        self.SetMinSize(self._baseline_size)
        
        # Bind resize handler to make the UI responsive
        self.Bind(wx.EVT_SIZE, self.on_resize)
        
        self.SetSize(self._baseline_size)
        self.SetTitle('MultiSOCIAL Toolbox')
        self.Centre()
        
    def _scale_font(self, base_point_size, scale):
        new_size = max(9, int(round(base_point_size * scale)))
        return wx.Font(new_size, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)

    def _scale_bold_font(self, base_point_size, scale):
        new_size = max(10, int(round(base_point_size * scale)))
        return wx.Font(new_size, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)

    def on_resize(self, event):
        # Establish baseline once
        if self._baseline_size is None:
            self._baseline_size = self.GetSize()
        cur_w, cur_h = self.GetSize()
        base_w, base_h = self._baseline_size
        # Compute scale preserving aspect
        # Do not scale down below baseline; allow growth only and use scrollbars when smaller
        scale_w = max(cur_w / max(1, base_w), 1.0)
        scale_h = max(cur_h / max(1, base_h), 1.0)
        scale = min(scale_w, scale_h, 2.0)

        # Scale title and logo fonts
        try:
            self.title.SetFont(self._scale_font(20, scale))
            self.logoLabel.SetFont(self._scale_bold_font(24, scale))
            self.placeholderVideoLabel.SetFont(self._scale_font(20, scale))
            self.placeholderAudioLabel.SetFont(self._scale_font(20, scale))
            self.multiPersonCheckbox.SetFont(self._scale_font(14, scale))
            if hasattr(self, 'frameStrideInput'):
                self.frameStrideInput.SetFont(self._scale_font(12, scale))
            if hasattr(self, 'downscaleCheckbox'):
                self.downscaleCheckbox.SetFont(self._scale_font(12, scale))
            # Frame threshold elements
            if hasattr(self, 'frameThresholdInput'):
                self.frameThresholdInput.SetFont(self._scale_font(12, scale))
            # Buttons
            button_font = self._scale_font(16, scale)
            self.convertBtn.SetFont(button_font)
            self.extractFeaturesBtn.SetFont(button_font)
            self.embedFeaturesBtn.SetFont(button_font)
            self.extractAudioFeaturesBtn.SetFont(button_font)
            self.extractTranscriptsBtn.SetFont(button_font)
            # Status label: scale font and wrap to panel width for responsiveness
            if hasattr(self, 'statusLabel') and self.statusLabel:
                self.statusLabel.SetFont(self._scale_font(12, scale))
                try:
                    wrap_width = max(200, int(cur_w * 0.9))
                    self.statusLabel.Wrap(wrap_width)
                    # Re-apply centering after Wrap, which can reset alignment
                    self.statusLabel.SetWindowStyleFlag(
                        self.statusLabel.GetWindowStyle() | wx.ALIGN_CENTER_HORIZONTAL
                    )
                except Exception:
                    pass
            # Progress bar height scaling
            if hasattr(self, 'progress') and self.progress:
                self.progress.SetMinSize((-1, max(14, int(18 * scale))))
        except Exception:
            pass

        # Relayout after scaling
        self.Layout()
        event.Skip()

    def set_status_message(self, message):
        """Safely update the status label from any thread."""
        if hasattr(self, 'statusLabel'):
            wx.CallAfter(self.statusLabel.SetLabel, message)


    def update_progress(self, value):
        """Update the progress bar."""
        wx.CallAfter(self.progress.SetValue, value)


    def ensure_output_folders(self, folder_path):
        """Ensures output directories exist inside the selected folder only when needed."""
    
        # Define folder paths
        self.converted_audio_folder = os.path.join(folder_path, "converted_audio")
        self.extracted_pose_folder = os.path.join(folder_path, "pose_features")
        self.embedded_pose_folder = os.path.join(folder_path, "embedded_pose")
        self.extracted_audio_folder = os.path.join(folder_path, "audio_features")
        self.extracted_transcripts_folder = os.path.join(folder_path, "transcripts")

        # Check if the selected folder has video files
        video_files = self.get_files_from_folder(folder_path, (".mp4", ".avi", ".mov"))
        if video_files:
            for folder in [self.converted_audio_folder, self.extracted_pose_folder, self.embedded_pose_folder]:
                os.makedirs(folder, exist_ok=True)
        else:
            self.converted_audio_folder = None
            self.extracted_pose_folder = None
            self.embedded_pose_folder = None

        # Check if the selected folder has audio files
        audio_files = self.get_files_from_folder(folder_path, (".wav",))
        if audio_files:
            for folder in [self.extracted_audio_folder, self.extracted_transcripts_folder]:
                os.makedirs(folder, exist_ok=True)
        else:
            self.extracted_audio_folder = None
            self.extracted_transcripts_folder = None

            
    def get_files_from_folder(self, folder_path, extensions):
        """Retrieve files with the specified extensions from the folder."""
        return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extensions)]
      
      
    def on_convert(self, event):
        """Convert all videos in a selected folder to WAV."""
        folder_path = self.folderPicker.GetPath()
        if not folder_path:
            wx.MessageBox("Please select a folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.ensure_output_folders(folder_path)
        video_files = self.get_files_from_folder(folder_path, (".mp4", ".avi", ".mov", ".mkv"))

        if not video_files:
            wx.MessageBox("No video files found in the selected folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Process each video file in a separate thread
        thread = threading.Thread(target=self.convert_all_videos_to_wav, args=(video_files,))
        thread.start()

    def convert_all_videos_to_wav(self, video_files):
        """Convert multiple videos to WAV format and save to output folder."""
        total_files = len(video_files)
    
        for i, video_file in enumerate(video_files):
            #print(f"Converting: {video_file}")
            file_name = os.path.basename(video_file)
            self.set_status_message(f"Converting to WAV: {file_name}")
            
            # Create progress callback for this video
            def make_progress_callback(video_index, total_videos):
                def progress_callback(conversion_progress):
                    # Calculate overall progress: (video_index-1)/total_videos + conversion_progress/total_videos
                    overall_progress = int(((video_index - 1) / total_videos) * 100 + (conversion_progress / total_videos))
                    self.update_progress(overall_progress)
                return progress_callback
            
            self.convert_to_wav(video_file, progress_callback=make_progress_callback(i + 1, total_files))

        wx.MessageBox("Video to audio conversion completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        self.update_progress(0)  # Reset progress bar

    def on_verify_consistency(self, event):
        folder_path = self.folderPicker.GetPath()
        if not folder_path:
            wx.MessageBox("Please select a folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Ensure folders and also create verification output dirs
        self.ensure_output_folders(folder_path)
        if not self.extracted_pose_folder or not self.embedded_pose_folder:
            wx.MessageBox("No pose or embedded outputs found. Extract and embed first.", "Error", wx.OK | wx.ICON_ERROR)
            return

        verification_dir = os.path.join(folder_path, "verification")
        os.makedirs(verification_dir, exist_ok=True)
        worst_frames_root = os.path.join(verification_dir, "worst_frames")
        os.makedirs(worst_frames_root, exist_ok=True)

        # Collect embedded videos and match CSVs by basename
        embedded_videos = self.get_files_from_folder(self.embedded_pose_folder, (".mp4", ".avi", ".mov"))
        if not embedded_videos:
            wx.MessageBox("No embedded videos found in embedded_pose/", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Launch background verification to keep UI responsive
        thread = threading.Thread(target=self._verify_consistency_batch, args=(embedded_videos, verification_dir, worst_frames_root))
        thread.start()

    def _verify_consistency_batch(self, embedded_videos, verification_dir, worst_frames_root):
        from verify_pose_embedding import verify, save_report
        import glob

        summary = []
        total = len(embedded_videos)
        for i, video in enumerate(embedded_videos, start=1):
            base = os.path.splitext(os.path.basename(video))[0]
            # Remove trailing "_pose" if present to get CSV base
            csv_base = base.replace("_pose", "")
            csv_pattern = os.path.join(self.extracted_pose_folder, f"{csv_base}*_ID_*.csv")
            csv_paths = sorted(glob.glob(csv_pattern))
            if not csv_paths:
                # Try single-person default
                single_csv = os.path.join(self.extracted_pose_folder, f"{csv_base}_ID_0.csv")
                if os.path.exists(single_csv):
                    csv_paths = [single_csv]

            if not csv_paths:
                self.set_status_message(f"⚠️ No CSVs found for {base}; skipping")
                continue

            self.set_status_message(f"🔎 Verifying: {os.path.basename(video)}")
            try:
                worst_dir = os.path.join(worst_frames_root, base)
                report = verify(
                    video_path=video,
                    csv_paths=csv_paths,
                    stride=max(1, int(self.frameStrideInput.GetValue())),
                    max_worst=10,
                    worst_dir=worst_dir,
                    processed_only=True,
                    metric='both',
                    hit_threshold=0.8,
                    ssim_threshold=0.98,
                    window=5,
                    conf_threshold=0.0,
                )
                out_json = os.path.join(verification_dir, f"{base}_report.json")
                out_csv = os.path.join(verification_dir, f"{base}_worst.csv")
                save_report(report, out_json, out_csv)
                summary.append({
                    "basename": base,
                    "frames_compared": report.get("frames_compared"),
                    "mean_hit_rate": report.get("mean_hit_rate"),
                    "min_hit_rate": report.get("min_hit_rate"),
                    "mean_ssim": report.get("mean_ssim"),
                    "min_ssim": report.get("min_ssim"),
                })
            except Exception as e:
                self.set_status_message(f"❌ Verification failed for {base}: {e}")

            # Progress update
            overall = int((i / max(1, total)) * 100)
            self.update_progress(overall)

        # Write summary JSON
        try:
            import json
            summary_path = os.path.join(verification_dir, "summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception:
            pass

        wx.CallAfter(wx.MessageBox, "Verification completed! See the 'verification' folder for reports.", "Success", wx.OK | wx.ICON_INFORMATION)
        self.update_progress(0)

    def convert_to_wav(self, filepath, progress_callback=None):
        """Convert a single video file to WAV using ffmpeg."""
        try:
            output_path = os.path.join(self.converted_audio_folder, os.path.splitext(os.path.basename(filepath))[0] + ".wav")
        
            # Debugging messages
            print(f"Processing video: {filepath}")
            print(f"Saving output as: {output_path}")
        
            # Run ffmpeg conversion with progress tracking
            if progress_callback:
                # Simulate progress for ffmpeg conversion (since ffmpeg doesn't provide real-time progress easily)
                import time
                progress_callback(0)
                time.sleep(0.1)  # Small delay to show progress start
                progress_callback(50)
            
            (
                ffmpeg
                .input(filepath)
                .output(output_path, format='wav', acodec='pcm_s16le')
                .run(overwrite_output=True)
            )
            
            if progress_callback:
                progress_callback(100)

            print(f"Conversion complete: {output_path}")

        except Exception as e:
            wx.MessageBox(f'Error converting {filepath}: {e}', 'Error', wx.OK | wx.ICON_ERROR)


    def on_extract_features(self, event):
        folder_path = self.folderPicker.GetPath()
        if not folder_path:
            wx.MessageBox("Please select a folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.ensure_output_folders(folder_path)
        video_files = self.get_files_from_folder(folder_path, (".mp4", ".avi"))
        
        if not video_files:
            wx.MessageBox("No video files found in the selected folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Run in a separate thread to keep UI responsive
        thread = threading.Thread(target=self.extract_pose_features_batch, args=(video_files,))
        thread.start()

    def extract_pose_features_batch(self, video_files):
        """Batch process all video files to extract pose features."""
        total_files = len(video_files)
        
        # Read processing options
        stride_val = 1
        try:
            stride_val = max(1, int(self.frameStrideInput.GetValue()))
        except Exception:
            stride_val = 1
        downscale_to = (1280, 720) if (hasattr(self, 'downscaleCheckbox') and self.downscaleCheckbox.GetValue()) else None

        pose_processor = PoseProcessor(self.extracted_pose_folder, status_callback=self.set_status_message, frame_threshold=self.frameThresholdInput.GetValue(), frame_stride=stride_val, downscale_to=downscale_to)
        pose_processor.set_multi_person_mode(self.multiPersonCheckbox.GetValue())

        for index, video_file in enumerate(video_files, start=1):
            self.set_status_message(f"📸 Extracting pose from: {os.path.basename(video_file)}")
            
            # Create progress callback for this video
            def make_progress_callback(video_index, total_videos):
                def progress_callback(frame_progress):
                    # Calculate overall progress: (video_index-1)/total_videos + frame_progress/total_videos
                    overall_progress = int(((video_index - 1) / total_videos) * 100 + (frame_progress / total_videos))
                    self.update_progress(overall_progress)
                return progress_callback
            
            pose_processor.extract_pose_features(video_file, progress_callback=make_progress_callback(index, total_files))

        wx.CallAfter(wx.MessageBox, "Pose feature extraction completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        self.update_progress(0)  # Reset progress bar
            
           
    def on_embed_poses(self, event):
        folder_path = self.folderPicker.GetPath()
        if not folder_path:
            wx.MessageBox("Please select a folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.ensure_output_folders(folder_path)
        video_files = self.get_files_from_folder(folder_path, (".mp4", ".avi", ".mov"))
        
      
        stride_val = 1
        try:
            stride_val = max(1, int(self.frameStrideInput.GetValue()))
        except Exception:
            stride_val = 1
        downscale_to = (1280, 720) if (hasattr(self, 'downscaleCheckbox') and self.downscaleCheckbox.GetValue()) else None

        pose_processor = PoseProcessor(output_csv_folder=self.extracted_pose_folder, output_video_folder=self.embedded_pose_folder, frame_threshold=self.frameThresholdInput.GetValue(), frame_stride=stride_val, downscale_to=downscale_to)
        pose_processor.set_multi_person_mode(self.multiPersonCheckbox.GetValue())
        
        if not video_files:
            wx.MessageBox("No video files found in the selected folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Run in a separate thread to keep UI responsive
        thread = threading.Thread(target=self.embed_pose_batch, args=(video_files, pose_processor))
        thread.start()

    def embed_pose_batch(self, video_files, pose_processor):
        total_files = len(video_files)

        for index, video_file in enumerate(video_files, start=1):
            self.set_status_message(f"🕺 Embedding poses for: {os.path.basename(video_file)}")
            
            # Create progress callback for this video
            def make_progress_callback(video_index, total_videos):
                def progress_callback(frame_progress):
                    # Calculate overall progress: (video_index-1)/total_videos + frame_progress/total_videos
                    overall_progress = int(((video_index - 1) / total_videos) * 100 + (frame_progress / total_videos))
                    self.update_progress(overall_progress)
                return progress_callback
            
            pose_processor.embed_pose_video(video_file, progress_callback=make_progress_callback(index, total_files))

        wx.CallAfter(wx.MessageBox, "Pose embedding completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        self.update_progress(0)  # Reset progress bar


    

    def on_extract_audio_features(self, event):
        """Extract audio features from all WAV files in the folder."""
        folder_path = self.folderPicker.GetPath()
        if not folder_path:
            wx.MessageBox("Please select a folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.ensure_output_folders(folder_path)
        audio_files = self.get_files_from_folder(folder_path, (".wav",))

        if not audio_files:
            wx.MessageBox("No WAV files found in the selected folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Initialize audio processor
        audio_processor = AudioProcessor(
            output_audio_features_folder=self.extracted_audio_folder,
            output_transcripts_folder=None,  # Not needed for feature extraction
            status_callback=self.set_status_message
        )

        thread = threading.Thread(target=self.extract_audio_features_batch, args=(audio_files, audio_processor))
        thread.start()

    def extract_audio_features_batch(self, audio_files, audio_processor):
        """Batch process all audio files to extract features."""
        def progress_callback(progress):
            self.update_progress(progress)
        
        try:
            audio_processor.extract_audio_features_batch(audio_files, progress_callback=progress_callback)
            wx.CallAfter(wx.MessageBox, "Audio feature extraction completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.CallAfter(wx.MessageBox, f"Error during audio feature extraction: {e}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            self.update_progress(0)  # Reset progress bar



    def on_extract_transcripts(self, event):
        """Extract transcripts from all WAV files in the folder."""
        folder_path = self.folderPicker.GetPath()
        if not folder_path:
            wx.MessageBox("Please select a folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.ensure_output_folders(folder_path)
        audio_files = self.get_files_from_folder(folder_path, (".wav",))

        if not audio_files:
            wx.MessageBox("No WAV files found in the selected folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Initialize audio processor
        audio_processor = AudioProcessor(
            output_audio_features_folder=None,  # Not needed for transcript extraction
            output_transcripts_folder=self.extracted_transcripts_folder,
            status_callback=self.set_status_message,
            auth_token="YOUR_HUGGING_FACE_TOKEN"  # Your Hugging Face token
        )

        # Run transcription in a separate thread
        thread = threading.Thread(target=self.extract_transcripts_batch, args=(audio_files, audio_processor))
        thread.start()


    def extract_transcripts_batch(self, audio_files, audio_processor):
        """Batch process all audio files to generate transcripts."""
        def progress_callback(progress):
            self.update_progress(progress)
        
        try:
            audio_processor.extract_transcripts_batch(audio_files, progress_callback=progress_callback)
            wx.CallAfter(wx.MessageBox, "Transcription extraction completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.CallAfter(wx.MessageBox, f"Error during transcript extraction: {e}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            self.update_progress(0)  # Reset progress bar




def main():
    # Ensure ffmpeg is available before the UI starts doing conversions
    if not ensure_ffmpeg_available():
        msg = (
            "ffmpeg was not found. Install it or let the app use a bundled one.\n\n"
            "macOS: brew install ffmpeg\n"
            "Linux: sudo apt-get install ffmpeg\n"
            "Windows: choco install ffmpeg\n\n"
            "Alternatively, ensure imageio-ffmpeg is installed (already in requirements)."
        )
        try:
            wx.MessageBox(msg, "ffmpeg not found", wx.OK | wx.ICON_ERROR)
        except Exception:
            print(msg)
    app = wx.App()
    frm = VideoToWavConverter(None)
    frm.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
