'''
This is the main script for multisocial app

'''

# Import necessary system and utility modules
import os
import threading

# Set up GPU environment specially for Mediapipe (specific for Saturn Cloud), if you use some other high performance computing platform check compatibility before usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Make sure the system uses the GPU
# Enable MPS fallback for Mac to prevent freezes on unsupported operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Third-party libraries (assumed pre-installed via requirements.txt)
import ffmpeg
import wx
import unicodedata
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Import the core processing classes
from pose import PoseProcessor
from audio import AudioProcessor
import gui_utils
from gui_utils import Theme
from ui_components import GradientPanel, GlassPanel, ElevatedLogoPanel, TooltipButton, CustomGauge

## All dependencies are expected to be installed ahead of time via requirements.txt

class VideoToWavConverter(wx.Frame):
    def __init__(self, *args, **kw):
        super(VideoToWavConverter, self).__init__(*args, **kw)
        
        self._init_state()
        self._init_ui()
        
    def _init_state(self):
        # Start at designed size; prevent shrinking below baseline
        self._baseline_size = None  # will be captured on first resize to drive responsive scaling
        # Track if a background process is running to prevent UI resets during tab switches
        self._process_running = False
        # File extensions constants
        self.VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
        self.AUDIO_EXTENSIONS = (".wav",)
        
    def _init_ui(self):
        pnl = GradientPanel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        # Add extra space above the title
        vbox.Add((0, 20))
        
        # Header (Logo and Title)
        self._create_header(pnl, vbox)
        
        # Mode toggle buttons
        self._create_mode_selection(pnl, vbox)
        
        # Folder Picker
        self._create_folder_picker(pnl, vbox)
        
        # Panels for Video and Audio options
        self.videoPanel = GlassPanel(pnl)
        self.audioPanel = GlassPanel(pnl)
        
        self._create_video_panel()
        self._create_audio_panel()
        
        # Assemble panels
        vbox.AddSpacer(8)
        vbox.Add(self.videoPanel, proportion=0, flag=wx.ALIGN_CENTER|wx.LEFT|wx.RIGHT, border=12)
        vbox.Add(self.audioPanel, proportion=0, flag=wx.ALIGN_CENTER|wx.LEFT|wx.RIGHT, border=12)
        vbox.AddStretchSpacer(1)
        
        # Status and Progress
        self._create_status_and_progress(pnl, vbox)
        
        pnl.SetSizer(vbox)
        pnl.Layout()
        
        # Final setup (sizing, binding, etc.)
        self._finalize_setup(vbox)

    def _create_header(self, pnl, vbox):
        new_width, new_height = wx.GetDisplaySize()
        logo_size = min(int(new_height * 0.08), 150)
        
        top_box = wx.BoxSizer(wx.HORIZONTAL)
        
        # Logo
        logo_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "MultiSOCIAL_logo.png")
        logo_image = wx.Image(logo_path, wx.BITMAP_TYPE_ANY)
        logo_image = logo_image.Scale(logo_size, logo_size, wx.IMAGE_QUALITY_HIGH)
        logo_bmp = wx.Bitmap(logo_image)
        
        elevated_logo = ElevatedLogoPanel(pnl, logo_bmp)
        elevated_logo.SetMinSize((logo_size + 20, logo_size + 20))
        
        top_box.AddStretchSpacer(1)
        top_box.Add(elevated_logo, flag=wx.ALIGN_CENTER | wx.ALL, border=15)
        top_box.AddStretchSpacer(1)

        vbox.Add(top_box, flag=wx.EXPAND | wx.TOP | wx.BOTTOM, border=10)
        
        # Title
        self.title = wx.StaticText(pnl, label="Welcome to", style=wx.ALIGN_CENTER)
        self.title.SetFont(Theme.get_font(20))
        self.title.SetForegroundColour(Theme.COLOR_TEXT_WHITE)
        vbox.Add(self.title, flag=wx.ALIGN_CENTER | wx.BOTTOM, border=10)

        # Logo Label
        self.logoLabel = wx.StaticText(pnl, label="MultiSOCIAL Toolbox", style=wx.ALIGN_CENTER)
        self.logoLabel.SetFont(Theme.get_font(24, bold=True))
        self.logoLabel.SetForegroundColour(Theme.COLOR_TEXT_WHITE)

        vbox.Add(self.logoLabel, flag=wx.ALIGN_CENTER|wx.TOP, border=5)

    def _create_mode_selection(self, pnl, vbox):
        mode_box = wx.BoxSizer(wx.HORIZONTAL)
        self.videoModeBtn = wx.Button(pnl, label="Video Options", size=(140, 35))
        self.audioModeBtn = wx.Button(pnl, label="Audio Options", size=(140, 35))
        
        mode_btn_font = Theme.get_font(11, bold=True)
        self.videoModeBtn.SetFont(mode_btn_font)
        self.audioModeBtn.SetFont(mode_btn_font)
        
        self.videoModeBtn.Bind(wx.EVT_BUTTON, lambda evt: self.switch_mode('video'))
        self.audioModeBtn.Bind(wx.EVT_BUTTON, lambda evt: self.switch_mode('audio'))
        
        mode_box.AddStretchSpacer(1)
        mode_box.Add(self.videoModeBtn, flag=wx.ALL, border=5)
        mode_box.Add(self.audioModeBtn, flag=wx.ALL, border=5)
        mode_box.AddStretchSpacer(1)
        vbox.Add(mode_box, flag=wx.EXPAND|wx.TOP|wx.BOTTOM, border=10)
        
        # Underlines
        self._video_underline = wx.Panel(pnl, size=(-1, 3))
        self._video_underline.SetBackgroundColour(Theme.COLOR_ACCENT_GREEN)
        self._audio_underline = wx.Panel(pnl, size=(-1, 3))
        self._audio_underline.SetBackgroundColour(Theme.COLOR_ACCENT_GREEN)
        
        mode_box.Insert(2, self._audio_underline, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=5)
        mode_box.Insert(1, self._video_underline, flag=wx.EXPAND | wx.LEFT | wx.RIGHT, border=5)
        
        self._video_underline.Hide()
        self._audio_underline.Hide()

    def _create_folder_picker(self, pnl, vbox):
        self.folderCaption = wx.StaticText(pnl, label="Select a folder containing VIDEO files", style=wx.ALIGN_CENTER)
        self.folderCaption.SetFont(Theme.get_font(13, bold=True))
        self.folderCaption.SetForegroundColour(Theme.COLOR_TEXT_WHITE)
        vbox.Add(self.folderCaption, flag=wx.ALIGN_CENTER|wx.LEFT|wx.RIGHT|wx.TOP, border=8)
        
        self.folderPicker = wx.DirPickerCtrl(pnl, message="Select a folder containing media files")
        self.folderPicker.Bind(wx.EVT_DIRPICKER_CHANGED, self.on_folder_changed)
        vbox.Add(self.folderPicker, flag=wx.ALIGN_CENTER | wx.ALL, border=10, proportion=0)

    def _create_video_panel(self):
        video_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Placeholder
        self.placeholderVideoLabel = wx.StaticText(self.videoPanel, label="If you have a video file:")
        self.placeholderVideoLabel.SetFont(Theme.get_font(20))
        video_sizer.AddSpacer(8)
        video_sizer.Add(self.placeholderVideoLabel, flag=wx.ALIGN_CENTER|wx.ALL, border=10)
        
        # Multi-person checkbox
        self.multiPersonCheckbox = wx.CheckBox(self.videoPanel, label="Enable Multi-Person Pose")
        self.multiPersonCheckbox.SetFont(Theme.get_font(14))
        self.multiPersonCheckbox.SetForegroundColour(Theme.COLOR_TEXT_WHITE)
        video_sizer.Add(self.multiPersonCheckbox, flag=wx.ALIGN_CENTER|wx.ALL, border=10)
        
        # Downsampling controls
        ds_box = wx.BoxSizer(wx.HORIZONTAL)
        ds_label = wx.StaticText(self.videoPanel, label="Process every k-th frame:")
        ds_label.SetFont(Theme.get_font(12))
        ds_label.SetForegroundColour(Theme.COLOR_TEXT_WHITE)
        ds_box.Add(ds_label, flag=wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border=8)
        
        self.frameStrideInput = wx.SpinCtrl(self.videoPanel, value="1", min=1, max=10)
        self.frameStrideInput.SetFont(Theme.get_font(12))
        ds_box.Add(self.frameStrideInput, flag=wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border=20)
        
        self.downscaleCheckbox = wx.CheckBox(self.videoPanel, label="Downscale to 720p for processing")
        self.downscaleCheckbox.SetFont(Theme.get_font(12))
        self.downscaleCheckbox.SetForegroundColour(Theme.COLOR_TEXT_WHITE)
        ds_box.Add(self.downscaleCheckbox, flag=wx.ALIGN_CENTER_VERTICAL)
        video_sizer.Add(ds_box, flag=wx.ALIGN_CENTER|wx.ALL, border=10)
        
        # Frame threshold
        frame_threshold_box = wx.BoxSizer(wx.HORIZONTAL)
        frame_threshold_label = wx.StaticText(self.videoPanel, label="Frame Threshold for Bounding Box Recalibration:")
        frame_threshold_label.SetFont(Theme.get_font(12))
        frame_threshold_label.SetForegroundColour(Theme.COLOR_TEXT_WHITE)
        frame_threshold_box.Add(frame_threshold_label, flag=wx.ALIGN_CENTER_VERTICAL|wx.RIGHT, border=10)
        
        self.frameThresholdInput = wx.SpinCtrl(self.videoPanel, value="10", min=1, max=100)
        self.frameThresholdInput.SetFont(Theme.get_font(12))
        frame_threshold_box.Add(self.frameThresholdInput, flag=wx.ALIGN_CENTER_VERTICAL)
        video_sizer.Add(frame_threshold_box, flag=wx.ALIGN_CENTER|wx.ALL, border=10)
        
        # Buttons
        button_font = Theme.get_font(16)
        
        self.convertBtn, hbox_convert = TooltipButton.create_with_icon(
            self.videoPanel, 
            'Convert video to audio', 
            'Converts video files (.mp4, .avi, .mov, .mkv) to audio files (.wav) for further processing',
            font=button_font,
            handler=self.on_convert
        )
        video_sizer.Add(hbox_convert, flag=wx.ALIGN_CENTER|wx.ALL, border=10)
        
        self.extractFeaturesBtn, hbox_extract_pose = TooltipButton.create_with_icon(
            self.videoPanel,
            'Extract Pose Features',
            'Extracts human pose landmarks and features from video files using MediaPipe. Supports single and multi-person detection.',
            font=button_font,
            handler=self.on_extract_features
        )
        video_sizer.Add(hbox_extract_pose, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        self.embedFeaturesBtn, hbox_embed_pose = TooltipButton.create_with_icon(
            self.videoPanel,
            'Embed Pose Features',
            'Creates vector embeddings from extracted pose features for machine learning and analysis purposes.',
            font=button_font,
            handler=self.on_embed_poses
        )
        video_sizer.Add(hbox_embed_pose, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        self.verifyBtn, hbox_verify = TooltipButton.create_with_icon(
            self.videoPanel,
            'Verify Consistency',
            'Verifies that embedded videos match extracted pose CSVs and saves a report with worst-frame thumbnails.',
            font=button_font,
            handler=self.on_verify_consistency
        )
        video_sizer.Add(hbox_verify, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        video_sizer.AddSpacer(8)
        
        self.videoPanel.SetSizer(video_sizer)

    def _create_audio_panel(self):
        audio_sizer = wx.BoxSizer(wx.VERTICAL)
        
        # Placeholder
        self.placeholderAudioLabel = wx.StaticText(self.audioPanel, label="If you have an audio file:")
        self.placeholderAudioLabel.SetFont(Theme.get_font(20))
        audio_sizer.AddSpacer(8)
        audio_sizer.Add(self.placeholderAudioLabel, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        # Diarization
        self.diarizationCheckbox = wx.CheckBox(self.audioPanel, label="Enable speaker diarization (requires Hugging Face token)")
        self.diarizationCheckbox.SetFont(Theme.get_font(12))
        self.diarizationCheckbox.SetForegroundColour(Theme.COLOR_TEXT_WHITE)
        audio_sizer.Add(self.diarizationCheckbox, flag=wx.ALIGN_CENTER|wx.ALL, border=6)
        
        # Buttons
        button_font = Theme.get_font(16)
        
        self.extractAudioFeaturesBtn, hbox_extract_audio = TooltipButton.create_with_icon(
            self.audioPanel,
            'Extract Audio Features',
            'Extracts acoustic features from audio files (.wav) including MFCC, spectral features, and prosodic characteristics.',
            font=button_font,
            handler=self.on_extract_audio_features
        )
        audio_sizer.Add(hbox_extract_audio, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        self.extractTranscriptsBtn, hbox_extract_transcripts = TooltipButton.create_with_icon(
            self.audioPanel,
            'Extract Transcripts',
            'Converts speech in audio files (.wav) to text transcripts using automatic speech recognition (ASR) technology.',
            font=button_font,
            handler=self.on_extract_transcripts
        )
        audio_sizer.Add(hbox_extract_transcripts, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        
        self.alignFeaturesBtn, hbox_align = TooltipButton.create_with_icon(
            self.audioPanel,
            'Align Features',
            'Aligns extracted audio features with word-level transcripts. Requires both features and transcripts.',
            font=button_font,
            handler=self.on_align_features
        )
        audio_sizer.Add(hbox_align, flag=wx.ALIGN_CENTER|wx.ALL, border=12)
        audio_sizer.AddSpacer(8)
        
        self.audioPanel.SetSizer(audio_sizer)

    def _create_status_and_progress(self, pnl, vbox):
        self.statusLabel = wx.StaticText(pnl, label="", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.statusLabel.SetForegroundColour(Theme.COLOR_TEXT_WHITE)
        try:
            self.statusLabel.SetFont(Theme.get_font(13, bold=True))
        except Exception:
            pass
        vbox.Add(self.statusLabel, flag=wx.EXPAND|wx.ALL, border=10)
        
        self.progress = CustomGauge(pnl, range=100)
        self.progress.SetMinSize((-1, 28))
        try:
            self.progress.SetForegroundColour(wx.Colour(33, 150, 243))
        except Exception:
            pass
        vbox.Add(self.progress, proportion=0, flag=wx.ALIGN_CENTER|wx.ALL, border=12)

    def _finalize_setup(self, vbox):
        # Capture a design baseline size
        best_min = vbox.CalcMin()
        self._baseline_size = (max(400, best_min.width), max(800, best_min.height))
        
        self.Bind(wx.EVT_SIZE, self.on_resize)
        
        screen_w, screen_h = wx.GetDisplaySize()
        max_w = max(600, screen_w - 80)
        max_h = max(600, screen_h - 80)
        base_w, base_h = self._baseline_size
        min_w = min(base_w, max_w)
        min_h = min(base_h, max_h)
        self.SetSizeHints(min_w, min_h, max_w, max_h)
        self.SetSize((min_w, min_h))
        self.SetTitle('MultiSOCIAL Toolbox')
        self.Centre()
        
        self.active_mode = 'video'
        # Try to load token from environment variable first
        self.hf_token = os.getenv("HF_TOKEN")
        self.switch_mode('video')
        self._update_panel_sizes()
        self.update_buttons_enabled()
        try:
            wx.CallAfter(self._apply_status_wrap_and_center)
        except Exception:
            pass

    def switch_mode(self, mode):
        self.active_mode = mode
        # Remove old button background coloring for modern look
        self.videoModeBtn.SetBackgroundColour(wx.NullColour)
        self.audioModeBtn.SetBackgroundColour(wx.NullColour)
        self.videoModeBtn.SetForegroundColour(wx.Colour(0, 0, 0))
        self.audioModeBtn.SetForegroundColour(wx.Colour(0, 0, 0))
        # Underline logic
        if mode == 'video':
            self.videoPanel.Show()
            self.audioPanel.Hide()
            self.folderCaption.SetLabel("Select a folder containing VIDEO files")
            self.videoModeBtn.Enable(False)
            self.audioModeBtn.Enable(True)
            # Underline: show under video, hide under audio
            self._video_underline.Show()
            self._audio_underline.Hide()
        else:
            self.videoPanel.Hide()
            self.audioPanel.Show()
            self.folderCaption.SetLabel("Select a folder containing AUDIO (.wav) files")
            self.videoModeBtn.Enable(True)
            self.audioModeBtn.Enable(False)
            # Underline: show under audio, hide under video
            self._video_underline.Hide()
            self._audio_underline.Show()
        # Refresh button and underline appearance
        self.videoModeBtn.Refresh()
        self.audioModeBtn.Refresh()
        self._video_underline.Refresh()
        self._audio_underline.Refresh()
        # Update panel sizes and relayout
        self._update_panel_sizes()
        self.Layout()
        # Relayout parent panel as well
        pnl = self.GetChildren()[0] if self.GetChildren() else None
        if pnl and hasattr(pnl, "Layout"):
            pnl.Layout()
        # Reset UX state on mode switch ONLY if no process is running
        if not self._process_running:
            if hasattr(self, 'statusLabel'):
                self.statusLabel.SetLabel("")
            self.update_progress(0)
        self.update_buttons_enabled()

    def on_folder_changed(self, event):
        self.ensure_output_folders(self.folderPicker.GetPath())
        self.update_buttons_enabled()

    def _update_panel_sizes(self):
        # Keep panels at a reasonable width fraction of the frame
        if not hasattr(self, 'videoPanel') or not hasattr(self, 'audioPanel'):
            return
        try:
            cur_w, cur_h = self.GetSize()
            target_w = max(380, int(cur_w * 0.60))
            # Cap to avoid overly wide panels on very large screens
            target_w = min(target_w, 900)
            # Let each panel size itself based on content, with a min/max cap
            if self.videoPanel:
                self.videoPanel.SetMinSize((target_w, -1))
                self.videoPanel.SetMaxSize((target_w, -1))
            if self.audioPanel:
                self.audioPanel.SetMinSize((target_w, -1))
                self.audioPanel.SetMaxSize((target_w, -1))
            # Match folder picker width to glass panels
            if hasattr(self, 'folderPicker') and self.folderPicker:
                self.folderPicker.SetMinSize((target_w, -1))
                self.folderPicker.SetMaxSize((target_w, -1))
            # Match progress bar width to glass panels
            if hasattr(self, 'progress') and self.progress:
                current_min_size = self.progress.GetMinSize()
                current_h = current_min_size.height if current_min_size.height > 0 else 28
                self.progress.SetMinSize((target_w, current_h))
                self.progress.SetMaxSize((target_w, current_h))
        except Exception:
            pass

    def update_buttons_enabled(self):
        folder_path = self.folderPicker.GetPath()
        has_folder = bool(folder_path)
        # Detect files
        video_files = []
        audio_files = []
        try:
            if has_folder:
                video_files = gui_utils.get_files_from_folder(folder_path, self.VIDEO_EXTENSIONS)
                audio_files = gui_utils.get_files_from_folder(folder_path, self.AUDIO_EXTENSIONS)
        except Exception:
            pass
        # Video buttons
        enable_video = has_folder and len(video_files) > 0
        for btn in [getattr(self, 'convertBtn', None), getattr(self, 'extractFeaturesBtn', None), getattr(self, 'embedFeaturesBtn', None), getattr(self, 'verifyBtn', None)]:
            if btn:
                btn.Enable(enable_video)
        # Audio buttons
        enable_audio = has_folder and len(audio_files) > 0
        for btn in [getattr(self, 'extractAudioFeaturesBtn', None), getattr(self, 'extractTranscriptsBtn', None), getattr(self, 'alignFeaturesBtn', None)]:
            if btn:
                btn.Enable(enable_audio)
        
    def _scale_font(self, base_point_size, scale):
        new_size = max(9, int(round(base_point_size * scale)))
        return Theme.get_font(new_size)

    def _scale_bold_font(self, base_point_size, scale):
        new_size = max(10, int(round(base_point_size * scale)))
        return Theme.get_font(new_size, bold=True)

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
            if hasattr(self, 'title') and self.title:
                self.title.SetFont(self._scale_font(20, scale))
            if hasattr(self, 'logoLabel') and self.logoLabel:
                self.logoLabel.SetFont(self._scale_bold_font(24, scale))
            if hasattr(self, 'placeholderVideoLabel') and self.placeholderVideoLabel:
                self.placeholderVideoLabel.SetFont(self._scale_font(20, scale))
            if hasattr(self, 'placeholderAudioLabel') and self.placeholderAudioLabel:
                self.placeholderAudioLabel.SetFont(self._scale_font(20, scale))
            if hasattr(self, 'folderCaption') and self.folderCaption:
                self.folderCaption.SetFont(self._scale_font(13, scale))
                try:
                    wrap_width = max(240, int(cur_w * 0.6))
                    self.folderCaption.Wrap(wrap_width)
                except Exception:
                    pass
            # Mode toggle buttons
            if hasattr(self, 'videoModeBtn') and self.videoModeBtn:
                self.videoModeBtn.SetFont(self._scale_font(11, scale))
            if hasattr(self, 'audioModeBtn') and self.audioModeBtn:
                self.audioModeBtn.SetFont(self._scale_font(11, scale))
            if hasattr(self, 'multiPersonCheckbox') and self.multiPersonCheckbox:
                self.multiPersonCheckbox.SetFont(self._scale_font(14, scale))
            if hasattr(self, 'diarizationCheckbox') and self.diarizationCheckbox:
                self.diarizationCheckbox.SetFont(self._scale_font(12, scale))
            if hasattr(self, 'frameStrideInput'):
                self.frameStrideInput.SetFont(self._scale_font(12, scale))
            if hasattr(self, 'downscaleCheckbox'):
                self.downscaleCheckbox.SetFont(self._scale_font(12, scale))
            # Frame threshold elements
            if hasattr(self, 'frameThresholdInput'):
                self.frameThresholdInput.SetFont(self._scale_font(12, scale))
            # Buttons (safely check existence)
            button_font = self._scale_font(16, scale)
            if hasattr(self, 'convertBtn') and self.convertBtn:
                self.convertBtn.SetFont(button_font)
            if hasattr(self, 'extractFeaturesBtn') and self.extractFeaturesBtn:
                self.extractFeaturesBtn.SetFont(button_font)
            if hasattr(self, 'embedFeaturesBtn') and self.embedFeaturesBtn:
                self.embedFeaturesBtn.SetFont(button_font)
            if hasattr(self, 'verifyBtn') and self.verifyBtn:
                self.verifyBtn.SetFont(button_font)
            if hasattr(self, 'extractAudioFeaturesBtn') and self.extractAudioFeaturesBtn:
                self.extractAudioFeaturesBtn.SetFont(button_font)
            if hasattr(self, 'extractTranscriptsBtn') and self.extractTranscriptsBtn:
                self.extractTranscriptsBtn.SetFont(button_font)
            if hasattr(self, 'alignFeaturesBtn') and self.alignFeaturesBtn:
                self.alignFeaturesBtn.SetFont(button_font)
            # Status label: scale font and wrap to panel width for responsiveness
            if hasattr(self, 'statusLabel') and self.statusLabel:
                self.statusLabel.SetFont(self._scale_font(12, scale))
                try:
                    wrap_width = max(200, int(cur_w * 0.9))
                    self.statusLabel.Wrap(wrap_width)
                    # Re-apply centering after Wrap, which can reset alignment
                    self.statusLabel.SetWindowStyleFlag(wx.ALIGN_CENTER_HORIZONTAL)
                except Exception:
                    pass
            # Progress bar height scaling
            if hasattr(self, 'progress') and self.progress:
                self.progress.SetMinSize((-1, max(20, int(28 * scale))))
        except Exception:
            pass

        # Relayout after scaling
        self._update_panel_sizes()
        self.Layout()
        event.Skip()

    def set_status_message(self, message):
        """Safely update the centered status label with emoji-free text prefixed by 'Status:'."""
        if hasattr(self, 'statusLabel'):
            # Remove emojis and most symbol-like non-ASCII chars
            try:
                text = str(message)
                sanitized = ''.join(
                    ch for ch in text
                    if not (ord(ch) > 127 and unicodedata.category(ch) in ('So', 'Sk', 'Cs'))
                )
            except Exception:
                # Fallback: strip non-ascii
                try:
                    sanitized = str(message).encode('ascii', 'ignore').decode()
                except Exception:
                    sanitized = str(message)

            final_text = f"Status: {sanitized}" if sanitized else "Status:"
            wx.CallAfter(self.statusLabel.SetLabel, final_text)
            # Wrap/center now (post event) and request layout
            wx.CallAfter(self._apply_status_wrap_and_center)

    def _apply_status_wrap_and_center(self):
        """One place to wrap, center, and layout the status label based on current frame width."""
        if not hasattr(self, 'statusLabel') or not self.statusLabel:
            return
        try:
            cur_w = self.GetSize()[0]
            wrap_width = max(300, int(cur_w * 0.9))
            self.statusLabel.Wrap(wrap_width)
        except Exception:
            try:
                self.statusLabel.Wrap(600)
            except Exception:
                pass
        try:
            self.statusLabel.SetWindowStyleFlag(wx.ALIGN_CENTER_HORIZONTAL)
        except Exception:
            pass
        # --- Added layout/refresh logic to fix initial left alignment ---
        try:
            parent = self.statusLabel.GetParent()
            if parent:
                parent.Layout()
            self.Layout()
            self.Refresh()
        except Exception:
            pass


    def update_progress(self, value):
        """Update the progress bar."""
        if hasattr(self, 'progress') and self.progress:
            wx.CallAfter(self.progress.SetValue, value)

    def make_overall_progress_cb(self, item_index, total_items):
        """Return a callback mapping per-item percent (0-100) to overall percent (0-100)."""
        total_items = max(1, int(total_items))
        item_index = max(1, int(item_index))
        def _cb(per_item_percent):
            try:
                per_item_fraction = float(per_item_percent) / 100.0
            except Exception:
                per_item_fraction = 0.0
            overall = int((((item_index - 1) + per_item_fraction) / total_items) * 100)
            self.update_progress(min(100, max(0, overall)))
        return _cb


    def ensure_output_folders(self, folder_path):
        """Ensures output directories exist inside the selected folder only when needed."""
        # Guard against empty/invalid path
        if not folder_path or not os.path.exists(folder_path):
            self.converted_audio_folder = None
            self.extracted_pose_folder = None
            self.embedded_pose_folder = None
            self.extracted_audio_folder = None
            self.extracted_transcripts_folder = None
            return
    
        # Define folder paths
        self.converted_audio_folder = os.path.join(folder_path, "converted_audio")
        self.extracted_pose_folder = os.path.join(folder_path, "pose_features")
        self.embedded_pose_folder = os.path.join(folder_path, "embedded_pose")
        self.extracted_audio_folder = os.path.join(folder_path, "audio_features")
        self.extracted_transcripts_folder = os.path.join(folder_path, "transcripts")

        # Check if the selected folder has video files
        video_files = gui_utils.get_files_from_folder(folder_path, (".mp4", ".avi", ".mov"))
        if video_files:
            try:
                for folder in [self.converted_audio_folder, self.extracted_pose_folder, self.embedded_pose_folder]:
                    os.makedirs(folder, exist_ok=True)
            except (OSError, PermissionError) as e:
                # Can't create folders, disable video processing
                self.converted_audio_folder = None
                self.extracted_pose_folder = None
                self.embedded_pose_folder = None
                wx.CallAfter(wx.MessageBox, f"Cannot create output folders: {e}", "Warning", wx.OK | wx.ICON_WARNING)
        else:
            self.converted_audio_folder = None
            self.extracted_pose_folder = None
            self.embedded_pose_folder = None

        # Check if the selected folder has audio files
        audio_files = gui_utils.get_files_from_folder(folder_path, (".wav",))
        if audio_files:
            try:
                for folder in [self.extracted_audio_folder, self.extracted_transcripts_folder]:
                    os.makedirs(folder, exist_ok=True)
            except (OSError, PermissionError) as e:
                # Can't create folders, disable audio processing
                self.extracted_audio_folder = None
                self.extracted_transcripts_folder = None
                wx.CallAfter(wx.MessageBox, f"Cannot create output folders: {e}", "Warning", wx.OK | wx.ICON_WARNING)
        else:
            self.extracted_audio_folder = None
            self.extracted_transcripts_folder = None

      
      
    def on_convert(self, event):
        """Convert all videos in a selected folder to WAV."""
        folder_path = self.folderPicker.GetPath()
        if not folder_path:
            wx.MessageBox("Please select a folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.ensure_output_folders(folder_path)
        video_files = gui_utils.get_files_from_folder(folder_path, self.VIDEO_EXTENSIONS)

        if not video_files:
            wx.MessageBox("No video files found in the selected folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Process each video file in a separate thread
        thread = threading.Thread(target=self.convert_all_videos_to_wav, args=(video_files,))
        thread.start()

    def convert_all_videos_to_wav(self, video_files):
        """Convert multiple videos to WAV format and save to output folder."""
        self._process_running = True
        try:
            total_files = len(video_files)
        
            for i, video_file in enumerate(video_files):
                #print(f"Converting: {video_file}")
                file_name = os.path.basename(video_file)
                self.set_status_message(f"Converting to WAV: {file_name}")
                
                # Overall progress callback (percent-based)
                self.convert_to_wav(video_file, progress_callback=self.make_overall_progress_cb(i + 1, total_files))

            wx.MessageBox("Video to audio conversion completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        finally:
            self._process_running = False
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
        embedded_videos = gui_utils.get_files_from_folder(self.embedded_pose_folder, (".mp4", ".avi", ".mov"))
        if not embedded_videos:
            wx.MessageBox("No embedded videos found in embedded_pose/", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Launch background verification to keep UI responsive
        thread = threading.Thread(target=self._verify_consistency_batch, args=(embedded_videos, verification_dir, worst_frames_root))
        thread.start()

    def _verify_consistency_batch(self, embedded_videos, verification_dir, worst_frames_root):
        from verify_pose_embedding import verify, save_report
        import glob

        self._process_running = True
        try:
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
        finally:
            self._process_running = False
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
        video_files = gui_utils.get_files_from_folder(folder_path, (".mp4", ".avi"))
        
        if not video_files:
            wx.MessageBox("No video files found in the selected folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Run in a separate thread to keep UI responsive
        thread = threading.Thread(target=self.extract_pose_features_batch, args=(video_files,))
        thread.start()

    def extract_pose_features_batch(self, video_files):
        """Batch process all video files to extract pose features."""
        self._process_running = True
        try:
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
                
                # Overall progress callback (percent-based)
                pose_processor.extract_pose_features(video_file, progress_callback=self.make_overall_progress_cb(index, total_files))

            wx.CallAfter(wx.MessageBox, "Pose feature extraction completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        finally:
            self._process_running = False
            self.update_progress(0)  # Reset progress bar
            
           
    def on_embed_poses(self, event):
        folder_path = self.folderPicker.GetPath()
        if not folder_path:
            wx.MessageBox("Please select a folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.ensure_output_folders(folder_path)
        video_files = gui_utils.get_files_from_folder(folder_path, (".mp4", ".avi", ".mov"))
        
      
        stride_val = 1
        try:
            stride_val = max(1, int(self.frameStrideInput.GetValue()))
        except Exception:
            stride_val = 1
        downscale_to = (1280, 720) if (hasattr(self, 'downscaleCheckbox') and self.downscaleCheckbox.GetValue()) else None

        pose_processor = PoseProcessor(output_csv_folder=self.extracted_pose_folder, output_video_folder=self.embedded_pose_folder, status_callback=self.set_status_message, frame_threshold=self.frameThresholdInput.GetValue(), frame_stride=stride_val, downscale_to=downscale_to)
        pose_processor.set_multi_person_mode(self.multiPersonCheckbox.GetValue())
        
        if not video_files:
            wx.MessageBox("No video files found in the selected folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Run in a separate thread to keep UI responsive
        thread = threading.Thread(target=self.embed_pose_batch, args=(video_files, pose_processor))
        thread.start()

    def embed_pose_batch(self, video_files, pose_processor):
        self._process_running = True
        try:
            total_files = len(video_files)

            for index, video_file in enumerate(video_files, start=1):
                self.set_status_message(f"🕺 Embedding poses for: {os.path.basename(video_file)}")
                
                # Overall progress callback (percent-based)
                pose_processor.embed_pose_video(video_file, progress_callback=self.make_overall_progress_cb(index, total_files))

            wx.CallAfter(wx.MessageBox, "Pose embedding completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        finally:
            self._process_running = False
            self.update_progress(0)  # Reset progress bar


    

    def on_extract_audio_features(self, event):
        """Extract audio features from all WAV files in the folder."""
        folder_path = self.folderPicker.GetPath()
        if not folder_path:
            wx.MessageBox("Please select a folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.ensure_output_folders(folder_path)
        audio_files = gui_utils.get_files_from_folder(folder_path, self.AUDIO_EXTENSIONS)

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
        self._process_running = True
        try:
            def progress_callback(progress):
                self.update_progress(progress)
            
            audio_processor.extract_audio_features_batch(audio_files, progress_callback=progress_callback)
            wx.CallAfter(wx.MessageBox, "Audio feature extraction completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.CallAfter(wx.MessageBox, f"Error during audio feature extraction: {e}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            self._process_running = False
            self.update_progress(0)  # Reset progress bar



    def on_extract_transcripts(self, event):
        """Extract transcripts from all WAV files in the folder."""
        folder_path = self.folderPicker.GetPath()
        if not folder_path:
            wx.MessageBox("Please select a folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.ensure_output_folders(folder_path)
        audio_files = gui_utils.get_files_from_folder(folder_path, self.AUDIO_EXTENSIONS)

        if not audio_files:
            wx.MessageBox("No WAV files found in the selected folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Determine diarization preference and collect token if needed
        enable_diarization = False
        if hasattr(self, 'diarizationCheckbox') and self.diarizationCheckbox.GetValue():
            enable_diarization = True
            # Check if we already have a token (from env or previous entry)
            if not self.hf_token:
                dlg = wx.TextEntryDialog(
                    self,
                    message=(
                        "Speaker diarization requires a Hugging Face token for pyannote.\n\n"
                        "Please see the README for setup steps. Once complete, enter your token here:\n"
                        "(It will be saved to .env for future use)"
                    ),
                    caption="Enter Hugging Face Token"
                )
                if dlg.ShowModal() == wx.ID_OK:
                    token_val = dlg.GetValue().strip()
                    if token_val:
                        self.hf_token = token_val
                        # Save to .env file
                        try:
                            env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
                            with open(env_path, "a") as f:
                                f.write(f"\nHF_TOKEN={token_val}\n")
                            print(f"Token saved to {env_path}")
                        except Exception as e:
                            print(f"Failed to save token to .env: {e}")
                    else:
                        self.hf_token = None
                dlg.Destroy()

            if not self.hf_token:
                wx.MessageBox(
                    "No token provided. Continuing without speaker diarization.",
                    "Info",
                    wx.OK | wx.ICON_INFORMATION
                )
                enable_diarization = False

        # Initialize audio processor
        audio_processor = AudioProcessor(
            output_audio_features_folder=None,
            output_transcripts_folder=self.extracted_transcripts_folder,
            status_callback=self.set_status_message,
            enable_speaker_diarization=enable_diarization,
            auth_token=self.hf_token if enable_diarization else None
        )

        # If diarization requested, try preloading to validate token early
        if enable_diarization:
            try:
                audio_processor.preload_speaker_diarizer()
            except Exception as e:
                wx.MessageBox(
                    f"Speaker diarization could not be enabled: {e}\nContinuing with transcript only.",
                    "Diarization Disabled",
                    wx.OK | wx.ICON_WARNING
                )
                audio_processor.enable_speaker_diarization = False

        # Run transcription in a separate thread
        thread = threading.Thread(target=self.extract_transcripts_batch, args=(audio_files, audio_processor))
        thread.start()


    def extract_transcripts_batch(self, audio_files, audio_processor):
        """Batch process all audio files to generate transcripts."""
        self._process_running = True
        try:
            def progress_callback(progress):
                self.update_progress(progress)
            
            audio_processor.extract_transcripts_batch(audio_files, progress_callback=progress_callback)
            wx.CallAfter(wx.MessageBox, "Transcription extraction completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            wx.CallAfter(wx.MessageBox, f"Error during transcript extraction: {e}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            self._process_running = False
            self.update_progress(0)  # Reset progress bar

    def on_align_features(self, event):
        folder_path = self.folderPicker.GetPath()
        if not folder_path:
            wx.MessageBox("Please select a folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.ensure_output_folders(folder_path)
        
        # Check if we have features and transcripts
        if not os.path.exists(self.extracted_audio_folder):
            wx.MessageBox("No audio features folder found. Please extract audio features first.", "Error", wx.OK | wx.ICON_ERROR)
            return
            
        # We don't strictly need transcripts folder to exist yet if we are going to generate them,
        # but we need audio files.
        audio_files = gui_utils.get_files_from_folder(folder_path, self.AUDIO_EXTENSIONS)
        if not audio_files:
            wx.MessageBox("No audio files found.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Run in thread
        thread = threading.Thread(target=self.align_features_batch, args=(audio_files,))
        thread.start()

    def align_features_batch(self, audio_files):
        self._process_running = True
        try:
            total_files = len(audio_files)
            
            # Initialize processor
            # We need token for PyAnnote if we were doing diarization, but for alignment we just need Whisper
            # We'll pass the token just in case
            audio_processor = AudioProcessor(
                output_audio_features_folder=self.extracted_audio_folder,
                output_transcripts_folder=self.extracted_transcripts_folder,
                status_callback=self.set_status_message,
                auth_token=self.hf_token
            )
            
            alignment_pairs = []
            
            for i, audio_file in enumerate(audio_files, start=1):
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                
                # 1. Ensure we have word-level transcript (JSON)
                json_path = os.path.join(self.extracted_transcripts_folder, f"{base_name}_words.json")
                if not os.path.exists(json_path):
                    self.set_status_message(f"📝 Generating word-level transcript for: {base_name}")
                    try:
                        # Force word timestamps
                        audio_processor.extract_transcript(audio_file, word_timestamps=True)
                    except Exception as e:
                        print(f"Failed to generate transcript for {base_name}: {e}")
                        continue
                
                # 2. Ensure we have features CSV
                # Feature files are usually named {base_name}.csv or similar in extracted_audio_folder
                # The AudioProcessor.extract_audio_features saves them as {base_name}.csv
                feature_csv = os.path.join(self.extracted_audio_folder, f"{base_name}.csv")
                if not os.path.exists(feature_csv):
                    self.set_status_message(f"🎵 Extracting audio features for: {base_name}")
                    try:
                        # We need to call extract_audio_features for this single file
                        # But AudioProcessor has batch method. Let's use the internal logic or just skip if missing?
                        # User should have clicked "Extract Audio Features" first.
                        # But we can try to be helpful.
                        # Actually, AudioProcessor.extract_audio_features takes a list.
                        # Let's just warn if missing for now to avoid complexity of re-extracting everything.
                        print(f"Features missing for {base_name}, skipping.")
                        continue
                    except Exception:
                        continue
                        
                # 3. Output path
                output_csv = os.path.join(self.extracted_audio_folder, f"{base_name}_aligned.csv")
                alignment_pairs.append((feature_csv, json_path, output_csv))
                
                self.update_progress(int((i / total_files) * 50)) # First 50% for prep

            if not alignment_pairs:
                wx.CallAfter(wx.MessageBox, "No valid pairs found to align. Ensure you have extracted audio features.", "Warning", wx.OK | wx.ICON_WARNING)
                return

            # Run alignment
            audio_processor.align_features_batch(
                alignment_pairs, 
                progress_callback=lambda p: self.update_progress(50 + int(p/2))
            )
            
            wx.CallAfter(wx.MessageBox, "Feature alignment completed!", "Success", wx.OK | wx.ICON_INFORMATION)
            
        except Exception as e:
            wx.CallAfter(wx.MessageBox, f"Alignment failed: {e}", "Error", wx.OK | wx.ICON_ERROR)
        finally:
            self._process_running = False
            self.update_progress(0)

def main():
    # Ensure ffmpeg is available before the UI starts doing conversions
    if not gui_utils.ensure_ffmpeg_available():
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
