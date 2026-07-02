'''
This is the main script for multisocial app

'''

# Import necessary system and utility modules
import glob
import json
import os
import threading

# Set up GPU environment specially for Mediapipe (specific for Saturn Cloud), if you use some other high performance computing platform check compatibility before usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Make sure the system uses the GPU
# Enable MPS fallback for Mac to prevent freezes on unsupported operations
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


# Third-party libraries (assumed pre-installed via the project package metadata)
import wx
import unicodedata
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

import gui_utils
import runtime_services
from gui_utils import Theme
from ui_components import (
    GradientPanel,
    GlassPanel,
    ElevatedLogoPanel,
    TooltipButton,
    CustomGauge,
    FlatButton,
    ToggleTabBar,
    CustomCheckBox,
    SectionCard,
)

_PoseProcessorCls = None


def _get_pose_processor_class():
    """Load pose/Mediapipe only when the user starts a video step (startup stays light on Windows)."""
    global _PoseProcessorCls
    if os.environ.get("MULTISOCIAL_IMPORT_SMOKE_TEST") == "1":
        return None
    if _PoseProcessorCls is None:
        from pose import PoseProcessor

        _PoseProcessorCls = PoseProcessor
    return _PoseProcessorCls


# Keep packaged import smoke test lightweight by avoiding heavy ML/native imports.
if os.environ.get("MULTISOCIAL_IMPORT_SMOKE_TEST") != "1":
    gui_utils.ensure_ffmpeg_available()
    from audio import AudioProcessor
else:
    AudioProcessor = None

# Enable High DPI on Windows
gui_utils.setup_high_dpi()

## All dependencies are expected to be installed ahead of time via the project package metadata.

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
        self._diarization_install_running = False
        self._cancel_event = threading.Event()
        self._scalable_widgets = []  # (widget, base_size, bold)
        self._panels_horizontal = None  # tri-state: None = not yet laid out
        # File extensions constants
        self.VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
        self.AUDIO_EXTENSIONS = (".wav", ".wave", ".aiff", ".aif", ".aifc", ".flac", ".caf", ".au", ".snd")

    def _format_supported_extensions(self, extensions):
        return ", ".join(extensions)

    def _register_scalable(self, widget, base_size, bold=False):
        if widget is not None:
            self._scalable_widgets.append((widget, base_size, bold))

    def _begin_process(self):
        self._cancel_event.clear()
        self._process_running = True
        self._set_process_controls_enabled(False)
        if hasattr(self, "cancelBtn") and self.cancelBtn:
            self.cancelBtn.Show()
            self.cancelBtn.Enable(True)
        self._relayout_main()

    def _end_process(self, cancelled=False):
        self._process_running = False
        if hasattr(self, "cancelBtn") and self.cancelBtn:
            self.cancelBtn.Hide()
            self.cancelBtn.Enable(False)
        self.update_progress(0)
        self._set_process_controls_enabled(True)
        if cancelled:
            self.set_status_message("Processing cancelled.")
        else:
            self._refresh_idle_hint()
        self.update_buttons_enabled()
        self._relayout_main()

    def _relayout_main(self):
        """Reflow the scrolled main panel so showing/hiding the Cancel button
        actually reclaims its space (frame-level Layout() alone won't)."""
        if hasattr(self, "mainPanel") and self.mainPanel:
            self.mainPanel.Layout()
            try:
                self.mainPanel.FitInside()
            except Exception:
                pass
        self.Layout()

    def _set_process_controls_enabled(self, enabled):
        if hasattr(self, "modeTabs") and self.modeTabs:
            self.modeTabs.EnableTabs(enabled)
        if hasattr(self, "folderPicker") and self.folderPicker:
            self.folderPicker.Enable(enabled)
        if not enabled:
            # While a task runs, only Cancel stays live. Every action button and
            # every settings control is disabled so nothing else can be started
            # or changed mid-run. Re-enabling is handled by update_buttons_enabled
            # (called from _end_process), which restores the correct per-file gating.
            for btn in [
                getattr(self, "convertBtn", None),
                getattr(self, "extractFeaturesBtn", None),
                getattr(self, "embedFeaturesBtn", None),
                getattr(self, "embedTranscriptBtn", None),
                getattr(self, "verifyBtn", None),
                getattr(self, "extractAudioFeaturesBtn", None),
                getattr(self, "extractTranscriptsBtn", None),
                getattr(self, "alignFeaturesBtn", None),
                getattr(self, "installDiarizationBtn", None),
            ]:
                if btn:
                    btn.Enable(False)
            for ctrl in [
                getattr(self, "multiPersonCheckbox", None),
                getattr(self, "downscaleCheckbox", None),
                getattr(self, "frameStrideInput", None),
                getattr(self, "frameThresholdInput", None),
                getattr(self, "diarizationCheckbox", None),
                getattr(self, "wordTimestampsCheckbox", None),
            ]:
                if ctrl is not None:
                    ctrl.Enable(False)

    def on_cancel(self, event):
        if not self._process_running:
            return
        self._cancel_event.set()
        self.set_status_message("Cancelling...")
        if hasattr(self, "cancelBtn") and self.cancelBtn:
            self.cancelBtn.Enable(False)
        
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
        
        # Panels for Video and Audio options (invisible grouping — cards provide chrome)
        self.videoPanel = GlassPanel(pnl, chrome=False)
        self.audioPanel = GlassPanel(pnl, chrome=False)
        
        self._create_video_panel()
        self._create_audio_panel()
        
        # Assemble panels (centered to a content column; width set in _update_panel_sizes)
        vbox.AddSpacer(self.FromDIP(Theme.SPACE_SM))
        vbox.Add(self.videoPanel, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=Theme.SPACE_SM)
        vbox.Add(self.audioPanel, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=Theme.SPACE_SM)
        vbox.AddSpacer(self.FromDIP(Theme.SPACE_LG))
        
        # Status and Progress
        self._create_status_and_progress(pnl, vbox)
        
        pnl.SetSizer(vbox)
        self.mainPanel = pnl
        self.mainPanel.Layout()
        self.mainPanel.FitInside()
        
        # Final setup (sizing, binding, etc.)
        self._finalize_setup(vbox)

    def _create_header(self, pnl, vbox):
        new_width, new_height = wx.GetDisplaySize()
        logo_size = min(int(new_height * 0.055), 96)

        # Header lives in a centered, width-clamped container (like the cards) so the
        # logo/title left edge lines up with the content column at every window size.
        self.headerBar = GlassPanel(pnl, chrome=False)
        header_row = wx.BoxSizer(wx.HORIZONTAL)

        logo_path = runtime_services.resource_path("assets", "MultiSOCIAL_logo.png")
        logo_image = wx.Image(logo_path, wx.BITMAP_TYPE_ANY)
        logo_image = logo_image.Scale(logo_size, logo_size, wx.IMAGE_QUALITY_HIGH)
        logo_bmp = wx.Bitmap(logo_image)

        elevated_logo = ElevatedLogoPanel(self.headerBar, logo_bmp)
        elevated_logo.SetMinSize((logo_size + 16, logo_size + 16))
        header_row.Add(elevated_logo, flag=wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, border=Theme.SPACE_LG)

        title_col = wx.BoxSizer(wx.VERTICAL)
        self.logoLabel = gui_utils.create_transparent_text(self.headerBar, label="MultiSOCIAL Toolbox", style=wx.ALIGN_LEFT)
        self.logoLabel.SetFont(Theme.get_font(Theme.FONT_DISPLAY, bold=True))
        self.logoLabel.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_WHITE))
        self._register_scalable(self.logoLabel, Theme.FONT_DISPLAY, bold=True)
        title_col.Add(self.logoLabel, flag=wx.ALIGN_LEFT)

        self.subtitleLabel = gui_utils.create_transparent_text(
            self.headerBar, label="Pose · Audio · Alignment", style=wx.ALIGN_LEFT
        )
        self.subtitleLabel.SetFont(Theme.get_font(Theme.FONT_SUBTITLE))
        self.subtitleLabel.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_MUTED))
        self._register_scalable(self.subtitleLabel, Theme.FONT_SUBTITLE)
        title_col.Add(self.subtitleLabel, flag=wx.ALIGN_LEFT | wx.TOP, border=Theme.SPACE_XS)

        header_row.Add(title_col, proportion=1, flag=wx.ALIGN_CENTER_VERTICAL)
        self.headerBar.SetSizer(header_row)
        vbox.Add(self.headerBar, flag=wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=Theme.SPACE_XL)

    def _create_mode_selection(self, pnl, vbox):
        self.modeTabs = ToggleTabBar(pnl)
        self.modeTabs.set_on_change(self.switch_mode)
        self._register_scalable(self.modeTabs.video_tab, Theme.FONT_BODY, bold=True)
        self._register_scalable(self.modeTabs.audio_tab, Theme.FONT_BODY, bold=True)
        vbox.Add(self.modeTabs, flag=wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=Theme.SPACE_LG)

    def _create_folder_picker(self, pnl, vbox):
        self.folderCard = SectionCard(pnl, heading="Data folder")
        folder_card = self.folderCard
        content = folder_card.content_sizer

        self.folderCaption = gui_utils.create_transparent_text(
            folder_card,
            label="Select a folder containing VIDEO files",
            style=wx.ALIGN_LEFT,
        )
        self.folderCaption.SetFont(Theme.get_font(Theme.FONT_BODY, bold=True))
        self.folderCaption.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_WHITE))
        self._register_scalable(self.folderCaption, Theme.FONT_BODY, bold=True)
        content.Add(self.folderCaption, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_SM)

        self.folderPicker = wx.DirPickerCtrl(
            folder_card, message="Select a folder containing media files"
        )
        self.folderPicker.Bind(wx.EVT_DIRPICKER_CHANGED, self.on_folder_changed)
        gui_utils.style_native_input(self.folderPicker)
        content.Add(self.folderPicker, flag=wx.EXPAND)

        vbox.Add(folder_card, flag=wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, border=Theme.SPACE_LG)

    def _create_video_panel(self):
        video_sizer = wx.BoxSizer(wx.VERTICAL)

        settings_card = SectionCard(self.videoPanel, heading="Settings")
        settings = settings_card.content_sizer

        # Toggles: stacked, left-aligned (full-width so the hover row reads cleanly)
        self.multiPersonCheckbox = CustomCheckBox(settings_card, label="Enable Multi-Person Pose")
        self.multiPersonCheckbox.SetFont(Theme.get_font(Theme.FONT_HEADING))
        self.multiPersonCheckbox.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_WHITE))
        self._register_scalable(self.multiPersonCheckbox, Theme.FONT_HEADING)
        self.multiPersonCheckbox.Bind(wx.EVT_CHECKBOX, lambda event: self.update_buttons_enabled())
        settings.Add(self.multiPersonCheckbox, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_SM)

        self.downscaleCheckbox = CustomCheckBox(settings_card, label="Downscale to 720p for processing")
        self.downscaleCheckbox.SetFont(Theme.get_font(Theme.FONT_BODY))
        self.downscaleCheckbox.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_WHITE))
        self._register_scalable(self.downscaleCheckbox, Theme.FONT_BODY)
        settings.Add(self.downscaleCheckbox, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_MD)

        # Labeled numeric inputs: 2-col grid (label | control) on a shared baseline.
        # Growable label column pushes the spin controls into one aligned right column.
        grid = wx.FlexGridSizer(rows=2, cols=2, vgap=self.FromDIP(Theme.SPACE_SM), hgap=self.FromDIP(Theme.SPACE_MD))
        grid.AddGrowableCol(0, 1)

        ds_label = gui_utils.create_transparent_text(settings_card, label="Process every k-th frame:")
        ds_label.SetFont(Theme.get_font(Theme.FONT_BODY))
        ds_label.SetForegroundColour(Theme.COLOR_TEXT_WHITE)
        self._register_scalable(ds_label, Theme.FONT_BODY)
        grid.Add(ds_label, flag=wx.ALIGN_CENTER_VERTICAL)

        self.frameStrideInput = wx.SpinCtrl(settings_card, value="1", min=1, max=10)
        self.frameStrideInput.SetFont(Theme.get_font(Theme.FONT_BODY))
        gui_utils.style_native_input(self.frameStrideInput)
        self._register_scalable(self.frameStrideInput, Theme.FONT_BODY)
        grid.Add(self.frameStrideInput, flag=wx.ALIGN_CENTER_VERTICAL)

        frame_threshold_label = gui_utils.create_transparent_text(
            settings_card, label="Frame Threshold for Bounding Box Recalibration:"
        )
        frame_threshold_label.SetFont(Theme.get_font(Theme.FONT_BODY))
        frame_threshold_label.SetForegroundColour(Theme.COLOR_TEXT_WHITE)
        self._register_scalable(frame_threshold_label, Theme.FONT_BODY)
        grid.Add(frame_threshold_label, flag=wx.ALIGN_CENTER_VERTICAL)

        self.frameThresholdInput = wx.SpinCtrl(settings_card, value="10", min=1, max=100)
        self.frameThresholdInput.SetFont(Theme.get_font(Theme.FONT_BODY))
        gui_utils.style_native_input(self.frameThresholdInput)
        self._register_scalable(self.frameThresholdInput, Theme.FONT_BODY)
        grid.Add(self.frameThresholdInput, flag=wx.ALIGN_CENTER_VERTICAL)

        settings.Add(grid, flag=wx.EXPAND)

        actions_card = SectionCard(self.videoPanel, heading="Actions")
        actions = actions_card.content_sizer
        button_font = Theme.get_font(Theme.FONT_BUTTON)

        self.convertBtn, hbox_convert = TooltipButton.create_with_icon(
            actions_card,
            'Convert video to audio',
            f'Converts video files ({self._format_supported_extensions(self.VIDEO_EXTENSIONS)}) to audio files (.wav) for further processing',
            font=button_font,
            handler=self.on_convert,
            variant=FlatButton.VARIANT_SECONDARY,
        )
        self._register_scalable(self.convertBtn, Theme.FONT_BUTTON)
        actions.Add(hbox_convert, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_SM)

        self.extractFeaturesBtn, hbox_extract_pose = TooltipButton.create_with_icon(
            actions_card,
            'Extract Pose Features',
            'Extracts human pose landmarks and features from video files using MediaPipe. Supports single and multi-person detection.',
            font=button_font,
            handler=self.on_extract_features,
        )
        self._register_scalable(self.extractFeaturesBtn, Theme.FONT_BUTTON)
        actions.Add(hbox_extract_pose, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_SM)

        self.embedFeaturesBtn, hbox_embed_pose = TooltipButton.create_with_icon(
            actions_card,
            'Embed Pose Features',
            "Overlays each tracked person's pose skeleton onto the video in its own color, with a legend. Landmark brightness reflects detection confidence (brighter = more confident). Run Extract Pose Features first.",
            font=button_font,
            handler=self.on_embed_poses,
            variant=FlatButton.VARIANT_SECONDARY,
        )
        self._register_scalable(self.embedFeaturesBtn, Theme.FONT_BUTTON)
        actions.Add(hbox_embed_pose, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_SM)

        # Sub-option of "Embed Transcript on Video": burn captions onto the
        # pose-embedded video instead of the raw source. Indented + hinted so it
        # reads as a modifier of the action button below it.
        self.captionPoseVideoCheckbox = CustomCheckBox(actions_card, label="Add captions to pose-embedded video")
        self.captionPoseVideoCheckbox.SetFont(Theme.get_font(Theme.FONT_BODY))
        self.captionPoseVideoCheckbox.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_WHITE))
        self.captionPoseVideoCheckbox.SetToolTip(
            "On: burn the transcript onto the pose-embedded video, saved as "
            "*_pose_captioned.mp4.\n"
            "Off: caption the original video (*_captioned.mp4).\n"
            "Requires Embed Pose Features and Extract Transcripts to have run first.\n"
            "If both single-person and multi-person pose videos exist, the newest one is used."
        )
        self._register_scalable(self.captionPoseVideoCheckbox, Theme.FONT_BODY)
        self.captionPoseVideoCheckbox.Bind(wx.EVT_CHECKBOX, lambda event: self.update_buttons_enabled())

        self.captionPoseVideoHint = gui_utils.create_transparent_text(
            actions_card, label="Uses newest pose video · outputs *_pose_captioned.mp4"
        )
        self.captionPoseVideoHint.SetFont(Theme.get_font(Theme.FONT_CAPTION))
        self.captionPoseVideoHint.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_MUTED))
        self._register_scalable(self.captionPoseVideoHint, Theme.FONT_CAPTION)

        caption_pose_opt = wx.BoxSizer(wx.VERTICAL)
        caption_pose_opt.Add(self.captionPoseVideoCheckbox, flag=wx.EXPAND)
        caption_pose_opt.Add(
            self.captionPoseVideoHint, flag=wx.TOP, border=Theme.SPACE_XS
        )
        actions.Add(
            caption_pose_opt,
            flag=wx.EXPAND | wx.LEFT | wx.BOTTOM,
            border=Theme.SPACE_MD,
        )

        self.embedTranscriptBtn, hbox_embed_transcript = TooltipButton.create_with_icon(
            actions_card,
            'Embed Transcript on Video',
            "Burns the turn-by-turn transcript (with speaker labels when diarization was used) onto the video as captions, for reviewing transcription accuracy. Run Extract Transcripts first.",
            font=button_font,
            handler=self.on_embed_transcript,
            variant=FlatButton.VARIANT_SECONDARY,
        )
        self._register_scalable(self.embedTranscriptBtn, Theme.FONT_BUTTON)
        actions.Add(hbox_embed_transcript, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_SM)

        self.verifyBtn, hbox_verify = TooltipButton.create_with_icon(
            actions_card,
            'Verify Pose Match',
            'Checks that pose CSVs match the embedded pose videos and saves a sampled QA report with worst-frame thumbnails.',
            font=button_font,
            handler=self.on_verify_consistency,
            variant=FlatButton.VARIANT_SECONDARY,
        )
        self._register_scalable(self.verifyBtn, Theme.FONT_BUTTON)
        actions.Add(hbox_verify, flag=wx.EXPAND)

        gap = self.FromDIP(Theme.SPACE_MD)
        video_sizer.Add(settings_card, proportion=0, flag=wx.EXPAND)
        video_sizer.Add((gap, gap))
        video_sizer.Add(actions_card, proportion=0, flag=wx.EXPAND)
        self.videoPanel.SetSizer(video_sizer)
        self._video_cards = (settings_card, actions_card)

    def _create_audio_panel(self):
        audio_sizer = wx.BoxSizer(wx.VERTICAL)

        settings_card = SectionCard(self.audioPanel, heading="Settings")
        settings = settings_card.content_sizer

        self.diarizationCheckbox = CustomCheckBox(settings_card, label="Enable speaker diarization")
        self.diarizationCheckbox.SetFont(Theme.get_font(Theme.FONT_BODY))
        self.diarizationCheckbox.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_WHITE))
        self._register_scalable(self.diarizationCheckbox, Theme.FONT_BODY)
        settings.Add(self.diarizationCheckbox, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_SM)

        self.wordTimestampsCheckbox = CustomCheckBox(
            settings_card, label="Save word-level timestamps (for Align Features)"
        )
        self.wordTimestampsCheckbox.SetFont(Theme.get_font(Theme.FONT_BODY))
        self.wordTimestampsCheckbox.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_WHITE))
        self._register_scalable(self.wordTimestampsCheckbox, Theme.FONT_BODY)
        settings.Add(self.wordTimestampsCheckbox, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_MD)

        self.diarizationStatusLabel = gui_utils.create_transparent_text(
            settings_card,
            label="Diarization status: checking optional component...",
            style=wx.ALIGN_LEFT,
        )
        self.diarizationStatusLabel.SetFont(Theme.get_font(Theme.FONT_CAPTION))
        self.diarizationStatusLabel.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_MUTED))
        self._register_scalable(self.diarizationStatusLabel, Theme.FONT_CAPTION)
        settings.Add(self.diarizationStatusLabel, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_SM)

        self.installDiarizationBtn = FlatButton(
            settings_card, label="Install Complete Toolbox", variant=FlatButton.VARIANT_SECONDARY
        )
        self.installDiarizationBtn.SetFont(Theme.get_font(Theme.FONT_CAPTION, bold=True))
        self.installDiarizationBtn.Bind(wx.EVT_BUTTON, self.on_install_diarization)
        self._register_scalable(self.installDiarizationBtn, Theme.FONT_CAPTION, bold=True)
        settings.Add(self.installDiarizationBtn, flag=wx.ALIGN_LEFT)

        actions_card = SectionCard(self.audioPanel, heading="Actions")
        actions = actions_card.content_sizer
        button_font = Theme.get_font(Theme.FONT_BUTTON)

        self.extractAudioFeaturesBtn, hbox_extract_audio = TooltipButton.create_with_icon(
            actions_card,
            'Extract Audio Features',
            f'Extracts acoustic features from supported audio files ({self._format_supported_extensions(self.AUDIO_EXTENSIONS)}) including MFCC, spectral features, and prosodic characteristics.',
            font=button_font,
            handler=self.on_extract_audio_features,
        )
        self._register_scalable(self.extractAudioFeaturesBtn, Theme.FONT_BUTTON)
        actions.Add(hbox_extract_audio, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_SM)

        self.extractTranscriptsBtn, hbox_extract_transcripts = TooltipButton.create_with_icon(
            actions_card,
            'Extract Transcripts',
            f'Converts speech in supported audio files ({self._format_supported_extensions(self.AUDIO_EXTENSIONS)}) to text transcripts using automatic speech recognition (ASR) technology.',
            font=button_font,
            handler=self.on_extract_transcripts,
            variant=FlatButton.VARIANT_SECONDARY,
        )
        self._register_scalable(self.extractTranscriptsBtn, Theme.FONT_BUTTON)
        actions.Add(hbox_extract_transcripts, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_SM)

        self.alignFeaturesBtn, hbox_align = TooltipButton.create_with_icon(
            actions_card,
            'Align Features',
            'Aligns extracted audio features with word-level transcripts. Requires both features and transcripts.',
            font=button_font,
            handler=self.on_align_features,
            variant=FlatButton.VARIANT_SECONDARY,
        )
        self._register_scalable(self.alignFeaturesBtn, Theme.FONT_BUTTON)
        actions.Add(hbox_align, flag=wx.EXPAND)

        gap = self.FromDIP(Theme.SPACE_MD)
        audio_sizer.Add(settings_card, proportion=0, flag=wx.EXPAND)
        audio_sizer.Add((gap, gap))
        audio_sizer.Add(actions_card, proportion=0, flag=wx.EXPAND)
        self.audioPanel.SetSizer(audio_sizer)
        self._audio_cards = (settings_card, actions_card)

    def _create_status_and_progress(self, pnl, vbox):
        self.footerCard = SectionCard(pnl, heading="Status")
        footer_card = self.footerCard
        footer = footer_card.content_sizer

        self.statusLabel = gui_utils.create_transparent_text(
            footer_card, label="", style=wx.ALIGN_LEFT
        )
        self.statusLabel.SetForegroundColour(Theme.colour(Theme.COLOR_TEXT_WHITE))
        self.statusLabel.SetFont(Theme.get_font(Theme.FONT_BODY, bold=True))
        self._register_scalable(self.statusLabel, Theme.FONT_BODY, bold=True)
        footer.Add(self.statusLabel, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_SM)

        self.progress = CustomGauge(footer_card, range=100)
        self.progress.SetMinSize((-1, self.FromDIP(34)))
        footer.Add(self.progress, flag=wx.EXPAND | wx.BOTTOM, border=Theme.SPACE_SM)

        self.cancelBtn = FlatButton(footer_card, label="Cancel", variant=FlatButton.VARIANT_DANGER)
        self.cancelBtn.SetFont(Theme.get_font(Theme.FONT_BODY, bold=True))
        self.cancelBtn.Bind(wx.EVT_BUTTON, self.on_cancel)
        self.cancelBtn.Hide()
        self._register_scalable(self.cancelBtn, Theme.FONT_BODY, bold=True)
        footer.Add(self.cancelBtn, flag=wx.EXPAND)

        vbox.Add(footer_card, flag=wx.ALIGN_CENTER_HORIZONTAL | wx.TOP | wx.BOTTOM, border=Theme.SPACE_LG)

    def _finalize_setup(self, vbox):
        # Capture a design baseline size
        best_min = vbox.CalcMin()
        self._baseline_size = (max(400, best_min.width), max(640, best_min.height))
        
        self.Bind(wx.EVT_SIZE, self.on_resize)
        
        screen_w, screen_h = wx.GetDisplaySize()
        max_w = max(600, screen_w - 80)
        max_h = max(600, screen_h - 80)
        base_w, base_h = self._baseline_size
        min_w = min(base_w, max_w)
        min_h = min(base_h, max_h)
        self.SetSizeHints(min_w, min_h, max_w, max_h)
        self.SetSize((min_w, min_h))
        self.SetTitle(runtime_services.get_app_title())
        
        # Set Window Icon
        try:
            logo_path = runtime_services.resource_path("assets", "MultiSOCIAL_logo.png")
            if os.path.exists(logo_path):
                icon = wx.Icon()
                icon.CopyFromBitmap(wx.Bitmap(logo_path, wx.BITMAP_TYPE_ANY))
                self.SetIcon(icon)
        except Exception:
            pass
            
        self.Centre()
        
        self.active_mode = 'video'
        # Try to load token from environment variable first
        self.hf_token = runtime_services.load_hf_token()
        self.switch_mode('video')
        self.refresh_diarization_state()
        self._update_panel_sizes()
        self.update_buttons_enabled()
        try:
            wx.CallAfter(self._apply_status_wrap_and_center)
        except Exception:
            pass

    def switch_mode(self, mode):
        self.active_mode = mode
        if hasattr(self, "modeTabs") and self.modeTabs:
            self.modeTabs.set_selected(mode)
        if mode == 'video':
            self.videoPanel.Show()
            self.audioPanel.Hide()
            self.folderCaption.SetLabel("Select a folder containing VIDEO files")
        else:
            self.videoPanel.Hide()
            self.audioPanel.Show()
            self.folderCaption.SetLabel(
                f"Select a folder containing AUDIO files ({self._format_supported_extensions(self.AUDIO_EXTENSIONS)})"
            )
        self._update_panel_sizes()
        self.Layout()
        pnl = self.GetChildren()[0] if self.GetChildren() else None
        if pnl and hasattr(pnl, "Layout"):
            pnl.Layout()
        if not self._process_running:
            self.update_progress(0)
            self._refresh_idle_hint()
        self.update_buttons_enabled()

    def _refresh_idle_hint(self):
        """Empty/idle-state guidance in the STATUS line: prompts for a folder,
        warns when none match, or confirms how many files are ready."""
        if self._process_running or not hasattr(self, 'statusLabel'):
            return
        folder = self.get_selected_folder_path()
        kind = "video" if self.active_mode == 'video' else "audio"
        if not folder:
            self.set_status_message("Select a data folder to begin.")
            return
        try:
            if self.active_mode == 'video':
                files = gui_utils.get_files_from_folder(
                    gui_utils.resolved_dataset_root(folder), self.VIDEO_EXTENSIONS
                )
            else:
                files = gui_utils.get_audio_files_for_processing(folder, self.AUDIO_EXTENSIONS)
        except Exception:
            files = []
        n = len(files)
        if n == 0:
            self.set_status_message(f"No {kind} files found in this folder.")
        else:
            self.set_status_message(f"Ready — {n} {kind} file{'s' if n != 1 else ''} found.")

    def on_folder_changed(self, event):
        normalized_path = self.get_selected_folder_path()
        if normalized_path and normalized_path != self.folderPicker.GetPath():
            try:
                self.folderPicker.SetPath(normalized_path)
            except Exception:
                pass
        self.ensure_output_folders(normalized_path)
        self.update_buttons_enabled()
        self._refresh_idle_hint()

    def get_selected_folder_path(self):
        return gui_utils.normalize_path(self.folderPicker.GetPath())

    def refresh_diarization_state(self):
        self.diarizationFeatureState = runtime_services.get_diarization_feature_state()
        if not hasattr(self, 'diarizationCheckbox'):
            return

        state = self.diarizationFeatureState
        status = state["status"]
        installed = bool(state["installed"])
        can_self_install = bool(state["can_self_install"])
        last_error = str(state.get("last_error") or "").strip()

        if installed:
            checkbox_label = "Enable speaker diarization (installed; requires Hugging Face token)"
            status_label = "Diarization status: installed and ready."
            button_label = "Optional Component Installed"
            button_enabled = False
        elif status == "installing":
            checkbox_label = "Enable speaker diarization (optional add-on)"
            status_label = "Diarization status: installing optional component..."
            button_label = "Installing Optional Component..."
            button_enabled = False
        elif status == "failed":
            checkbox_label = "Enable speaker diarization (optional add-on)"
            detail = f" Last error: {last_error}" if last_error else ""
            if can_self_install:
                status_label = f"Diarization status: install failed.{detail}"
                button_label = "Retry Complete Install"
                button_enabled = True
            else:
                status_label = f"Diarization status: not available in this build.{detail}"
                button_label = "Use a Complete Build"
                button_enabled = False
        else:
            checkbox_label = "Enable speaker diarization (optional add-on)"
            if can_self_install:
                status_label = "Diarization status: not installed. Click Install Complete Toolbox for one-click setup."
                button_label = "Install Complete Toolbox"
                button_enabled = True
            else:
                status_label = "Diarization status: not installed in this build. Use a Complete installer."
                button_label = "Use a Complete Build"
                button_enabled = False

        self.diarizationCheckbox.SetLabel(checkbox_label)
        if not installed and status != "installing":
            self.diarizationCheckbox.SetValue(False)
        self.diarizationStatusLabel.SetLabel(status_label)
        self.installDiarizationBtn.SetLabel(button_label)
        self.installDiarizationBtn.Enable(
            button_enabled and not self._diarization_install_running and self._folder_allows_controls()
        )

        try:
            self.audioPanel.Layout()
            self.mainPanel.Layout()
            self.Layout()
        except Exception:
            pass

    def _update_panel_sizes(self):
        """Size the centered content column and choose 1- vs 2-column panel layout.

        Below the breakpoint the Settings/Actions cards stack (narrow column);
        at/above it they sit side-by-side and the column widens to use the frame,
        which removes the large side margins on desktop-sized windows.
        """
        if not hasattr(self, 'videoPanel') or not hasattr(self, 'audioPanel'):
            return
        try:
            cur_w, _cur_h = self.GetSize()
            horizontal = cur_w >= self.FromDIP(860)
            if horizontal:
                target_w = min(int(cur_w * 0.90), self.FromDIP(1180))
                target_w = max(target_w, self.FromDIP(720))
            else:
                target_w = min(max(self.FromDIP(360), int(cur_w * 0.70)), self.FromDIP(640))

            # All content blocks share one width so their edges align.
            tab_h = self.FromDIP(50)
            for block in (
                getattr(self, 'headerBar', None),
                getattr(self, 'modeTabs', None),
                getattr(self, 'folderCard', None),
                self.videoPanel,
                self.audioPanel,
                getattr(self, 'footerCard', None),
            ):
                if not block:
                    continue
                if block is getattr(self, 'modeTabs', None):
                    block.SetMinSize((target_w, tab_h))
                    block.SetMaxSize((target_w, tab_h))
                else:
                    block.SetMinSize((target_w, -1))
                    block.SetMaxSize((target_w, 10000))

            self._apply_panel_orientation(horizontal)
        except Exception:
            pass

    def _apply_panel_orientation(self, horizontal):
        """Rebuild each panel's sizer only when the 1-/2-column mode flips.

        A fresh BoxSizer (rather than SetOrientation) keeps this working across
        wxPython versions. Cards are reused — only the sizer is replaced."""
        if horizontal == self._panels_horizontal:
            return
        pairs = (
            (self.videoPanel, getattr(self, '_video_cards', None)),
            (self.audioPanel, getattr(self, '_audio_cards', None)),
        )
        try:
            for panel, cards in pairs:
                if not panel or not cards:
                    continue
                settings_card, actions_card = cards
                gap = self.FromDIP(Theme.SPACE_MD)
                if horizontal:
                    # Equal-width columns, each at its natural height (top-aligned)
                    # so the shorter Settings card doesn't stretch with empty space.
                    sizer = wx.BoxSizer(wx.HORIZONTAL)
                    card_flag = wx.ALIGN_TOP
                    proportion = 1
                else:
                    sizer = wx.BoxSizer(wx.VERTICAL)
                    card_flag = wx.EXPAND
                    proportion = 0
                sizer.Add(settings_card, proportion=proportion, flag=card_flag)
                sizer.Add((gap, gap))
                sizer.Add(actions_card, proportion=proportion, flag=card_flag)
                panel.SetSizer(sizer, deleteOld=True)
                panel.Layout()
            self._panels_horizontal = horizontal
        except Exception:
            # Leave whatever orientation is currently applied; never crash layout.
            pass

    def _folder_allows_controls(self):
        return bool(self.get_selected_folder_path())

    def update_buttons_enabled(self):
        if self._process_running:
            return
        folder_path = self.get_selected_folder_path()
        has_folder = bool(folder_path)
        video_files = []
        audio_files = []
        has_pose_csv = False
        has_transcript_srt = False
        has_embedded_video = False
        has_all_embedded_pose = False
        try:
            if has_folder:
                workspace_root = gui_utils.resolved_dataset_root(folder_path)
                video_files = gui_utils.get_files_from_folder(workspace_root, self.VIDEO_EXTENSIONS)
                audio_files = gui_utils.get_audio_files_for_processing(folder_path, self.AUDIO_EXTENSIONS)
                pose_dir = os.path.join(workspace_root, "pose_features")
                pose_multi_person = bool(
                    getattr(self, 'multiPersonCheckbox', None)
                    and self.multiPersonCheckbox.GetValue()
                )
                has_pose_csv = self._has_all_pose_csvs(pose_dir, video_files, pose_multi_person)
                transcripts_dir = gui_utils.transcripts_output_folder(folder_path)
                has_transcript_srt = self._has_all_transcript_srts(transcripts_dir, video_files)
                embedded_dir = os.path.join(workspace_root, "embedded_pose")
                has_embedded_video = self._has_any_embedded_video(embedded_dir)
                has_all_embedded_pose = self._has_all_embedded_pose_videos(embedded_dir, video_files)
        except Exception:
            pass

        enable_video = has_folder and len(video_files) > 0
        enable_audio = has_folder and len(audio_files) > 0
        caption_pose_requested = bool(
            getattr(self, 'captionPoseVideoCheckbox', None)
            and self.captionPoseVideoCheckbox.GetValue()
        )

        for ctrl in [
            getattr(self, 'multiPersonCheckbox', None),
            getattr(self, 'downscaleCheckbox', None),
            getattr(self, 'frameStrideInput', None),
            getattr(self, 'frameThresholdInput', None),
            getattr(self, 'captionPoseVideoCheckbox', None),
            getattr(self, 'diarizationCheckbox', None),
            getattr(self, 'wordTimestampsCheckbox', None),
        ]:
            if ctrl is not None:
                ctrl.Enable(has_folder)

        for btn in [
            getattr(self, 'convertBtn', None),
            getattr(self, 'extractFeaturesBtn', None),
        ]:
            if btn:
                btn.Enable(enable_video)

        # Embed depends on extraction having run: keep it locked (greyed, with a
        # hover hint) until pose CSVs exist for the selected videos.
        embed_btn = getattr(self, 'embedFeaturesBtn', None)
        if embed_btn:
            embed_btn.Enable(enable_video)
            if enable_video and not has_pose_csv:
                embed_btn.set_locked(
                    True,
                    "Run 'Extract Pose Features' first. Pose data is saved in the "
                    "'pose_features' folder.",
                )
            else:
                embed_btn.set_locked(False)

        # Transcript embed depends on transcription having produced an .srt sidecar.
        embed_transcript_btn = getattr(self, 'embedTranscriptBtn', None)
        if embed_transcript_btn:
            embed_transcript_btn.Enable(enable_video)
            if enable_video and not has_transcript_srt:
                embed_transcript_btn.set_locked(
                    True,
                    "Run 'Extract Transcripts' (Audio tab) first. Transcripts are saved "
                    "as '.srt' files in the 'transcripts' folder.",
                )
            elif enable_video and caption_pose_requested and not has_all_embedded_pose:
                embed_transcript_btn.set_locked(
                    True,
                    "Run 'Embed Pose Features' first. Captions will be added to videos in the 'embedded_pose' folder.",
                )
            else:
                embed_transcript_btn.set_locked(False)

        # Verify needs pose-embedded videos to compare against their CSVs.
        verify_btn = getattr(self, 'verifyBtn', None)
        if verify_btn:
            verify_btn.Enable(enable_video)
            if enable_video and not has_embedded_video:
                verify_btn.set_locked(
                    True,
                    "Run 'Embed Pose on Video' first. Results are saved in the "
                    "'embedded_pose' folder.",
                )
            else:
                verify_btn.set_locked(False)

        for btn in [
            getattr(self, 'extractAudioFeaturesBtn', None),
            getattr(self, 'extractTranscriptsBtn', None),
            getattr(self, 'alignFeaturesBtn', None),
        ]:
            if btn:
                btn.Enable(enable_audio)

        self.refresh_diarization_state()

    def _has_all_pose_csvs(self, pose_dir, video_files, multi_person=False):
        """True if every selected video has extracted pose CSVs.

        Mirrors the mode-specific naming used by pose.find_pose_csv_paths
        without importing the heavy pose module on every folder change.
        """
        if not video_files or not pose_dir or not os.path.isdir(pose_dir):
            return False
        for video in video_files:
            base = os.path.splitext(os.path.basename(video))[0]
            pattern = f"{base}_multi_ID_*.csv" if multi_person else f"{base}_ID_*.csv"
            if not glob.glob(os.path.join(pose_dir, pattern)):
                return False
        return True

    def _has_all_transcript_srts(self, transcripts_dir, video_files):
        """True if every selected video has a matching transcript .srt sidecar."""
        if not video_files or not transcripts_dir or not os.path.isdir(transcripts_dir):
            return False
        for video in video_files:
            base = os.path.splitext(os.path.basename(video))[0]
            if not os.path.exists(os.path.join(transcripts_dir, f"{base}.srt")):
                return False
        return True

    def _embedded_pose_video_path(self, embedded_dir, source_video):
        """Return the newest embedded-pose video path for a source video, if present."""
        if not embedded_dir or not source_video:
            return None
        base = os.path.splitext(os.path.basename(source_video))[0]
        candidates = [
            os.path.join(embedded_dir, f"{base}_pose.mp4"),
            os.path.join(embedded_dir, f"{base}_multi_pose.mp4"),
        ]
        existing = [candidate for candidate in candidates if os.path.exists(candidate)]
        if not existing:
            return None
        return max(existing, key=lambda path: os.path.getmtime(path))

    @staticmethod
    def _pose_source_base_from_embedded_video(video_path):
        """Return the original source basename for an embedded pose video."""
        base = os.path.splitext(os.path.basename(video_path))[0]
        for suffix in ("_multi_pose", "_pose"):
            if base.endswith(suffix):
                return base[: -len(suffix)]
        return base

    @staticmethod
    def _embedded_pose_video_is_multi(video_path):
        """True when an embedded pose video was produced from multi-person CSVs."""
        base = os.path.splitext(os.path.basename(video_path))[0]
        return base.endswith("_multi_pose")

    def _pose_csv_paths_for_embedded_video(self, video_path):
        """Return CSV paths matching the embedded video's pose mode."""
        csv_base = self._pose_source_base_from_embedded_video(video_path)
        is_multi = self._embedded_pose_video_is_multi(video_path)
        pattern = f"{csv_base}_multi_ID_*.csv" if is_multi else f"{csv_base}_ID_*.csv"
        return sorted(glob.glob(os.path.join(self.extracted_pose_folder, pattern)))

    def _pose_stride_for_embedded_video(self, video_path):
        """Return extraction stride for an embedded pose video, defaulting older outputs to 1."""
        csv_base = self._pose_source_base_from_embedded_video(video_path)
        suffix = "_multi" if self._embedded_pose_video_is_multi(video_path) else ""
        meta_path = os.path.join(self.extracted_pose_folder, f"{csv_base}{suffix}_meta.json")
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                return max(1, int(json.load(handle).get("frame_stride", 1)))
        except Exception:
            return 1

    def _has_all_embedded_pose_videos(self, embedded_dir, video_files):
        """True if every selected source video has a matching embedded pose video."""
        if not video_files or not embedded_dir or not os.path.isdir(embedded_dir):
            return False
        return all(self._embedded_pose_video_path(embedded_dir, video) for video in video_files)

    def _has_any_embedded_video(self, embedded_dir):
        """True if the embedded_pose folder holds at least one video to verify."""
        if not embedded_dir or not os.path.isdir(embedded_dir):
            return False
        return bool(gui_utils.get_files_from_folder(embedded_dir, self.VIDEO_EXTENSIONS))

    def on_resize(self, event):
        if self._baseline_size is None:
            self._baseline_size = self.GetSize()
        cur_w, cur_h = self.GetSize()
        base_w, base_h = self._baseline_size
        scale_w = max(cur_w / max(1, base_w), 1.0)
        scale_h = max(cur_h / max(1, base_h), 1.0)
        # Cap growth tightly: typography should stay consistent across window sizes
        # (responsive layout, not ballooning fonts, fills large windows). 2x looked
        # inconsistent and could overflow fixed-width cards.
        scale = min(scale_w, scale_h, 1.25)

        try:
            for widget, base_size, bold in self._scalable_widgets:
                if not widget:
                    continue
                # Some registered targets are lightweight font proxies (segmented
                # tab labels) with no IsShown(); treat those as always visible and
                # never let one widget abort the whole scaling pass.
                try:
                    is_shown = widget.IsShown() if hasattr(widget, "IsShown") else True
                    if is_shown:
                        new_size = max(9, int(round(base_size * scale)))
                        widget.SetFont(Theme.get_font(new_size, bold=bold))
                except Exception:
                    continue
            if hasattr(self, 'folderCaption') and self.folderCaption:
                try:
                    self.folderCaption.Wrap(max(240, int(cur_w * 0.6)))
                except Exception:
                    pass
            if hasattr(self, 'diarizationStatusLabel') and self.diarizationStatusLabel:
                try:
                    self.diarizationStatusLabel.Wrap(max(240, int(cur_w * 0.55)))
                except Exception:
                    pass
            if hasattr(self, 'statusLabel') and self.statusLabel:
                try:
                    wrap_width = max(200, int(cur_w * 0.9))
                    self.statusLabel.Wrap(wrap_width)
                    self.statusLabel.SetWindowStyleFlag(wx.ALIGN_CENTER_HORIZONTAL)
                except Exception:
                    pass
        except Exception:
            pass

        self._update_panel_sizes()
        self.mainPanel.Layout()
        self.mainPanel.FitInside()
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

    def on_install_diarization(self, event):
        if self._process_running:
            return
        state = runtime_services.get_diarization_feature_state()
        if state["installed"]:
            wx.MessageBox(
                "Optional diarization support is already installed.",
                "Installed",
                wx.OK | wx.ICON_INFORMATION,
            )
            return

        if not state["can_self_install"]:
            wx.MessageBox(
                "This build cannot install diarization in-place. Use a Complete build or rerun the installer with the complete profile.",
                "Complete Build Required",
                wx.OK | wx.ICON_INFORMATION,
            )
            return

        if self._diarization_install_running:
            return

        thread = threading.Thread(target=self._install_diarization_worker, daemon=True)
        thread.start()

    def _install_diarization_worker(self):
        self._diarization_install_running = True
        runtime_services.update_feature_state(runtime_services.DIARIZATION_FEATURE, status="installing", requested=True, last_error="")
        wx.CallAfter(self.refresh_diarization_state)
        try:
            success, error_message = runtime_services.install_diarization_support(
                status_callback=self.set_status_message,
                progress_callback=self.update_progress,
            )
            if success:
                wx.CallAfter(self._handle_diarization_install_success)
            else:
                wx.CallAfter(self._handle_diarization_install_failure, error_message or "Unknown install error")
        finally:
            self._diarization_install_running = False
            wx.CallAfter(self.refresh_diarization_state)
            wx.CallAfter(self.update_progress, 0)

    def _handle_diarization_install_success(self):
        self.refresh_diarization_state()
        self.diarizationCheckbox.SetValue(True)
        wx.MessageBox(
            "Optional speaker diarization support has been installed.\n\nRestarting the app is recommended before running diarized transcripts.",
            "Install Complete",
            wx.OK | wx.ICON_INFORMATION,
        )

    def _handle_diarization_install_failure(self, error_message):
        self.refresh_diarization_state()
        wx.MessageBox(
            f"Could not install optional diarization support:\n\n{error_message}",
            "Install Failed",
            wx.OK | wx.ICON_ERROR,
        )

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
        folder_path = gui_utils.normalize_path(folder_path)
        # Guard against empty/invalid path
        if not folder_path or not os.path.isdir(folder_path):
            self.converted_audio_folder = None
            self.extracted_pose_folder = None
            self.embedded_pose_folder = None
            self.extracted_audio_folder = None
            self.extracted_transcripts_folder = None
            self.captioned_video_folder = None
            return

        workspace_root = gui_utils.resolved_dataset_root(folder_path)
        converted_audio_dir = os.path.join(workspace_root, "converted_audio")

        # Canonical paths (actually created only when the folder has matching media).
        self.extracted_pose_folder = os.path.join(workspace_root, "pose_features")
        self.embedded_pose_folder = os.path.join(workspace_root, "embedded_pose")
        self.captioned_video_folder = os.path.join(workspace_root, "captioned_video")
        self.extracted_audio_folder = os.path.join(workspace_root, "audio_features")
        self.extracted_transcripts_folder = gui_utils.transcripts_output_folder(folder_path)

        video_files = gui_utils.get_files_from_folder(workspace_root, self.VIDEO_EXTENSIONS)
        if video_files:
            self.converted_audio_folder = converted_audio_dir
            try:
                os.makedirs(converted_audio_dir, exist_ok=True)
                os.makedirs(self.extracted_pose_folder, exist_ok=True)
                os.makedirs(self.embedded_pose_folder, exist_ok=True)
                os.makedirs(self.captioned_video_folder, exist_ok=True)
            except (OSError, PermissionError) as e:
                self.converted_audio_folder = None
                self.extracted_pose_folder = None
                self.embedded_pose_folder = None
                self.captioned_video_folder = None
                wx.CallAfter(wx.MessageBox, f"Cannot create output folders: {e}", "Warning", wx.OK | wx.ICON_WARNING)
        else:
            self.converted_audio_folder = None
            self.extracted_pose_folder = None
            self.embedded_pose_folder = None
            self.captioned_video_folder = None

        audio_files = gui_utils.get_audio_files_for_processing(folder_path, self.AUDIO_EXTENSIONS)
        if audio_files:
            try:
                for folder in [self.extracted_audio_folder, self.extracted_transcripts_folder]:
                    os.makedirs(folder, exist_ok=True)
            except (OSError, PermissionError) as e:
                self.extracted_audio_folder = None
                self.extracted_transcripts_folder = None
                wx.CallAfter(wx.MessageBox, f"Cannot create output folders: {e}", "Warning", wx.OK | wx.ICON_WARNING)
        else:
            self.extracted_audio_folder = None
            self.extracted_transcripts_folder = None

      
      
    def on_convert(self, event):
        """Convert all videos in a selected folder to WAV."""
        if self._process_running:
            return
        folder_path = self.get_selected_folder_path()
        if not folder_path:
            wx.MessageBox("Please select a folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.ensure_output_folders(folder_path)
        workspace_root = gui_utils.resolved_dataset_root(folder_path)
        video_files = gui_utils.get_files_from_folder(workspace_root, self.VIDEO_EXTENSIONS)

        if not video_files:
            wx.MessageBox("No video files found in the selected folder.", "Error", wx.OK | wx.ICON_ERROR)
            return

        # Process each video file in a separate thread
        self._begin_process()
        thread = threading.Thread(target=self.convert_all_videos_to_wav, args=(video_files,))
        thread.start()

    def convert_all_videos_to_wav(self, video_files):
        """Convert multiple videos to WAV format and save to output folder."""
        cancelled = False
        failures = []
        try:
            total_files = len(video_files)

            for i, video_file in enumerate(video_files):
                if self._cancel_event.is_set():
                    cancelled = True
                    break
                file_name = os.path.basename(video_file)
                self.set_status_message(f"Converting to WAV: {file_name}")
                try:
                    self.convert_to_wav(
                        video_file,
                        progress_callback=self.make_overall_progress_cb(i + 1, total_files),
                        cancel_check=lambda: self._cancel_event.is_set(),
                    )
                except Exception as e:
                    failures.append(f"{file_name}: {e}")
                    self.set_status_message(f"Failed to convert {file_name}: {e}")

            if not cancelled:
                if failures:
                    preview = "\n".join(failures[:5])
                    if len(failures) > 5:
                        preview += f"\n...and {len(failures) - 5} more."
                    wx.CallAfter(
                        wx.MessageBox,
                        "Video to audio conversion finished with errors.\n\n" + preview,
                        "Conversion Warning",
                        wx.OK | wx.ICON_WARNING,
                    )
                else:
                    wx.CallAfter(wx.MessageBox, "Video to audio conversion completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        finally:
            wx.CallAfter(self._end_process, cancelled)

    def on_verify_consistency(self, event):
        if self._process_running:
            return
        folder_path = self.get_selected_folder_path()
        if not folder_path:
            wx.MessageBox("Please select a folder before verifying consistency.", "No Folder Selected", wx.OK | wx.ICON_INFORMATION)
            return

        # Ensure folders and also create verification output dirs
        self.ensure_output_folders(folder_path)
        if not self.extracted_pose_folder or not self.embedded_pose_folder:
            wx.MessageBox(
                "There are no pose results to verify yet.\n\n"
                "Run 'Extract Pose Features' and then 'Embed Pose on Video' first.",
                "Nothing to Verify",
                wx.OK | wx.ICON_INFORMATION,
            )
            return

        verification_dir = os.path.join(gui_utils.resolved_dataset_root(folder_path), "verification")
        os.makedirs(verification_dir, exist_ok=True)
        worst_frames_root = os.path.join(verification_dir, "worst_frames")
        os.makedirs(worst_frames_root, exist_ok=True)

        # Collect embedded videos and match CSVs by basename
        embedded_videos = gui_utils.get_files_from_folder(self.embedded_pose_folder, self.VIDEO_EXTENSIONS)
        if not embedded_videos:
            wx.MessageBox(
                "No pose-embedded videos were found to verify.\n\n"
                "Run 'Embed Pose on Video' first. Results are saved in the "
                "'embedded_pose' folder inside your selected folder.",
                "Embedded Videos Needed",
                wx.OK | wx.ICON_INFORMATION,
            )
            return

        # Launch background verification to keep UI responsive
        self._begin_process()
        thread = threading.Thread(target=self._verify_consistency_batch, args=(embedded_videos, verification_dir, worst_frames_root))
        thread.start()

    def _verify_consistency_batch(self, embedded_videos, verification_dir, worst_frames_root):
        from verify_pose_embedding import verify, save_report

        cancelled = False
        try:
            summary = []
            total = len(embedded_videos)
            for i, video in enumerate(embedded_videos, start=1):
                if self._cancel_event.is_set():
                    cancelled = True
                    break
                base = os.path.splitext(os.path.basename(video))[0]
                csv_paths = self._pose_csv_paths_for_embedded_video(video)

                if not csv_paths:
                    self.set_status_message(f"⚠️ No CSVs found for {base}; skipping")
                    continue

                self.set_status_message(f"🔎 Verifying pose match: {os.path.basename(video)}")
                try:
                    worst_dir = os.path.join(worst_frames_root, base)
                    def verify_progress(local_progress, file_index=i, file_total=total):
                        overall = int(((file_index - 1) / max(1, file_total)) * 100 + (local_progress / max(1, file_total)))
                        self.update_progress(overall)

                    report = verify(
                        video_path=video,
                        csv_paths=csv_paths,
                        stride=self._pose_stride_for_embedded_video(video),
                        max_worst=10,
                        worst_dir=worst_dir,
                        processed_only=True,
                        metric='hit_rate',
                        hit_threshold=0.8,
                        window=5,
                        conf_threshold=0.0,
                        max_frames=300,
                        progress_callback=verify_progress,
                        cancel_check=lambda: self._cancel_event.is_set(),
                    )
                    out_json = os.path.join(verification_dir, f"{base}_report.json")
                    out_csv = os.path.join(verification_dir, f"{base}_worst.csv")
                    save_report(report, out_json, out_csv)
                    summary.append({
                        "basename": base,
                        "frames_compared": report.get("frames_compared"),
                        "eligible_frames": report.get("eligible_frames"),
                        "sampled": report.get("sampled"),
                        "max_frames": report.get("max_frames"),
                        "mean_hit_rate": report.get("mean_hit_rate"),
                        "min_hit_rate": report.get("min_hit_rate"),
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

            if not cancelled:
                wx.CallAfter(
                    wx.MessageBox,
                    "Pose match verification completed. See the 'verification' folder for reports and worst-frame thumbnails.",
                    "Success",
                    wx.OK | wx.ICON_INFORMATION,
                )
        finally:
            wx.CallAfter(self._end_process, cancelled)

    def convert_to_wav(self, filepath, progress_callback=None, cancel_check=None):
        """Convert a single video file to WAV using ffmpeg (with real progress)."""
        filepath = gui_utils.normalize_path(filepath)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(filepath)
        if not self.converted_audio_folder:
            raise FileNotFoundError("converted_audio output folder is unavailable")
        os.makedirs(self.converted_audio_folder, exist_ok=True)
        output_path = os.path.join(self.converted_audio_folder, os.path.splitext(os.path.basename(filepath))[0] + ".wav")
        ffmpeg_cmd = gui_utils.get_ffmpeg_executable() or "ffmpeg"

        if progress_callback:
            progress_callback(0)

        cmd = [
            ffmpeg_cmd, "-y",
            "-i", filepath,
            "-vn", "-acodec", "pcm_s16le", "-f", "wav",
            output_path,
        ]
        result = gui_utils.run_ffmpeg_with_progress(
            cmd, progress_callback=progress_callback, cancel_check=cancel_check
        )
        if result is False:
            return None

        print(f"Conversion complete: {output_path}")
        return output_path


    def on_extract_features(self, event):
        if self._process_running:
            return
        folder_path = self.get_selected_folder_path()
        if not folder_path:
            wx.MessageBox("Please select a folder before extracting pose features.", "No Folder Selected", wx.OK | wx.ICON_INFORMATION)
            return

        self.ensure_output_folders(folder_path)
        workspace_root = gui_utils.resolved_dataset_root(folder_path)
        video_files = gui_utils.get_files_from_folder(workspace_root, self.VIDEO_EXTENSIONS)

        if not video_files:
            wx.MessageBox(
                "No video files found in the selected folder.\n\n"
                "Add videos (.mp4, .mov, .avi, .mkv, .m4v) to the folder, then try again.",
                "No Videos Found", wx.OK | wx.ICON_INFORMATION,
            )
            return

        if not self.extracted_pose_folder:
            wx.MessageBox(
                "The app could not create the 'pose_features' folder for your results.\n\n"
                "Check that you have permission to write to the selected folder, then try again.",
                "Cannot Save Results",
                wx.OK | wx.ICON_ERROR,
            )
            return

        PoseCls = _get_pose_processor_class()
        if PoseCls is None:
            wx.MessageBox("Pose extraction is unavailable in this launch mode.", "Error", wx.OK | wx.ICON_ERROR)
            return

        stride_val = 1
        try:
            stride_val = max(1, int(self.frameStrideInput.GetValue()))
        except Exception:
            stride_val = 1
        downscale_to = (1280, 720) if (hasattr(self, "downscaleCheckbox") and self.downscaleCheckbox.GetValue()) else None

        try:
            pose_processor = PoseCls(
                self.extracted_pose_folder,
                status_callback=self.set_status_message,
                frame_threshold=self.frameThresholdInput.GetValue(),
                frame_stride=stride_val,
                downscale_to=downscale_to,
            )
            pose_processor.set_multi_person_mode(self.multiPersonCheckbox.GetValue())
        except Exception as e:
            wx.MessageBox(
                (
                    "Could not load pose processing (Mediapipe). On Windows this is often missing Visual C++ "
                    "runtime DLLs.\n\n"
                    f"{e}"
                ),
                "Pose Engine Error",
                wx.OK | wx.ICON_ERROR,
            )
            return

        self._begin_process()
        thread = threading.Thread(target=self.extract_pose_features_batch, args=(video_files, pose_processor))
        thread.start()

    def extract_pose_features_batch(self, video_files, pose_processor):
        """Batch process all video files to extract pose features."""
        cancelled = False
        try:
            total_files = len(video_files)

            for index, video_file in enumerate(video_files, start=1):
                if self._cancel_event.is_set():
                    cancelled = True
                    break
                self.set_status_message(f"Extracting pose from: {os.path.basename(video_file)}")
                result = pose_processor.extract_pose_features(
                    video_file,
                    progress_callback=self.make_overall_progress_cb(index, total_files),
                    cancel_check=self._cancel_event.is_set,
                )
                if result is False:
                    cancelled = True
                    break

            if not cancelled:
                wx.CallAfter(wx.MessageBox, "Pose feature extraction completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        finally:
            wx.CallAfter(self._end_process, cancelled)
            
           
    def on_embed_poses(self, event):
        if self._process_running:
            return
        folder_path = self.get_selected_folder_path()
        if not folder_path:
            wx.MessageBox("Please select a folder before embedding pose on video.", "No Folder Selected", wx.OK | wx.ICON_INFORMATION)
            return

        self.ensure_output_folders(folder_path)
        workspace_root = gui_utils.resolved_dataset_root(folder_path)
        video_files = gui_utils.get_files_from_folder(workspace_root, self.VIDEO_EXTENSIONS)

        if not video_files:
            wx.MessageBox(
                "No video files found in the selected folder.\n\n"
                "Add videos (.mp4, .mov, .avi, .mkv, .m4v) to the folder, then try again.",
                "No Videos Found", wx.OK | wx.ICON_INFORMATION,
            )
            return

        if not self.extracted_pose_folder or not self.embedded_pose_folder:
            wx.MessageBox(
                "The app could not create the output folders for pose videos.\n\n"
                "Check that you have permission to write to the selected folder, then try again.",
                "Cannot Save Results",
                wx.OK | wx.ICON_ERROR,
            )
            return

        from pose import find_pose_csv_paths

        multi_person = bool(
            hasattr(self, "multiPersonCheckbox") and self.multiPersonCheckbox.GetValue()
        )

        missing_csv_videos = [
            os.path.basename(video)
            for video in video_files
            if not find_pose_csv_paths(self.extracted_pose_folder, video, multi_person=multi_person)
        ]
        if missing_csv_videos:
            preview = ", ".join(missing_csv_videos[:3])
            if len(missing_csv_videos) > 3:
                preview += f", and {len(missing_csv_videos) - 3} more"
            wx.MessageBox(
                "Pose data is missing for these videos:\n"
                f"{preview}.\n\n"
                "Run 'Extract Pose Features' first. Pose data is saved as CSV files "
                "in the 'pose_features' folder inside your selected folder.",
                "Pose Data Needed",
                wx.OK | wx.ICON_INFORMATION,
            )
            return

        PoseCls = _get_pose_processor_class()
        if PoseCls is None:
            wx.MessageBox("Pose embedding is unavailable in this launch mode.", "Error", wx.OK | wx.ICON_ERROR)
            return

        stride_val = 1
        try:
            stride_val = max(1, int(self.frameStrideInput.GetValue()))
        except Exception:
            stride_val = 1
        downscale_to = (1280, 720) if (hasattr(self, "downscaleCheckbox") and self.downscaleCheckbox.GetValue()) else None

        try:
            pose_processor = PoseCls(
                output_csv_folder=self.extracted_pose_folder,
                output_video_folder=self.embedded_pose_folder,
                status_callback=self.set_status_message,
                frame_threshold=self.frameThresholdInput.GetValue(),
                frame_stride=stride_val,
                downscale_to=downscale_to,
            )
            pose_processor.set_multi_person_mode(self.multiPersonCheckbox.GetValue())
        except Exception as e:
            wx.MessageBox(
                (
                    "Could not load pose processing (Mediapipe). On Windows this is often missing Visual C++ "
                    "runtime DLLs.\n\n"
                    f"{e}"
                ),
                "Pose Engine Error",
                wx.OK | wx.ICON_ERROR,
            )
            return

        self._begin_process()
        thread = threading.Thread(target=self.embed_pose_batch, args=(video_files, pose_processor))
        thread.start()

    def embed_pose_batch(self, video_files, pose_processor):
        cancelled = False
        failed = False
        try:
            total_files = len(video_files)

            for index, video_file in enumerate(video_files, start=1):
                if self._cancel_event.is_set():
                    cancelled = True
                    break
                self.set_status_message(f"Embedding poses for: {os.path.basename(video_file)}")
                result = pose_processor.embed_pose_video(
                    video_file,
                    progress_callback=self.make_overall_progress_cb(index, total_files),
                    cancel_check=lambda: self._cancel_event.is_set(),
                )
                if result is False:
                    cancelled = True
                    break
                if result is None:
                    failed = True
                    self.set_status_message(
                        f"Skipped embedding for {os.path.basename(video_file)}: no pose CSV found."
                    )

            if cancelled:
                return
            if failed:
                wx.CallAfter(
                    wx.MessageBox,
                    "Pose embedding finished with errors. Some videos were skipped because CSVs were missing.",
                    "Warning",
                    wx.OK | wx.ICON_WARNING,
                )
                return
            wx.CallAfter(wx.MessageBox, "Pose embedding completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        finally:
            wx.CallAfter(self._end_process, cancelled)

    def on_embed_transcript(self, event):
        if self._process_running:
            return
        folder_path = self.get_selected_folder_path()
        if not folder_path:
            wx.MessageBox("Please select a folder before embedding transcripts on video.", "No Folder Selected", wx.OK | wx.ICON_INFORMATION)
            return

        self.ensure_output_folders(folder_path)
        workspace_root = gui_utils.resolved_dataset_root(folder_path)
        video_files = gui_utils.get_files_from_folder(workspace_root, self.VIDEO_EXTENSIONS)

        if not video_files:
            wx.MessageBox(
                "No video files found in the selected folder.\n\n"
                "Add videos (.mp4, .mov, .avi, .mkv, .m4v) to the folder, then try again.",
                "No Videos Found", wx.OK | wx.ICON_INFORMATION,
            )
            return

        if not self.captioned_video_folder:
            wx.MessageBox(
                "The app could not create the 'captioned_video' folder for your results.\n\n"
                "Check that you have permission to write to the selected folder, then try again.",
                "Cannot Save Results",
                wx.OK | wx.ICON_ERROR,
            )
            return

        transcripts_dir = gui_utils.transcripts_output_folder(folder_path)
        caption_pose_video = bool(
            hasattr(self, 'captionPoseVideoCheckbox') and self.captionPoseVideoCheckbox.GetValue()
        )
        missing_srt_videos = [
            os.path.basename(video)
            for video in video_files
            if not os.path.exists(
                os.path.join(transcripts_dir, os.path.splitext(os.path.basename(video))[0] + ".srt")
            )
        ]
        if missing_srt_videos:
            preview = ", ".join(missing_srt_videos[:3])
            if len(missing_srt_videos) > 3:
                preview += f", and {len(missing_srt_videos) - 3} more"
            wx.MessageBox(
                "Transcripts are missing for these videos:\n"
                f"{preview}.\n\n"
                "On the Audio tab, run 'Extract Transcripts' first. Each transcript is "
                "saved as a '.srt' file in the 'transcripts' folder inside your selected folder.",
                "Transcripts Needed",
                wx.OK | wx.ICON_INFORMATION,
            )
            return

        if caption_pose_video:
            missing_pose_videos = [
                os.path.basename(video)
                for video in video_files
                if not self._embedded_pose_video_path(self.embedded_pose_folder, video)
            ]
            if missing_pose_videos:
                preview = ", ".join(missing_pose_videos[:3])
                if len(missing_pose_videos) > 3:
                    preview += f", and {len(missing_pose_videos) - 3} more"
                wx.MessageBox(
                    "Pose-embedded videos are missing for these videos:\n"
                    f"{preview}.\n\n"
                    "Run 'Embed Pose Features' first. Captions will be added to videos in the "
                    "'embedded_pose' folder.",
                    "Pose Videos Needed",
                    wx.OK | wx.ICON_INFORMATION,
                )
                return

        ffmpeg_exe = gui_utils.get_subtitle_capable_ffmpeg()
        if not ffmpeg_exe:
            wx.MessageBox(
                "Captions cannot be added because no compatible video engine (FFmpeg with "
                "subtitle support) was found on this computer.\n\n"
                "Reinstall the app or install FFmpeg, then try again.",
                "Video Engine Unavailable",
                wx.OK | wx.ICON_ERROR,
            )
            return

        self._begin_process()
        thread = threading.Thread(
            target=self.embed_transcript_batch,
            args=(video_files, transcripts_dir, ffmpeg_exe, caption_pose_video),
        )
        thread.start()

    def embed_transcript_batch(self, video_files, transcripts_dir, ffmpeg_exe, caption_pose_video=False):
        import captions

        cancelled = False
        failed = False
        try:
            total_files = len(video_files)
            for index, video_file in enumerate(video_files, start=1):
                if self._cancel_event.is_set():
                    cancelled = True
                    break

                base = os.path.splitext(os.path.basename(video_file))[0]
                srt_path = os.path.join(transcripts_dir, base + ".srt")
                input_video = video_file
                output_suffix = "_captioned.mp4"
                if caption_pose_video:
                    input_video = self._embedded_pose_video_path(self.embedded_pose_folder, video_file)
                    output_suffix = "_pose_captioned.mp4"
                    if not input_video:
                        failed = True
                        self.set_status_message(f"Missing pose-embedded video for {os.path.basename(video_file)}")
                        continue
                out_path = os.path.join(self.captioned_video_folder, base + output_suffix)

                self.set_status_message(f"Embedding transcript for: {os.path.basename(input_video)}")
                try:
                    result = captions.burn_subtitles(
                        input_video,
                        srt_path,
                        out_path,
                        ffmpeg_exe,
                        progress_callback=self.make_overall_progress_cb(index, total_files),
                        cancel_check=lambda: self._cancel_event.is_set(),
                    )
                except Exception as e:
                    failed = True
                    self.set_status_message(f"Failed to caption {os.path.basename(video_file)}: {e}")
                    continue

                if result is False:
                    cancelled = True
                    break

            if cancelled:
                return
            if failed:
                wx.CallAfter(
                    wx.MessageBox,
                    "Transcript embedding finished with errors. Some videos were skipped.",
                    "Warning",
                    wx.OK | wx.ICON_WARNING,
                )
                return
            wx.CallAfter(wx.MessageBox, "Transcript embedding completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        finally:
            wx.CallAfter(self._end_process, cancelled)

    def on_extract_audio_features(self, event):
        """Extract audio features from all supported audio files in the folder."""
        if self._process_running:
            return
        folder_path = self.get_selected_folder_path()
        if not folder_path:
            wx.MessageBox("Please select a folder before extracting audio features.", "No Folder Selected", wx.OK | wx.ICON_INFORMATION)
            return

        self.ensure_output_folders(folder_path)
        audio_files = gui_utils.get_audio_files_for_processing(folder_path, self.AUDIO_EXTENSIONS)

        if not audio_files:
            wx.MessageBox(
                "No audio files found in the selected folder.\n\n"
                "If you have videos, run 'Convert to WAV' on the Video tab first. "
                "Converted audio is saved in the 'converted_audio' folder.",
                "No Audio Found", wx.OK | wx.ICON_INFORMATION,
            )
            return

        if not self.extracted_audio_folder:
            wx.MessageBox(
                "The app could not create the 'audio_features' folder for your results.\n\n"
                "Check that you have permission to write to the selected folder, then try again.",
                "Cannot Save Results",
                wx.OK | wx.ICON_ERROR,
            )
            return

        # Initialize audio processor
        audio_processor = AudioProcessor(
            output_audio_features_folder=self.extracted_audio_folder,
            output_transcripts_folder=None,  # Not needed for feature extraction
            status_callback=self.set_status_message
        )

        self._begin_process()
        thread = threading.Thread(target=self.extract_audio_features_batch, args=(audio_files, audio_processor))
        thread.start()

    def extract_audio_features_batch(self, audio_files, audio_processor):
        """Batch process all audio files to extract features."""
        cancelled = False
        try:
            def progress_callback(progress):
                self.update_progress(progress)

            audio_processor.extract_audio_features_batch(
                audio_files,
                progress_callback=progress_callback,
                cancel_check=self._cancel_event.is_set,
            )
            cancelled = self._cancel_event.is_set()
            if not cancelled:
                wx.CallAfter(wx.MessageBox, "Audio feature extraction completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            if not self._cancel_event.is_set():
                wx.CallAfter(wx.MessageBox, f"Error during audio feature extraction: {e}", "Error", wx.OK | wx.ICON_ERROR)
            else:
                cancelled = True
        finally:
            wx.CallAfter(self._end_process, cancelled)



    def on_extract_transcripts(self, event):
        """Extract transcripts from all supported audio files in the folder."""
        if self._process_running:
            return
        folder_path = self.get_selected_folder_path()
        if not folder_path:
            wx.MessageBox("Please select a folder before extracting transcripts.", "No Folder Selected", wx.OK | wx.ICON_INFORMATION)
            return

        self.ensure_output_folders(folder_path)
        audio_files = gui_utils.get_audio_files_for_processing(folder_path, self.AUDIO_EXTENSIONS)

        if not audio_files:
            wx.MessageBox(
                "No audio files found in the selected folder.\n\n"
                "If you have videos, run 'Convert to WAV' on the Video tab first. "
                "Converted audio is saved in the 'converted_audio' folder.",
                "No Audio Found", wx.OK | wx.ICON_INFORMATION,
            )
            return

        if not self.extracted_transcripts_folder:
            wx.MessageBox(
                "The app could not create the 'transcripts' folder for your results.\n\n"
                "Check that you have permission to write to the selected folder, then try again.",
                "Cannot Save Results",
                wx.OK | wx.ICON_ERROR,
            )
            return

        # Determine diarization preference and collect token if needed
        enable_diarization = False
        if hasattr(self, 'diarizationCheckbox') and self.diarizationCheckbox.GetValue():
            diarization_state = runtime_services.get_diarization_feature_state()
            if not diarization_state["installed"]:
                if diarization_state["can_self_install"]:
                    install_now = wx.MessageBox(
                        "Speaker diarization is not installed yet.\n\nWould you like MultiSOCIAL Toolbox to install the optional component now?",
                        "Install Optional Diarization",
                        wx.YES_NO | wx.ICON_QUESTION,
                    )
                    if install_now == wx.YES:
                        self.on_install_diarization(None)
                    return
                wx.MessageBox(
                    "Speaker diarization is not installed in this build. Use a Complete build or rerun the installer with the complete profile.",
                    "Diarization Not Installed",
                    wx.OK | wx.ICON_INFORMATION,
                )
                return

            enable_diarization = True
            # Check if we already have a token (from env or previous entry)
            if not self.hf_token:
                dlg = wx.TextEntryDialog(
                    self,
                    message=(
                        "Speaker diarization requires a Hugging Face token for pyannote.\n\n"
                        "Please see the README for setup steps. Once complete, enter your token here:\n"
                        "(It will be saved in local app settings for future use)"
                    ),
                    caption="Enter Hugging Face Token"
                )
                if dlg.ShowModal() == wx.ID_OK:
                    token_val = dlg.GetValue().strip()
                    if token_val:
                        self.hf_token = token_val
                        try:
                            runtime_services.save_hf_token(token_val)
                            print("Token saved to local app settings")
                        except Exception as e:
                            print(f"Failed to save token to app settings: {e}")
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

        # Word-level timestamps are opt-in: they precompute the JSON sidecar Align Features
        # needs, so alignment doesn't have to re-run transcription file-by-file later.
        word_timestamps = bool(
            hasattr(self, 'wordTimestampsCheckbox') and self.wordTimestampsCheckbox.GetValue()
        )

        # Run transcription in a separate thread
        self._begin_process()
        thread = threading.Thread(
            target=self.extract_transcripts_batch,
            args=(audio_files, audio_processor, word_timestamps),
        )
        thread.start()


    def extract_transcripts_batch(self, audio_files, audio_processor, word_timestamps=False):
        """Batch process all audio files to generate transcripts."""
        cancelled = False
        try:
            def progress_callback(progress):
                self.update_progress(progress)

            audio_processor.extract_transcripts_batch(
                audio_files,
                progress_callback=progress_callback,
                word_timestamps=word_timestamps,
                cancel_check=self._cancel_event.is_set,
            )
            cancelled = self._cancel_event.is_set()
            if not cancelled:
                wx.CallAfter(wx.MessageBox, "Transcription extraction completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        except Exception as e:
            if not self._cancel_event.is_set():
                wx.CallAfter(wx.MessageBox, f"Error during transcript extraction: {e}", "Error", wx.OK | wx.ICON_ERROR)
            else:
                cancelled = True
        finally:
            wx.CallAfter(self._end_process, cancelled)

    def on_align_features(self, event):
        if self._process_running:
            return
        folder_path = self.get_selected_folder_path()
        if not folder_path:
            wx.MessageBox("Please select a folder before aligning features.", "No Folder Selected", wx.OK | wx.ICON_INFORMATION)
            return

        self.ensure_output_folders(folder_path)

        # Check if we have features and transcripts
        if not self.extracted_audio_folder or not os.path.exists(self.extracted_audio_folder):
            wx.MessageBox(
                "Audio features haven't been extracted yet.\n\n"
                "Run 'Extract Audio Features' first. Results are saved in the "
                "'audio_features' folder inside your selected folder.",
                "Audio Features Needed", wx.OK | wx.ICON_INFORMATION,
            )
            return

        # We don't strictly need transcripts folder to exist yet if we are going to generate them,
        # but we need audio files.
        audio_files = gui_utils.get_audio_files_for_processing(folder_path, self.AUDIO_EXTENSIONS)
        if not audio_files:
            wx.MessageBox(
                "No audio files found in the selected folder.\n\n"
                "If you have videos, run 'Convert to WAV' on the Video tab first.",
                "No Audio Found", wx.OK | wx.ICON_INFORMATION,
            )
            return

        # Run in thread
        self._begin_process()
        thread = threading.Thread(target=self.align_features_batch, args=(audio_files,))
        thread.start()

    def align_features_batch(self, audio_files):
        cancelled = False
        try:
            total_files = len(audio_files)
            
            # Initialize processor. Alignment only needs word-level Whisper output,
            # so diarization is explicitly disabled: otherwise the AudioProcessor
            # default (enabled) would offload Whisper and attempt pyannote for every
            # file we auto-transcribe here, which is slow and needs an HF token.
            audio_processor = AudioProcessor(
                output_audio_features_folder=self.extracted_audio_folder,
                output_transcripts_folder=self.extracted_transcripts_folder,
                status_callback=self.set_status_message,
                enable_speaker_diarization=False,
                auth_token=self.hf_token
            )
            
            alignment_pairs = []
            prep_errors = []  # (base_name, reason) for files we couldn't prepare

            for i, audio_file in enumerate(audio_files, start=1):
                if self._cancel_event.is_set():
                    cancelled = True
                    break
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
                        prep_errors.append((base_name, f"transcript failed: {e}"))
                        continue
                    # extract_transcript only warns (doesn't raise) if the JSON can't be
                    # written, so confirm the sidecar actually exists before pairing it.
                    if not os.path.exists(json_path):
                        print(f"Word-level JSON was not produced for {base_name}; skipping.")
                        prep_errors.append((base_name, "word-level JSON was not produced"))
                        continue

                # 2. Ensure we have features CSV
                # Feature files are usually named {base_name}.csv or similar in extracted_audio_folder
                # The AudioProcessor.extract_audio_features saves them as {base_name}.csv
                feature_csv = os.path.join(self.extracted_audio_folder, f"{base_name}.csv")
                if not os.path.exists(feature_csv):
                    self.set_status_message(f"🎵 Extracting audio features for: {base_name}")
                    try:
                        # Auto-extract missing features for this file
                        audio_processor.extract_audio_features(audio_file)
                        # Verify the file was created
                        if not os.path.exists(feature_csv):
                            print(f"Feature extraction completed but file not found: {feature_csv}, skipping.")
                            prep_errors.append((base_name, "feature CSV was not produced"))
                            continue
                    except Exception as e:
                        print(f"Failed to extract features for {base_name}: {e}")
                        prep_errors.append((base_name, f"feature extraction failed: {e}"))
                        continue

                # 3. Output path
                output_csv = os.path.join(self.extracted_audio_folder, f"{base_name}_aligned.csv")
                alignment_pairs.append((feature_csv, json_path, output_csv))

                self.update_progress(int((i / total_files) * 50)) # First 50% for prep

            if cancelled:
                pass
            elif not alignment_pairs:
                detail = "\n".join(f"• {name}: {reason}" for name, reason in prep_errors[:10])
                if len(prep_errors) > 10:
                    detail += f"\n…and {len(prep_errors) - 10} more (see console/log)."
                message = "No valid pairs found to align. Ensure you have extracted audio features."
                if detail:
                    message += "\n\nWhat went wrong:\n" + detail
                wx.CallAfter(wx.MessageBox, message, "Warning", wx.OK | wx.ICON_WARNING)
            elif not self._cancel_event.is_set():
                audio_processor.align_features_batch(
                    alignment_pairs,
                    progress_callback=lambda p: self.update_progress(50 + int(p / 2)),
                )
                success_message = f"Feature alignment completed for {len(alignment_pairs)} file(s)!"
                if prep_errors:
                    success_message += (
                        f"\n\n{len(prep_errors)} file(s) were skipped (see console/log for details)."
                    )
                wx.CallAfter(wx.MessageBox, success_message, "Success", wx.OK | wx.ICON_INFORMATION)
            else:
                cancelled = True
            
        except Exception as e:
            if not self._cancel_event.is_set():
                wx.CallAfter(wx.MessageBox, f"Alignment failed: {e}", "Error", wx.OK | wx.ICON_ERROR)
            else:
                cancelled = True
        finally:
            wx.CallAfter(self._end_process, cancelled)

def main():
    if os.environ.get("MULTISOCIAL_IMPORT_SMOKE_TEST") == "1":
        profile = runtime_services.get_build_profile().lower()
        if profile == "complete":
            try:
                import pyannote.audio
                print("Import smoke test passed (complete profile).")
            except ImportError as e:
                import sys
                print(f"ERROR: pyannote.audio import failed: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print("Import smoke test passed (standard profile).")
        return

    # Create wx.App FIRST before any wx calls (including MessageBox)
    app = wx.App()
    
    # Now we can safely show message boxes
    if not gui_utils.ensure_ffmpeg_available():
        msg = (
            "ffmpeg was not found. Install it or let the app use a bundled one.\n\n"
            "macOS: brew install ffmpeg\n"
            "Linux: sudo apt-get install ffmpeg\n"
            "Windows: choco install ffmpeg\n\n"
            "Alternatively, ensure imageio-ffmpeg is installed (already in requirements)."
        )
        wx.MessageBox(msg, "ffmpeg not found", wx.OK | wx.ICON_ERROR)

    print(f"Startup diagnostics: {runtime_services.get_startup_diagnostics()}")
    
    frm = VideoToWavConverter(None)
    frm.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
