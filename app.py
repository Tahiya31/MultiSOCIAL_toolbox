'''
This is the main script for multisocial app

'''


# Import necessary system and utility modules
import os
import threading


# Third-party libraries (assumed pre-installed via requirements.txt)
import ffmpeg
import opensmile
import wx
import librosa
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
 

# Import the core pose processing class
from pose import PoseProcessor

# Set up GPU environment specially for Mediapipe (specific for Saturn Cloud), if you use some other high performance computing platform check compatibility before usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Make sure the system uses the GPU


## All dependencies are expected to be installed ahead of time via requirements.txt

# (Optional) Helper for Windows FFmpeg setup was removed to avoid runtime installs


class GradientPanel(wx.Panel):
    def __init__(self, parent):
        super(GradientPanel, self).__init__(parent)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        rect = self.GetClientRect()
        dc.GradientFillLinear(rect, '#00695C', '#2E7D32', wx.NORTH)  # Dark Teal to Medium Forest Green


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
        self.tooltip_text = tooltip_text
        self.tooltip = None
        
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
    
    def OnPaint(self, event):
        dc = wx.PaintDC(self)
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
        
        #status update (top)
        self.statusLabel = wx.StaticText(pnl, label="", style=wx.ALIGN_CENTER_HORIZONTAL)
        self.statusLabel.SetForegroundColour('#FFFFFF')
        vbox.Add(self.statusLabel, flag=wx.EXPAND|wx.ALL, border=10)
        
        # Add extra space above the title
        vbox.Add((0, 30))  # Add a 30-pixel high spacer, adjust as needed
		
		  
        new_width, new_height = wx.GetDisplaySize()
        logo_size = min(int(new_height * 0.08), 150)
        
        # Top layout for logo and title
        top_box = wx.BoxSizer(wx.HORIZONTAL)
        
        # Logo image
        logo_path = "MultiSOCIAL_logo.png"  # Path to your logo image
        logo_image = wx.Image(logo_path, wx.BITMAP_TYPE_ANY)
        logo_image = logo_image.Scale(logo_size, logo_size, wx.IMAGE_QUALITY_HIGH)
        logo_bmp = wx.Bitmap(logo_image)
        
        logo_bitmap = wx.StaticBitmap(pnl, bitmap = logo_bmp)
        
        top_box.AddStretchSpacer(1)
        top_box.Add(logo_bitmap, flag=wx.ALIGN_CENTER | wx.ALL, border=10)
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
        
        pose_processor = PoseProcessor(self.extracted_pose_folder, status_callback=self.set_status_message)
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
        
      
        pose_processor = PoseProcessor(output_csv_folder=self.extracted_pose_folder,output_video_folder=self.embedded_pose_folder)
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

        thread = threading.Thread(target=self.extract_audio_features_batch, args=(audio_files,))
        thread.start()

    def extract_audio_features_batch(self, audio_files):
        """Batch process all audio files to extract features."""
        total_files = len(audio_files)

        for i, audio_file in enumerate(audio_files):
            self.set_status_message(f"🎧 Extracting audio from: {os.path.basename(audio_file)}")
            print(f"Extracting features from: {audio_file}")
            
            # Create progress callback for this audio file
            def make_progress_callback(audio_index, total_audios):
                def progress_callback(extraction_progress):
                    # Calculate overall progress: (audio_index-1)/total_audios + extraction_progress/total_audios
                    overall_progress = int(((audio_index - 1) / total_audios) * 100 + (extraction_progress / total_audios))
                    self.update_progress(overall_progress)
                return progress_callback
            
            self.extract_audio_features(audio_file, progress_callback=make_progress_callback(i + 1, total_files))

        wx.MessageBox("Audio feature extraction completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        self.update_progress(0)  # Reset progress bar

    def extract_audio_features(self, filepath, progress_callback=None):
        """Extracts audio features from a single WAV file."""
        try:
            if progress_callback:
                progress_callback(0)
            
            feature_set_name = opensmile.FeatureSet.ComParE_2016
            feature_level_name = opensmile.FeatureLevel.LowLevelDescriptors

            if progress_callback:
                progress_callback(25)
            
            smile = opensmile.Smile(feature_set=feature_set_name, feature_level=feature_level_name)
            y, sr = librosa.load(filepath)
            
            if progress_callback:
                progress_callback(50)
            
            features = smile.process_signal(y, sr)
            
            if progress_callback:
                progress_callback(75)

            output_csv = os.path.join(self.extracted_audio_folder, os.path.splitext(os.path.basename(filepath))[0] + ".csv")
            features.to_csv(output_csv, index=False)
            
            if progress_callback:
                progress_callback(100)

            print(f"Saved audio features: {output_csv}")

        except Exception as e:
            wx.MessageBox(f'Error extracting audio features from {filepath}: {e}', 'Error', wx.OK | wx.ICON_ERROR)


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

        # Run transcription in a separate thread
        thread = threading.Thread(target=self.extract_transcripts_batch, args=(audio_files,))
        thread.start()


    def extract_transcripts_batch(self, audio_files):
        """Batch process all audio files to generate transcripts."""
        total_files = len(audio_files)

        #print(f"Found {total_files} WAV files for transcription.")

        for i, audio_file in enumerate(audio_files):
            #print(f"Starting transcription for: {audio_file}")
            #file_name = os.path.basename(audio_file)
            self.set_status_message(f"🗣️ Transcribing: {os.path.basename(audio_file)}")
            
            # Create progress callback for this audio file
            def make_progress_callback(audio_index, total_audios):
                def progress_callback(transcription_progress):
                    # Calculate overall progress: (audio_index-1)/total_audios + transcription_progress/total_audios
                    overall_progress = int(((audio_index - 1) / total_audios) * 100 + (transcription_progress / total_audios))
                    self.update_progress(overall_progress)
                return progress_callback
        
            try:
                self.extract_transcripts(audio_file, progress_callback=make_progress_callback(i + 1, total_files))
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

        #print("All transcriptions completed.")
        #self.set_status_message("Transcription complete.")
        wx.MessageBox("Transcription extraction completed!", "Success", wx.OK | wx.ICON_INFORMATION)
        self.update_progress(0)  # Reset progress bar

    def extract_transcripts(self, filepath, progress_callback=None):
        """Transcribes a single WAV file using Whisper."""
        try:
            if progress_callback:
                progress_callback(0)
            
            print(f"Loading Whisper model for {filepath}...")
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            if progress_callback:
                progress_callback(10)

            model_id = "distil-whisper/distil-large-v3"

            if progress_callback:
                progress_callback(20)

            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)

            if progress_callback:
                progress_callback(40)

            processor = AutoProcessor.from_pretrained(model_id)

            if progress_callback:
                progress_callback(50)

            pipe = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                max_new_tokens=128,
                chunk_length_s=25,
                batch_size=16,
                torch_dtype=torch_dtype,
                device=device,
            )

            if progress_callback:
                progress_callback(70)

            print(f"Transcribing {filepath}...")
            result = pipe(filepath)
            transcript = result['text']

            if progress_callback:
                progress_callback(90)

            output_txt = os.path.join(self.extracted_transcripts_folder, os.path.splitext(os.path.basename(filepath))[0] + ".txt")

            with open(output_txt, 'w') as f:
                f.write(transcript)

            if progress_callback:
                progress_callback(100)

            print(f"Saved transcript: {output_txt}")

        except Exception as e:
            print(f" Error transcribing {filepath}: {e}")
            wx.MessageBox(f'Error transcribing {filepath}: {e}', 'Error', wx.OK | wx.ICON_ERROR)



def main():
    app = wx.App()
    frm = VideoToWavConverter(None)
    frm.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
