import subprocess
import sys
import os

#import wx
#import opensmile
#import librosa
#import mediapipe as mp
#import pandas as pd
#import cv2
import base64
import sys
import torch
from scipy.io.wavfile import read
import wx.lib.agw.gradientbutton as GB
import wx.adv as hl
import threading
import time

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure ffmpeg-python, pydub, wxPython, librosa, mediapipe, opencv-python, and SpeechRecognition are installed

try:
    import opensmile
except ImportError:
    install("opensmile pandas opencv-python torch scipy torch")
    import opensmile
    
    
try:
    import ffmpeg
except ImportError:
    install("ffmpeg-python")
    import ffmpeg

try:
    import wx
except ImportError:
    install("wxPython")
    import wx
    
try:
    import librosa
except ImportError:
    install("librosa")
    import librosa

try:
    import mediapipe as mp
except ImportError:
    install("mediapipe")
    import mediapipe as mp

try:
    import cv2
except ImportError:
    install("opencv-python")
    import cv2

try:
    import torch, transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
except ImportError:
    install("torch")
    install("transformers")
    install("accelerate") 
    install("datasets[audio]")
    import torch
    import transformers
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    from transformers import pipeline

class GradientPanel(wx.Panel):
    def __init__(self, parent):
        super(GradientPanel, self).__init__(parent)
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        rect = self.GetClientRect()
        dc.GradientFillLinear(rect, '#7FFFD4', '#5F9EA0', wx.NORTH)  # Aqua green gradient


class VideoToWavConverter(wx.Frame):
    def __init__(self, *args, **kw):
        super(VideoToWavConverter, self).__init__(*args, **kw)
        
        pnl = GradientPanel(self)
        
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        # Add extra space above the title
        vbox.Add((0, 30))  # Add a 30-pixel high spacer, adjust as needed

        # Title
        title = wx.StaticText(pnl, label="Welcome to", style=wx.ALIGN_CENTER)
        title_font = wx.Font(14, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        title.SetFont(title_font)
        title.SetForegroundColour('#FFFFFF')

        # Logo
        logo = wx.StaticText(pnl, label="MultiSOCIAL Toolbox", style=wx.ALIGN_CENTER)
        font = wx.Font(24, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        logo.SetFont(font)
        logo.SetForegroundColour('#FFFFFF')

        vbox.Add(title, flag=wx.ALIGN_CENTER|wx.BOTTOM, border=5)  # Adjusted the bottom border
        vbox.Add(logo, flag=wx.ALIGN_CENTER|wx.TOP, border=5)  # Adjusted the top border

        # File Picker
        self.filePicker = wx.FilePickerCtrl(pnl, message="Select a video or an audio file", wildcard="Video files (*.mp4;*.avi;*.mov;*.mkv)|*.mp4;*.avi;*.mov;*.mkv|WAV files (*.wav)|*.wav")
        vbox.Add(self.filePicker, flag=wx.EXPAND|wx.ALL, border=10)
        
        # Placeholder above buttons
        placeholder_above = wx.StaticText(pnl, label="If you have a video file:")
        placeholder_above_font = wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        placeholder_above.SetFont(placeholder_above_font)
        vbox.Add(placeholder_above, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        # Button Font
        button_font = wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        placeholder_font = wx.Font(12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)

        # Convert Button with Placeholder
        hbox_convert = wx.BoxSizer(wx.HORIZONTAL)
        #placeholder_convert = wx.StaticText(pnl, label="If you have a video file:")
        #placeholder_convert.SetFont(placeholder_font)
        #hbox_convert.Add(placeholder_convert, flag=wx.ALIGN_CENTER|wx.ALL, border=10)
        
        self.convertBtn = wx.Button(pnl, label='Convert video to audio')
        self.convertBtn.SetFont(button_font)
        self.convertBtn.Bind(wx.EVT_BUTTON, self.on_convert)
        hbox_convert.Add(self.convertBtn, flag=wx.ALIGN_CENTER|wx.ALL, border=10)
        
        vbox.Add(hbox_convert, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        # Extract Pose Features Button with Placeholder
        hbox_extract_pose = wx.BoxSizer(wx.HORIZONTAL)
        #placeholder_pose = wx.StaticText(pnl, label="To extract pose features:")
        #placeholder_pose.SetFont(placeholder_font)
        #hbox_extract_pose.Add(placeholder_pose, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        self.extractFeaturesBtn = wx.Button(pnl, label='Extract Pose Features')
        self.extractFeaturesBtn.SetFont(button_font)
        self.extractFeaturesBtn.Bind(wx.EVT_BUTTON, self.on_extract_features)
        hbox_extract_pose.Add(self.extractFeaturesBtn, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        vbox.Add(hbox_extract_pose, flag=wx.ALIGN_CENTER|wx.ALL, border=10)
        
        # Placeholder middle
        placeholder_middle = wx.StaticText(pnl, label="If you have an audio file:")
        placeholder_middle.SetFont(placeholder_font)
        vbox.Add(placeholder_middle, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        # Extract Audio Features Button with Placeholder
        hbox_extract_audio = wx.BoxSizer(wx.HORIZONTAL)
        #placeholder_audio = wx.StaticText(pnl, label="To extract audio features:")
        #placeholder_audio.SetFont(placeholder_font)
        #hbox_extract_audio.Add(placeholder_audio, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        self.extractAudioFeaturesBtn = wx.Button(pnl, label='Extract Audio Features')
        self.extractAudioFeaturesBtn.SetFont(button_font)
        self.extractAudioFeaturesBtn.Bind(wx.EVT_BUTTON, self.on_extract_audio_features)
        hbox_extract_audio.Add(self.extractAudioFeaturesBtn, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        vbox.Add(hbox_extract_audio, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        # Extract Transcripts Button with Placeholder
        hbox_extract_transcripts = wx.BoxSizer(wx.HORIZONTAL)
        #placeholder_transcripts = wx.StaticText(pnl, label="To extract transcripts:")
        #placeholder_transcripts.SetFont(placeholder_font)
        #hbox_extract_transcripts.Add(placeholder_transcripts, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        self.extractTranscriptsBtn = wx.Button(pnl, label='Extract Transcripts')
        self.extractTranscriptsBtn.SetFont(button_font)
        self.extractTranscriptsBtn.Bind(wx.EVT_BUTTON, self.on_extract_transcripts)
        hbox_extract_transcripts.Add(self.extractTranscriptsBtn, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        vbox.Add(hbox_extract_transcripts, flag=wx.ALIGN_CENTER|wx.ALL, border=10)

        # Progress Bar
        self.progress = wx.Gauge(pnl, range=100, style=wx.GA_HORIZONTAL)
        vbox.Add(self.progress, proportion=1, flag=wx.EXPAND|wx.ALL, border=10)
        
        
        # Logo image
        #logo_path = "MultiSOCIAL_logo.png"  # Path to your logo image
        #logo_bmp = wx.Bitmap(logo_path, wx.BITMAP_TYPE_ANY)
        #logo_bitmap = wx.StaticBitmap(pnl, bitmap=logo_bmp)

        # GitHub link
        #github_link = hl.HyperlinkCtrl(pnl, id=wx.ID_ANY, label="GitHub", url="https://github.com")
        #github_icon_path = "github_icon.png"  # Path to your GitHub icon image
        #github_bmp = wx.Bitmap(github_icon_path, wx.BITMAP_TYPE_ANY)
        #github_bitmap = wx.StaticBitmap(pnl, bitmap=github_bmp)

        # Horizontal box sizer to arrange logo and GitHub link
        #hbox = wx.BoxSizer(wx.HORIZONTAL)
        #hbox.Add(logo_bitmap, flag=wx.ALL|wx.ALIGN_LEFT, border=5)
        #hbox.AddStretchSpacer(1)  # Add stretchable space between logo and GitHub link
        #hbox.Add(github_link, flag=wx.ALL|wx.ALIGN_RIGHT, border=5)
        #hbox.Add(github_bitmap, flag=wx.ALL|wx.ALIGN_RIGHT, border=5)
        
        pnl.SetSizer(vbox)
        
        self.SetSize((400, 600))  # Adjusted the size to accommodate new elements
        self.SetTitle('MultiSOCIAL Toolbox')
        self.Centre()

    def update_progress(self, value):
        wx.CallAfter(self.progress.SetValue, value)

    def on_convert(self, event):
        filepath = self.filePicker.GetPath()
        if not filepath:
            wx.MessageBox('Please select a file.', 'Error', wx.OK | wx.ICON_ERROR)
            return
        
        self.progress.SetValue(0)

        def convert_task():
        	try:
        		# Simulating a long-running task
        		for i in range(1, 101):
        			time.sleep(0.05)  # Simulate work by sleeping
        			self.update_progress(i)
        			
        			
        		#perform conversion
        		self.convert_to_wav(filepath)
        	except:
        		wx.CallAfter(wx.MessageBox, 'Conversion completed!', 'Info', wx.OK | wx.ICON_INFORMATION)

        thread = threading.Thread(target=convert_task)
        thread.start()
        
        
    def convert_to_wav(self, filepath):
        try:
            output_path = os.path.splitext(filepath)[0] + ".wav"
            video = ffmpeg.input(filepath)
            audio = video.audio
            ffmpeg.output(audio, output_path).run(overwrite_output=True)
            wx.MessageBox(f'Video has been converted to {output_path}', 'Success', wx.OK | wx.ICON_INFORMATION)
            self.save_file(output_path)
        except Exception as e:
            wx.CallAfter(wx.MessageBox(f'An error occurred: {e}', 'Error', wx.OK | wx.ICON_ERROR))

    def save_file(self, output_path):
        with wx.FileDialog(self, "Save WAV file", wildcard="WAV files (*.wav)|*.wav", style=wx.FD_SAVE |
		        wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal() == wx.ID_CANCEL:
                return     # the user changed their mind
            
            save_path = fileDialog.GetPath()
            if save_path:
                os.rename(output_path, save_path)
                wx.MessageBox(f'File saved to {save_path}', 'Saved', wx.OK | wx.ICON_INFORMATION)

    def on_extract_features(self, event):
        filepath = self.filePicker.GetPath()
        if filepath and filepath.lower().endswith('.mp4'):
            self.extract_features(filepath)
        else:
            wx.MessageBox('Please select a video file.', 'Error', wx.OK | wx.ICON_ERROR)
    
    def extract_features(self, filepath):
        try:
            mp_pose = mp.solutions.pose

            # Define a function to extract pose features
            def extract_pose_features(video_path):
                # Initialize MediaPipe Pose model
                with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5) as pose:
                    # Open video file
                    cap = cv2.VideoCapture(video_path)
                    frame_number = 0
                    frames = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append((frame_number, frame))
                        frame_number += 1
                    cap.release()

                    # Extract pose features for each frame
                    pose_features = []
                    for frame_number, frame in frames:
                        # Convert frame to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        # Process the frame with MediaPipe Pose
                        results = pose.process(frame_rgb)
                        # Extract pose landmarks
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            pose_landmarks = [frame_number] + [coord for lmk in landmarks for coord in (lmk.x, lmk.y, lmk.z, lmk.visibility)]
                            pose_features.append(pose_landmarks)
                        else:
                            pose_landmarks = [frame_number] + [0, 0, 0, 0] * 33  # Fill with zeros if no landmarks detected
                            pose_features.append(pose_landmarks)
                    return pose_features

            # Extract pose features
            pose_features = extract_pose_features(filepath)

            # Column names
            columns = [
                'frame_number', 'Nose_x', 'Nose_y', 'Nose_z', 'Nose_conf',
                'Left_eye_inner_x', 'Left_eye_inner_y', 'Left_eye_inner_z', 'Left_eye_inner_conf',
                'Left_eye_x', 'Left_eye_y', 'Left_eye_z', 'Left_eye_conf',
                'Left_eye_outer_x', 'Left_eye_outer_y', 'Left_eye_outer_z', 'Left_eye_outer_conf',
                'Right_eye_inner_x', 'Right_eye_inner_y', 'Right_eye_inner_z', 'Right_eye_inner_conf',
                'Right_eye_x', 'Right_eye_y', 'Right_eye_z', 'Right_eye_conf',
                'Right_eye_outer_x', 'Right_eye_outer_y', 'Right_eye_outer_z', 'Right_eye_outer_conf',
                'Left_ear_x', 'Left_ear_y', 'Left_ear_z', 'Left_ear_conf',
                'Right_ear_x', 'Right_ear_y', 'Right_ear_z', 'Right_ear_conf',
                'Mouth_left_x', 'Mouth_left_y', 'Mouth_left_z', 'Mouth_left_conf',
                'Mouth_right_x', 'Mouth_right_y', 'Mouth_right_z', 'Mouth_right_conf',
                'Left_shoulder_x', 'Left_shoulder_y', 'Left_shoulder_z', 'Left_shoulder_conf',
                'Right_shoulder_x', 'Right_shoulder_y', 'Right_shoulder_z', 'Right_shoulder_conf',
                'Left_elbow_x', 'Left_elbow_y', 'Left_elbow_z', 'Left_elbow_conf',
                'Right_elbow_x', 'Right_elbow_y', 'Right_elbow_z', 'Right_elbow_conf',
                'Left_wrist_x', 'Left_wrist_y', 'Left_wrist_z', 'Left_wrist_conf',
                'Right_wrist_x', 'Right_wrist_y', 'Right_wrist_z', 'Right_wrist_conf',
                'Left_pinky_x', 'Left_pinky_y', 'Left_pinky_z', 'Left_pinky_conf',
                'Right_pinky_x', 'Right_pinky_y', 'Right_pinky_z', 'Right_pinky_conf',
                'Left_index_x', 'Left_index_y', 'Left_index_z', 'Left_index_conf',
                'Right_index_x', 'Right_index_y', 'Right_index_z', 'Right_index_conf',
                'Left_thumb_x', 'Left_thumb_y', 'Left_thumb_z', 'Left_thumb_conf',
                'Right_thumb_x', 'Right_thumb_y', 'Right_thumb_z', 'Right_thumb_conf',
                'Left_hip_x', 'Left_hip_y', 'Left_hip_z', 'Left_hip_conf',
                'Right_hip_x', 'Right_hip_y', 'Right_hip_z', 'Right_hip_conf',
                'Left_knee_x', 'Left_knee_y', 'Left_knee_z', 'Left_knee_conf',
                'Right_knee_x', 'Right_knee_y', 'Right_knee_z', 'Right_knee_conf',
                'Left_ankle_x', 'Left_ankle_y', 'Left_ankle_z', 'Left_ankle_conf',
                'Right_ankle_x', 'Right_ankle_y', 'Right_ankle_z', 'Right_ankle_conf',
                'Left_heel_x', 'Left_heel_y', 'Left_heel_z', 'Left_heel_conf',
                'Right_heel_x', 'Right_heel_y', 'Right_heel_z', 'Right_heel_conf',
                'Left_foot_index_x', 'Left_foot_index_y', 'Left_foot_index_z', 'Left_foot_index_conf',
                'Right_foot_index_x', 'Right_foot_index_y', 'Right_foot_index_z', 'Right_foot_index_conf'
            ]

            # Convert pose features to DataFrame
            pose_df = pd.DataFrame(pose_features, columns=columns)

            # Ask user to select a path to save the output CSV file
            with wx.FileDialog(self, "Save Pose Feature File", wildcard="CSV files (*.csv)|*.csv", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return  # User changed their mind
                output_path = fileDialog.GetPath()

                # Save pose features as CSV
                if output_path:
                    pose_df.to_csv(output_path, index=False)
                    wx.MessageBox(f'Pose features extracted and saved to {output_path}', 'Success', wx.OK | wx.ICON_INFORMATION)

        except Exception as e:
            wx.MessageBox(f'An error occurred during pose feature extraction: {e}', 'Error', wx.OK | wx.ICON_ERROR)

    def on_extract_audio_features(self, event):
        filepath = self.filePicker.GetPath()
        if filepath and filepath.lower().endswith('.wav'):
            self.extract_audio_features(filepath)
        else:
        	wx.MessageBox('Please select a WAV file.', 'Error', wx.OK | wx.ICON_ERROR)
			            
			         
    def extract_audio_features(self, filepath):
        
        try:
        	feature_set_name = opensmile.FeatureSet.ComParE_2016
        	feature_level_name=opensmile.FeatureLevel.LowLevelDescriptors
        
        	smile = opensmile.Smile(feature_set=feature_set_name,feature_level=feature_level_name)
        	y, sr = librosa.load(filepath)
        	features = smile.process_signal(y, sr)
        	features_df = pd.DataFrame(features, columns=smile.feature_names)
        	with wx.FileDialog(self, "Save Audio Feature File", wildcard="CSV files (*.csv)|*.csv", style=wx.FD_SAVE|wx.FD_OVERWRITE_PROMPT) as fileDialog:
        		if fileDialog.ShowModal() == wx.ID_CANCEL:
        			return  # User changed their mind
        		output_path = fileDialog.GetPath()
        		
        		if output_path:
        			features_df.to_csv(output_path, index=False)
        			wx.MessageBox(f'Audio features extracted and saved to {output_path}', 'Success', wx.OK | wx.ICON_INFORMATION)

        except Exception as e:
            wx.MessageBox(f'An error occurred during audio feature extraction: {e}', 'Error', wx.OK | wx.ICON_ERROR)

    def on_extract_transcripts(self, event):
        filepath = self.filePicker.GetPath()
        if filepath and filepath.lower().endswith('.wav'):
            self.extract_transcripts(filepath)
        else:
            wx.MessageBox('Please select a WAV file.', 'Error', wx.OK | wx.ICON_ERROR)
            
    def extract_transcripts(self, filepath):
        try:
            # Check for GPU availability
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Load the model and tokenizer
            model_id = "distil-whisper/distil-large-v3"
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
            )
            model.to(device)

            processor = AutoProcessor.from_pretrained(model_id)

            # Create the transcription pipeline
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

            # Perform transcription
            result = pipe(filepath)
            transcript = result['text']

            with wx.FileDialog(self, "Save Transcript File", wildcard="Text files (*.txt)|*.txt", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
                if fileDialog.ShowModal() == wx.ID_CANCEL:
                    return  # User changed their mind
                output_path = fileDialog.GetPath()

                if output_path:
                    with open(output_path, 'w') as f:
                        f.write(transcript)
                    wx.MessageBox(f'Transcript extracted and saved to {output_path}', 'Success', wx.OK | wx.ICON_INFORMATION)

        except Exception as e:
            wx.MessageBox(f'An error occurred during transcript extraction: {e}', 'Error', wx.OK | wx.ICON_ERROR)
            
            
          


def main():
    app = wx.App()
    frm = VideoToWavConverter(None)
    frm.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()
