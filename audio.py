'''
This is the script for extracting audio features and transcripts from audio files
using OpenSMILE and Whisper

'''

import os
import torch
import librosa
import opensmile
import pandas as pd
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline




class AudioProcessor:
    def __init__(self, output_audio_features_folder, output_transcripts_folder, status_callback=None):
        """
        Initialize the AudioProcessor.
        
        Args:
            output_audio_features_folder (str): Path to folder for saving audio features CSV files
            output_transcripts_folder (str): Path to folder for saving transcript text files
            status_callback (callable, optional): Callback function for status updates
        """
        self.output_audio_features_folder = output_audio_features_folder
        self.output_transcripts_folder = output_transcripts_folder
        self.status_callback = status_callback
        
        # Ensure output directories exist (only if they are not None)
        if self.output_audio_features_folder is not None:
            os.makedirs(self.output_audio_features_folder, exist_ok=True)
        if self.output_transcripts_folder is not None:
            os.makedirs(self.output_transcripts_folder, exist_ok=True)
        
        # Initialize device for Whisper
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Whisper model will be loaded lazily when needed
        self.whisper_model = None
        self.whisper_processor = None
        self.whisper_pipe = None

    def set_status_message(self, message):
        """Safely update status message using callback if available."""
        if self.status_callback:
            self.status_callback(message)

    def extract_audio_features(self, filepath, progress_callback=None):
        """
        Extract audio features from a single WAV file using OpenSMILE.
        
        Args:
            filepath (str): Path to the WAV file
            progress_callback (callable, optional): Callback function for progress updates
            
        Returns:
            str: Path to the saved CSV file with features
        """
        if self.output_audio_features_folder is None:
            raise ValueError("Audio features output folder not configured")
            
        try:
            if progress_callback:
                progress_callback(0)
            
            # Configure OpenSMILE feature extraction
            feature_set_name = opensmile.FeatureSet.ComParE_2016
            feature_level_name = opensmile.FeatureLevel.LowLevelDescriptors

            if progress_callback:
                progress_callback(10)
            
            # Initialize OpenSMILE processor
            smile = opensmile.Smile(feature_set=feature_set_name, feature_level=feature_level_name)
            
            if progress_callback:
                progress_callback(20)
            
            # Load audio file using librosa
            y, sr = librosa.load(filepath)
            
            if progress_callback:
                progress_callback(40)
            
            # Extract features using OpenSMILE
            features = smile.process_signal(y, sr)
            
            if progress_callback:
                progress_callback(70)

            # Add timestamp columns as the leftmost columns
            # Calculate precise frame duration based on actual audio length
            audio_duration = len(y) / sr  # Total audio duration in seconds
            num_frames = len(features)
            frame_duration = audio_duration / num_frames  # Actual frame duration
            
            # Create timestamps for each frame
            timestamps_seconds = [i * frame_duration for i in range(num_frames)]
            timestamps_milliseconds = [t * 1000 for t in timestamps_seconds]  # Convert to milliseconds
            timestamps_formatted = [f"{int(t//60):02d}:{t%60:06.3f}" for t in timestamps_seconds]  # MM:SS.mmm format
            
            # Insert timestamp columns at the beginning
            features.insert(0, 'Timestamp_Seconds', timestamps_seconds)
            features.insert(1, 'Timestamp_Milliseconds', timestamps_milliseconds)
            features.insert(2, 'Timestamp_Formatted', timestamps_formatted)
            
            # Save features to CSV with original OpenSMILE column names and timestamps
            output_csv = os.path.join(
                self.output_audio_features_folder, 
                os.path.splitext(os.path.basename(filepath))[0] + ".csv"
            )
            features.to_csv(output_csv, index=False)
            
            if progress_callback:
                progress_callback(100)

            print(f"Saved audio features: {output_csv}")
            return output_csv

        except Exception as e:
            error_msg = f'Error extracting audio features from {filepath}: {e}'
            print(error_msg)
            raise Exception(error_msg)

    def extract_audio_features_batch(self, audio_files, progress_callback=None):
        """
        Batch process multiple audio files to extract features.
        
        Args:
            audio_files (list): List of paths to WAV files
            progress_callback (callable, optional): Callback function for progress updates
        """
        total_files = len(audio_files)
        
        for i, audio_file in enumerate(audio_files):
            self.set_status_message(f"🎧 Extracting audio features from: {os.path.basename(audio_file)}")
            print(f"Extracting features from: {audio_file}")
            
            # Create progress callback for this audio file
            def make_progress_callback(audio_index, total_audios):
                def file_progress_callback(extraction_progress):
                    if progress_callback:
                        # Calculate overall progress: (audio_index-1)/total_audios + extraction_progress/total_audios
                        overall_progress = int(((audio_index - 1) / total_audios) * 100 + (extraction_progress / total_audios))
                        progress_callback(overall_progress)
                return file_progress_callback
            
            try:
                self.extract_audio_features(audio_file, progress_callback=make_progress_callback(i + 1, total_files))
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue

    def _load_whisper_model(self, progress_callback=None):
        """
        Load Whisper model and processor (lazy loading).
        
        Args:
            progress_callback (callable, optional): Callback function for progress updates
        """
        if self.whisper_model is not None:
            return  # Already loaded
            
        if progress_callback:
            progress_callback(5)

        model_id = "distil-whisper/distil-large-v3"

        if progress_callback:
            progress_callback(10)

        # Load model
        self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, 
            torch_dtype=self.torch_dtype, 
            low_cpu_mem_usage=True, 
            use_safetensors=True
        )
        
        if progress_callback:
            progress_callback(25)
            
        self.whisper_model.to(self.device)

        if progress_callback:
            progress_callback(40)

        # Load processor
        self.whisper_processor = AutoProcessor.from_pretrained(model_id)

        if progress_callback:
            progress_callback(55)

        # Create pipeline
        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
            tokenizer=self.whisper_processor.tokenizer,
            feature_extractor=self.whisper_processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=25,
            batch_size=16,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        if progress_callback:
            progress_callback(70)

    def extract_transcript(self, filepath, progress_callback=None):
        """
        Extract transcript from a single WAV file using Whisper.
        
        Args:
            filepath (str): Path to the WAV file
            progress_callback (callable, optional): Callback function for progress updates
            
        Returns:
            str: Path to the saved transcript file
        """
        if self.output_transcripts_folder is None:
            raise ValueError("Transcripts output folder not configured")
            
        try:
            if progress_callback:
                progress_callback(0)
            
            print(f"Loading Whisper model for {filepath}...")
            
            # Load Whisper model (lazy loading)
            self._load_whisper_model(progress_callback)

            print(f"Transcribing {filepath}...")
            
            # Transcribe audio
            result = self.whisper_pipe(filepath)
            transcript = result['text']

            if progress_callback:
                progress_callback(90)

            # Save transcript to text file
            output_txt = os.path.join(
                self.output_transcripts_folder, 
                os.path.splitext(os.path.basename(filepath))[0] + ".txt"
            )

            with open(output_txt, 'w') as f:
                f.write(transcript)

            if progress_callback:
                progress_callback(100)

            print(f"Saved transcript: {output_txt}")
            return output_txt

        except Exception as e:
            error_msg = f'Error transcribing {filepath}: {e}'
            print(error_msg)
            raise Exception(error_msg)

    def extract_transcripts_batch(self, audio_files, progress_callback=None):
        """
        Batch process multiple audio files to generate transcripts.
        
        Args:
            audio_files (list): List of paths to WAV files
            progress_callback (callable, optional): Callback function for progress updates
        """
        total_files = len(audio_files)

        for i, audio_file in enumerate(audio_files):
            self.set_status_message(f"🗣️ Transcribing: {os.path.basename(audio_file)}")
            
            # Create progress callback for this audio file
            def make_progress_callback(audio_index, total_audios):
                def file_progress_callback(transcription_progress):
                    if progress_callback:
                        # Calculate overall progress: (audio_index-1)/total_audios + transcription_progress/total_audios
                        overall_progress = int(((audio_index - 1) / total_audios) * 100 + (transcription_progress / total_audios))
                        progress_callback(overall_progress)
                return file_progress_callback
        
            try:
                self.extract_transcript(audio_file, progress_callback=make_progress_callback(i + 1, total_files))
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue

    def process_audio_file(self, filepath, extract_features=True, extract_transcript=True, progress_callback=None):
        """
        Process a single audio file for both features and transcript.
        
        Args:
            filepath (str): Path to the WAV file
            extract_features (bool): Whether to extract audio features
            extract_transcript (bool): Whether to extract transcript
            progress_callback (callable, optional): Callback function for progress updates
            
        Returns:
            dict: Dictionary with paths to saved files
        """
        results = {}
        
        if extract_features:
            if progress_callback:
                progress_callback(0)
            results['features'] = self.extract_audio_features(filepath, progress_callback)
        
        if extract_transcript:
            if progress_callback:
                progress_callback(50)
            results['transcript'] = self.extract_transcript(filepath, progress_callback)
        
        return results

    def process_audio_batch(self, audio_files, extract_features=True, extract_transcript=True, progress_callback=None):
        """
        Batch process multiple audio files for both features and transcripts.
        
        Args:
            audio_files (list): List of paths to WAV files
            extract_features (bool): Whether to extract audio features
            extract_transcript (bool): Whether to extract transcript
            progress_callback (callable, optional): Callback function for progress updates
        """
        if extract_features:
            self.extract_audio_features_batch(audio_files, progress_callback)
        
        if extract_transcript:
            self.extract_transcripts_batch(audio_files, progress_callback)
