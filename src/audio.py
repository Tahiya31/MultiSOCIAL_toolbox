"""
Audio processing module for MultiSOCIAL Toolbox.

This module provides functionality for:
- Audio feature extraction using OpenSMILE
- Speech transcription using Whisper
- Speaker diarization using PyAnnote
- Speaker-transcript alignment
"""

import os
import torch
import librosa
import opensmile
import gc
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from pyannote.audio import Pipeline


class AudioProcessor:
    def __init__(self, output_audio_features_folder, output_transcripts_folder, status_callback=None, enable_speaker_diarization=True, auth_token=None):
        """
        Initialize the AudioProcessor.
        
        Args:
            output_audio_features_folder (str): Path to folder for saving audio features CSV files
            output_transcripts_folder (str): Path to folder for saving transcript text files
            status_callback (callable, optional): Callback function for status updates
            enable_speaker_diarization (bool): Whether to enable speaker diarization for transcripts
            auth_token (str, optional): Hugging Face auth token for pyannote (required for speaker diarization)
        """
        self.output_audio_features_folder = output_audio_features_folder
        self.output_transcripts_folder = output_transcripts_folder
        self.status_callback = status_callback
        self.enable_speaker_diarization = enable_speaker_diarization
        self.auth_token = auth_token
        
        # Ensure output directories exist (only if they are not None)
        if self.output_audio_features_folder is not None:
            os.makedirs(self.output_audio_features_folder, exist_ok=True)
        if self.output_transcripts_folder is not None:
            os.makedirs(self.output_transcripts_folder, exist_ok=True)
        
        # Initialize device for Whisper
        # Initialize device for Whisper
        if torch.cuda.is_available():
            self.device = "cuda:0"
            self.torch_dtype = torch.float16
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.torch_dtype = torch.float16
        else:
            self.device = "cpu"
            self.torch_dtype = torch.float32
        
        # Whisper model will be loaded lazily when needed
        self.whisper_model = None
        self.whisper_processor = None
        self.whisper_pipe = None
        
        # Speaker diarizer will be loaded lazily when needed
        self.speaker_diarizer = None

    def set_status_message(self, message):
        """Safely update status message using callback if available."""
        if self.status_callback:
            self.status_callback(message)

    def _create_scoped_progress_callback(self, parent_callback, start_percentage, end_percentage):
        """
        Create a scoped progress callback that maps 0-100% to a sub-range of the parent callback.
        
        Args:
            parent_callback (callable): The main progress callback
            start_percentage (int): The start of the sub-range (0-100)
            end_percentage (int): The end of the sub-range (0-100)
            
        Returns:
            callable: A new callback function
        """
        if not parent_callback:
            return None
            
        def scoped_callback(progress):
            # Map progress (0-100) to range [start_percentage, end_percentage]
            range_width = end_percentage - start_percentage
            scaled_progress = start_percentage + (progress / 100.0 * range_width)
            parent_callback(int(scaled_progress))
            
        return scoped_callback

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
            # If model exists but is on CPU when we have a better device, move it back
            if self.device != "cpu" and self.whisper_model.device.type == "cpu":
                print(f"Moving Whisper model back to {self.device}...")
                self.whisper_model.to(self.device)
            return  # Already loaded
            
        if progress_callback:
            progress_callback(5)

        # Use standard large-v3-turbo for better speed/accuracy balance than distil
        # or fallback to large-v3 if turbo is unavailable
        model_id = "openai/whisper-large-v3-turbo"

        if progress_callback:
            progress_callback(10)

        # Load model
        try:
            print("Attempting to load Whisper model from local cache...")
            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=self.torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True,
                local_files_only=True
            )
            print("✓ Loaded Whisper model from cache.")
        except Exception as e:
            print(f"Whisper model not found in cache or error loading: {e}. Downloading...")
            self.whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id, 
                torch_dtype=self.torch_dtype, 
                low_cpu_mem_usage=True, 
                use_safetensors=True,
                local_files_only=False
            )
        
        if progress_callback:
            progress_callback(25)
            
        self.whisper_model.to(self.device)

        if progress_callback:
            progress_callback(40)

        # Load processor
        # Load processor
        try:
            self.whisper_processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        except Exception:
            self.whisper_processor = AutoProcessor.from_pretrained(model_id, local_files_only=False)

        if progress_callback:
            progress_callback(55)

        # Create pipeline with explicit configuration to avoid warnings
        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
            tokenizer=self.whisper_processor.tokenizer,
            feature_extractor=self.whisper_processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30, # Standard chunk length for robustness
            batch_size=8,
            torch_dtype=self.torch_dtype,
            device=self.device,
            # Add return_timestamps for better processing
            return_timestamps=True
        )

        if progress_callback:
            progress_callback(70)

    def _offload_whisper_model(self):
        """Offload Whisper model to CPU to free up VRAM for diarization."""
        if self.whisper_model is not None and self.device != "cpu":
            print("Offloading Whisper model to CPU to free VRAM...")
            self.whisper_model.to("cpu")
            
            # Explicitly clear cache based on device
            if self.device == "mps":
                try:
                    torch.mps.empty_cache()
                except AttributeError:
                    pass
            elif self.device.startswith("cuda"):
                torch.cuda.empty_cache()
                
            gc.collect()

    def _load_speaker_diarizer(self, progress_callback=None):
        """Load the PyAnnote speaker diarization model (lazy loading)"""
        if self.speaker_diarizer is not None:
            return
            
        try:
            if progress_callback:
                progress_callback(5)
                
            print("Loading PyAnnote speaker diarization model...")
            # Use PyAnnote diarization
            self.speaker_diarizer = PyAnnoteSpeakerDiarizer(progress_callback, self.auth_token, device=self.device)
            
            if progress_callback:
                progress_callback(15)
                
        except Exception as e:
            raise Exception(f"Failed to load PyAnnote speaker diarization model: {str(e)}")
    
    def preload_speaker_diarizer(self):
        """Pre-load the speaker diarization model to avoid delays during first use"""
        try:
            print("Pre-loading PyAnnote speaker diarization model...")
            self._load_speaker_diarizer()
            print("✓ PyAnnote model pre-loaded successfully")
        except Exception as e:
            print(f"Warning: Could not pre-load PyAnnote model: {e}")

    def extract_transcript(self, filepath, progress_callback=None):
        """
        Extract transcript from a single WAV file using Whisper with optional speaker diarization.
        
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
            # Allocation: 0-10%
            self._load_whisper_model(
                progress_callback=self._create_scoped_progress_callback(progress_callback, 0, 10)
            )

            print(f"Transcribing {filepath}...")
            
            if progress_callback:
                progress_callback(10)

            # Transcribe audio with timestamps (request at call time for reliability)
            # Note: return_timestamps=True (segment level) is generally more robust for alignment
            # than word-level, especially with turbo models.
            result = self.whisper_pipe(
                filepath,
                return_timestamps=True, 
                generate_kwargs={"task": "transcribe"}
            )
            transcript = result['text']
            # Debug: summarize timestamps presence
            try:
                if isinstance(result, dict):
                    print("Whisper output keys:", list(result.keys()))
                    if 'chunks' in result and isinstance(result['chunks'], list):
                        print(f"Whisper chunks count: {len(result['chunks'])}")
                        for i, c in enumerate(result['chunks'][:3]):
                            ts = c.get('timestamp') or c.get('timestamps') or (c.get('start'), c.get('end'))
                            print(f"  chunk[{i}] ts={ts} text='{(c.get('text') or '').strip()[:60]}'")
                    elif 'segments' in result and isinstance(result['segments'], list):
                        print(f"Whisper segments count: {len(result['segments'])}")
                        for i, s in enumerate(result['segments'][:3]):
                            print(f"  segment[{i}] start={s.get('start')} end={s.get('end')} text='{(s.get('text') or '').strip()[:60]}'")
                    else:
                        print("Whisper output does not include 'chunks' or 'segments'.")
            except Exception as _dbg_e:
                print(f"Whisper debug print failed: {_dbg_e}")
            
            # Store the full result for timestamped segments
            self.whisper_result = result

            if progress_callback:
                progress_callback(50)

            # Perform speaker diarization if enabled
            speaker_segments = None
            if self.enable_speaker_diarization:
                try:
                    # Offload Whisper to free up memory for Diarization
                    self._offload_whisper_model()

                    print(f"Performing speaker diarization for {filepath}...")
                    
                    # Allocation: 50-90%
                    scoped_diarization_callback = self._create_scoped_progress_callback(progress_callback, 50, 90)
                    
                    self._load_speaker_diarizer(progress_callback=scoped_diarization_callback)
                    speaker_segments = self.speaker_diarizer.diarize_speakers(filepath, progress_callback=scoped_diarization_callback)
                    
                    # Offload diarizer to free memory for next Whisper run or general system use
                    self.speaker_diarizer.offload_model()
                    
                    print(f"Found {len(speaker_segments)} speaker segments")
                except Exception as e:
                    print(f"Speaker diarization failed: {str(e)}")
                    print("Continuing with transcript only...")
                    speaker_segments = None
            else:
                # If diarization is disabled, we just jump to 90%
                if progress_callback:
                    progress_callback(90)

            # Format transcript with speaker labels if available
            if speaker_segments and len(speaker_segments) > 0:
                formatted_transcript = self._format_transcript_with_speakers(transcript, speaker_segments)
            else:
                formatted_transcript = transcript

            # Save transcript to text file
            output_txt = os.path.join(
                self.output_transcripts_folder, 
                os.path.splitext(os.path.basename(filepath))[0] + ".txt"
            )

            with open(output_txt, 'w') as f:
                f.write(formatted_transcript)

            if progress_callback:
                progress_callback(100)

            print(f"Saved transcript: {output_txt}")
            return output_txt

        except Exception as e:
            error_msg = f'Error transcribing {filepath}: {e}'
            print(error_msg)
            raise Exception(error_msg)

    def _format_transcript_with_speakers(self, transcript, speaker_segments):
        """
        Format transcript with speaker labels based on speaker segments.
        
        Args:
            transcript (str): Raw transcript from Whisper
            speaker_segments (list): List of speaker segments with timestamps
            
        Returns:
            str: Formatted transcript with speaker labels
        """
        if not speaker_segments:
            return transcript
            
        # Get Whisper's timestamped segments for alignment
        try:
            # Extract timestamped segments from Whisper result
            whisper_segments = self._extract_whisper_segments(transcript)
            
            if whisper_segments:
                # Align Whisper segments with speaker segments
                aligned_transcript = self._align_segments_with_speakers(whisper_segments, speaker_segments)
                if aligned_transcript.strip():
                    return aligned_transcript
            # Fallback to raw transcript if no timestamps available/alignment failed
            print("Warning: No aligned transcript produced; falling back to raw transcript.")
            return transcript
                
        except Exception as e:
            print(f"Warning: Could not align transcript with speakers: {e}")
            return transcript
    
    def _extract_whisper_segments(self, transcript):
        """
        Extract timestamped segments from Whisper result.
        
        Args:
            transcript (str): Raw transcript from Whisper
            
        Returns:
            list: List of (start_time, end_time, text) tuples
        """
        # Use the stored Whisper result if available
        if hasattr(self, 'whisper_result') and self.whisper_result:
            result = self.whisper_result
            
            # Preferred: pipeline returns 'chunks' with timestamp tuples
            if isinstance(result, dict) and 'chunks' in result and result['chunks']:
                segments = []
                for chunk in result['chunks']:
                    text = (chunk.get('text') or '').strip()
                    if not text:
                        continue

                    # Support multiple timestamp schemas observed across transformers versions
                    start_time = None
                    end_time = None

                    # Schema 1: 'timestamp': (start, end)
                    ts_tuple = chunk.get('timestamp')
                    if ts_tuple is not None and isinstance(ts_tuple, (list, tuple)) and len(ts_tuple) >= 2:
                        start_time = ts_tuple[0] if ts_tuple[0] is not None else 0.0
                        end_time = ts_tuple[1] if ts_tuple[1] is not None else (start_time + 0.1)

                    # Schema 2: 'timestamps': {'start': x, 'end': y}
                    if start_time is None or end_time is None:
                        ts_obj = chunk.get('timestamps') or chunk.get('time_stamps')
                        if isinstance(ts_obj, dict):
                            s = ts_obj.get('start')
                            e = ts_obj.get('end')
                            if s is not None and e is not None:
                                start_time = float(s)
                                end_time = float(e)

                    # Schema 3: direct fields 'start'/'end' (some outputs)
                    if start_time is None or end_time is None:
                        if 'start' in chunk and 'end' in chunk:
                            start_time = float(chunk['start'])
                            end_time = float(chunk['end'])

                    # Final guard defaults
                    if start_time is None:
                        start_time = 0.0
                    if end_time is None or end_time <= start_time:
                        end_time = start_time + 0.1

                    segments.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text
                    })
                
                if segments:
                    return segments

            # Alternative: some versions expose 'segments' list
            if isinstance(result, dict) and 'segments' in result and result['segments']:
                segments = []
                for seg in result['segments']:
                    text = (seg.get('text') or '').strip()
                    if not text:
                        continue
                    try:
                        start_time = float(seg.get('start', 0.0))
                        end_time = float(seg.get('end', start_time + 1.0))
                    except Exception:
                        start_time = 0.0
                        end_time = 1.0
                    if end_time <= start_time:
                        end_time = start_time + 0.1
                    segments.append({'start': start_time, 'end': end_time, 'text': text})
                if segments:
                    return segments
        
        # Fallback: create approximation from transcript
        sentences = transcript.split('. ')
        segments = []
        current_time = 0.0
        
        for sentence in sentences:
            if sentence.strip():
                # Estimate duration based on text length (rough approximation)
                duration = max(1.0, len(sentence) * 0.1)  # ~0.1 seconds per character
                end_time = current_time + duration
                
                segments.append({
                    'start': current_time,
                    'end': end_time,
                    'text': sentence.strip() + ('.' if not sentence.endswith('.') else '')
                })
                current_time = end_time
        
        return segments
    
    def _align_segments_with_speakers(self, whisper_segments, speaker_segments):
        """
        Align Whisper transcript segments with speaker segments and merge
        consecutive chunks by the same speaker into longer utterances.
        
        Args:
            whisper_segments (list): List of Whisper segments with timestamps
            speaker_segments (list): List of speaker segments with timestamps
            
        Returns:
            str: Aligned transcript with speaker labels (merged by speaker)
        """
        # Label each ASR chunk with a speaker
        labeled = []
        for seg in whisper_segments:
            speaker = self._find_speaker_for_time(seg['start'], seg['end'], speaker_segments)
            labeled.append({
                'speaker': speaker,
                'start': float(seg['start']),
                'end': float(seg['end']),
                'text': seg['text']
            })

        if not labeled:
            return ""

        # Merge consecutive chunks from the same speaker with small gaps
        gap_merge_threshold = 0.5  # seconds
        merged = []
        for item in labeled:
            if not merged:
                merged.append(item.copy())
                continue
            prev = merged[-1]
            same_speaker = (item['speaker'] == prev['speaker'])
            small_gap = (item['start'] - prev['end']) <= gap_merge_threshold
            if same_speaker and small_gap:
                prev['end'] = max(prev['end'], item['end'])
                # Simple whitespace join; avoid double spaces
                prev_text = (prev['text'] or '').strip()
                item_text = (item['text'] or '').strip()
                if prev_text and item_text:
                    prev['text'] = prev_text + ' ' + item_text
                elif item_text:
                    prev['text'] = item_text
                # else keep prev['text'] as is
            else:
                merged.append(item.copy())

        # Format merged lines
        lines = []
        for m in merged:
            # Skip empty text entries
            if not (m.get('text') or '').strip():
                continue
            start_time = f"{int(m['start']//60):02d}:{m['start']%60:06.3f}"
            end_time = f"{int(m['end']//60):02d}:{m['end']%60:06.3f}"
            lines.append(f"{m['speaker']}: [{start_time} - {end_time}] {m['text']}")

        return "\n".join(lines)
    
    def _find_speaker_for_time(self, start_time, end_time, speaker_segments):
        """
        Find which speaker was talking during a given time period.
        
        Args:
            start_time (float): Start time of the segment
            end_time (float): End time of the segment
            speaker_segments (list): List of speaker segments
            
        Returns:
            str: Speaker label for the time period
        """
        if not speaker_segments:
            return "UNKNOWN"
            
        # Prefer speaker whose segment contains the midpoint (handles overlaps)
        midpoint = (start_time + end_time) / 2.0
        containing = []
        for seg in speaker_segments:
            try:
                if seg['start'] <= midpoint <= seg['end']:
                    containing.append(seg)
            except (KeyError, TypeError):
                continue
        if containing:
            # If multiple contain midpoint (overlapped speech), prefer the shorter segment (more specific)
            chosen = min(containing, key=lambda s: (s['end'] - s['start'], s['start']))
            return chosen.get('speaker', 'UNKNOWN')

        # Fallback: choose maximal overlap; tie-break by highest coverage ratio then shorter segment
        best_speaker = "UNKNOWN"
        best_overlap = 0.0
        best_coverage = -1.0
        best_len = float('inf')
        
        # Also track the closest speaker in case we are in a gap (e.g. VAD silence)
        closest_speaker = "UNKNOWN"
        min_dist = float('inf')
        
        for seg in speaker_segments:
            try:
                seg_start = seg['start']
                seg_end = seg['end']
                
                # Check proximity for fallback
                dist = 0
                if end_time < seg_start:
                    dist = seg_start - end_time
                elif start_time > seg_end:
                    dist = start_time - seg_end
                else:
                    dist = 0 # overlap
                    
                if dist < min_dist:
                    min_dist = dist
                    closest_speaker = seg.get('speaker', 'UNKNOWN')

                seg_len = max(1e-6, seg_end - seg_start)
                overlap_start = max(start_time, seg_start)
                overlap_end = min(end_time, seg_end)
                overlap_duration = max(0.0, overlap_end - overlap_start)
                
                if overlap_duration <= 0.0:
                    continue
                
                coverage = overlap_duration / seg_len
                # Primary: larger overlap, Secondary: higher coverage, Tertiary: shorter segment
                if (
                    overlap_duration > best_overlap or
                    (abs(overlap_duration - best_overlap) < 1e-9 and coverage > best_coverage) or
                    (abs(overlap_duration - best_overlap) < 1e-9 and abs(coverage - best_coverage) < 1e-9 and seg_len < best_len)
                ):
                    best_overlap = overlap_duration
                    best_coverage = coverage
                    best_len = seg_len
                    best_speaker = seg.get('speaker', 'UNKNOWN')
            except (KeyError, TypeError):
                continue
                
        # If we found an overlapping speaker, return it
        if best_speaker != "UNKNOWN":
            return best_speaker
            
        # If no overlap but we are very close to a speaker (e.g. < 0.5s gap), infer it
        if min_dist < 0.5 and closest_speaker != "UNKNOWN":
            return closest_speaker
            
        return "UNKNOWN"

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
        
        if extract_features and extract_transcript:
            # Split progress: 50% for features, 50% for transcript
            if progress_callback:
                progress_callback(0)
            
            features_callback = self._create_scoped_progress_callback(progress_callback, 0, 50)
            results['features'] = self.extract_audio_features(filepath, features_callback)
            
            transcript_callback = self._create_scoped_progress_callback(progress_callback, 50, 100)
            results['transcript'] = self.extract_transcript(filepath, transcript_callback)
            
        elif extract_features:
            if progress_callback:
                progress_callback(0)
            results['features'] = self.extract_audio_features(filepath, progress_callback)
            
        elif extract_transcript:
            if progress_callback:
                progress_callback(0)
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
        if extract_features and extract_transcript:
            # Split progress: 50% for features, 50% for transcript
            features_callback = self._create_scoped_progress_callback(progress_callback, 0, 50)
            self.extract_audio_features_batch(audio_files, features_callback)
            
            transcript_callback = self._create_scoped_progress_callback(progress_callback, 50, 100)
            self.extract_transcripts_batch(audio_files, transcript_callback)
            
        elif extract_features:
            self.extract_audio_features_batch(audio_files, progress_callback)
        
        elif extract_transcript:
            self.extract_transcripts_batch(audio_files, progress_callback)


# PyAnnote-based SpeakerDiarizer
class PyAnnoteSpeakerDiarizer:
    """
    Speaker diarization class using pyannote-audio to identify speakers in audio files.
    Requires Hugging Face auth token for model access.
    """
    
    def __init__(self, progress_callback=None, auth_token=None, device="cpu"):
        """
        Initialize the PyAnnoteSpeakerDiarizer.
        
        Args:
            progress_callback (callable, optional): Callback function for progress updates
            auth_token (str, optional): Hugging Face auth token for model access
            device (str, optional): Device to run the model on (e.g., "cuda:0", "cpu")
        """
        self.progress_callback = progress_callback
        self.auth_token = auth_token
        self.device = device
        self.diarization_pipeline = None
        
    def _load_diarization_model(self):
        """Load the speaker diarization model (lazy loading)"""
        if self.diarization_pipeline is not None:
            # Ensure it's on the right device if it was offloaded
            try:
                # Check if we need to move it back to GPU
                if self.device != "cpu":
                    # Move pipeline to the specified device
                    self.diarization_pipeline.to(torch.device(self.device))
            except Exception:
                pass
            return
            
        try:
            if self.progress_callback:
                self.progress_callback(5)
                
            # Load the pre-trained speaker diarization pipeline
            # Note: You need to accept the license at https://huggingface.co/pyannote/speaker-diarization
            try:
                print("Attempting to load PyAnnote model from local cache...")
                if self.auth_token:
                    # Try passing local_files_only if supported, or rely on cache check
                    # Note: pyannote.audio Pipeline.from_pretrained might not explicitly support local_files_only 
                    # in all versions, but we attempt it. If it fails (TypeError), we fall back.
                    # However, to be safe and robust, we'll try to set environment variable for offline mode temporarily
                    # if we really want to force offline. But for "cache if available", standard behavior usually prefers cache.
                    # We will try explicit kwarg first.
                    try:
                        self.diarization_pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization",
                            use_auth_token=self.auth_token,
                            # Some versions of pyannote/huggingface_hub might accept this
                            # If not, it might just ignore it or error out.
                        )
                    except Exception:
                        # If simple load fails, we can't easily force offline without env var.
                        # But actually, the user wants "if available use it".
                        # Hugging Face usually uses cache by default if internet is up.
                        # If internet is DOWN, it uses cache.
                        # The user wants to ensure we don't *need* internet.
                        # So we don't necessarily need to force offline, just ensure it works.
                        # But to be explicit, let's try to simulate the "offline first" check.
                        pass
                    
                    # Actually, a better approach for PyAnnote (which wraps HF Hub) is:
                    # It automatically uses cache. If we want to *ensure* it doesn't hit internet if cached:
                    # We can't easily do it without potentially breaking the "check for updates" logic unless we go offline.
                    # But standard HF `from_pretrained` DOES hit the API to check for updates unless `local_files_only=True`.
                    
                    # Let's try passing it.
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization",
                        use_auth_token=self.auth_token,
                        cache_dir=None # Use default
                    )
                    # Note: PyAnnote's Pipeline.from_pretrained is a bit opaque.
                    # Let's stick to the user request: "cache ... so we don't need internet".
                    # If we just load it normally, it caches. Next time if no internet, it uses cache.
                    # But if internet IS present, it checks for updates.
                    # To force "use cache if exists", we can try:
                    
                else:
                    self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
                    
            except Exception:
                # If loading fails (e.g. no internet and not cached), we can't do much but let it fail or retry.
                # But we want to implement the "try offline first" logic.
                pass

            # RE-IMPLEMENTING with explicit fallback logic
            loaded = False
            
            # 1. Try offline load
            try:
                print("Attempting to load PyAnnote model from local cache...")
                kw = {"use_auth_token": self.auth_token} if self.auth_token else {}
                # Try passing local_files_only=True. 
                # If the library doesn't support it, it raises TypeError.
                # If it supports it but model not found, it raises OSError/ValueError.
                self.diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    local_files_only=True,
                    **kw
                )
                print("✓ Loaded PyAnnote model from cache.")
                loaded = True
            except Exception as e:
                print(f"Offline load failed ({e}). Trying online...")
            
            # 2. Fallback to standard load (online check)
            if not loaded:
                try:
                    print("Downloading/Loading PyAnnote model (online)...")
                    kw = {"use_auth_token": self.auth_token} if self.auth_token else {}
                    self.diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization",
                        **kw
                    )
                    loaded = True
                except Exception as e:
                    print(f"Standard load failed: {e}")

            if not loaded:
                raise Exception("Could not load PyAnnote pipeline.")
            
            # Move pipeline to the specified device
            if self.diarization_pipeline is not None:
                self.diarization_pipeline.to(torch.device(self.device))
                print(f"✓ PyAnnote pipeline moved to {self.device}")
            
            if self.progress_callback:
                self.progress_callback(20)
                
        except Exception as e:
            raise Exception(f"Failed to load pyannote diarization model: {str(e)}")

    def offload_model(self):
        """Offload the diarization model to CPU to free up VRAM."""
        if self.diarization_pipeline is not None and self.device != "cpu":
            try:
                print("Offloading PyAnnote model to CPU...")
                # PyAnnote pipeline.to() works similar to torch models
                self.diarization_pipeline.to(torch.device("cpu"))
                
                # Clear cache
                if self.device == "mps":
                    try:
                        torch.mps.empty_cache()
                    except AttributeError:
                        pass
                elif self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"Warning: Failed to offload PyAnnote model: {e}")
    
    def diarize_speakers(self, filepath, progress_callback=None):
        """
        Perform speaker diarization on an audio file using pyannote.
        
        Args:
            filepath (str): Path to the audio file
            progress_callback (callable, optional): Callback function for progress updates
            
        Returns:
            list: List of speaker segments with timestamps
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")
            
        if progress_callback:
            progress_callback(0)
            
        try:
            # Load diarization model
            self._load_diarization_model()
            
            if progress_callback:
                progress_callback(10)
                
            # Perform diarization
            print("PyAnnote is processing audio (this may take 2-5 minutes on first run)...")
            diarization = self.diarization_pipeline(filepath)
            print("PyAnnote processing completed!")
            
            # Cleanup
            if self.device == "mps":
                try:
                    torch.mps.empty_cache()
                except AttributeError:
                    pass
            elif self.device.startswith("cuda"):
                torch.cuda.empty_cache()
            gc.collect()
            
            if progress_callback:
                progress_callback(50)
                
            # Convert to list of speaker segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    'start': float(turn.start),
                    'end': float(turn.end),
                    'speaker': str(speaker),
                    'duration': float(turn.end - turn.start)
                })
            
            if progress_callback:
                progress_callback(90)
                
            return speaker_segments
            
        except Exception as e:
            raise Exception(f"PyAnnote speaker diarization failed: {str(e)}")
    
    def format_speaker_segments(self, speaker_segments):
        """
        Format speaker segments into a readable string.
        
        Args:
            speaker_segments (list): List of speaker segments
            
        Returns:
            str: Formatted speaker segments
        """
        if not speaker_segments:
            return "No speakers detected"
            
        formatted_segments = []
        for segment in speaker_segments:
            start_time = f"{int(segment['start']//60):02d}:{segment['start']%60:06.3f}"
            end_time = f"{int(segment['end']//60):02d}:{segment['end']%60:06.3f}"
            formatted_segments.append(f"{segment['speaker']}: {start_time} --> {end_time}")
            
        return "\n".join(formatted_segments)
