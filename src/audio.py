"""
Audio processing module for MultiSOCIAL Toolbox.

This module provides functionality for:
- Audio feature extraction using OpenSMILE
- Speech transcription using Whisper
- Speaker diarization using PyAnnote
- Speaker-transcript alignment
"""

import os
import json
import torch
import opensmile
import gc
import numpy as np
from scipy.io import wavfile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from runtime_services import DIARIZATION_MODEL_ID


_SCIPY_WAV_EXTENSIONS = {".wav", ".wave"}


def _pcm_audio_to_mono_float32(audio, sampling_rate):
    """Normalize arbitrary PCM/float waveform to mono float32 in [-1, 1] (approximate for float clips)."""
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        scale = float(np.iinfo(audio.dtype).max)
        if audio.ndim > 1:
            audio = audio.astype(np.float32).mean(axis=1)
        if scale > 0:
            audio = audio / scale
        else:
            audio = audio.astype(np.float32)
    else:
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32, copy=False)
    return audio, int(sampling_rate)


def _decode_audio_file(filepath):
    """
    Decode a supported audio file to mono float32 without FFmpeg.

    Hugging Face ASR passes string paths through ffmpeg_read(bytes(...)); SciPy fails on some
    WAV variants (e.g. certain extensible headers). libsndfile via soundfile covers those cases
    and also supports additional low-risk formats like AIFF, FLAC, CAF, AU, and SND.
    """
    filepath = os.path.normpath(os.path.abspath(filepath))
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    errors = []

    if ext in _SCIPY_WAV_EXTENSIONS:
        try:
            sr, audio = wavfile.read(filepath)
            return _pcm_audio_to_mono_float32(audio, sr)
        except Exception as e:
            errors.append(f"scipy.io.wavfile: {e}")

    try:
        import soundfile as sf

        audio, sr = sf.read(filepath, dtype="float32", always_2d=False)
        return _pcm_audio_to_mono_float32(audio, sr)
    except ImportError:
        errors.append("soundfile: not installed (required for non-WAV inputs and some WAV formats on Windows)")
    except Exception as e:
        errors.append(f"soundfile: {e}")

    raise RuntimeError(
        f"Could not decode this audio file ({ext or 'unknown extension'}) without FFmpeg-based loaders.\n\n"
        + "\n".join(errors)
        + "\n\nTry re-exporting as WAV, AIFF, or FLAC, or ensure the 'soundfile' dependency is bundled."
    )


def _load_audio_for_opensmile(filepath):
    """Load a supported audio file into a mono float32 array suitable for OpenSMILE."""
    return _decode_audio_file(filepath)


def _whisper_pipeline_audio_input(filepath):
    """
    Build HF ASR inputs dict so preprocess() never receives a str path.

    A string path makes transformers read raw bytes then call ffmpeg_read(), which requires FFmpeg.
    """
    audio, sr = _decode_audio_file(filepath)
    return {"array": audio, "sampling_rate": sr}


class AudioProcessor:
    def __init__(self, output_audio_features_folder, output_transcripts_folder, status_callback=None, enable_speaker_diarization=False, auth_token=None):
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
            
            # Audio input is decoded via scipy for WAV and soundfile/libsndfile otherwise.
            y, sr = _load_audio_for_opensmile(filepath)
            
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

    def extract_audio_features_batch(self, audio_files, progress_callback=None, cancel_check=None):
        """
        Batch process multiple audio files to extract features.
        
        Args:
            audio_files (list): List of paths to WAV files
            progress_callback (callable, optional): Callback function for progress updates
            cancel_check (callable, optional): Return True to stop before the next file
        """
        total_files = len(audio_files)
        
        for i, audio_file in enumerate(audio_files):
            if cancel_check and cancel_check():
                break
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

        def load_whisper_from_pretrained(local_files_only):
            kwargs = {
                "torch_dtype": self.torch_dtype,
                "use_safetensors": True,
                "local_files_only": local_files_only,
            }
            try:
                return AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    low_cpu_mem_usage=True,
                    **kwargs,
                )
            except ImportError as e:
                if "low_cpu_mem_usage=True" not in str(e):
                    raise
                print("Accelerate is not installed; loading Whisper without low_cpu_mem_usage.")
                return AutoModelForSpeechSeq2Seq.from_pretrained(
                    model_id,
                    low_cpu_mem_usage=False,
                    **kwargs,
                )

        # Load model
        try:
            print("Attempting to load Whisper model from local cache...")
            self.whisper_model = load_whisper_from_pretrained(local_files_only=True)
            print("✓ Loaded Whisper model from cache.")
        except Exception as e:
            print(f"Whisper model not found in cache or error loading: {e}. Downloading...")
            self.whisper_model = load_whisper_from_pretrained(local_files_only=False)
        
        if progress_callback:
            progress_callback(25)
            
        self.whisper_model.to(self.device)

        if progress_callback:
            progress_callback(40)

        # Load processor
        try:
            self.whisper_processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
        except Exception:
            self.whisper_processor = AutoProcessor.from_pretrained(model_id, local_files_only=False)

        if progress_callback:
            progress_callback(55)

        # Create pipeline with explicit configuration to avoid warnings
        # Diarization mode favors peak-memory safety over throughput. A high ASR
        # batch size can spike RAM on long files before pyannote even starts.
        whisper_batch_size = 1 if self.enable_speaker_diarization else 8
        self.whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model=self.whisper_model,
            tokenizer=self.whisper_processor.tokenizer,
            feature_extractor=self.whisper_processor.feature_extractor,
            # No max_new_tokens cap: a low cap (was 128) truncates long 30s chunks and
            # breaks word-level timestamp alignment. Let Whisper use its full target window.
            chunk_length_s=30, # Standard chunk length for robustness
            batch_size=whisper_batch_size,
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
            self._clear_torch_cache()

    def _clear_torch_cache(self):
        """Release Python and torch allocator caches after large model/audio work."""
        if self.device == "mps":
            try:
                torch.mps.empty_cache()
            except AttributeError:
                pass
        elif self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        gc.collect()

    def _clear_whisper_model(self):
        """Fully release Whisper objects so pyannote can run with lower peak RAM."""
        self.whisper_pipe = None
        self.whisper_processor = None
        self.whisper_model = None
        self.whisper_result = None
        self._clear_torch_cache()

    def _clear_speaker_diarizer(self):
        """Fully release pyannote objects after a diarized file finishes."""
        if self.speaker_diarizer is not None:
            try:
                self.speaker_diarizer.offload_model()
            finally:
                self.speaker_diarizer = None
        self._clear_torch_cache()

    def _speaker_diarization_device(self):
        """Prefer CPU for pyannote on Apple Silicon to avoid unified-memory spikes."""
        if self.device == "mps":
            return "cpu"
        return self.device

    def _compact_whisper_result(self, whisper_result, transcript):
        """Keep only text and timestamped chunks needed for transcript/SRT formatting."""
        compact = {"text": transcript, "chunks": []}
        self.whisper_result = whisper_result
        for seg in self._extract_whisper_segments(transcript):
            compact["chunks"].append({
                "text": str(seg.get("text") or ""),
                "timestamp": (float(seg.get("start", 0.0)), float(seg.get("end", 0.0))),
            })
        return compact

    def _write_word_json_sidecar(self, audio_file, whisper_result):
        """Write raw word-level Whisper output for Align Features."""
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_json = os.path.join(self.output_transcripts_folder, f"{base_name}_words.json")
        try:
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(whisper_result, f, indent=2, ensure_ascii=False)
            print(f"Saved word-level info: {output_json}")
        except Exception as e:
            print(f"Warning: Could not save word-level JSON for {base_name}: {e}")

    def _write_srt_sidecar(self, filepath, transcript, whisper_result, speaker_segments=None):
        """Write a readable SRT sidecar next to the transcript text file."""
        try:
            from captions import write_srt

            self.whisper_result = whisper_result
            srt_segments = self._extract_whisper_segments(transcript)
            output_srt = os.path.join(
                self.output_transcripts_folder,
                os.path.splitext(os.path.basename(filepath))[0] + ".srt",
            )
            label_fn = None
            if speaker_segments:
                label_fn = lambda s, e: self._find_speaker_for_time(s, e, speaker_segments)
            return write_srt(srt_segments, output_srt, speaker_label_fn=label_fn)
        except Exception as e:
            print(f"Warning: Could not write SRT sidecar for {os.path.basename(filepath)}: {e}")
            return None

    def _load_speaker_diarizer(self, progress_callback=None):
        """Load the PyAnnote speaker diarization model (lazy loading)"""
        if self.speaker_diarizer is not None:
            # Ensure preload paths actually verify model import/load state.
            self.speaker_diarizer._load_diarization_model()
            return
            
        try:
            if progress_callback:
                progress_callback(5)
                
            print("Loading PyAnnote speaker diarization model...")
            # Use PyAnnote diarization
            self.speaker_diarizer = PyAnnoteSpeakerDiarizer(
                progress_callback,
                self.auth_token,
                device=self._speaker_diarization_device(),
            )
            # Force an actual model load so "preload" validates availability.
            self.speaker_diarizer._load_diarization_model()
            
            if progress_callback:
                progress_callback(15)
                
        except Exception as e:
            raise Exception(f"Failed to load PyAnnote speaker diarization model: {str(e)}")
    
    def preload_speaker_diarizer(self):
        """Validate diarization setup without keeping pyannote resident in memory."""
        print("Pre-loading PyAnnote speaker diarization model...")
        try:
            self._load_speaker_diarizer()
            print("✓ PyAnnote model pre-loaded successfully")
        finally:
            self._clear_speaker_diarizer()

    def extract_transcript(self, filepath, progress_callback=None, word_timestamps=False):
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
            # However, if word_timestamps is requested (for alignment feature), we use "word".
            ts_mode = "word" if word_timestamps else True
            
            result = self.whisper_pipe(
                _whisper_pipeline_audio_input(filepath),
                return_timestamps=ts_mode, 
                generate_kwargs={"task": "transcribe"}
            )
            transcript = result['text']

            compact_result = self._compact_whisper_result(result, transcript)

            if progress_callback:
                progress_callback(50)

            if self.enable_speaker_diarization and word_timestamps:
                self._write_word_json_sidecar(filepath, result)

            # Perform speaker diarization if enabled
            speaker_segments = None
            if self.enable_speaker_diarization:
                try:
                    del result
                    self._clear_whisper_model()

                    print(f"Performing speaker diarization for {filepath}...")
                    
                    # Allocation: 50-90%
                    scoped_diarization_callback = self._create_scoped_progress_callback(progress_callback, 50, 90)
                    
                    self._load_speaker_diarizer(progress_callback=scoped_diarization_callback)
                    speaker_segments = self.speaker_diarizer.diarize_speakers(filepath, progress_callback=scoped_diarization_callback)
                    
                    # Offload diarizer to free memory for next Whisper run or general system use
                    self._clear_speaker_diarizer()
                    
                    print(f"Found {len(speaker_segments)} speaker segments")
                except Exception as e:
                    print(f"Speaker diarization failed: {str(e)}")
                    print("Continuing with transcript only...")
                    speaker_segments = None
            else:
                # If diarization is disabled, we just jump to 90%
                if progress_callback:
                    progress_callback(90)

            ok, err = self._save_transcript_outputs(
                filepath,
                transcript,
                compact_result if self.enable_speaker_diarization else result,
                speaker_segments,
                False if self.enable_speaker_diarization else word_timestamps,
            )
            if not ok:
                raise OSError(err)

            if progress_callback:
                progress_callback(100)

            output_txt = os.path.join(
                self.output_transcripts_folder,
                os.path.splitext(os.path.basename(filepath))[0] + ".txt"
            )
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

    def _format_plain_transcript(self, transcript):
        """
        Format a non-diarized transcript into readable line-broken segments.

        Args:
            transcript (str): Raw transcript from Whisper

        Returns:
            str: Line-broken transcript using Whisper's chunk timing when available
        """
        try:
            whisper_segments = self._extract_whisper_segments(transcript)
        except Exception:
            whisper_segments = []

        if whisper_segments:
            lines = []
            for seg in whisper_segments:
                text = str(seg.get('text') or '').strip()
                if text:
                    lines.append(text)
            if lines:
                return "\n".join(lines)

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

    def _save_transcript_outputs(self, audio_file, transcript, whisper_result, speaker_segments, word_timestamps):
        """Write the .txt transcript (+ .srt sidecar, + _words.json when requested) for one file.

        Shared by the diarization-on (deferred) and diarization-off (streamed) save paths
        so both produce identical output. Returns (saved_bool, error_msg_or_None).
        """
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_txt = os.path.join(self.output_transcripts_folder, f"{base_name}.txt")

        # _extract_whisper_segments (used by both formatters and the SRT writer) reads
        # self.whisper_result, so set it for this file before formatting.
        self.whisper_result = whisper_result
        if speaker_segments and len(speaker_segments) > 0:
            formatted_transcript = self._format_transcript_with_speakers(transcript, speaker_segments)
        else:
            formatted_transcript = self._format_plain_transcript(transcript)

        try:
            with open(output_txt, 'w', encoding='utf-8', newline='\n') as f:
                f.write(formatted_transcript)
            print(f"Saved transcript: {output_txt}")
            self._write_srt_sidecar(audio_file, transcript, whisper_result, speaker_segments)
        except OSError as e:
            print(f"Error saving transcript for {audio_file}: {e}")
            return False, f"{output_txt}: {e}"

        # When word-level timestamps were requested, write the JSON sidecar that
        # Align Features consumes (so it won't have to re-transcribe this file).
        if word_timestamps:
            self._write_word_json_sidecar(audio_file, whisper_result)

        return True, None

    def _extract_transcripts_batch_with_diarization(self, audio_files, progress_callback=None, word_timestamps=False, cancel_check=None):
        """Memory-bounded diarized batch path: transcribe, clear Whisper, diarize, save per file."""
        total_files = len(audio_files)
        os.makedirs(self.output_transcripts_folder, exist_ok=True)
        save_errors = []
        transcription_errors = []
        saved_count = 0

        for i, audio_file in enumerate(audio_files):
            if cancel_check and cancel_check():
                return

            file_start = (i / total_files) * 100
            file_span = 100 / total_files

            def file_progress(local_progress):
                if progress_callback:
                    progress_callback(int(file_start + (local_progress / 100.0 * file_span)))

            transcript = None
            compact_result = None
            audio_path = os.path.normpath(os.path.abspath(audio_file))

            try:
                self.set_status_message(f"🗣️ Transcribing ({i+1}/{total_files}): {os.path.basename(audio_file)}")
                file_progress(0)
                self._load_whisper_model()
                file_progress(10)

                if not os.path.isfile(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")

                ts_mode = "word" if word_timestamps else True
                result = self.whisper_pipe(
                    _whisper_pipeline_audio_input(audio_path),
                    return_timestamps=ts_mode,
                    generate_kwargs={"task": "transcribe"}
                )
                transcript = result['text']
                compact_result = self._compact_whisper_result(result, transcript)
                if word_timestamps:
                    self._write_word_json_sidecar(audio_file, result)

                del result
                self._clear_whisper_model()
                file_progress(50)

                self.set_status_message(f"🎭 Diarizing ({i+1}/{total_files}): {os.path.basename(audio_file)}")
                speaker_segments = None
                try:
                    self._load_speaker_diarizer()
                    speaker_segments = self.speaker_diarizer.diarize_speakers(audio_path)
                except Exception as e:
                    print(f"Error diarizing {audio_file}: {e}")
                finally:
                    self._clear_speaker_diarizer()
                file_progress(90)

                ok, err = self._save_transcript_outputs(
                    audio_file, transcript, compact_result, speaker_segments, False
                )
                if ok:
                    saved_count += 1
                elif err:
                    save_errors.append(err)
                file_progress(100)

            except Exception as e:
                print(f"Error transcribing {audio_file}: {e}")
                transcription_errors.append((audio_file, str(e)))
            finally:
                transcript = None
                compact_result = None
                self._clear_whisper_model()
                self._clear_torch_cache()

        if saved_count == 0:
            detail_bits = [
                f"{os.path.basename(path)}: {err}"
                for path, err in transcription_errors[:8]
            ]
            detail = "\n".join(detail_bits) if detail_bits else "(see console / log for details)"
            raise RuntimeError(
                "Speech recognition failed for every file; no transcripts were produced.\n\n" + detail
            )

        if save_errors:
            raise RuntimeError(
                "Could not write one or more transcript files:\n\n" + "\n".join(save_errors)
            )

        if progress_callback:
            progress_callback(100)

    def extract_transcripts_batch(self, audio_files, progress_callback=None, word_timestamps=False, cancel_check=None):
        """
        Batch process multiple audio files to generate transcripts.

        OPTIMIZED: Loads each model once for all files instead of per-file,
        significantly reducing processing time for batches.

        Args:
            audio_files (list): List of paths to WAV files
            progress_callback (callable, optional): Callback function for progress updates
            word_timestamps (bool): When True, request word-level timestamps and write a
                ``{base}_words.json`` sidecar per file so Align Features can reuse it instead
                of re-transcribing later.
            cancel_check (callable, optional): Return True to stop before the next file
        """
        if not audio_files:
            return

        if self.output_transcripts_folder is None:
            raise ValueError(
                "Transcripts output folder is not configured. "
                "The transcript output folder is not configured (<dataset>/transcripts). Select your dataset folder again."
            )

        if self.enable_speaker_diarization:
            return self._extract_transcripts_batch_with_diarization(
                audio_files, progress_callback, word_timestamps, cancel_check
            )

        total_files = len(audio_files)
        
        # Diarization-off path keeps Whisper loaded once for speed and saves each
        # file immediately so batch memory stays flat.
        transcription_start = 10
        transcription_end = 95
        
        # ===== PHASE 1: Load Whisper model ONCE =====
        self.set_status_message("🔄 Loading speech recognition model...")
        if progress_callback:
            progress_callback(0)
        
        try:
            self._load_whisper_model()
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            raise
        
        if progress_callback:
            progress_callback(transcription_start)
        
        # ===== PHASE 2: Transcribe files and save immediately =====
        transcription_range = transcription_end - transcription_start

        os.makedirs(self.output_transcripts_folder, exist_ok=True)
        save_errors = []
        saved_count = 0
        transcribed_ok = 0

        transcription_errors = []
        for i, audio_file in enumerate(audio_files):
            if cancel_check and cancel_check():
                return
            self.set_status_message(f"🗣️ Transcribing ({i+1}/{total_files}): {os.path.basename(audio_file)}")

            transcript, result = None, None
            try:
                audio_path = os.path.normpath(os.path.abspath(audio_file))
                if not os.path.isfile(audio_path):
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                # Transcribe without loading/offloading model.
                # word-level timestamps (for Align Features) when requested, else segment-level.
                ts_mode = "word" if word_timestamps else True
                result = self.whisper_pipe(
                    _whisper_pipeline_audio_input(audio_path),
                    return_timestamps=ts_mode,
                    generate_kwargs={"task": "transcribe"}
                )
                transcript = result['text']
            except Exception as e:
                print(f"Error transcribing {audio_file}: {e}")
                transcription_errors.append((audio_file, str(e)))

            if transcript is not None:
                transcribed_ok += 1
                ok, err = self._save_transcript_outputs(
                    audio_file, transcript, result, None, word_timestamps
                )
                if ok:
                    saved_count += 1
                elif err:
                    save_errors.append(err)

            # Update progress
            if progress_callback:
                file_progress = ((i + 1) / total_files) * transcription_range
                progress_callback(int(transcription_start + file_progress))

        if transcribed_ok == 0:
            detail_bits = [
                f"{os.path.basename(path)}: {err}"
                for path, err in transcription_errors[:8]
            ]
            detail = "\n".join(detail_bits) if detail_bits else "(see console / log for details)"
            raise RuntimeError(
                "Speech recognition failed for every file; no transcripts were produced.\n\n" + detail
            )

        if save_errors:
            raise RuntimeError(
                "Could not write one or more transcript files:\n\n" + "\n".join(save_errors)
            )

        if transcribed_ok > 0 and saved_count == 0:
            raise RuntimeError(
                "Transcription succeeded but no .txt files were saved (unexpected). "
                f"Output folder: {self.output_transcripts_folder}"
            )

        if progress_callback:
            progress_callback(100)

    def align_features(self, features_csv, transcript_json, output_csv):
        """
        Align audio features with word-level transcript.
        
        Args:
            features_csv (str): Path to audio features CSV
            transcript_json (str): Path to word-level transcript JSON
            output_csv (str): Path to save aligned CSV
        """
        import pandas as pd
        import json

        # Load features
        try:
            df_features = pd.read_csv(features_csv)
            if 'Timestamp_Seconds' not in df_features.columns:
                raise ValueError("Features CSV missing 'Timestamp_Seconds' column")
        except Exception as e:
            raise Exception(f"Failed to load features CSV: {e}")

        # Load transcript words
        try:
            with open(transcript_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            words = []
            # Parse words from Whisper pipeline output
            # Structure depends on return_timestamps="word"
            # Usually result['chunks'] contains words directly or segments with words
            
            chunks = data.get('chunks', [])
            if not chunks and 'segments' in data:
                chunks = data['segments']
                
            for chunk in chunks:
                # If it's a word-level chunk
                text = chunk.get('text', '').strip()
                timestamp = chunk.get('timestamp')
                
                start = None
                end = None
                
                if isinstance(timestamp, (list, tuple)) and len(timestamp) >= 2:
                    start, end = timestamp
                else:
                    # Try dict format
                    ts = chunk.get('timestamps')
                    if isinstance(ts, dict):
                        start = ts.get('start')
                        end = ts.get('end')
                
                if start is not None and end is not None and text:
                    words.append({
                        'word': text,
                        'start': float(start),
                        'end': float(end)
                    })
                    
        except Exception as e:
            raise Exception(f"Failed to parse transcript JSON: {e}")

        if not words:
            print("Warning: No words found in transcript JSON for alignment.")
            return

        # Align
        aligned_rows = []
        feature_cols = [c for c in df_features.columns if c not in ['Timestamp_Seconds', 'Timestamp_Milliseconds', 'Timestamp_Formatted', 'name', 'frameTime']]
        
        for w in words:
            w_start = w['start']
            w_end = w['end']
            
            # Filter features within this time range
            mask = (df_features['Timestamp_Seconds'] >= w_start) & (df_features['Timestamp_Seconds'] <= w_end)
            subset = df_features.loc[mask, feature_cols]
            
            if subset.empty:
                # If word is too short or falls between frames, take nearest frame
                # Find index of nearest timestamp
                nearest_idx = (df_features['Timestamp_Seconds'] - w_start).abs().idxmin()
                subset = df_features.loc[[nearest_idx], feature_cols]
            
            # Compute mean of features
            means = subset.mean()
            
            row = {
                'word': w['word'],
                'start_time': w_start,
                'end_time': w_end,
                'duration': w_end - w_start
            }
            # Add feature means
            for col, val in means.items():
                row[col] = val
                
            aligned_rows.append(row)

        # Save to CSV
        df_aligned = pd.DataFrame(aligned_rows)
        df_aligned.to_csv(output_csv, index=False)
        print(f"Saved aligned features: {output_csv}")

    def align_features_batch(self, alignment_pairs, progress_callback=None):
        """
        Batch align features.
        
        Args:
            alignment_pairs (list): List of tuples (features_csv, transcript_json, output_csv)
            progress_callback (callable): Progress callback
        """
        total = len(alignment_pairs)
        for i, (feat_csv, trans_json, out_csv) in enumerate(alignment_pairs):
            try:
                self.set_status_message(f"🔗 Aligning: {os.path.basename(out_csv)}")
                self.align_features(feat_csv, trans_json, out_csv)
            except Exception as e:
                print(f"Error aligning {os.path.basename(out_csv)}: {e}")
            
            if progress_callback:
                progress_callback(int((i + 1) / total * 100))


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
                if self.device != "cpu":
                    self.diarization_pipeline.to(torch.device(self.device))
            except Exception:
                pass
            return
            
        try:
            if self.progress_callback:
                self.progress_callback(5)

            if not self.auth_token:
                raise Exception(
                    "A Hugging Face token is required for speaker diarization."
                )

            kw = {"use_auth_token": self.auth_token}

            try:
                from pyannote.audio import Pipeline
            except ModuleNotFoundError as e:
                raise Exception(
                    "Optional diarization support is not installed. Install the Complete toolbox profile first."
                ) from e
            except Exception as e:
                raise Exception(
                    f"Failed to import pyannote.audio dependencies: {e}"
                ) from e

            try:
                print("Loading PyAnnote speaker diarization pipeline...")
                self.diarization_pipeline = Pipeline.from_pretrained(
                    DIARIZATION_MODEL_ID,
                    **kw
                )
            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"PyAnnote loading failed with trace:\n{error_trace}")
                raise Exception(
                    f"Could not load the PyAnnote pipeline. Check that your Hugging Face token is valid, "
                    f"that you accepted the pyannote model terms, and that the app can reach Hugging Face. "
                    f"Original error: {repr(e)}"
                ) from e
            
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
