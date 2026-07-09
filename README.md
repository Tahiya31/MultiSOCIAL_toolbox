# MultiSOCIAL_toolbox

Toolbox for multimodal interaction analysis using video, audio, and transcript information.

MultiSOCIAL Toolbox helps you process recordings of human interaction. It can extract body pose, create pose-overlay videos, convert video to audio, extract speech features, transcribe audio, optionally add speaker labels, align words with audio features, and burn transcript captions onto video.

Current release/runtime baseline:
- Python `3.10.x` for source installs
- `Standard` profile: base toolbox
- `Complete` profile: base toolbox plus optional speaker diarization support

## Quick links

- [Installation](#installation)
- [Which version should I choose?](#which-version-should-i-choose)
- [Usage](#usage)
- [Video files](#video-files)
- [Audio files](#audio-files)
- [Troubleshooting](#troubleshooting)

## Installation

There are two ways to use MultiSOCIAL Toolbox:

1. **Packaged desktop release**  
   Recommended for most users. You do not need to install Python manually.
2. **Source install using the launcher scripts**  
   Mainly for contributors or users running directly from the repository. You must have Python `3.10.x` installed first.

## Option 1: Packaged desktop release

Packaged builds are attached to the repo's **GitHub Releases** page.

1. Open the repository on GitHub.
2. Click **Releases** on the right side of the repo page.
3. Open the latest release.
4. Download the build for your system:
   - macOS:
     - `MultiSOCIAL-Standard-macOS.zip`
     - `MultiSOCIAL-Complete-macOS.zip`
   - Windows:
     - `MultiSOCIAL-Standard-windows.zip`
     - `MultiSOCIAL-Complete-windows.zip`

### macOS packaged release steps

1. Download the `.zip` file from **Releases**.
2. Double-click the `.zip` to extract it.
3. Open **Terminal**.
4. Remove the quarantine flag from the extracted app:
   - `xattr -dr com.apple.quarantine "/path/to/MultiSOCIAL-Standard.app"`
   - or
   - `xattr -dr com.apple.quarantine "/path/to/MultiSOCIAL-Complete.app"`
5. Open the app:
   - double-click the `.app`, or
   - right-click the `.app` and choose **Open**

If a future macOS release is signed and notarized, the quarantine-removal step may not be needed.

### Windows packaged release steps

1. Download the `.zip` file from **Releases**.
2. Extract the `.zip`.
3. Open the extracted folder.
4. Launch the app:
   - `MultiSOCIAL-Standard.exe`, or
   - `MultiSOCIAL-Complete.exe`

If Windows shows a trust warning on an unsigned build:

1. Click **More info**.
2. Click **Run anyway**.

If a future Windows release is signed, this warning should be reduced or removed.

## Option 2: Source install from the repository

This path uses the launcher scripts in the repo and requires Python `3.10.x`.

Do not use Python `3.11` or `3.12` for this project.

### macOS source install

1. Download or clone the repository.
2. Open **Terminal**.
3. Go to the repository folder:
   - `cd /path/to/MultiSOCIAL_toolbox`
4. Run:
   - `bash run_app.sh`
5. Choose one profile when prompted:
   - `Standard`
   - `Complete`

The script will create or reuse `.venv`, install the correct dependencies, and launch the app.

### Windows source install

1. Download or clone the repository.
2. Install Python `3.10.x`.
3. Open **Command Prompt**.
4. Go to the repository folder:
   - `cd \path\to\MultiSOCIAL_toolbox`
5. Run:
   - `run_app.bat`
6. Choose one profile when prompted:
   - `Standard`
   - `Complete`

The script will create or reuse `.venv`, install the correct dependencies, and launch the app.

## Which version should I choose?

- Choose `Standard` if you want the main toolbox features without speaker diarization.
- Choose `Complete` if you also want speaker diarization support.

If you are unsure, start with `Standard`.

## Usage

Once launched, the MultiSOCIAL Toolbox application looks like this.

<img src="./assets/ApplicationUI.png" width="350">

The toolbox takes two types of input:

- Audio: `.wav`, `.wave`, `.aiff`, `.aif`, `.aifc`, `.flac`, `.caf`, `.au`, `.snd`
- Video: `.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v`

Select a folder with your files, then run the steps you need. The toolbox creates output folders inside the selected folder.

## How the toolbox keeps your workflow safe

The toolbox runs one task at a time and guides you through the right order of steps.

- **One task at a time.** While a task is running, every other button and setting is disabled and only **Cancel** stays active.
- **Steps unlock when their inputs exist.** Buttons that depend on earlier results stay greyed out until the required files are present on disk. Hover over a greyed-out button to see what is missing.
- **Clear messages.** If a step cannot run, a plain-language message tells you what is missing, what to do next, and which folder the required files belong in.

Specifically:

- **Embed Pose Features** unlocks once **Extract Pose Features** has produced pose CSVs in `pose_features` for the selected mode: single-person `*_ID_*.csv` or multi-person `*_multi_ID_*.csv`.
- **Embed Transcript on Video** unlocks once **Extract Transcripts** has produced matching `.srt` files in `transcripts` for every selected video.
- If **Add captions to pose-embedded video** is checked, **Embed Transcript on Video** also waits for matching videos in `embedded_pose`.
- **Verify Pose Match** unlocks once **Embed Pose Features** has produced videos in `embedded_pose`.

## Video files

### Convert video to audio

Use this step if you have supported video files and want `.wav` audio files.

1. Use **Browse** to select the folder with your video files.
2. Press **Convert video to audio**.
3. When the WAV files are ready, the toolbox shows a confirmation message.

The output is saved in `converted_audio`.

### Extract Pose Features

Use this step to extract body key-points from video. The toolbox uses [MediaPipe Pose](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md) and returns 33 body pose landmarks.

1. Use **Browse** to select the folder with your video files.
2. Choose pose options if needed.
3. Press **Extract Pose Features**.

Important options:

- **Enable Multi-person Pose**: use this when your video has multiple people and you want pose information for each person. Keep this checkbox in the same state for **Extract Pose Features**, **Embed Pose Features**, and **Verify Pose Match** so the toolbox uses the matching CSV set.
- **Process every k-th frame**: skip frames to speed up processing. For example, `k=2` processes every other frame. The progress bar and status line follow source video frames, so they still reach the end of the clip when stride is greater than 1.
- **Downscale to 720p**: reduce video resolution during processing for faster extraction.
- **Frame Threshold for Bounding Box Recalibration**: controls how often person bounding boxes are re-detected in multi-person mode.

Outputs:

- Single-person mode creates files such as `session01_ID_0.csv`.
- Multi-person mode creates one CSV per tracked person, such as `session01_multi_ID_0.csv`, `session01_multi_ID_1.csv`, and so on.
- A metadata sidecar records extraction settings such as frame stride.
- When processing a folder of videos, the toolbox continues through the batch and shows a warning listing any files that failed or produced no CSVs.

CSV format:

- Each row represents a processed frame.
- Each row includes the frame number and person ID.
- For each of the 33 body landmarks, the CSV includes:
  - `x` and `y`: landmark coordinates normalized to `[0.0, 1.0]` by image width and height.
  - `z`: landmark depth, with the midpoint of the hips as the origin. Smaller values are closer to the camera.
  - `confidence`: value in `[0.0, 1.0]` indicating how likely the landmark is visible.

### Embed Pose Features

Use this step after **Extract Pose Features** to draw pose landmarks onto the original video.

1. Use **Browse** to select the same folder as your video files.
2. Keep **Enable Multi-person Pose** set to the same mode used during extraction.
3. Press **Embed Pose Features**.

The toolbox reads pose CSVs from `pose_features` and writes pose-overlay videos to `embedded_pose`.

Details:

- The button stays greyed out until pose CSVs exist for every selected video in the current mode.
- Each tracked person gets a stable color with an on-screen legend.
- Landmark brightness reflects detection confidence.
- Embed automatically reuses the frame stride recorded during extraction, so the overlay stays aligned with the CSV rows.
- Output naming follows the mode: `*_pose.mp4` for single-person CSVs and `*_multi_pose.mp4` for multi-person CSVs.
- When embedding a folder of videos, the toolbox continues through the batch and shows a warning listing any files that were skipped or failed.

### Embed Transcript on Video

Use this step after **Extract Transcripts** to burn transcript captions onto video.

1. Run **Extract Transcripts** on the Audio tab. This saves `.txt` and `.srt` files in `transcripts`.
2. Use **Browse** on the Video tab to select the same folder as your video files.
3. Press **Embed Transcript on Video**.

The captioned video is written to `captioned_video` as `<name>_captioned.mp4`. The original video is left untouched.

Optional:

- Check **Add captions to pose-embedded video** to burn captions onto the pose-overlay video instead of the raw source.
- Pose-captioned output is written as `*_pose_captioned.mp4`.
- If both single-person and multi-person pose-overlay videos exist, the newest embedded pose video is used.

If the button stays greyed out, rerun **Extract Transcripts** and make sure each audio file has the same base name as its video. For example, `session01.wav` produces captions for `session01.mp4`.

### Verify Pose Match

After extracting pose CSVs and generating embedded pose videos, use **Verify Pose Match** to compare them.

Pre-requisites:

- `pose_features` must contain CSVs generated from **Extract Pose Features** for the same mode as the embedded video.
- `embedded_pose` must contain pose-overlay videos generated from **Embed Pose Features**.

Verification matches each embedded video to the correct CSV set and reads the extraction `frame_stride` from the matching metadata sidecar.

The tool creates a `verification` folder with:

- per-video JSON and CSV reports
- `worst_frames` for quick inspection
- `summary.json` aggregating all runs

Use this to spot pose drift or embedding issues before downstream analysis.

### Typical video outputs

You should see these folders within the same folder as your input video. Each folder is created only when the matching step runs.

- `converted_audio`: WAV files produced by **Convert video to audio**
- `pose_features`: CSV pose feature files produced by **Extract Pose Features**
- `embedded_pose`: rendered pose-overlay videos produced by **Embed Pose Features**
- `captioned_video`: captioned videos produced by **Embed Transcript on Video**

For a folder with several videos, the typical output looks like this:

```text
YourVideoFolder/
  session01.mp4
  session02.mp4
  converted_audio/
    session01.wav
    session02.wav
  pose_features/
    session01_ID_0.csv
    session01_meta.json
    session02_ID_0.csv
    session02_meta.json
  embedded_pose/
    session01_pose.mp4
    session02_pose.mp4
  transcripts/
    session01.txt
    session01.srt
    session01_words.json
    session02.txt
    session02.srt
    session02_words.json
  audio_features/
    session01.csv
    session01_aligned.csv
    session02.csv
    session02_aligned.csv
  captioned_video/
    session01_captioned.mp4
    session01_pose_captioned.mp4
    session02_captioned.mp4
    session02_pose_captioned.mp4
  verification/
    summary.json
    session01_pose_report.json
    session01_pose_worst.csv
    worst_frames/
```

If **Enable Multi-person Pose** is used, pose CSVs and embedded pose videos include `_multi`, for example `session01_multi_ID_0.csv` and `session01_multi_pose.mp4`.

## Audio files

### Extract Audio Features

Use this step to extract speech features from audio. The toolbox uses [OpenSMILE](https://audeering.github.io/opensmile-python/) with the ComParE 2016 feature set.

1. Use **Browse** to select the folder with your audio files. You can select audio in `converted_audio` as well.
2. Press **Extract Audio Features**.
3. When the features are ready, the toolbox shows a confirmation message.

Feature CSV format:

- Each row represents a frame/sample.
- Each file includes three timestamp columns: `Timestamp_Seconds`, `Timestamp_Milliseconds`, and `Timestamp_Formatted`.
- The timestamp columns are followed by the 65 ComParE 2016 feature columns.

### Extract Transcripts

Use this step to transcribe audio. The toolbox uses [Whisper Large V3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) for automatic speech recognition, with GPU/MPS acceleration when available.

1. Use **Browse** to select the folder with your audio files.
2. Press **Extract Transcripts**.
3. Each file is saved to `transcripts` as soon as it finishes, so you can inspect partial batch results while the rest are still processing.

Outputs:

- `.txt`: readable transcript
- `.srt`: caption file used by **Embed Transcript on Video**
- `_words.json`: word-level timestamps, created when word timestamps are requested or when **Align Features** needs them

### Optional: Speaker diarization with PyAnnote

Speaker diarization labels who is speaking in the transcript. In the GUI, enable the speaker diarization checkbox before running **Extract Transcripts**.

- If diarization is not installed yet and your build supports self-install, click **Install Complete Toolbox** in the audio panel.
- Source installs can also select the `Complete` profile from `run_app.sh` or `run_app.bat`.

If the checkbox is off, only Whisper runs and a plain transcript is saved.

If the checkbox is on, Whisper runs and PyAnnote adds speaker labels. The first time, you will be prompted for a Hugging Face access token because PyAnnote models require one. Diarized output is saved directly in the transcript file with inline speaker segments, for example:

```text
SPEAKER_00: [00:00.000 - 00:07.000] What about things that you're afraid of? ...
SPEAKER_01: [00:11.000 - 00:19.000] It's a hard question. I'm not really scared of anything...
```

How to get and use the Hugging Face token:

1. Create or sign in to a Hugging Face account: `https://huggingface.co`
2. Accept the model licenses:
   - `https://huggingface.co/pyannote/speaker-diarization`
   - `https://huggingface.co/pyannote/segmentation`
3. Create an access token: go to `https://huggingface.co/settings/tokens`, choose **New token**, set scope to **Read**, and copy the token.
4. In MultiSOCIAL Toolbox, paste the token when prompted and confirm.

Notes:

- The app stores the token in local app settings for future runs.
- You can still set `HF_TOKEN` in your environment or a local `.env` file if you prefer.
- If you cancel or the token is invalid, the app continues with transcript only.
- The first diarization run may download models and can take a few minutes.
- Once models are downloaded, they are cached locally for later use.
- Speaker labels are heuristic, such as `SPEAKER_00` and `SPEAKER_01`; they do not identify real names.

### Align Features

This step aligns extracted audio features with word-level transcripts. It is useful for analyzing acoustic features of specific words.

1. Run **Extract Audio Features** first to generate `.csv` feature files.
2. Press **Align Features**.

The toolbox:

- creates or reuses a detailed word-level transcript (`_words.json`)
- merges audio features with each word based on timestamps
- saves the result as `_aligned.csv` in `audio_features`

Features are averaged over the duration of each word.

### Typical audio outputs

For a folder with several audio files, the typical output looks like this:

```text
YourAudioFolder/
  session01.wav
  session02.wav
  audio_features/
    session01.csv
    session01_aligned.csv
    session02.csv
    session02_aligned.csv
  transcripts/
    session01.txt
    session01.srt
    session01_words.json
    session02.txt
    session02.srt
    session02_words.json
```

## Troubleshooting

### A button is greyed out

Greyed-out buttons usually mean a required file is missing. Hover over the button for the exact next step.

- **Embed Pose Features** needs matching pose CSVs in `pose_features` for the current **Enable Multi-person Pose** setting.
- **Embed Transcript on Video** needs matching `.srt` files in `transcripts`.
- If **Add captions to pose-embedded video** is checked, **Embed Transcript on Video** also needs matching videos in `embedded_pose`.
- **Verify Pose Match** needs pose-overlay videos in `embedded_pose` plus CSVs that match each video's mode: `*_pose.mp4` with `*_ID_*.csv`, or `*_multi_pose.mp4` with `*_multi_ID_*.csv`.

### Multi-person pose looks wrong or buttons stay locked

Make sure **Enable Multi-person Pose** matches how you extracted pose data. Single-person CSVs such as `session_ID_0.csv` and multi-person CSVs such as `session_multi_ID_0.csv` are treated separately. Toggling the checkbox after extraction will not unlock **Embed Pose Features** until the matching files exist.

If **Enable Multi-person Pose** fails in a packaged build with a message about missing YOLOv5 weights, reinstall from the latest Release build. Multi-person detection uses `yolov5s.pt` bundled inside the app and does not download weights at runtime.

### Captions are missing or the caption button stays locked

Run **Extract Transcripts** for audio files with the same base name as the videos. For example:

```text
session01.mp4
converted_audio/session01.wav
transcripts/session01.srt
```

If the `.srt` file is missing, rerun **Extract Transcripts**. If you want captions on the pose-overlay video, run **Embed Pose Features** first.

### Speaker diarization is slow or uses a lot of memory

Speaker diarization is optional and can be slower than plain transcription, especially on long files.

- Turn off **Enable speaker diarization** if you only need captions or a plain transcript.
- Diarization is only needed when you want speaker labels such as `SPEAKER_00` and `SPEAKER_01`.
- Word timestamps are not required for captions; they are mainly used by **Align Features**.
- On macOS, the app runs diarization on CPU to avoid large memory spikes.

### FFmpeg or video engine errors

The packaged app and current source setup try a bundled FFmpeg fallback automatically, so most users should not need to install FFmpeg manually.

- **Embed Transcript on Video** needs an FFmpeg build with the `subtitles` filter. The app checks your system FFmpeg first and falls back to the bundled build when the system copy cannot burn captions.
- If **Convert video to audio**, **Embed Transcript on Video**, or caption burning reports a video-engine error, relaunch the app once so it can re-check the bundled FFmpeg.
- If the bundled fallback is unavailable on your machine, install FFmpeg and add it to `PATH`.
- On Windows, you can start from [ffmpeg.org](https://ffmpeg.org/download.html) and these [PATH setup steps](https://phoenixnap.com/kb/ffmpeg-windows).

### Warnings in the terminal

Some libraries print warnings about model loading, package paths, or cache folders. These warnings are usually safe to ignore if the requested output files are created.

- For packaged releases, only investigate warnings if a feature fails.
- For source installs, rerun `run_app.sh` or `run_app.bat` if a tool appears to be missing.

### Pose output looks wrong

Run **Verify Pose Match** after **Embed Pose Features**. It creates a `verification` folder with a report and worst-frame thumbnails so you can quickly inspect whether the CSV and embedded video match.

If a multi-person video looks like it used the wrong overlay, remember that **Add captions to pose-embedded video** uses the newest pose-overlay video when both single-person and multi-person versions exist.

## Acknowledgement

We thank the authors and developers of `MediaPipe`, `OpenSMILE`, `YOLOv5`, and `Whisper` for their open-source contributions used in MultiSOCIAL Toolbox.

## Team

MultiSOCIAL Toolbox is developed by Tahiya Chowdhury, Veronica Romero, Alexandra Paxton, and Muneeb Nafees.

## Disclaimer

Automated tools can be inaccurate and should be used after human verification for correctness.

## Help us improve this toolbox!

[Please leave your feedback in this form.](https://docs.google.com/forms/d/e/1FAIpQLScGkEu-LfLAa_IGNOXG25trtMf8k12FFPymObBRDmLdPkAvxQ/viewform)
