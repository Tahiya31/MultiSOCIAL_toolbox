# MultiSOCIAL_toolbox
Toolbox for multimodal interaction analysis for text, audio, and video information.

Current release/runtime baseline:
- Python `3.10.x`
- `Standard` profile: base toolbox
- `Complete` profile: base toolbox plus optional speaker diarization support

## How can I use MultiSOCIAL Toolbox?

The toolbox allows you to process audio and video files of conversation.

# Installation

There are two ways to use MultiSOCIAL Toolbox:

1. **Packaged desktop release**
   This is the recommended option for most users. You do **not** need to install Python manually.
2. **Source install using the launcher scripts**
   This is mainly for contributors or users running directly from the repository. You **must** have Python `3.10.x` installed first.

## Option 1: Packaged desktop releases

Packaged builds are attached to the repo's **GitHub Releases** page.

- `Standard` build: base toolbox
- `Complete` build: base toolbox plus diarization support

Official user downloads should come from **Releases**.
The GitHub Actions **Build Releases** workflow also produces artifacts, but those are mainly useful for branch/fork testing before an official release is published.
Workflow artifacts now download as a single zip that extracts directly into one app folder. For Windows, the `.exe` is at the top level of that extracted folder.

### Where to find them

1. Open the repository on GitHub.
2. Click **Releases** on the right side of the repo page.
3. Open the latest release.
4. Download the artifact for your system:
   * macOS:
     * `MultiSOCIAL-Standard-macos.zip`
     * `MultiSOCIAL-Complete-macos.zip`
   * Windows:
     * `MultiSOCIAL-Standard-windows.zip`
     * `MultiSOCIAL-Complete-windows.zip`

### Branch or fork testing artifacts

1. Open the **Actions** tab on GitHub.
2. Open the relevant **Build Releases** workflow run.
3. Download the artifact for your system.
4. Extract the downloaded zip once.
5. Open the extracted app folder:
   * Windows: the `.exe` is directly inside that folder
   * macOS: the `.app` is directly inside that folder

### macOS packaged release steps

1. Download the `.zip` file from **Releases**.
2. Double-click the `.zip` to extract it.
3. Open **Terminal**.
4. Remove the quarantine flag from the extracted app:
   * `xattr -dr com.apple.quarantine "/path/to/MultiSOCIAL-Standard.app"`
   * or
   * `xattr -dr com.apple.quarantine "/path/to/MultiSOCIAL-Complete.app"`
5. Open the app:
   * double-click the `.app`, or
   * right-click the `.app` and choose **Open**

If your macOS build is signed and notarized in a future release, the quarantine-removal step may not be needed.

### Windows packaged release steps

1. Download the `.zip` file from **Releases**.
2. Extract the `.zip`.
3. Open the extracted folder.
4. Launch the app:
   * `MultiSOCIAL-Standard.exe`, or
   * `MultiSOCIAL-Complete.exe`

If Windows shows a trust warning on an unsigned build:
1. Click **More info**
2. Click **Run anyway**

If the build is signed in a future release, this warning should be reduced or removed.

## Option 2: Source install from the repository

This path uses the launcher scripts in the repo and requires Python `3.10.x`.

### Prerequisite

- Install Python `3.10.x`

Do not use Python `3.11` or `3.12` for this project.

### macOS source install

1. Download or clone the repository.
2. Open **Terminal**.
3. Go to the repository folder:
   * `cd /path/to/MultiSOCIAL_toolbox`
4. Run:
   * `bash run_app.sh`
5. Choose one profile when prompted:
   * `Standard`
   * `Complete`

The script will:
- create or reuse `.venv`
- install the correct dependencies
- launch the app

### Windows source install

1. Download or clone the repository.
2. Install Python `3.10.x`.
3. Open **Command Prompt**.
4. Go to the repository folder:
   * `cd \path\to\MultiSOCIAL_toolbox`
5. Run:
   * `run_app.bat`
6. Choose one profile when prompted:
   * `Standard`
   * `Complete`

The script will:
- create or reuse `.venv`
- install the correct dependencies
- launch the app

## Which version should I choose?

- Choose `Standard` if you want the main toolbox features without diarization.
- Choose `Complete` if you also want speaker diarization support.

If you are unsure, start with `Standard`.


# Usage
Once launched, MultiSOCIAL Toolbox application looks like this.

<img src="./assets/ApplicationUI.png" width="350">

The toolbox takes two types of input: audio (`.wav`, `.wave`, `.aiff`, `.aif`, `.aifc`, `.flac`, `.caf`, `.au`, `.snd`) and video (`.mp4`, `.avi`, `.mov`, `.mkv`, `.m4v`).

## How the toolbox keeps your workflow safe
The toolbox runs one task at a time and guides you through the right order of steps.

* **One task at a time.** While a task is running, every other button and setting is disabled and only **Cancel** stays active. This prevents two tasks from running at once or interfering with each other, and means rapid or accidental clicks can't start a second job.
* **Steps unlock when their inputs exist.** Buttons that depend on earlier results stay greyed out until the required files are actually present on disk (the toolbox checks the files, not just whether you clicked a button before). Hover over a greyed-out button to see what's missing and what to do next. Specifically:
  * **Embed Pose Features** unlocks once **Extract Pose Features** has produced pose CSVs in the `pose_features` folder.
  * **Embed Transcript on Video** unlocks once **Extract Transcripts** has produced matching `.srt` files in the `transcripts` folder for every selected video.
  * **Verify Consistency** unlocks once **Embed Pose Features** has produced videos in the `embedded_pose` folder.
* **Clear messages.** If a step can't run, a plain-language message tells you what is missing, what to do next, and which folder the required files belong in.

## Video file
**Convert video to audio** If you have a supported video file of human interaction and would like to convert it to an audio file in `.wav` format, this step is for you.
  * Use the ``Browse`` button to locate your input video file.
  * Then press **Convert video to audio** button.
  * Once the .wav file is ready, a dialogue box will let you know the output file is ready.

You should see these folders within the same folder as your input video (each is created only when the matching step runs).
  * `converted_audio`: Contains the WAV files produced by **Convert video to audio**.
  * `pose_features`: Contains the CSV pose feature files produced by **Extract Pose Features**.
  * `embedded_pose`: Contains the rendered pose-overlay videos produced by **Embed Pose Features**.
  * `captioned_video`: Contains the captioned videos produced by **Embed Transcript on Video**.


**Extract Pose Features** If you are interested in extracting pose or body key-points from the video, this step uses [MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md) to achieve this. This step returns 33 body pose land marks. For more details on MediaPipe, please check out the [official page](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md). 
  * Use the ``Browse`` button to locate your input video file.
  * Then press **Extract Pose Features** button. This step may take some time.
  * Note: If your video has multiple people, you must select **Enable Multi-person Pose** for getting pose information of each person.
  * **Performance Options:**
    * **Process every k-th frame**: Skip frames to speed up processing (e.g., k=2 processes every other frame).
    * **Downscale to 720p**: Reduce video resolution during processing for faster extraction.
    * **Frame Threshold for Bounding Box Recalibration**: Controls how often person bounding boxes are re-detected in multi-person mode.
  * Once the pose features are extracted, you can find them in **pose_features** folder created before.
  * **For multi-person mode**: Each output csv file will represent a single person's information (files will be named as {name of original video file}_multi_ID_0, {name of original video file}_multi_ID_1, etc.).
  * **CSV format** Each row represents a frame, each column represents features. For each of the 33 body land marks, you should see 4 columns:
    * x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
    * z: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
    * confidence: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.

**Embed Pose Features** Overlay body key-points from extracted pose CSVs onto each video frame (run **Extract Pose Features** first). The **Embed Pose Features** button stays greyed out until extraction has produced pose CSVs for your video; hover over it for a reminder.
  * Use the ``Browse`` button to locate your input video file.
  * Then press **Embed Pose Features** button. The toolbox reads pose CSVs from `pose_features/` and draws landmarks onto the video. Each tracked person gets a stable color with an on-screen legend, and each landmark's brightness reflects its detection confidence: brighter means more confident, dimmer means less confident.
  * Embed automatically reuses the frame stride recorded at extraction time, so the overlay stays aligned with the CSV rows regardless of the stride selected when embedding.
  * Once all the frames are processed, an output video will appear in the **embedded_pose** folder where your input video is located.

**Embed Transcript on Video** Burn the turn-by-turn transcript onto the video as captions so you can watch the video and read the transcript in sync — useful for evaluating transcription accuracy. The **Embed Transcript on Video** button stays greyed out until **Extract Transcripts** (on the Audio tab) has produced matching `.srt` files for the selected videos; hover over it for a reminder.
  * First run **Extract Transcripts** on the Audio tab. This now also saves a `.srt` caption file (one caption per spoken segment, prefixed with the speaker label when diarization is enabled) into the `transcripts` folder, alongside the existing `.txt` transcript.
  * Use the ``Browse`` button to select the same folder as your input video, then press **Embed Transcript on Video**.
  * The captioned video is written to the **captioned_video** folder as `<name>_captioned.mp4`. The audio is copied unchanged and the original video is left untouched; only a new captioned copy is created.
  * If the button stays greyed out, rerun **Extract Transcripts** and make sure each audio file has the same base name as its video (for example, `session01.wav` produces captions for `session01.mp4`).
  * Optional: check **Add captions to pose-embedded video** to burn captions onto the pose overlay instead of the raw source (`*_pose_captioned.mp4` in `captioned_video/`).

### Verify Consistency (Pose QA)
After extracting pose CSVs and generating embedded pose videos, you can run **Verify Consistency** to compare them.
  * Pre-requisites:
    * `pose_features/` must contain CSVs generated from **Extract Pose Features**.
    * `embedded_pose/` must contain pose-overlay videos generated from **Embed Pose Features**.
  * The tool creates a `verification/` folder in your dataset directory with:
    * per-video JSON + CSV reports (landmark hit rate against the embedded overlay)
    * `worst_frames/` subfolder containing reference frames for quick inspection
    * `summary.json` aggregating all runs
  * Use this to spot pose drift or embedding issues before downstream analysis.

## Audio file
**Extract Audio Features** If you are interested in extracting speech features from human speech during interaction, this step uses [OpenSMILE](https://audeering.github.io/opensmile-python/) to achieve this. This step currently uses predetermined feature sets (ComParE 2016) from OpenSMILE. For more details on OpenSMILE, please check their official [documentation page](https://audeering.github.io/opensmile-python/).
  * Use the ``Browse`` button to locate your supported input audio file. You can select audio located in **converted_audio** as well.
  * Then press **Extract Audio Features** button.
  * Once the audio features are extracted, a dialogue box will let you know the output file is ready.

  You should see two folders within your input folder (containing audio) now.
  * audio_features: This will contain all the csv files containing audio features from the **Extract Audio Features** option.
  * transcripts: This will contain all the .txt files containing transcriptions of the audio from the **Extract Transcripts** option
    
  *  **CSV format for Audio feature** Each file includes three timestamp columns (`Timestamp_Seconds`, `Timestamp_Milliseconds`, `Timestamp_Formatted`) followed by the 65 ComParE 2016 feature columns. Each row represents a frame/sample.

**Extract Transcripts** If you are interested in extracting transcript of the conversation, this step now uses [Whisper Large V3 Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) for automatic speech recognition (with GPU/MPS acceleration when available). For more details on Whisper, please check their official documentation page [here](https://github.com/openai/whisper).
  * Use the ``Browse`` button to locate your input audio file.
  * Then press **Extract Transcripts** button.
  * Each file is saved to the **transcripts** folder as soon as it finishes (`.txt`, plus `.srt` when segments are available), so you can inspect partial batch results while the rest are still processing.
  * Once the full batch completes, a dialogue box confirms success.
  * Alongside each `.txt` transcript, the toolbox also saves a `.srt` caption file in the same `transcripts` folder. This drives the **Embed Transcript on Video** step on the Video tab (see above).

### Optional: Speaker diarization with PyAnnote
You can optionally label who is speaking in the transcript (speaker diarization). In the GUI, enable the checkbox for speaker diarization before running **Extract Transcripts**.

- If diarization is not installed yet and your build supports self-install, click **Install Complete Toolbox** in the audio panel.
- Source installs can also select the `Complete` profile directly from `run_app.sh` or `run_app.bat`.

- **If the checkbox is OFF**: Only Whisper runs and a plain transcript is saved.
- **If the checkbox is ON**: Whisper runs and PyAnnote is used to add speaker labels. The first time, you will be prompted for a Hugging Face access token because PyAnnote models require one. The diarized output is saved directly in the transcript file with inline speaker segments, e.g.:

```
SPEAKER_00: [00:00.000 - 00:07.000] What about things that you're afraid of? ...
SPEAKER_01: [00:11.000 - 00:19.000] It's a hard question. I'm not really scared of anything...
```

How to get and use the Hugging Face token (one-time setup):
1. Create/sign in to a Hugging Face account: ``https://huggingface.co``
2. Accept the model licenses (both pages):
   * ``https://huggingface.co/pyannote/speaker-diarization``
   * ``https://huggingface.co/pyannote/segmentation``
3. Create an access token: go to ``https://huggingface.co/settings/tokens`` → **New token** (scope: "Read") → copy the token.
4. In MultiSOCIAL Toolbox, when prompted, paste the token and confirm. The app stores it in local app settings for future runs.
   * **Tip**: You can still set `HF_TOKEN` in your environment or a local `.env` file if you prefer.

Notes:
- If you cancel or the token is invalid, the app continues with transcript only (no diarization).
- The first diarization run may download models and can take a few minutes.
- **Offline Mode**: Once models are downloaded, they are cached locally. Subsequent runs will use the cached models, allowing offline usage.
- Speaker labels are heuristic (e.g., ``SPEAKER_00``, ``SPEAKER_01``) and do not identify real names.

### Audio-Transcript Alignment
This feature aligns your extracted audio features (from OpenSMILE) with word-level transcripts (from Whisper). This is useful for analyzing acoustic features of specific words.

1.  **Extract Audio Features**: Run this first to generate the `.csv` feature files.
2.  **Align Features**: Click the **Align Features** button.
    *   It generates a detailed word-level transcript (`_words.json`).
    *   It merges the audio features with each word based on timestamps.
    *   The result is saved as `_aligned.csv` in the `audio_features` folder.
    *   **Note**: Features are averaged over the duration of each word.

# Troubleshooting

* I am running into error in the **Convert video to audio** step that says ``An error occured [WinError 2]: The system cannot find the file specified.``

  Or

  in **Extract Transcript** step that says ``An error occured during transcript extraction: ffmpeg was not found but required to load audio file form filename``

  * The packaged app and current source setup try a bundled `ffmpeg` fallback automatically, so most users should not need to install `ffmpeg` manually.
  * If you still see this error, first relaunch the app once so it can re-check the bundled binary.
  * Only if the bundled fallback is unavailable on your machine should you install `ffmpeg` yourself and add it to `PATH`. For Windows you can start from [ffmpeg.org](https://ffmpeg.org/download.html) and these [PATH setup steps](https://phoenixnap.com/kb/ffmpeg-windows).
 
* I am seeing warnings suggesting to set the path to certain package directories installed by this toolbox.
  * For packaged releases, you should usually ignore these warnings unless a feature is failing at runtime.
  * For source installs, if a tool is truly missing, add it to `PATH` or rerun `run_app.sh` / `run_app.bat` so the environment is recreated cleanly.
 
## Acknowledgement

We thank the authors and developers of ``MediaPipe``, ``OpenSMILE``, ``YOLOv5`` and ``whisper`` for their awesome contributions and making their code open-sourced which we use to develop ``MultiSOCIAL toolbox``. 

## Team

``MultiSOCIAL toolbox`` is developed by Tahiya Chowdhury, Veronica Romero, Alexandra Paxton and Muneeb Nafees.

## Disclaimer

Automated tools can be inaccurate and should be used after human verification for correctness.

## Help us improve this toolbox!
[Please leave your feedback in this form.](https://docs.google.com/forms/d/e/1FAIpQLScGkEu-LfLAa_IGNOXG25trtMf8k12FFPymObBRDmLdPkAvxQ/viewform)

 

  
