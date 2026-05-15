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

## Video file
**Convert video to audio** If you have a supported video file of human interaction and would like to convert it to an audio file in `.wav` format, this step is for you.
  * Use the ``Browse`` button to locate your input video file.
  * Then press **Convert video to audio** button.
  * Once the .wav file is ready, a dialogue box will let you know the output file is ready.

You should see three folders within the same folder as your input video.
  * `converted_audio`: Contains the WAV files produced by **Convert video to audio**.
  * `pose_features`: Contains the CSV pose feature files produced by **Extract Pose Features**.
  * `embedded_pose`: Contains the rendered pose-overlay videos produced by **Embed Pose Features**.


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

**Embed Pose Features** If you are interested in embedding body key-points extracted from Mediapipe on each frames, this step uses [MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md) to achieve this. 
  * Use the ``Browse`` button to locate your input video file.
  * Then press **Embed Pose Features** button. The toolbox will process each frame and embed body key-points onto the video.
  * Once all the frames are processed, an output video will appear in the **embedded_pose** folder where your input video is located.

### Verify Consistency (Pose QA)
After extracting pose CSVs and generating embedded pose videos, you can run **Verify Consistency** to compare them.
  * Pre-requisites:
    * `pose_features/` must contain CSVs generated from **Extract Pose Features**.
    * `embedded_pose/` must contain pose-overlay videos generated from **Embed Pose Features**.
  * The tool creates a `verification/` folder in your dataset directory with:
    * per-video JSON + CSV reports (hit rate, SSIM, etc.)
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
  * Once the transcript is extracted, a dialogue box will let you know the output file is ready.
  * You can find them in **transcripts** folder created before.

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
 
* I am getting an error during installation similar to ``ERROR: Failed building wheel for pi-heif`` or ``error: failed-wheel-build-for-install``
  * Why this happens: ``pi-heif`` provides HEIF/AVIF image support in Python and needs the system ``libheif`` library when a prebuilt wheel is not available for your OS/Python/architecture. On some machines, ``pip`` falls back to building from source, which fails without ``libheif`` present.
  * What to do:
    * macOS:
      * Install Homebrew if you don't have it yet by following the instructions here: [Homebrew](https://brew.sh).
      * Then install ``libheif``:
        * ``brew install libheif``
      * Re-run the toolbox setup command (``bash run_app.sh``).
    * Windows:
      * Install vcpkg (Microsoft's C/C++ package manager). Quick start: [vcpkg getting started](https://learn.microsoft.com/en-us/vcpkg/get_started/get-started).
      * In ``Command Prompt`` or ``PowerShell``:
        * ``git clone https://github.com/microsoft/vcpkg.git``
        * ``cd vcpkg``
        * ``.\bootstrap-vcpkg.bat``
        * (Optional, if you need specific codec backends for HEIF/AVIF) e.g., install AOM: ``vcpkg install aom:x64-windows``
        * Install libheif: ``vcpkg install libheif:x64-windows``
        * (Optional) Integrate with Visual Studio: ``vcpkg integrate install``
      * Re-run the toolbox setup command (``run_app.bat``).
  * After installing ``libheif``, ``pip`` can either use a compatible prebuilt wheel or successfully build ``pi-heif`` from source.


## Acknowledgement

We thank the authors and developers of ``MediaPipe``, ``OpenSMILE``, ``YOLOv5`` and ``whisper`` for their awesome contributions and making their code open-sourced which we use to develop ``MultiSOCIAL toolbox``. 

## Team

``MultiSOCIAL toolbox`` is developed by Tahiya Chowdhury, Veronica Romero, Alexandra Paxton and Muneeb Nafees.

## Disclaimer

Automated tools can be inaccurate and should be used after human verification for correctness.

## Help us improve this toolbox!
[Please leave your feedback in this form.](https://docs.google.com/forms/d/e/1FAIpQLScGkEu-LfLAa_IGNOXG25trtMf8k12FFPymObBRDmLdPkAvxQ/viewform)

 

  
