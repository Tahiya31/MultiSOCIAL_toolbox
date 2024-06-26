# MultiSOCIAL_toolbox
Toolbox for multimodal interaction analysis for text, audio, and video information.

## How can I use MultiSOCIAL Toolbox?

The toolbox allows you to process audio and video files of conversation.

# Installation

## MAC
1. Download the toolbox code by going to [this link](https://github.com/Tahiya31/MultiSOCIAL_toolbox) and click on **Code**.
2. Click on **Download Zip** to download the entire code folder. (You can also use ``git clone`` command to clone the repository.)
3. Open the **Terminal** application.
4. Go to the folder where **MultiSOCIAL_toolbox** is saved. (On Mac OS, typically the location is ``../Users/(name of the user)/Downloads/``)
   * You can run ``cd Downloads/MultiSOCIAL_toolbox`` to achieve this.
5. Run ``python app.py.``
   * If you run into error saying **Python is not recognized**, then you will need to install Python, the programming language our toolbox is written on, on your computer.
   * Install the latest version of Python for your MAC computer from [this link](https://www.python.org/downloads/macos/)
6. If the script above executes properly, all necessary packages should be installed and the MultiSOCIAL app should launch.      


## WINDOWS

1. Download the toolbox code by going to [this link](https://github.com/Tahiya31/MultiSOCIAL_toolbox) and click on **Code**.
2. Click on **Download Zip** to download the entire code folder. (You can also use ``git clone`` command to clone the repository.)
3. Open **Windows Command Prompt** application.
4. Go to the folder where **MultiSOCIAL_toolbox** is saved. (Typically this location is ``../Users/(name of the user)/Downloads/``)
   * You can run ``cd Downloads/MultiSOCIAL_toolbox`` to achieve this.

5. Run ``python app.py.``
   * If you run into error saying **Python is not recognized**, then you will need to install Python, the programming language our toolbox is written on, on your computer.
   * Install the latest version of Python for your WINDOWS computer from [this link](https://www.python.org/downloads/windows/)
6. If the script above executes properly, all necessary packages should be installed and the MultiSOCIAL app should launch.


# Usage
Once launched, MultiSOCIAL Toolbox application looks like this.

![alt text](./ApplicationUI.png =100x20) 

The toolbox takes two types of input: audio (.wav file) and video (.mp4 file).

## Video file
**Convert video to audio** If you have a video file of human interaction and would like to convert it to a audio file in .wav format, this step is for you.
  * Use the ``Browse`` button to locate your input vudeo file.
  * Then press **Convert video to audio** button.
  * Once the .wav file is ready, a dialogue box will prompt you to provide a location and name to save output .wav file.

**Extract Pose Features** If you are interested in extracting pose or body key-points from the video, this step uses [MediaPipe](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md) to achieve this. This step returns 33 body pose land marks. For more details on MediaPipe, please check out the [official page](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md). 
  * Use the ``Browse`` button to locate your input video file.
  * Then press **Extract Pose Features** button. This step may take some time.
  * Once the pose features are extracted, a dialogue box will appeare asking to provide a location and name to save the output .csv file.
  * **CSV format** Each row represents a frame, each column represents features. For each of the 33 body land marks, you should see 4 columns:
  * x and y: Landmark coordinates normalized to [0.0, 1.0] by the image width and height respectively.
  * z: Represents the landmark depth with the depth at the midpoint of hips being the origin, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.
  * confidence: A value in [0.0, 1.0] indicating the likelihood of the landmark being visible (present and not occluded) in the image.

## Audio file
**Extract Audio Features** If you are interested in extracting speech features from human speech during interaction, this step uses [OpenSMILE](https://audeering.github.io/opensmile-python/) to achieve this. This step currently uses predetermined feature sets (ComParE 2016) from OpenSMILE. For more details on OpenSMILE, please check their official [documentation page](https://audeering.github.io/opensmile-python/).
  * Use the ``Browse`` button to locate your input audio file.
  * Then press **Extract Audio Features** button.
  * Once the audio features are extracted, a dialogue box will appeare asking to provide a location and name to save the output .csv file.
  *  **CSV format** Each row represents a sample, each column represents features. For ComParE 2016, you should see 65 feature columns.

**Extract Transcripts** If you are interested in extracting transcript of the conversation, this step uses [whisper](https://github.com/openai/whisper) and [distil-whisper](https://github.com/huggingface/distil-whisper) for automatically recognize speech and transcribe. For more details on whisper, please check their offcial documentation page [here](https://github.com/openai/whisper).
  * Use the ``Browse`` button to locate your input audio file.
  * Then press **Extract Transcripts** button.
  * Once the transcript is extracted, a dialogue box will appeare asking to provide a location and name to save the output .txt file.

# Troubleshooting

* I am running into error in the **Convert video to audio** step that says ``An error occured [WinError 2]: The system cannot find the file specified.``

  Or

  in **Extract Transcript** step that says ``An error occured during transcript extraction: ffmpeg was not found but required to load audio file form filename``

  * We need ffmpeg framework for these two steps. For Windows we need to ffmpeg from [here](https://ffmpeg.org/download.html).
  * Follow the steps described here to add [ffmpeg](https://phoenixnap.com/kb/ffmpeg-windows) to the environment PATH so that it can be used from Command Prompt.
  * You may need to close **Command Prompt** and re-open to allow this change to take effect.
 
* I am seeing warnings suggesting to set the path to certain package directories installed by this toolbox.
  * You can follow the link to add ``ffmpeg`` to the environment PATH above or [this link](https://stackoverflow.com/questions/44272416/how-to-add-a-folder-to-path-environment-variable-in-windows-10-with-screensho) to add them.
 

  
