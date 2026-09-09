"""Run native-operation checks from the packaged macOS application executable.

This entry point is selected only by the fork-tag CI environment.  It bypasses
the wx event loop but uses the same in-process backend, bundled dependencies,
and packaged resources that the normal macOS application uses.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import time
import urllib.request
import wave
from pathlib import Path

from analysis_backend import get_backend


PERSON_FIXTURE_URL = (
    "https://raw.githubusercontent.com/ultralytics/yolov5/v7.0/data/images/bus.jpg"
)
PERSON_FIXTURE_GIT_BLOB = "b43e311165c785f000eb7493ff8fb662d06a3f83"
WHISPER_ATTEMPTS = 3


def _write_tone(path: Path, seconds: int) -> None:
    sample_rate = 16000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        chunk = b"".join(
            int(1000 * math.sin(2 * math.pi * 220 * index / sample_rate)).to_bytes(
                2, "little", signed=True
            )
            for index in range(sample_rate)
        )
        for _ in range(seconds):
            output.writeframesraw(chunk)


def _download_person_fixture(path: Path) -> None:
    error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(PERSON_FIXTURE_URL, timeout=60) as response:
                content = response.read()
            break
        except Exception as exc:
            error = exc
            if attempt < 2:
                time.sleep(2**attempt)
    else:
        raise RuntimeError("Could not download the pinned person fixture") from error

    digest = hashlib.sha1(
        f"blob {len(content)}\0".encode("ascii") + content
    ).hexdigest()
    if digest != PERSON_FIXTURE_GIT_BLOB:
        raise RuntimeError(f"Person fixture hash mismatch: {digest}")
    path.write_bytes(content)


def _make_person_video(workspace: Path) -> Path:
    import gui_utils

    image = workspace / "bus.jpg"
    video = workspace / "person.avi"
    _download_person_fixture(image)
    # Resolve through the same helper the GUI uses so the bundled binary is
    # located and marked executable exactly as it is for a real macOS user.
    ffmpeg_exe = gui_utils.get_ffmpeg_executable()
    if not ffmpeg_exe:
        raise RuntimeError("No ffmpeg executable is available to the packaged app")
    subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-loop",
            "1",
            "-i",
            str(image),
            "-t",
            "2",
            "-vf",
            "scale=640:-2",
            "-r",
            "10",
            "-c:v",
            "mjpeg",
            str(video),
        ],
        cwd=str(workspace),
        check=True,
        timeout=120,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return video


def _assert_success(outcome: dict, label: str) -> None:
    if outcome.get("cancelled") or outcome.get("failed") or not outcome.get("succeeded"):
        raise RuntimeError(f"{label} did not succeed: {outcome!r}")


def _run_whisper(audio, tone: Path, transcripts: Path) -> None:
    """Retry only bounded transient Hub/download failures around an atomic operation."""
    error: Exception | None = None
    for attempt in range(1, WHISPER_ATTEMPTS + 1):
        try:
            outcome = audio.extract_transcripts_batch([str(tone)])
            _assert_success(outcome, "Whisper ASR")
            if not list(transcripts.glob("*.txt")):
                raise RuntimeError("Whisper did not commit a transcript")
            return
        except Exception as exc:
            error = exc
            if attempt < WHISPER_ATTEMPTS:
                delay = 10 * attempt
                print(
                    f"Whisper ASR attempt {attempt}/{WHISPER_ATTEMPTS} failed; "
                    f"retrying in {delay}s.",
                    flush=True,
                )
                time.sleep(delay)
    raise RuntimeError(f"Whisper ASR failed after {WHISPER_ATTEMPTS} attempts") from error


def _run_diarization(backend, tone: Path, workspace: Path) -> None:
    """Exercise the Complete-only PyAnnote path with the fork-scoped CI token."""
    token = os.environ.get("MULTISOCIAL_CI_HF_TOKEN")
    if not token:
        raise RuntimeError("MULTISOCIAL_CI_HF_TOKEN is required for Complete macOS E2E")
    diarized = workspace / "diarized"
    diarized.mkdir(exist_ok=True)
    audio = backend.create_audio_processor(
        None,
        str(diarized),
        enable_speaker_diarization=True,
        auth_token=token,
    )
    outcome = audio.extract_transcripts_batch([str(tone)])
    _assert_success(outcome, "Complete diarization")
    if not list(diarized.glob("*.txt")):
        raise RuntimeError("Complete diarization did not commit a transcript")


def main() -> None:
    workspace_env = os.environ.get("MULTISOCIAL_MACOS_E2E_WORKSPACE")
    if not workspace_env:
        raise RuntimeError("MULTISOCIAL_MACOS_E2E_WORKSPACE is required for packaged E2E")
    workspace = Path(workspace_env).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    backend = get_backend()
    tone = workspace / "tone.wav"
    _write_tone(tone, 1)
    video = _make_person_video(workspace)

    features = workspace / "features"
    transcripts = workspace / "transcripts"
    pose_single = workspace / "pose-single"
    pose_multi = workspace / "pose-multi"
    embed = workspace / "embed"
    for directory in (features, transcripts, pose_single, pose_multi, embed):
        directory.mkdir(exist_ok=True)

    audio = backend.create_audio_processor(str(features), str(transcripts))
    feature_outcome = audio.extract_audio_features_batch([str(tone)])
    _assert_success(feature_outcome, "OpenSMILE features")
    features_csv = features / "tone.csv"
    if not features_csv.is_file():
        raise RuntimeError("OpenSMILE did not commit tone.csv")

    words = workspace / "tone_words.json"
    words.write_text(
        json.dumps({"chunks": [{"text": "tone", "timestamp": [0.0, 0.5]}]}),
        encoding="utf-8",
    )
    aligned = workspace / "aligned.csv"
    alignment_outcome = audio.align_features_batch([(str(features_csv), str(words), str(aligned))])
    _assert_success(alignment_outcome, "Feature alignment")
    if not aligned.is_file():
        raise RuntimeError("Feature alignment did not commit aligned.csv")

    single = backend.create_pose_processor(str(pose_single), str(embed))
    if not single.extract_pose_features(str(video)):
        raise RuntimeError("MediaPipe single-person pose was cancelled")
    if not list(pose_single.glob("person_ID_*.csv")):
        raise RuntimeError("Single-person pose did not commit a pose CSV")

    multi = backend.create_pose_processor(str(pose_multi))
    multi.set_multi_person_mode(True)
    if not multi.extract_pose_features(str(video)):
        raise RuntimeError("YOLO/Torch multi-person pose was cancelled")
    if not list(pose_multi.glob("person_multi_ID_*.csv")):
        raise RuntimeError("Multi-person pose did not commit a pose CSV")

    embedded = single.embed_pose_video(str(video))
    if not embedded or not Path(embedded).is_file():
        raise RuntimeError("Pose-video embedding did not commit an output video")

    cancelled = workspace / "cancelled"
    cancelled.mkdir(exist_ok=True)
    long_tone = workspace / "long-tone.wav"
    _write_tone(long_tone, 30)
    cancellation_checks = 0

    def cancel_check() -> bool:
        nonlocal cancellation_checks
        cancellation_checks += 1
        return cancellation_checks >= 3

    cancel_audio = backend.create_audio_processor(str(cancelled), str(transcripts))
    cancellation = cancel_audio.extract_audio_features_batch(
        [str(long_tone)] * 4, cancel_check=cancel_check
    )
    if not cancellation.get("cancelled"):
        raise RuntimeError(f"Cancellation was not acknowledged: {cancellation!r}")
    if any(cancelled.iterdir()):
        raise RuntimeError("Cancellation left committed or staged audio output")

    _run_whisper(audio, tone, transcripts)
    if os.environ.get("MULTISOCIAL_BUILD_PROFILE") == "complete":
        _run_diarization(backend, tone, workspace)
    print("Packaged macOS native E2E passed.", flush=True)


if __name__ == "__main__":
    main()
