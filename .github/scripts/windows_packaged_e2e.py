"""Exercise native operations through the packaged Windows GUI boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import urllib.request
import wave
from pathlib import Path

from run_windows_packaged_request import run_request

PERSON_FIXTURE_URL = (
    "https://raw.githubusercontent.com/ultralytics/yolov5/v7.0/data/images/bus.jpg"
)
PERSON_FIXTURE_GIT_BLOB = "b43e311165c785f000eb7493ff8fb662d06a3f83"


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


def _find_ffmpeg(worker_root: Path) -> Path:
    candidates = sorted(worker_root.rglob("ffmpeg*.exe"))
    if not candidates:
        raise FileNotFoundError("Bundled worker ffmpeg executable is missing")
    return candidates[0]


def _invoke(
    gui: Path,
    workspace: Path,
    name: str,
    operation: str,
    payload: dict,
    *,
    cancel_after_seconds: float | None = None,
    verify_heavy_pose_asset: bool = False,
) -> dict:
    request = workspace / f"{name}-request.json"
    result = workspace / f"{name}-result.json"
    value = {"operation": operation, "payload": payload}
    if operation == "probe":
        value["timeout_seconds"] = 60
    if cancel_after_seconds is not None:
        value["cancel_after_seconds"] = cancel_after_seconds
    request.write_text(json.dumps(value), encoding="utf-8")
    response = run_request(
        gui,
        request,
        result,
        verify_heavy_pose_asset=verify_heavy_pose_asset,
        timeout=75 if operation == "probe" else 1200,
    )
    return dict(response["result"])


def _assert_success(result: dict, label: str) -> None:
    if result.get("cancelled") or result.get("failed") or not result.get("succeeded"):
        raise RuntimeError(f"{label} did not succeed: {result!r}")


def _run_diarization(gui: Path, workspace: Path, tone: Path) -> None:
    destination = workspace / "diarized"
    destination.mkdir(exist_ok=True)
    result = _invoke(
        gui,
        workspace,
        "diarization",
        "extract_transcripts",
        {
            "audio_files": [str(tone.resolve())],
            "output_transcripts_folder": str(destination.resolve()),
            "word_timestamps": False,
            "enable_diarization": True,
        },
    )
    _assert_success(result, "Complete diarization")
    if not list(destination.glob("*.txt")):
        raise RuntimeError("Complete diarization did not commit a transcript")


def _run_whisper(gui: Path, workspace: Path, tone: Path, attempts: int) -> None:
    """Run packaged ASR with bounded retries for transient Hub transport failures.

    The worker writes output atomically, so a failed attempt cannot be mistaken for
    a completed transcription.  A later attempt can safely reuse Hugging Face's
    verified, resumable cache entries.
    """
    transcripts = workspace / "transcripts"
    transcripts.mkdir(exist_ok=True)
    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            whisper_result = _invoke(
                gui,
                workspace,
                f"whisper-{attempt}",
                "extract_transcripts",
                {
                    "audio_files": [str(tone.resolve())],
                    "output_transcripts_folder": str(transcripts.resolve()),
                    "word_timestamps": False,
                    "enable_diarization": False,
                },
            )
            _assert_success(whisper_result, "Whisper ASR")
            if not list(transcripts.glob("*.txt")):
                raise RuntimeError("Whisper did not commit a transcript")
            return
        except Exception as exc:
            error = exc
            if attempt == attempts:
                break
            delay = 10 * attempt
            print(
                f"Whisper ASR attempt {attempt}/{attempts} failed; "
                f"retrying in {delay}s.",
                flush=True,
            )
            time.sleep(delay)
    raise RuntimeError(f"Whisper ASR failed after {attempts} attempts") from error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", type=Path, required=True)
    parser.add_argument("--profile", choices=("standard", "complete"), required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--include-whisper", action="store_true")
    parser.add_argument("--only-whisper", action="store_true")
    parser.add_argument("--whisper-attempts", type=int, default=1)
    parser.add_argument("--include-diarization", action="store_true")
    parser.add_argument("--only-diarization", action="store_true")
    args = parser.parse_args()
    if args.whisper_attempts < 1:
        parser.error("--whisper-attempts must be at least 1")

    gui = args.gui.resolve()
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    tone = workspace / "tone.wav"
    if not tone.is_file():
        _write_tone(tone, 1)
    if args.only_diarization:
        _run_diarization(gui, workspace, tone)
        return

    if args.only_whisper:
        _run_whisper(gui, workspace, tone, args.whisper_attempts)
        return

    worker_root = gui.parent / "worker"
    person_image = workspace / "bus.jpg"
    person_video = workspace / "person.avi"
    _download_person_fixture(person_image)
    ffmpeg = _find_ffmpeg(worker_root)
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-loop",
            "1",
            "-i",
            str(person_image),
            "-t",
            "2",
            "-vf",
            "scale=640:-2",
            "-r",
            "10",
            "-c:v",
            "mjpeg",
            str(person_video),
        ],
        cwd=str(workspace),
        check=True,
        timeout=120,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    for attempt in range(1, 11):
        result = _invoke(
            gui,
            workspace,
            f"probe-{attempt}",
            "probe",
            {"profile": args.profile},
            verify_heavy_pose_asset=attempt == 1,
        )
        runtime = result.get("runtime", {}).get("worker_runtime", {})
        if (
            not runtime.get("private_runtime")
            or not runtime.get("mediapipe_bindings")
            or runtime.get("loaded_module_violations")
            or runtime.get("native_module_violations")
            or runtime.get("external_module_violations")
        ):
            raise RuntimeError(f"Cold probe {attempt} failed provenance: {runtime!r}")

    features = workspace / "features"
    features.mkdir(exist_ok=True)
    result = _invoke(
        gui,
        workspace,
        "features",
        "extract_audio_features",
        {
            "audio_files": [str(tone.resolve())],
            "output_audio_features_folder": str(features.resolve()),
        },
    )
    _assert_success(result, "OpenSMILE features")
    if not (features / "tone.csv").is_file():
        raise RuntimeError("OpenSMILE did not commit tone.csv")

    words = workspace / "tone_words.json"
    words.write_text(
        json.dumps({"chunks": [{"text": "tone", "timestamp": [0.0, 0.5]}]}),
        encoding="utf-8",
    )
    aligned = workspace / "aligned.csv"
    alignment_result = _invoke(
        gui,
        workspace,
        "alignment",
        "align_features",
        {
            "alignment_pairs": [[
                str((features / "tone.csv").resolve()),
                str(words.resolve()),
                str(aligned.resolve()),
            ]],
        },
    )
    _assert_success(alignment_result, "Feature alignment")
    if not aligned.is_file():
        raise RuntimeError("Feature alignment did not commit aligned.csv")

    pose_single = workspace / "pose-single"
    pose_multi = workspace / "pose-multi"
    embed = workspace / "embed"
    for directory in (pose_single, pose_multi, embed):
        directory.mkdir(exist_ok=True)
    single_result = _invoke(
        gui,
        workspace,
        "pose-single",
        "extract_pose",
        {
            "video_files": [str(person_video.resolve())],
            "output_csv_folder": str(pose_single.resolve()),
            "multi_person": False,
        },
    )
    _assert_success(single_result, "MediaPipe single-person pose")
    if not list(pose_single.glob("person_ID_*.csv")):
        raise RuntimeError("Single-person pose did not commit a pose CSV")

    multi_result = _invoke(
        gui,
        workspace,
        "pose-multi",
        "extract_pose",
        {
            "video_files": [str(person_video.resolve())],
            "output_csv_folder": str(pose_multi.resolve()),
            "multi_person": True,
        },
    )
    _assert_success(multi_result, "YOLO/Torch multi-person pose")
    if not list(pose_multi.glob("person_multi_ID_*.csv")):
        raise RuntimeError("Multi-person pose did not commit a pose CSV")

    embed_result = _invoke(
        gui,
        workspace,
        "embed",
        "embed_pose",
        {
            "video_files": [str(person_video.resolve())],
            "output_csv_folder": str(pose_single.resolve()),
            "output_video_folder": str(embed.resolve()),
            "multi_person": False,
        },
    )
    _assert_success(embed_result, "Pose-video embedding")
    if not any(path.is_file() for path in embed.iterdir()):
        raise RuntimeError("Pose-video embedding did not commit an output video")

    cancelled = workspace / "cancelled"
    cancelled.mkdir(exist_ok=True)
    long_tone = workspace / "long-tone.wav"
    _write_tone(long_tone, 60)
    cancel_result = _invoke(
        gui,
        workspace,
        "cancel",
        "extract_audio_features",
        {
            "audio_files": [str(long_tone.resolve())] * 4,
            "output_audio_features_folder": str(cancelled.resolve()),
        },
        cancel_after_seconds=0.1,
    )
    if not cancel_result.get("cancelled"):
        raise RuntimeError(f"Cancellation was not acknowledged: {cancel_result!r}")
    if any(cancelled.iterdir()):
        raise RuntimeError("Cancellation left committed or staged output files")

    if args.include_whisper:
        _run_whisper(gui, workspace, tone, args.whisper_attempts)

    if args.include_diarization:
        _run_diarization(gui, workspace, tone)


if __name__ == "__main__":
    main()
