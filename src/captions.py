"""Turn-by-turn caption support: write SRT from Whisper segments and burn it onto video.

Used by:
- audio.py: writes a .srt sidecar beside each transcript .txt during transcription.
- app.py (Video tab): burns the .srt onto the source video via ffmpeg.
"""

import os
import re
import subprocess


_PUNCTUATION_ENDINGS = (".", "!", "?", ";", ":")


def _format_srt_timestamp(seconds):
    """Seconds (float) -> 'HH:MM:SS,mmm' SRT timecode."""
    if seconds is None or seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3600 * 1000)
    minutes, millis = divmod(millis, 60 * 1000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segment_start_end(segment):
    start = float(segment.get("start") or 0.0)
    end = float(segment.get("end") or start)
    if end <= start:
        end = start + 0.1
    return start, end


def group_caption_segments(segments, speaker_label_fn=None, max_gap=0.75, max_duration=6.0):
    """Merge short/word-level ASR chunks into readable caption cues."""
    grouped = []
    current = None

    for seg in segments or []:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue

        start, end = _segment_start_end(seg)
        label = None
        if speaker_label_fn is not None:
            try:
                label = speaker_label_fn(start, end)
            except Exception:
                label = None

        if current is None:
            current = {"start": start, "end": end, "text": text, "speaker": label}
            continue

        gap = start - current["end"]
        duration = end - current["start"]
        previous_text = current["text"].rstrip()
        should_flush = (
            gap > max_gap
            or duration > max_duration
            or label != current.get("speaker")
            or previous_text.endswith(_PUNCTUATION_ENDINGS)
        )

        if should_flush:
            grouped.append(current)
            current = {"start": start, "end": end, "text": text, "speaker": label}
        else:
            current["end"] = max(current["end"], end)
            current["text"] = f"{previous_text} {text}".strip()

    if current is not None:
        grouped.append(current)

    return grouped


def write_srt(segments, output_path, speaker_label_fn=None):
    """Write an SRT file from Whisper segments.

    Args:
        segments (list): dicts with 'start', 'end', 'text' (from
            AudioProcessor._extract_whisper_segments).
        output_path (str): destination .srt path.
        speaker_label_fn (callable, optional): fn(start, end) -> speaker label.
            When provided, each cue is prefixed with "<label>: ". Pass
            AudioProcessor._find_speaker_for_time bound with speaker_segments.

    Returns:
        str | None: output_path, or None if there were no usable segments.
    """
    cues = []
    index = 1
    for seg in group_caption_segments(segments, speaker_label_fn=speaker_label_fn):
        text = str(seg.get('text') or '').strip()
        if not text:
            continue
        start, end = _segment_start_end(seg)

        label = seg.get("speaker")
        if label:
            text = f"{label}: {text}"

        cues.append(
            f"{index}\n"
            f"{_format_srt_timestamp(start)} --> {_format_srt_timestamp(end)}\n"
            f"{text}\n"
        )
        index += 1

    if not cues:
        return None

    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write("\n".join(cues) + "\n")
    return output_path


def _escape_subtitles_filename(name):
    """Escape a (bare) filename for use inside ffmpeg's subtitles= filter value."""
    # Order matters: backslash first. Colon and single-quote break the filter syntax.
    name = name.replace('\\', '\\\\')
    name = name.replace(':', r'\:')
    name = name.replace("'", r"\'")
    return name


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_TIME_RE = re.compile(r"time=\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def _hms_to_seconds(h, m, s):
    return int(h) * 3600 + int(m) * 60 + float(s)


def burn_subtitles(video_path, srt_path, out_path, ffmpeg_exe,
                   progress_callback=None, cancel_check=None):
    """Burn an SRT onto a video (re-encode video, copy audio) via ffmpeg.

    Returns:
        str: out_path on success.
        False: if cancelled via cancel_check.
    Raises:
        RuntimeError: if ffmpeg fails.
    """
    srt_dir = os.path.dirname(os.path.abspath(srt_path))
    srt_name = os.path.basename(srt_path)
    sub_arg = _escape_subtitles_filename(srt_name)
    vf = (
        f"subtitles={sub_arg}:force_style="
        "'FontSize=18,Outline=1,Shadow=0,MarginV=24'"
    )

    cmd = [
        ffmpeg_exe, "-y",
        "-i", os.path.abspath(video_path),
        "-vf", vf,
        "-c:a", "copy",
        os.path.abspath(out_path),
    ]

    # Run with cwd at the SRT folder so the filter sees a bare relative filename
    # (sidesteps Windows drive-colon and absolute-path escaping headaches).
    proc = subprocess.Popen(
        cmd,
        cwd=srt_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1,
    )

    total_seconds = None
    stderr_tail = []
    try:
        for line in proc.stderr:
            stderr_tail.append(line)
            if len(stderr_tail) > 40:
                stderr_tail.pop(0)

            if cancel_check and cancel_check():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                return False

            if total_seconds is None:
                m = _DURATION_RE.search(line)
                if m:
                    total_seconds = _hms_to_seconds(*m.groups())

            if progress_callback and total_seconds:
                t = _TIME_RE.search(line)
                if t:
                    elapsed = _hms_to_seconds(*t.groups())
                    pct = max(0, min(99, int((elapsed / total_seconds) * 100)))
                    progress_callback(pct)
    finally:
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to burn subtitles (exit {proc.returncode}).\n"
            + "".join(stderr_tail).strip()
        )

    if progress_callback:
        progress_callback(100)
    return out_path
