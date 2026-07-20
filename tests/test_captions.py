from __future__ import annotations


def _cue_times(srt_text):
    times = []
    for block in srt_text.strip().split("\n\n"):
        start, end = block.splitlines()[1].split(" --> ")

        def seconds(value):
            hours, minutes, rest = value.replace(",", ".").split(":")
            return int(hours) * 3600 + int(minutes) * 60 + float(rest)

        times.append((seconds(start), seconds(end)))
    return times


def test_write_srt_writes_timestamps_and_text(tmp_path):
    import captions

    out = tmp_path / "clip.srt"

    result = captions.write_srt(
        [{"start": 1.25, "end": 3.5, "text": "Hello there."}],
        str(out),
    )

    assert result == str(out)
    assert out.read_text(encoding="utf-8") == (
        "1\n"
        "00:00:01,250 --> 00:00:03,500\n"
        "Hello there.\n\n"
    )


def test_write_srt_prefixes_speaker_label(tmp_path):
    import captions

    out = tmp_path / "clip.srt"

    captions.write_srt(
        [{"start": 0.0, "end": 1.0, "text": "Hello."}],
        str(out),
        speaker_label_fn=lambda start, end: "SPEAKER_00",
    )

    assert "SPEAKER_00: Hello." in out.read_text(encoding="utf-8")


def test_write_srt_groups_word_level_chunks(tmp_path):
    import captions

    out = tmp_path / "clip.srt"
    chunks = [
        {"start": 0.0, "end": 0.2, "text": "hello"},
        {"start": 0.2, "end": 0.4, "text": "world."},
        {"start": 0.5, "end": 0.7, "text": "next"},
        {"start": 0.7, "end": 1.0, "text": "turn"},
    ]

    captions.write_srt(chunks, str(out))
    text = out.read_text(encoding="utf-8")

    assert "hello world." in text
    assert "next turn" in text
    assert "hello\n\n2\n" not in text
    assert text.count("-->") == 2


def test_write_srt_normalizes_overlapping_and_short_cues(tmp_path):
    import captions

    out = tmp_path / "clip.srt"
    captions.write_srt(
        [
            {"start": 2.0, "end": 2.1, "text": "second."},
            {"start": 0.0, "end": 1.0, "text": "first."},
            {"start": 0.8, "end": 0.8, "text": "overlap."},
        ],
        str(out),
    )

    times = _cue_times(out.read_text(encoding="utf-8"))
    assert times == sorted(times)
    assert all(end - start >= 0.25 for start, end in times)
    assert all(next_start >= end for (_, end), (next_start, _) in zip(times, times[1:]))


def test_escape_subtitles_filename_handles_filter_special_chars():
    import captions

    assert captions._escape_subtitles_filename(r"speaker's C:\clip.srt") == r"speaker\'s C\:\\clip.srt"


def test_burn_subtitles_builds_filter_and_reports_progress(tmp_path, monkeypatch):
    import captions

    video = tmp_path / "clip.mp4"
    srt = tmp_path / "speaker's.srt"
    out = tmp_path / "captioned.mp4"
    video.write_bytes(b"fake")
    srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHi\n", encoding="utf-8")
    captured = {}

    class FakeProc:
        returncode = 0
        stderr = [
            "Duration: 00:00:10.00\n",
            "frame=1 time=00:00:05.00 bitrate=1kbits/s\n",
        ]

        def wait(self, *args, **kwargs):
            return self.returncode

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["cwd"] = kwargs["cwd"]
        return FakeProc()

    monkeypatch.setattr(captions.subprocess, "Popen", fake_popen)
    progress = []

    result = captions.burn_subtitles(str(video), str(srt), str(out), "/fake/ffmpeg", progress.append)

    assert result == str(out)
    assert captured["cmd"][0] == "/fake/ffmpeg"
    assert captured["cwd"] == str(tmp_path)
    vf = captured["cmd"][captured["cmd"].index("-vf") + 1]
    assert r"subtitles=speaker\'s.srt" in vf
    assert progress == [50, 100]
