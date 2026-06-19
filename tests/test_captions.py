from __future__ import annotations


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
