from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pandas as pd


def _stub_transcription(processor, audio, monkeypatch, result):
    """Wire a processor so extract_transcripts_batch runs without a real model/audio.

    Returns the dict that captures the ``return_timestamps`` argument the pipeline saw.
    """
    captured = {}

    monkeypatch.setattr(processor, "_load_whisper_model", lambda *a, **k: None)
    monkeypatch.setattr(
        audio,
        "_whisper_pipeline_audio_input",
        lambda path: {"array": np.zeros(4, dtype=np.float32), "sampling_rate": 16000},
    )

    def fake_pipe(audio_input, return_timestamps=None, generate_kwargs=None):
        captured["return_timestamps"] = return_timestamps
        return result

    processor.whisper_pipe = fake_pipe
    return captured


def test_pcm_audio_to_mono_float32_converts_integer_stereo(import_audio):
    audio = import_audio
    pcm = np.array([[0, 1000], [2000, 3000]], dtype=np.int16)

    samples, sr = audio._pcm_audio_to_mono_float32(pcm, 16000)

    assert sr == 16000
    assert samples.dtype == np.float32
    assert samples.shape == (2,)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= 1.0)


def test_decode_audio_file_raises_for_missing_path(import_audio, tmp_path):
    audio = import_audio

    missing = tmp_path / "missing.wav"
    try:
        audio._decode_audio_file(str(missing))
    except FileNotFoundError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")


def test_decode_audio_file_uses_scipy_for_wav(monkeypatch, import_audio, tmp_path):
    audio = import_audio
    wav = tmp_path / "sample.wav"
    wav.write_text("not-real", encoding="utf-8")
    monkeypatch.setattr(audio.wavfile, "read", lambda path: (8000, np.array([0, 1000], dtype=np.int16)))

    samples, sr = audio._decode_audio_file(str(wav))

    assert sr == 8000
    assert samples.shape == (2,)


def test_decode_audio_file_falls_back_to_soundfile_for_non_wav(monkeypatch, import_audio, tmp_path):
    audio = import_audio
    flac = tmp_path / "sample.flac"
    flac.write_text("not-real", encoding="utf-8")
    fake_soundfile = types.SimpleNamespace(read=lambda *a, **k: (np.array([0.25, 0.5], dtype=np.float32), 22050))
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    samples, sr = audio._decode_audio_file(str(flac))

    assert sr == 22050
    assert np.allclose(samples, np.array([0.25, 0.5], dtype=np.float32))


def test_whisper_pipeline_audio_input_wraps_decoded_audio(monkeypatch, import_audio):
    audio = import_audio
    monkeypatch.setattr(audio, "_decode_audio_file", lambda path: (np.array([1.0, 2.0], dtype=np.float32), 44100))

    payload = audio._whisper_pipeline_audio_input("/tmp/fake.wav")

    assert payload["sampling_rate"] == 44100
    assert payload["array"].tolist() == [1.0, 2.0]


def test_extract_audio_features_writes_timestamped_csv(monkeypatch, import_audio, tmp_path):
    audio = import_audio
    processor = audio.AudioProcessor(
        output_audio_features_folder=str(tmp_path),
        output_transcripts_folder=None,
        status_callback=None,
    )
    monkeypatch.setattr(audio, "_load_audio_for_opensmile", lambda path: (np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32), 2))

    progress_updates = []
    out_csv = processor.extract_audio_features("clip.flac", progress_callback=progress_updates.append)
    df = pd.read_csv(out_csv)

    assert os.path.basename(out_csv) == "clip.csv"
    assert df.columns[:3].tolist() == [
        "Timestamp_Seconds",
        "Timestamp_Milliseconds",
        "Timestamp_Formatted",
    ]
    assert progress_updates[0] == 0
    assert progress_updates[-1] == 100


def _make_processor(audio, tmp_path):
    return audio.AudioProcessor(
        output_audio_features_folder=str(tmp_path),
        output_transcripts_folder=str(tmp_path),
        status_callback=None,
        enable_speaker_diarization=False,
    )


def test_extract_transcripts_batch_saves_word_json_when_requested(monkeypatch, import_audio, tmp_path):
    audio = import_audio
    processor = _make_processor(audio, tmp_path)
    word_result = {
        "text": "hello world",
        "chunks": [
            {"text": "hello", "timestamp": (0.0, 0.5)},
            {"text": "world", "timestamp": (0.5, 1.0)},
        ],
    }
    captured = _stub_transcription(processor, audio, monkeypatch, word_result)

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF0000")

    processor.extract_transcripts_batch([str(clip)], word_timestamps=True)

    # Word-level mode was requested at call time...
    assert captured["return_timestamps"] == "word"
    # ...the JSON sidecar Align Features needs was written...
    json_path = tmp_path / "clip_words.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["chunks"][0]["text"] == "hello"
    # ...and the plain .txt transcript is still produced.
    assert (tmp_path / "clip.txt").exists()


def test_extract_transcripts_batch_segment_mode_writes_no_word_json(monkeypatch, import_audio, tmp_path):
    audio = import_audio
    processor = _make_processor(audio, tmp_path)
    captured = _stub_transcription(
        processor, audio, monkeypatch, {"text": "hello world", "chunks": []}
    )

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF0000")

    processor.extract_transcripts_batch([str(clip)])  # word_timestamps defaults to False

    assert captured["return_timestamps"] is True
    assert not (tmp_path / "clip_words.json").exists()
    assert (tmp_path / "clip.txt").exists()


def test_align_features_maps_words_to_feature_means(import_audio, tmp_path):
    audio = import_audio
    processor = _make_processor(audio, tmp_path)

    features = pd.DataFrame(
        {
            "Timestamp_Seconds": [0.0, 0.25, 0.5, 0.75],
            "f0": [1.0, 3.0, 5.0, 7.0],
        }
    )
    feature_csv = tmp_path / "clip.csv"
    features.to_csv(feature_csv, index=False)

    words = {
        "text": "hello world",
        "chunks": [
            {"text": "hello", "timestamp": (0.0, 0.5)},
            {"text": "world", "timestamp": (0.5, 1.0)},
        ],
    }
    words_json = tmp_path / "clip_words.json"
    words_json.write_text(json.dumps(words), encoding="utf-8")

    out_csv = tmp_path / "clip_aligned.csv"
    processor.align_features(str(feature_csv), str(words_json), str(out_csv))

    result = pd.read_csv(out_csv)
    assert list(result["word"]) == ["hello", "world"]
    # "hello" spans [0.0, 0.5] -> frames 1.0, 3.0, 5.0 -> mean 3.0
    assert result.loc[0, "f0"] == 3.0
    # "world" spans [0.5, 1.0] -> frames 5.0, 7.0 -> mean 6.0
    assert result.loc[1, "f0"] == 6.0


def test_extract_audio_features_batch_cancel_check(import_audio, tmp_path, monkeypatch):
    audio = import_audio
    processor = _make_processor(audio, tmp_path)
    processed = []

    def fake_extract(path, progress_callback=None):
        processed.append(path)
        return str(tmp_path / f"{os.path.basename(path)}.csv")

    monkeypatch.setattr(processor, "extract_audio_features", fake_extract)

    files = []
    for i in range(3):
        wav = tmp_path / f"clip{i}.wav"
        wav.write_bytes(b"RIFF")
        files.append(str(wav))

    checks = {"n": 0}

    def cancel_check():
        checks["n"] += 1
        return checks["n"] > 1

    processor.extract_audio_features_batch(files, cancel_check=cancel_check)
    assert len(processed) == 1


def test_extract_transcripts_batch_cancel_check(monkeypatch, import_audio, tmp_path):
    audio = import_audio
    processor = _make_processor(audio, tmp_path)
    _stub_transcription(processor, audio, monkeypatch, {"text": "hi", "chunks": []})

    files = []
    for i in range(3):
        wav = tmp_path / f"clip{i}.wav"
        wav.write_bytes(b"RIFF")
        files.append(str(wav))

    transcribed = []
    original_pipe = processor.whisper_pipe

    def counting_pipe(*args, **kwargs):
        transcribed.append(args)
        return original_pipe(*args, **kwargs)

    processor.whisper_pipe = counting_pipe

    checks = {"n": 0}

    def cancel_check():
        checks["n"] += 1
        return checks["n"] > 1

    processor.extract_transcripts_batch(files, cancel_check=cancel_check)
    assert len(transcribed) == 1
