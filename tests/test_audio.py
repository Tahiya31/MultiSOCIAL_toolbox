from __future__ import annotations

import os
import sys
import types

import numpy as np
import pandas as pd


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
