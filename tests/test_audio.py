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
    assert df["Timestamp_Seconds"].tolist() == [0.0, 0.01]
    assert progress_updates[0] == 0
    assert progress_updates[-1] == 100


def _make_processor(audio, tmp_path):
    return audio.AudioProcessor(
        output_audio_features_folder=str(tmp_path),
        output_transcripts_folder=str(tmp_path),
        status_callback=None,
        enable_speaker_diarization=False,
    )


def test_audio_processor_defaults_diarization_off(import_audio, tmp_path):
    audio = import_audio
    processor = audio.AudioProcessor(
        output_audio_features_folder=str(tmp_path),
        output_transcripts_folder=str(tmp_path),
    )

    assert processor.enable_speaker_diarization is False


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
    assert (tmp_path / "clip.txt").read_text(encoding="utf-8") == "hello world"
    srt_text = (tmp_path / "clip.srt").read_text(encoding="utf-8")
    assert "hello world" in srt_text


def test_plain_transcript_groups_word_timestamp_chunks(monkeypatch, import_audio, tmp_path):
    audio = import_audio
    processor = _make_processor(audio, tmp_path)
    result = {
        "text": "hello world. next phrase",
        "chunks": [
            {"text": "hello", "timestamp": (0.0, 0.2)},
            {"text": "world.", "timestamp": (0.2, 0.4)},
            {"text": "next", "timestamp": (0.5, 0.7)},
            {"text": "phrase", "timestamp": (0.7, 0.9)},
        ],
    }
    _stub_transcription(processor, audio, monkeypatch, result)
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")

    processor.extract_transcripts_batch([str(clip)], word_timestamps=True)

    assert (tmp_path / "clip.txt").read_text(encoding="utf-8") == "hello world.\nnext phrase"


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
    assert (tmp_path / "clip.srt").exists()


def test_extract_transcripts_batch_reports_mixed_file_outcomes(monkeypatch, import_audio, tmp_path):
    audio = import_audio
    processor = _make_processor(audio, tmp_path)
    _stub_transcription(processor, audio, monkeypatch, {"text": "saved", "chunks": []})

    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    good.write_bytes(b"RIFF")
    bad.write_bytes(b"RIFF")
    original_pipe = processor.whisper_pipe

    def fake_pipe(*args, **kwargs):
        if not hasattr(fake_pipe, "called"):
            fake_pipe.called = True
            return original_pipe(*args, **kwargs)
        raise RuntimeError("decode failed")

    processor.whisper_pipe = fake_pipe
    outcome = processor.extract_transcripts_batch([str(good), str(bad)])

    assert outcome == {
        "succeeded": [str(good)],
        "failed": [(str(bad), "decode failed")],
        "cancelled": False,
    }
    assert (tmp_path / "good.txt").exists()
    assert not (tmp_path / "bad.txt").exists()


def test_clear_whisper_model_releases_large_objects(import_audio, tmp_path):
    audio = import_audio
    processor = _make_processor(audio, tmp_path)
    processor.whisper_pipe = object()
    processor.whisper_processor = object()
    processor.whisper_model = object()
    processor.whisper_result = {"chunks": [{"text": "large"}]}

    processor._clear_whisper_model()

    assert processor.whisper_pipe is None
    assert processor.whisper_processor is None
    assert processor.whisper_model is None
    assert processor.whisper_result is None


def test_preload_speaker_diarizer_validates_then_releases_model(import_audio, tmp_path):
    audio = import_audio
    processor = audio.AudioProcessor(
        output_audio_features_folder=str(tmp_path),
        output_transcripts_folder=str(tmp_path),
        enable_speaker_diarization=True,
        auth_token="token",
    )

    class FakeDiarizer:
        def __init__(self):
            self.loaded = False

        def _load_diarization_model(self):
            self.loaded = True

        def offload_model(self):
            pass

    fake = FakeDiarizer()
    processor.speaker_diarizer = fake

    processor.preload_speaker_diarizer()

    assert fake.loaded is True
    assert processor.speaker_diarizer is None


def test_speaker_diarization_uses_cpu_on_mps(import_audio, tmp_path):
    audio = import_audio
    processor = audio.AudioProcessor(
        output_audio_features_folder=str(tmp_path),
        output_transcripts_folder=str(tmp_path),
        enable_speaker_diarization=True,
    )
    processor.device = "mps"

    assert processor._speaker_diarization_device() == "cpu"


def test_whisper_batch_size_is_one_when_diarization_enabled(monkeypatch, import_audio, tmp_path):
    audio = import_audio
    processor = audio.AudioProcessor(
        output_audio_features_folder=str(tmp_path),
        output_transcripts_folder=str(tmp_path),
        enable_speaker_diarization=True,
    )
    processor.device = "cpu"
    processor.torch_dtype = "float32"

    class FakeModel:
        device = type("Device", (), {"type": "cpu"})()

        def to(self, device):
            return self

    fake_processor = type(
        "FakeProcessor",
        (),
        {
            "tokenizer": object(),
            "feature_extractor": object(),
        },
    )()
    captured = {}

    monkeypatch.setattr(audio.AutoModelForSpeechSeq2Seq, "from_pretrained", lambda *a, **k: FakeModel())
    monkeypatch.setattr(audio.AutoProcessor, "from_pretrained", lambda *a, **k: fake_processor)
    monkeypatch.setattr(audio, "pipeline", lambda *a, **k: captured.update(k) or object())

    processor._load_whisper_model()

    assert captured["batch_size"] == 1


def test_extract_transcripts_batch_diarization_saves_each_file_and_clears_whisper(monkeypatch, import_audio, tmp_path):
    audio = import_audio
    processor = audio.AudioProcessor(
        output_audio_features_folder=str(tmp_path),
        output_transcripts_folder=str(tmp_path),
        status_callback=None,
        enable_speaker_diarization=True,
        auth_token="token",
    )
    monkeypatch.setattr(
        audio,
        "_whisper_pipeline_audio_input",
        lambda path: {"array": np.zeros(4, dtype=np.float32), "sampling_rate": 16000},
    )

    pipe_calls = []
    previous_file_saved = []

    def fake_load_whisper():
        idx = len(pipe_calls)

        def fake_pipe(*args, **kwargs):
            pipe_calls.append(idx)
            return {
                "text": f"hello file {idx}",
                "chunks": [{"text": f"hello file {idx}", "timestamp": (0.0, 1.0)}],
            }

        processor.whisper_model = object()
        processor.whisper_processor = object()
        processor.whisper_pipe = fake_pipe

    class FakeDiarizer:
        def diarize_speakers(self, path):
            index = len(previous_file_saved)
            if index > 0:
                previous_file_saved.append((tmp_path / f"clip{index - 1}.txt").exists())
            else:
                previous_file_saved.append(False)
            assert processor.whisper_pipe is None
            assert processor.whisper_model is None
            return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

        def offload_model(self):
            pass

    monkeypatch.setattr(processor, "_load_whisper_model", fake_load_whisper)
    monkeypatch.setattr(processor, "_load_speaker_diarizer", lambda *a, **k: setattr(processor, "speaker_diarizer", FakeDiarizer()))

    files = []
    for i in range(2):
        wav = tmp_path / f"clip{i}.wav"
        wav.write_bytes(b"RIFF")
        files.append(str(wav))

    processor.extract_transcripts_batch(files)

    assert pipe_calls == [0, 1]
    assert previous_file_saved == [False, True]
    assert processor.whisper_pipe is None
    assert processor.whisper_model is None
    assert "SPEAKER_00:" in (tmp_path / "clip0.txt").read_text(encoding="utf-8")
    assert "SPEAKER_00:" in (tmp_path / "clip0.srt").read_text(encoding="utf-8")
    assert (tmp_path / "clip1.txt").exists()


def test_diarized_transcripts_report_mixed_file_outcomes(monkeypatch, import_audio, tmp_path):
    audio = import_audio
    processor = audio.AudioProcessor(
        output_audio_features_folder=str(tmp_path),
        output_transcripts_folder=str(tmp_path),
        status_callback=None,
        enable_speaker_diarization=True,
        auth_token="token",
    )
    monkeypatch.setattr(
        audio,
        "_whisper_pipeline_audio_input",
        lambda path: {"array": np.zeros(4, dtype=np.float32), "sampling_rate": 16000},
    )
    calls = []

    def fake_load_whisper():
        def fake_pipe(*args, **kwargs):
            calls.append(None)
            if len(calls) == 2:
                raise RuntimeError("model failed")
            return {"text": "saved", "chunks": [{"text": "saved", "timestamp": (0.0, 1.0)}]}

        processor.whisper_model = object()
        processor.whisper_processor = object()
        processor.whisper_pipe = fake_pipe

    class FakeDiarizer:
        def diarize_speakers(self, path):
            return []

        def offload_model(self):
            pass

    monkeypatch.setattr(processor, "_load_whisper_model", fake_load_whisper)
    monkeypatch.setattr(
        processor,
        "_load_speaker_diarizer",
        lambda *a, **k: setattr(processor, "speaker_diarizer", FakeDiarizer()),
    )
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    good.write_bytes(b"RIFF")
    bad.write_bytes(b"RIFF")

    outcome = processor.extract_transcripts_batch([str(good), str(bad)])

    assert outcome["succeeded"] == [str(good)]
    assert outcome["failed"] == [(str(bad), "model failed")]
    assert outcome["cancelled"] is False


def test_extract_transcripts_batch_diarization_failure_writes_plain_outputs(monkeypatch, import_audio, tmp_path):
    audio = import_audio
    processor = audio.AudioProcessor(
        output_audio_features_folder=str(tmp_path),
        output_transcripts_folder=str(tmp_path),
        status_callback=None,
        enable_speaker_diarization=True,
        auth_token="token",
    )
    monkeypatch.setattr(
        audio,
        "_whisper_pipeline_audio_input",
        lambda path: {"array": np.zeros(4, dtype=np.float32), "sampling_rate": 16000},
    )

    def fake_load_whisper():
        processor.whisper_model = object()
        processor.whisper_processor = object()
        processor.whisper_pipe = lambda *a, **k: {
            "text": "plain words",
            "chunks": [{"text": "plain words", "timestamp": (0.0, 1.0)}],
        }

    class FailingDiarizer:
        def diarize_speakers(self, path):
            raise RuntimeError("diarization boom")

        def offload_model(self):
            pass

    monkeypatch.setattr(processor, "_load_whisper_model", fake_load_whisper)
    monkeypatch.setattr(processor, "_load_speaker_diarizer", lambda *a, **k: setattr(processor, "speaker_diarizer", FailingDiarizer()))

    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"RIFF")

    processor.extract_transcripts_batch([str(clip)])

    assert (tmp_path / "clip.txt").read_text(encoding="utf-8") == "plain words"
    assert "plain words" in (tmp_path / "clip.srt").read_text(encoding="utf-8")


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


def test_align_features_rejects_empty_word_timestamps(import_audio, tmp_path):
    audio = import_audio
    processor = _make_processor(audio, tmp_path)
    feature_csv = tmp_path / "clip.csv"
    pd.DataFrame({"Timestamp_Seconds": [0.0], "f0": [1.0]}).to_csv(feature_csv, index=False)
    words_json = tmp_path / "clip_words.json"
    words_json.write_text(json.dumps({"chunks": []}), encoding="utf-8")

    try:
        processor.align_features(str(feature_csv), str(words_json), str(tmp_path / "aligned.csv"))
    except ValueError as exc:
        assert "No usable word timestamps" in str(exc)
    else:
        raise AssertionError("Expected empty word timestamps to fail alignment")


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

    outcome = processor.extract_audio_features_batch(files, cancel_check=cancel_check)
    assert len(processed) == 1
    assert outcome["cancelled"] is True


def test_extract_audio_features_batch_reports_per_file_failure(import_audio, tmp_path, monkeypatch):
    audio = import_audio
    processor = _make_processor(audio, tmp_path)

    def fake_extract(path, progress_callback=None):
        if path.endswith("bad.wav"):
            raise RuntimeError("decode failed")
        return path + ".csv"

    monkeypatch.setattr(processor, "extract_audio_features", fake_extract)
    outcome = processor.extract_audio_features_batch(["good.wav", "bad.wav"])

    assert outcome["succeeded"] == ["good.wav"]
    assert outcome["failed"] == [("bad.wav", "decode failed")]


def test_align_features_batch_reports_per_file_failure(import_audio, tmp_path, monkeypatch):
    audio = import_audio
    processor = _make_processor(audio, tmp_path)
    monkeypatch.setattr(processor, "align_features", lambda *args: (_ for _ in ()).throw(RuntimeError("bad data")))

    outcome = processor.align_features_batch([("features.csv", "words.json", "aligned.csv")])

    assert outcome["succeeded"] == []
    assert outcome["failed"] == [("aligned.csv", "bad data")]


def test_align_features_batch_requires_written_output(import_audio, tmp_path, monkeypatch):
    audio = import_audio
    processor = _make_processor(audio, tmp_path)
    out_csv = tmp_path / "missing.csv"
    monkeypatch.setattr(processor, "align_features", lambda *args: str(out_csv))

    outcome = processor.align_features_batch([("features.csv", "words.json", str(out_csv))])

    assert outcome["succeeded"] == []
    assert outcome["failed"] == [(str(out_csv), "Alignment did not write its output CSV")]


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


def test_align_segments_with_speakers_sorts_out_of_order_chunks(import_audio):
    audio = import_audio
    processor = audio.AudioProcessor.__new__(audio.AudioProcessor)

    # Whisper can emit chunks out of chronological order on long audio.
    whisper_segments = [
        {"start": 10.0, "end": 11.0, "text": "second"},
        {"start": 2.0, "end": 3.0, "text": "first"},
        {"start": 20.0, "end": 21.0, "text": "third"},
    ]
    speaker_segments = [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 9.0, "end": 12.0, "speaker": "SPEAKER_01"},
        {"start": 19.0, "end": 22.0, "speaker": "SPEAKER_02"},
    ]

    out = processor._align_segments_with_speakers(whisper_segments, speaker_segments)
    lines = out.splitlines()

    assert lines[0].endswith("first")
    assert lines[1].endswith("second")
    assert lines[2].endswith("third")
