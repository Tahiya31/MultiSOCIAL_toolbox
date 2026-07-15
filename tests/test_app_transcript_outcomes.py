from __future__ import annotations

from pathlib import Path


def test_transcript_worker_reports_partial_outcomes():
    app_source = (Path(__file__).resolve().parents[1] / "src" / "app.py").read_text(encoding="utf-8")

    assert 'failed = outcome["failed"]' in app_source
    assert 'succeeded = outcome["succeeded"]' in app_source
    assert '"Transcription Warning" if succeeded else "Transcription Failed"' in app_source
    assert "Transcripts were created for {len(succeeded)} of {len(audio_files)} file(s)." in app_source
