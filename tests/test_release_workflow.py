from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_release_workflow_supports_main_and_tag_pushes():
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "branches:" in workflow
    assert "- main" in workflow
    assert 'tags:' in workflow
    assert '- "v*"' in workflow


def test_release_workflow_has_upstream_auto_release_guardrails():
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "UPSTREAM_REPOSITORY: Tahiya31/MultiSOCIAL_toolbox" in workflow
    assert "should_publish_release" in workflow
    assert "Derive and validate release metadata" in workflow
    assert "already exists. Bump pyproject.toml before merging to main." in workflow
    assert "needs: [prepare, macos-build, windows-build, windows-client-compat]" in workflow


def test_release_workflow_requires_python_tests_before_packaging():
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "python-tests:" in workflow
    assert "Run full Python test suite" in workflow
    assert "run: python -m pytest" in workflow
    assert workflow.count("needs: [prepare, python-tests]") == 2


def test_release_workflow_runs_packaged_macos_e2e_only_for_fork_tags():
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    runner = (REPO_ROOT / "src" / "macos_packaged_e2e.py").read_text(encoding="utf-8")

    assert "Run fork-tag packaged macOS native E2E" in workflow
    assert "github.repository != env.UPSTREAM_REPOSITORY" in workflow
    assert "MULTISOCIAL_MACOS_PACKAGED_E2E: \"1\"" in workflow
    assert "MULTISOCIAL_MACOS_E2E_WORKSPACE" in workflow
    assert "WHISPER_ATTEMPTS = 3" in runner
    assert "Person fixture hash mismatch" in runner
    assert "Cancellation left committed or staged audio output" in runner


def test_transformers_hook_keeps_required_runtime_quantizers():
    hook = (REPO_ROOT / "hooks" / "hook-transformers.py").read_text(encoding="utf-8")

    assert '"transformers.quantizers"' not in hook


def test_release_workflow_uploads_versioned_artifacts():
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "name: MultiSOCIAL-${{ env.APP_VERSION }}-${{ matrix.profile }}-${{ runner.os }}" in workflow
    assert workflow.count("name: MultiSOCIAL-${{ env.APP_VERSION }}-${{ matrix.profile }}-${{ runner.os }}") == 2
    assert "release-asset-" not in workflow
    assert "Create qualified release and upload four assets" in workflow
    assert "tag_name: ${{ needs.prepare.outputs.release_tag }}" in workflow


def test_windows_release_builds_install_only_from_committed_lock():
    workflow = (REPO_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    standard_lock = (REPO_ROOT / "requirements" / "locks" / "windows-standard-py310.txt").read_text(encoding="utf-8")
    complete_lock = (REPO_ROOT / "requirements" / "locks" / "windows-complete-py310.txt").read_text(encoding="utf-8")
    complete_bootstrap = (REPO_ROOT / "requirements" / "locks" / "windows-complete-bootstrap.txt").read_text(encoding="utf-8")
    gui_lock = (REPO_ROOT / "requirements" / "locks" / "windows-gui-py310.txt").read_text(encoding="utf-8")

    assert "Create physically separate GUI and worker environments" in workflow
    assert "Build static native worker launcher" in workflow
    assert "packaging/build_windows_worker_launcher.py" in workflow
    assert "pip install --require-hashes --no-deps" in workflow
    assert "windows-${MULTISOCIAL_BUILD_PROFILE}-py310.txt" in workflow
    assert "windows-gui-py310.txt" in workflow
    assert "windows-2025" in workflow
    assert "windows-client-compat" in workflow
    assert "windows-2022" in workflow
    assert "Relocate bundle to an installation-like path" in workflow
    assert "Run extracted canonical ZIP through packaged boundary" in workflow
    assert "repetition: [1, 2, 3]" in workflow
    assert "Verify committed Windows packaging inputs" in workflow
    assert "git ls-files --error-unmatch" in workflow
    assert 'YOLO_AUTOINSTALL: "false"' in workflow
    assert 'YOLOv5_AUTOINSTALL: "false"' in workflow
    assert workflow.count("packaging/audit_windows_environment.py") == 3
    assert "Build private embedded Windows worker runtime" in workflow
    assert "packaging/build_windows_embedded_worker.py" in workflow
    assert "packaging/windows_gui.spec" in workflow
    assert "MULTISOCIAL_WHISPER_MODEL_ID: openai/whisper-tiny" in workflow
    assert "MULTISOCIAL_WHISPER_MODEL_REVISION: 169d4a4341b33bc18d8881c4b69c2e104e1cc0af" in workflow
    assert "HF_HUB_DOWNLOAD_TIMEOUT: \"120\"" in workflow
    assert "Restore immutable Whisper E2E cache" in workflow
    assert "--whisper-attempts 3" in workflow
    assert "needs.prepare.outputs.should_publish_release == 'true' || inputs.run_windows_complete_e2e" in workflow
    assert "windows_packaged_e2e.py" in workflow
    assert "Diagnose direct native launcher-to-worker probe" in workflow
    assert "Diagnose packaged GUI-to-worker probe" in workflow
    assert "Measure cold packaged GUI-to-worker MediaPipe startup" in workflow
    assert '"timeout_seconds":300' in workflow
    assert "--preserve-diagnostics" in workflow
    assert "Diagnose relocated native launcher-to-worker probe" in workflow
    assert "--worker \"dist/${app_name}/worker/MultiSOCIAL-Worker-Launcher.exe\"" in workflow
    assert "--timeout 60" in workflow
    assert "Expand-Archive" in workflow
    assert "path: release-artifacts/*-windows.zip" in workflow
    assert "blank.avi" not in workflow
    e2e = (REPO_ROOT / ".github" / "scripts" / "windows_packaged_e2e.py").read_text(encoding="utf-8")
    assert "PERSON_FIXTURE_GIT_BLOB" in e2e
    assert "for attempt in range(1, 11)" in e2e
    assert "cancel_after_seconds=0.1" in e2e
    assert "_assert_success(single_result" in e2e
    assert "_assert_success(multi_result" in e2e
    assert "--hash=sha256:" in standard_lock
    assert "--hash=sha256:" in complete_lock
    assert "--hash=sha256:" in gui_lock
    assert "opencv-contrib-python==4.11.0.86" in standard_lock
    assert "opencv-python==" not in standard_lock
    assert "opencv-python-headless==" not in standard_lock
    assert "wxpython==" not in standard_lock.lower()
    assert "wxpython==4.2.3" in gui_lock.lower()
    assert "torch==" not in gui_lock.lower()
    assert "pyannote.core==5.0.0" in complete_bootstrap
    assert 'destination.glob("*.txt")' in e2e
    assert 'transcripts.glob("*.txt")' in e2e
    assert "--whisper-attempts" in e2e
    assert "Whisper ASR failed after" in e2e
    assert 'glob("*.json")' not in e2e
    assert '"align_features"' in e2e
    assert "Feature alignment did not commit aligned.csv" in e2e

    macos_e2e = (REPO_ROOT / "src" / "macos_packaged_e2e.py").read_text(encoding="utf-8")
    assert "_run_diarization" in macos_e2e
    assert "MULTISOCIAL_CI_HF_TOKEN" in macos_e2e
    assert 'MULTISOCIAL_BUILD_PROFILE") == "complete"' in macos_e2e
    assert "MULTISOCIAL_CI_HF_TOKEN: ${{ secrets.MULTISOCIAL_CI_HF_TOKEN }}" in workflow
