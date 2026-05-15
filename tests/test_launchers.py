from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_unix_launcher_supports_python_fallbacks_and_offline_skip():
    script = (REPO_ROOT / "run_app.sh").read_text(encoding="utf-8")

    assert 'command -v "python${DESIRED_PYTHON}"' in script
    assert "elif command -v python3" in script
    assert "elif command -v python " in script
    assert 'if [ "$NEEDS_INSTALL" -eq 1 ]; then' in script
    assert script.index('if [ "$NEEDS_INSTALL" -eq 1 ]; then') < script.index('python -m pip install --upgrade pip')


def test_windows_launcher_keeps_network_install_inside_stamp_check():
    script = (REPO_ROOT / "run_app.bat").read_text(encoding="utf-8")

    assert 'if "%NEEDS_INSTALL%"=="1" (' in script
    assert "python -m pip install --upgrade pip" in script
    assert script.index('if "%NEEDS_INSTALL%"=="1" (') < script.index("python -m pip install --upgrade pip")
    assert 'py -%DESIRED_PYTHON% --version' in script
