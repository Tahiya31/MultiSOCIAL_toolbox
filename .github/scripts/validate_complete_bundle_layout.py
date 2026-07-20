import os
import sys
from pathlib import Path

def main():
    profile = os.environ.get("MULTISOCIAL_BUILD_PROFILE", "").lower()
    if profile != "complete":
        print("Skipping complete bundle layout validation: profile is not 'complete'")
        sys.exit(0)

    runner_os = os.environ.get("RUNNER_OS", "macOS")
    
    if runner_os == "Windows":
        app_name = "MultiSOCIAL-Complete"
        root = Path("dist") / app_name
    else:
        app_name = "MultiSOCIAL-Complete.app"
        root = Path("dist") / app_name

    print(f"Validating complete bundle layout in: {root}")

    if not root.exists():
        print(f"ERROR: Expected dist folder missing: {root}", file=sys.stderr)
        sys.exit(1)

    # Convert all paths to a flat list of posix strings to avoid rglob quirks on different OSes
    all_files = [p.as_posix() for p in root.rglob("*") if p.is_file()]

    # 1. Check for lightning_fabric version.info
    if not any("lightning_fabric" in f and "version.info" in f for f in all_files):
        print(f"ERROR: Missing lightning_fabric/version.info in packaged app. Did collect_data_files('lightning_fabric') fail?", file=sys.stderr)
        sys.exit(1)
    
    # 2. Check for pyannote.audio metadata
    if not any("pyannote.audio" in f and ".dist-info/METADATA" in f for f in all_files):
        print(f"ERROR: Missing pyannote.audio METADATA in packaged app. Did copy_metadata('pyannote.audio') fail?", file=sys.stderr)
        sys.exit(1)

    # 3. Check for speechbrain metadata
    if not any("speechbrain" in f and ".dist-info/METADATA" in f for f in all_files):
        print(f"ERROR: Missing speechbrain METADATA in packaged app. Did copy_metadata('speechbrain') fail?", file=sys.stderr)
        sys.exit(1)

    # 4. Heavy model is required by multi-person ROI pose and must never be
    # fetched at runtime in a packaged app.
    if not any(f.endswith("mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite") for f in all_files):
        print("ERROR: Missing bundled MediaPipe Heavy pose model.", file=sys.stderr)
        sys.exit(1)

    print("Complete bundle layout validation passed: required metadata and data files are present.")

if __name__ == "__main__":
    main()
