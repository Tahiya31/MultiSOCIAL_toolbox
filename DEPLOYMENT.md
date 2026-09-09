# Deployment Guide

This document is for maintainers and contributors using the automated MultiSOCIAL Toolbox build pipeline.

## Release model

- Fork tags are for contributor test builds. Commit the intended changes in the fork, then push a tag such as `v1.2.2-test1`; GitHub Actions starts automatically.
- A fork tag creates **exactly four** downloadable test artifacts: macOS and Windows, each in `Standard` and `Complete` profiles. It never creates an upstream release record or publishes user-facing release assets.
- Merges to `upstream/main` automatically build the official desktop release using the exact version in `pyproject.toml` at merge time.
- GitHub Actions workflow: `.github/workflows/release.yml`
- Official user downloads are the assets attached to **GitHub Releases** on the upstream repository.

## What gets built

Each tagged release builds four artifacts:

- macOS `Standard`
- macOS `Complete`
- Windows `Standard`
- Windows `Complete`

Profiles:

- `Standard`: base toolbox
- `Complete`: base toolbox plus diarization support

## How official upstream releases work

1. Update `pyproject.toml` in the PR to the exact release version you want:

```toml
version = "1.0.0"
```

2. Merge that PR into `upstream/main`.

3. GitHub Actions will automatically:
   - read `version = "1.0.0"` from `pyproject.toml`
   - build the macOS and Windows packages
   - qualify Windows with three clean builds per profile and a Windows 2022 relocation test
   - create the upstream GitHub Release `v1.0.0` only after every qualification job passes
   - upload the release assets to that upstream release

No manual tag push is required for upstream releases.

## Important rules

- The version in `pyproject.toml` is the source of truth for upstream releases.
- Packaged macOS builds stamp `CFBundleShortVersionString` / `CFBundleVersion` from that same value; CI fails if they diverge.
- Each merged upstream release version must be new.
  - Example: if `v1.0.0` already exists upstream, another merge with `version = "1.0.0"` will fail the release workflow.
- Do not publish official user builds from a fork tag.
- Use fork tags only for pre-release testing.

## Bundled models that CI checks

Every packaged build must include:

| Asset | Why |
|-------|-----|
| `assets/yolov5s.pt` | Multi-person person detection (YOLOv5). Frozen builds refuse runtime download. |
| `assets/pose_landmark_heavy.tflite` | MediaPipe Heavy pose model for multi-person ROI pose (`model_complexity=2`). Packaged as `mediapipe/modules/pose_landmark/pose_landmark_heavy.tflite`. |

Post-build checks:

- `validate_complete_bundle_layout.py` asserts the Heavy model is present in Complete-profile layouts.
- Release workflow verifies the Heavy model path on Standard and Complete artifacts.
- Packaged import smoke test runs with `MULTISOCIAL_VERIFY_HEAVY_POSE_ASSET=1` so missing Heavy aborts startup.
- macOS builds assert Info.plist version matches `pyproject.toml`.
- Windows GUI and worker builds use separate hashed locks, virtual environments, PyInstaller processes, work directories, and onedir runtimes. The GUI lock contains no native-analysis packages; the worker lock contains no wxPython and exactly one `cv2` provider (`opencv-contrib-python`).
- Each Windows profile must pass three clean builds. Every build performs ten cold native-worker probes plus OpenSMILE, MediaPipe, YOLO/Torch, pose-embedding, and staged-output checks. The canonical build also runs Whisper ASR.
- The canonical Windows ZIP is downloaded and relocated to a path containing spaces and non-ASCII characters on a Windows 2022 runner, then exercised through both the GUI launcher and worker. These checks do not create additional user-visible artifacts.
- Official Complete releases require the secret-backed diarization E2E. Fork maintainers can opt into the same check with `run_windows_complete_e2e`.

## Windows packaging boundary

- `MultiSOCIAL.spec` is macOS-only.
- `packaging/windows_gui.spec` analyzes only the shared GUI and worker client.
- `packaging/build_windows_embedded_worker.py` creates the private worker runtime from the locked worker environment without wxPython.
- `packaging/assemble_windows.py` copies the finished worker onedir beneath `worker/`; it never merges PyInstaller TOCs.
- Any new Windows GUI import of Torch, Transformers, MediaPipe, OpenCV, OpenSMILE, YOLO, or diarization code fails during `Analysis`, before assembly.

## Testing workflow before upstream release

Recommended flow for contributors:

1. Push packaging changes to a fork branch.
2. Create a temporary test tag on the fork, for example:

```bash
git tag v1.2.2-test1
git push origin v1.2.2-test1
```

3. Let GitHub Actions build the test artifacts in your fork.
4. Confirm Actions passed the Heavy-model and macOS version checks.
5. Test all four outputs on real machines (especially **Enable Multi-person Pose** in packaged builds).
6. Open the PR only after the fork artifacts behave correctly.

## First-run notes for users

- macOS:
  - users may need to remove quarantine on downloaded unsigned `.app` bundles:

```bash
xattr -dr com.apple.quarantine "/path/to/MultiSOCIAL-Standard.app"
```

- Windows:
  - unsigned builds may show a SmartScreen warning
  - users can open **More info** and then **Run anyway**

Signing/notarization can reduce these warnings in future releases, but the current workflow does not require signing to build and publish.
