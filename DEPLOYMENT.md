# Deployment Guide

This document is for maintainers and contributors using the automated MultiSOCIAL Toolbox build pipeline.

## Release model

- Fork tags are for contributor test builds.
- Push a tag like `v1.0.0-test1` in your fork to produce build artifacts in your fork's Actions run.
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
   - create the upstream GitHub Release `v1.0.0`
   - build the macOS and Windows packages
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

## Testing workflow before upstream release

Recommended flow for contributors:

1. Push packaging changes to a fork branch.
2. Create a temporary test tag on the fork, for example:

```bash
git tag v1.2.1-test1
git push origin v1.2.1-test1
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
