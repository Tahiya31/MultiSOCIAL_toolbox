# Deployment Guide

This document is for maintainers and contributors using the automated MultiSOCIAL Toolbox build pipeline.

## Release model

- Fork tags are for contributor test builds.
- Push a tag like `v0.2.0-test7` in your fork to produce build artifacts in your fork's Actions run.
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
version = "0.2.1"
```

2. Merge that PR into `upstream/main`.

3. GitHub Actions will automatically:
   - read `version = "0.2.1"` from `pyproject.toml`
   - create the upstream GitHub Release `v0.2.1`
   - build the macOS and Windows packages
   - upload the release assets to that upstream release

No manual tag push is required for upstream releases.

## Important rules

- The version in `pyproject.toml` is the source of truth for upstream releases.
- Each merged upstream release version must be new.
  - Example: if `v0.2.1` already exists upstream, another merge with `version = "0.2.1"` will fail the release workflow.
- Do not publish official user builds from a fork tag.
- Use fork tags only for pre-release testing.

## Testing workflow before upstream release

Recommended flow for contributors:

1. Push packaging changes to a fork branch.
2. Create a temporary test tag on the fork, for example:

```bash
git tag v0.2.0-test7
git push origin v0.2.0-test7
```

3. Let GitHub Actions build the test artifacts in your fork.
4. Test all four outputs on real machines.
5. Open the PR only after the fork artifacts behave correctly.

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
