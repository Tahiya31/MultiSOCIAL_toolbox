# Deployment Guide

This document is for maintainers publishing official MultiSOCIAL Toolbox builds from `upstream/main`.

## Release model

- Normal PR merges to `upstream/main` do **not** publish a user release by themselves.
- Official desktop builds are published from a Git tag that matches the app version.
- GitHub Actions workflow: `.github/workflows/release.yml`
- Official user downloads are the assets attached to **GitHub Releases**.

## What gets built

Each tagged release builds four artifacts:

- macOS `Standard`
- macOS `Complete`
- Windows `Standard`
- Windows `Complete`

Profiles:

- `Standard`: base toolbox
- `Complete`: base toolbox plus diarization support

## Before releasing

1. Make sure the PR is merged into `upstream/main`.
2. Confirm the packaging workflow is passing on the merged code.
3. Decide the next version number using the existing version format in `pyproject.toml`.

## How to publish an official release

1. Checkout the upstream main branch:

```bash
git checkout main
git pull upstream main
```

2. Update the version in `pyproject.toml`:

```toml
version = "0.2.1"
```

3. Commit the version bump:

```bash
git add pyproject.toml
git commit -m "Bump version to 0.2.1"
```

4. Tag that exact commit:

```bash
git tag v0.2.1
```

5. Push the branch and tag to the upstream repository:

```bash
git push upstream main
git push upstream v0.2.1
```

6. GitHub Actions will automatically:
   - build the macOS and Windows packages
   - upload the artifacts
   - publish the GitHub Release assets for that tag

## Important rules

- The Git tag and `pyproject.toml` version must match.
  - Example: `version = "0.2.1"` must be released as tag `v0.2.1`
- Do not publish official user builds from a fork tag.
- Use fork/branch tags only for pre-release testing.

## Testing workflow before upstream release

Recommended flow for contributors:

1. Push packaging changes to a fork branch.
2. Create a temporary test tag on the fork, for example:

```bash
git tag v0.2.0-test7
git push origin v0.2.0-test7
```

3. Let GitHub Actions build the test artifacts.
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
