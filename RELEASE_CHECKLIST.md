# Release Checklist

Follow these steps when preparing a new version of **Latent Self**.

## 1. Build binaries
1. Ensure your environment has all dependencies installed.
2. Run `pyinstaller latent_self.spec`.
3. Test the resulting executable in `dist/` on the target platforms.

## 2. Update the changelog
1. Edit `CHANGELOG.md` and move items from **Unreleased** under a new version heading:
   ```
   ## [X.Y.Z] - YYYY-MM-DD
   ```
2. Summarise notable changes.
3. Commit the updated changelog.

## 3. Tag the release
1. Create an annotated tag:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```
2. Optionally draft a GitHub release and upload the built binaries.

