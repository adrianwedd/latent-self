# Release Checklist

Follow these steps when preparing a new version of **Latent Self**.

## 1. Update the changelog
1. Edit `CHANGELOG.md` and move items from **Unreleased** under a new version heading:
   ```
   ## [X.Y.Z] - YYYY-MM-DD
   ```
2. Summarise notable changes.
3. Commit the updated changelog.

## 2. Tag the release
1. Create an annotated tag:
   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

## 3. Build binaries
1. Ensure your environment has all dependencies installed.
2. Run `pyinstaller latent_self.spec`.
3. Test the resulting executable in `dist/` on the target platforms.

## 4. Verify the installer script
1. Run `deploy/install_kiosk.sh` on a test system using the newly built binary.
2. Confirm the application installs to `/opt/latent-self` and the service starts without errors.
3. Check the script for clean error handling and idempotent behaviour.

## 5. Create a GitHub release
1. On GitHub, navigate to **Releases** and click **Draft a new release**, or run `scripts/github_release.sh vX.Y.Z` if you have the `gh` CLI installed.
2. Choose the newly pushed tag (`vX.Y.Z`) and set the release title to the same version.
3. Copy the changelog notes for this version into the description field.
4. Upload the PyInstaller artifacts from `dist/` (e.g., `LatentSelf.exe` for Windows, `.dmg` for macOS or the `.tar.gz` for Linux).
5. Publish the release.

