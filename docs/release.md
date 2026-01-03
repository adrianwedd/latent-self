# Release Process

This document outlines how to publish a new version of **Latent Self** on GitHub.

1. Update `CHANGELOG.md` and commit the changes.
2. Tag the commit with the version number, e.g. `git tag -a v0.4.1 -m "v0.4.1"`.
3. Build the binaries with PyInstaller on each supported OS.
4. Run `deploy/install_kiosk.sh` on a test machine to verify installation.
5. Use the GitHub CLI script to create the release and upload the artifacts:
   ```bash
   scripts/github_release.sh v0.4.1
   ```
6. Inspect the release on GitHub and publish it.
