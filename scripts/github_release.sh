#!/usr/bin/env bash
# Create a GitHub release and upload binaries using the gh CLI.
# Usage: ./scripts/github_release.sh vX.Y.Z

set -euo pipefail

if ! command -v gh >/dev/null; then
    echo "gh CLI is required: https://cli.github.com/" >&2
    exit 1
fi

TAG="${1:?Tag required, e.g. v0.4.1}"
NOTES=$(mktemp)

# Extract release notes from CHANGELOG.md
awk "/## \[$TAG\]/,/## \[/" CHANGELOG.md | sed '$d' > "$NOTES"

# Gather built artifacts from dist/
shopt -s nullglob
FILES=(dist/*.exe dist/*.tar.gz dist/*.dmg)

# Create the release and upload assets
gh release create "$TAG" "${FILES[@]}" \
    --title "$TAG" \
    --notes-file "$NOTES"

rm "$NOTES"
