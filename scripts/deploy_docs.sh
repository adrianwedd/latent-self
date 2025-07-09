#!/usr/bin/env bash
# Deploy versioned documentation using mike.
set -euo pipefail

VERSION=$(git describe --tags --abbrev=0)

mike deploy --update-aliases "$VERSION" latest stable
mike set-default stable
