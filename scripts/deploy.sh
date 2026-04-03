#!/usr/bin/env bash
set -euo pipefail

# Determine version tag: use git tag if on an exact tag, otherwise vYYYYMMDD-<hash>
if TAG=$(git describe --tags --exact-match 2>/dev/null); then
    VERSION="$TAG"
else
    SHORT_HASH=$(git rev-parse --short HEAD)
    VERSION="v$(date +%Y%m%d)-${SHORT_HASH}"
fi

COMMIT=$(git rev-parse --short HEAD)

echo "==> Building bot (commit: ${COMMIT})"
GIT_COMMIT="${COMMIT}" docker compose build bot

# Get the image name that compose actually built
BUILT_IMAGE=$(docker compose images bot --format json | python3 -c "import sys,json; print(json.load(sys.stdin)[0]['Repository'])")

echo "==> Tagging ${BUILT_IMAGE} → qq-bot:${VERSION}, qq-bot:latest"
docker tag "${BUILT_IMAGE}" "qq-bot:${VERSION}"
docker tag "${BUILT_IMAGE}" "qq-bot:latest"

echo "==> Deploying"
GIT_COMMIT="${COMMIT}" docker compose up bot -d

echo "==> Done: qq-bot:${VERSION}"
