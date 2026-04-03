#!/usr/bin/env bash
set -euo pipefail

# Determine version tag: use git tag if on an exact tag, otherwise vYYYYMMDD-<hash>
if TAG=$(git describe --tags --exact-match 2>/dev/null); then
    VERSION="$TAG"
else
    SHORT_HASH=$(git rev-parse --short HEAD)
    VERSION="v$(date +%Y%m%d)-${SHORT_HASH}"
fi

IMAGE="qq-bot:${VERSION}"

echo "==> Building ${IMAGE}"
docker compose build --build-arg GIT_COMMIT="$(git rev-parse --short HEAD)" bot

echo "==> Tagging ${IMAGE}"
docker tag qq-bot "${IMAGE}"

echo "==> Deploying ${IMAGE}"
GIT_COMMIT="$(git rev-parse --short HEAD)" docker compose up bot -d --build

echo "==> Done: ${IMAGE}"
