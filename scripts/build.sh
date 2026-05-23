#!/bin/sh
# Build Telemak via xcodebuild (required — see telemak-build-system memory).
# `swift build` doesn't compile the Metal kernels mlx-swift needs at runtime.

set -eu

CONFIGURATION="${1:-Debug}"
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
DERIVED="${ROOT}/.xcbuild"

cd "$ROOT"

xcodebuild \
  -scheme Telemak \
  -configuration "$CONFIGURATION" \
  -derivedDataPath "$DERIVED" \
  -destination 'platform=macOS' \
  -skipMacroValidation \
  build \
  | grep -vE '^(2026|note: |Note:|\s+[A-Z][a-z]+ \(in target)' \
  || true

BINARY="$DERIVED/Build/Products/$CONFIGURATION/telemak"
if [ -x "$BINARY" ]; then
  echo "✓ built $BINARY"
else
  echo "✗ build failed (no binary at $BINARY)" >&2
  exit 1
fi
