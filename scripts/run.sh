#!/bin/sh
# Run the built telemak binary. Args after the script are passed through.

set -eu

CONFIGURATION="${TELEMAK_CONFIG:-Debug}"
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
BINARY="$ROOT/.xcbuild/Build/Products/$CONFIGURATION/telemak"

if [ ! -x "$BINARY" ]; then
  echo "binary not found at $BINARY — run ./scripts/build.sh first" >&2
  exit 1
fi

exec "$BINARY" "$@"
