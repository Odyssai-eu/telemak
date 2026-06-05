#!/bin/sh
# Build a signed Telemak Monitor.app and package it in a drag-to-Applications DMG.

set -eu

CONFIGURATION="${1:-Release}"
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
DIST="$ROOT/dist"
VERSION="$(sed -n 's/^public let telemakVersion = "\(.*\)"/\1/p' "$ROOT/Sources/TelemakVersion/Version.swift")"
if [ -z "$VERSION" ]; then
  VERSION="0.0.0"
fi

cd "$ROOT"

if [ "${TELEMAK_PACKAGE_SKIP_BUILD:-0}" != "1" ]; then
  "$ROOT/scripts/build.sh" "$CONFIGURATION"
fi
"$ROOT/scripts/build-monitor-app.sh" "$CONFIGURATION"

APP="$DIST/Telemak Monitor.app"
STAGING="$DIST/monitor-dmg-staging"
DMG="$DIST/Telemak-Monitor-$VERSION.dmg"
VOLNAME="Telemak Monitor $VERSION"

rm -rf "$STAGING" "$DMG"
mkdir -p "$STAGING"
cp -R "$APP" "$STAGING/Telemak Monitor.app"
ln -s /Applications "$STAGING/Applications"

hdiutil create \
  -volname "$VOLNAME" \
  -srcfolder "$STAGING" \
  -ov \
  -format UDZO \
  "$DMG"

IDENTITY="${CODESIGN_IDENTITY:-Telemak Developer (Odyssai-eu)}"
if security find-identity -v -p codesigning 2>/dev/null | grep -q "$IDENTITY"; then
  codesign --sign "$IDENTITY" --force "$DMG"
fi

rm -rf "$STAGING"
echo "built $DMG"
