#!/bin/sh
# Build a signed Telemak.app and package it in a drag-to-Applications DMG.
# For a Developer ID build, also notarize + staple both the .app and the
# DMG so Gatekeeper accepts a double-click install on any Mac.

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
"$ROOT/scripts/build-menubar-app.sh" "$CONFIGURATION"

APP="$DIST/Telemak.app"
STAGING="$DIST/dmg-staging"
DMG="$DIST/Telemak-$VERSION.dmg"
VOLNAME="Telemak $VERSION"

IDENTITY="${CODESIGN_IDENTITY:-Developer ID Application: Dupont Sophie (U2YXX868N2)}"
NOTARY_PROFILE="${NOTARY_PROFILE:-telemak-notary}"

# Notarization only applies to a real Developer ID signature. Self-signed
# node builds (CODESIGN_IDENTITY="Telemak Developer …") can't be notarized
# and don't need it — their cert is trust-anchored on the nodes — so gate
# on both the identity kind and the presence of the stored notary profile.
can_notarize() {
  case "$IDENTITY" in
    "Developer ID"*) ;;
    *) return 1 ;;
  esac
  xcrun notarytool history --keychain-profile "$NOTARY_PROFILE" >/dev/null 2>&1
}

# 1. Notarize + staple the .app itself — this is the artifact that ends up
#    in /Applications, so it must carry its own ticket for an offline launch.
if can_notarize; then
  echo "↳ notarizing $APP"
  ditto -c -k --keepParent "$APP" "$DIST/Telemak-app.zip"
  xcrun notarytool submit "$DIST/Telemak-app.zip" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$APP"
  rm -f "$DIST/Telemak-app.zip"
else
  echo "↳ skipping app notarization (self-signed build or no '$NOTARY_PROFILE' profile)"
fi

# 2. Build the DMG from the (now stapled) app.
rm -rf "$STAGING" "$DMG"
mkdir -p "$STAGING"
cp -R "$APP" "$STAGING/Telemak.app"
ln -s /Applications "$STAGING/Applications"

hdiutil create \
  -volname "$VOLNAME" \
  -srcfolder "$STAGING" \
  -ov \
  -format UDZO \
  "$DMG"

if security find-identity -v -p codesigning 2>/dev/null | grep -q "$IDENTITY"; then
  codesign --sign "$IDENTITY" --force --timestamp "$DMG"
fi

# 3. Notarize + staple the DMG itself, then verify Gatekeeper accepts it.
if can_notarize; then
  echo "↳ notarizing $DMG"
  xcrun notarytool submit "$DMG" --keychain-profile "$NOTARY_PROFILE" --wait
  xcrun stapler staple "$DMG"
  xcrun stapler validate "$DMG" || true
  spctl -a -t open --context context:primary-signature -vv "$DMG" || true
fi

rm -rf "$STAGING"
echo "built $DMG"
