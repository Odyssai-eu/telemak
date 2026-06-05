#!/bin/sh
# Wrap the Telemak Monitor SwiftPM executable into a standalone macOS app.
#
# Output: dist/Telemak Monitor.app

set -eu

CONFIGURATION="${1:-Release}"
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

BINARY="$ROOT/.xcbuild/Build/Products/$CONFIGURATION/telemak-monitor"
if [ ! -x "$BINARY" ]; then
  echo "x monitor binary not found at $BINARY"
  echo "  run: ./scripts/build.sh $CONFIGURATION"
  exit 1
fi

APP="$ROOT/dist/Telemak Monitor.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$BINARY" "$APP/Contents/MacOS/Telemak Monitor"
chmod 755 "$APP/Contents/MacOS/Telemak Monitor"

VERSION="$(sed -n 's/^public let telemakVersion = "\(.*\)"/\1/p' "$ROOT/Sources/TelemakVersion/Version.swift")"
if [ -z "$VERSION" ]; then
  VERSION="0.0.0"
fi

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>Telemak Monitor</string>
    <key>CFBundleExecutable</key>
    <string>Telemak Monitor</string>
    <key>CFBundleIdentifier</key>
    <string>eu.odyssai.telemak.monitor</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Telemak Monitor</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>NSHumanReadableCopyright</key>
    <string>OdyssAI - Apache 2.0.</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

if [ "$CONFIGURATION" = "Release" ]; then
  IDENTITY="${CODESIGN_IDENTITY:-Telemak Developer (Odyssai-eu)}"
  if security find-identity -v -p codesigning 2>/dev/null | grep -q "$IDENTITY"; then
    echo "signing Telemak Monitor.app with identity '$IDENTITY'"
    codesign --sign "$IDENTITY" --force --options runtime "$APP"
  else
    echo "warning: codesign identity '$IDENTITY' not found; app bundle remains ad-hoc/unsigned"
  fi
fi

/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" 2>/dev/null || true

echo "built $APP"
