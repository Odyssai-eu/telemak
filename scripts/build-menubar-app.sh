#!/bin/sh
# Wrap the telemak-menubar SwiftPM executable into a proper .app bundle.
#
# Output: dist/Telemak.app
# Install: drag to /Applications, optionally add to Login Items.
#
# Distribution caveat: this .app is NOT codesigned. macOS Gatekeeper will
# show "can't be opened because Apple cannot check it for malicious
# software" on first launch — right-click → Open is the workaround.

set -eu

CONFIGURATION="${1:-Release}"
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

BINARY="$ROOT/.xcbuild/Build/Products/$CONFIGURATION/telemak-menubar"
if [ ! -x "$BINARY" ]; then
  echo "✗ binary not found at $BINARY"
  echo "  run: ./scripts/build.sh $CONFIGURATION"
  exit 1
fi

APP="$ROOT/dist/Telemak.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$BINARY" "$APP/Contents/MacOS/Telemak"

VERSION="0.2.0"

cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>en</string>
    <key>CFBundleDisplayName</key>
    <string>Telemak</string>
    <key>CFBundleExecutable</key>
    <string>Telemak</string>
    <key>CFBundleIdentifier</key>
    <string>eu.odyssai.telemak.menubar</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>Telemak</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>LSMinimumSystemVersion</key>
    <string>14.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSHumanReadableCopyright</key>
    <string>OdyssAI — Apache 2.0.</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Make sure macOS picks up the new bundle (clear the LaunchServices cache
# entry for the path; harmless if the cache wasn't aware of it yet).
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" 2>/dev/null || true

echo "✓ built $APP"
echo ""
echo "Install:"
echo "  mv $APP /Applications/"
echo "  open /Applications/Telemak.app          # first time: right-click → Open if Gatekeeper warns"
echo ""
echo "Optional: add to Login Items via System Settings → General → Login Items"
