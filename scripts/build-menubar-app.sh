#!/bin/sh
# Wrap the telemak-menubar SwiftPM executable into a proper .app bundle
# that can install the bundled Telemak runtime on first launch.
#
# Output: dist/Telemak.app
# Install: drag to /Applications, optionally add to Login Items.
#
set -eu

CONFIGURATION="${1:-Release}"
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$ROOT"

BINARY="$ROOT/.xcbuild/Build/Products/$CONFIGURATION/telemak-menubar"
if [ ! -x "$BINARY" ]; then
  echo "x binary not found at $BINARY"
  echo "  run: ./scripts/build.sh $CONFIGURATION"
  exit 1
fi
CLI="$ROOT/.xcbuild/Build/Products/$CONFIGURATION/telemak"
if [ ! -x "$CLI" ]; then
  echo "x CLI binary not found at $CLI"
  echo "  run: ./scripts/build.sh $CONFIGURATION"
  exit 1
fi
if [ ! -d "$ROOT/.xcbuild/Build/Products/$CONFIGURATION/mlx-swift_Cmlx.bundle" ]; then
  echo "x MLX Metal bundle not found at $ROOT/.xcbuild/Build/Products/$CONFIGURATION/mlx-swift_Cmlx.bundle"
  echo "  run: ./scripts/build.sh $CONFIGURATION"
  exit 1
fi

APP="$ROOT/dist/Telemak.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp "$BINARY" "$APP/Contents/MacOS/Telemak"
cp "$CLI" "$APP/Contents/Resources/telemak"
cp "$BINARY" "$APP/Contents/Resources/telemak-menubar"
find "$ROOT/.xcbuild/Build/Products/$CONFIGURATION" -maxdepth 1 -name '*.bundle' -type d | while IFS= read -r bundle; do
  cp -R "$bundle" "$APP/Contents/Resources/$(basename "$bundle")"
done
chmod 755 "$APP/Contents/MacOS/Telemak" "$APP/Contents/Resources/telemak" "$APP/Contents/Resources/telemak-menubar"

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
    <string>Telemak</string>
    <key>CFBundleExecutable</key>
    <string>Telemak</string>
    <key>CFBundleIdentifier</key>
    <string>eu.odyssai.telemak</string>
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
    <string>OdyssAI - Apache 2.0.</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

cat > "$APP/Contents/Resources/uninstall.sh" <<'EOF'
#!/bin/sh
set -eu

SERVER_LABEL="eu.odyssai.telemak"
MENUBAR_LABEL="eu.odyssai.telemak.menubar"
UID_VALUE="$(id -u)"
HOME_DIR="$HOME"

launchctl bootout "gui/$UID_VALUE/$SERVER_LABEL" 2>/dev/null || true
launchctl bootout "gui/$UID_VALUE/$MENUBAR_LABEL" 2>/dev/null || true
rm -f "$HOME_DIR/Library/LaunchAgents/$SERVER_LABEL.plist"
rm -f "$HOME_DIR/Library/LaunchAgents/$MENUBAR_LABEL.plist"

printf "Remove ~/telemak runtime directory too? [y/N] "
read answer
case "$answer" in
  y|Y|yes|YES)
    rm -rf "$HOME_DIR/telemak"
    echo "Removed ~/telemak"
    ;;
  *)
    echo "Kept ~/telemak"
    ;;
esac

echo "Telemak LaunchAgents removed."
EOF
chmod 755 "$APP/Contents/Resources/uninstall.sh"

if [ "$CONFIGURATION" = "Release" ]; then
  IDENTITY="${CODESIGN_IDENTITY:-Telemak Developer (Odyssai-eu)}"
  if security find-identity -v -p codesigning 2>/dev/null | grep -q "$IDENTITY"; then
    echo "signing Telemak.app with identity '$IDENTITY'"
    codesign --sign "$IDENTITY" --force --options runtime "$APP/Contents/Resources/telemak"
    codesign --sign "$IDENTITY" --force --options runtime "$APP/Contents/Resources/telemak-menubar"
    codesign --sign "$IDENTITY" --force --options runtime --deep "$APP"
  else
    echo "warning: codesign identity '$IDENTITY' not found; app bundle remains ad-hoc/unsigned"
  fi
fi

# Make sure macOS picks up the new bundle (clear the LaunchServices cache
# entry for the path; harmless if the cache wasn't aware of it yet).
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP" 2>/dev/null || true

echo "built $APP"
