# Telemak Release

## Build DMG

Use `xcodebuild` through the repo scripts. Do not use `swift build`.

```bash
./scripts/package-dmg.sh Release
```

Output:

```text
dist/Telemak-X.Y.Z.dmg
```

The package script:

- builds `telemak`, `telemak-menubar`, and `mlx-swift_Cmlx.bundle`
- builds `dist/Telemak.app`
- embeds the CLI runtime in `Telemak.app/Contents/Resources/`
- signs binaries and the app when `Telemak Developer (Odyssai-eu)` is available
- creates a drag-to-Applications DMG

## Signing

Default identity:

```bash
Telemak Developer (Odyssai-eu)
```

Override:

```bash
CODESIGN_IDENTITY="Other Identity" ./scripts/package-dmg.sh Release
```

Verify:

```bash
codesign -dv --verbose=4 dist/Telemak.app 2>&1 | grep -E "Signature|Authority|TeamIdentifier"
codesign -dv --verbose=4 dist/Telemak-*.dmg 2>&1 | grep -E "Signature|Authority|TeamIdentifier"
```

If `codesign` returns `errSecInternalComponent` over SSH, rebuild from an
interactive macOS session on the signing Mac. The certificate private key is
visible to `security find-identity` over SSH but can still be denied by
Keychain access control.

## Smoke

On a clean target Mac:

```bash
open dist/Telemak-*.dmg
```

Drag `Telemak.app` to `/Applications`, open it, run the installer, then:

```bash
curl http://localhost:8003/health
launchctl print gui/$(id -u)/eu.odyssai.telemak
launchctl print gui/$(id -u)/eu.odyssai.telemak.menubar
```
