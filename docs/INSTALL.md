# Telemak Install

## Install

1. Open `Telemak-X.Y.Z.dmg`.
2. Drag `Telemak.app` to `/Applications`.
3. Open `/Applications/Telemak.app`.
4. Choose the models directory. Default is `/Volumes/models/odysseus` when present, otherwise `~/Telemak-Models`.
5. Click `Install`.

The app copies the runtime to `~/telemak/Release/`, writes the LaunchAgents, starts the HTTP server on `:8003`, and starts the menu bar monitor.

## macOS Permissions

If models are on `/Volumes/...`, allow Telemak in:

- System Settings -> Privacy & Security -> Full Disk Access
- System Settings -> Privacy & Security -> Files and Folders -> Removable Volumes

Then use the menu bar icon to restart Telemak.

## Verify

```bash
curl http://localhost:8003/health
```

Expected: JSON with `"status":"ok"`.

## Uninstall

```bash
/Applications/Telemak.app/Contents/Resources/uninstall.sh
```

The script stops LaunchAgents, removes plists, and asks before removing `~/telemak`.
