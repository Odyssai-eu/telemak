#!/bin/sh
# Build Release once, deploy to one canary first, then roll out everywhere.

set -eu

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
# shellcheck source=scripts/telemak-hosts.sh
. "$ROOT/scripts/telemak-hosts.sh"

CONFIGURATION="Release"
DEFAULT_CANARY="$(telemak_hosts | awk -F'|' 'NF { print $1; exit }')"
CANARY="${TELEMAK_DEPLOY_CANARY:-$DEFAULT_CANARY}"
SKIP_BUILD=0
CANARY_ONLY=0

usage() {
  cat <<'EOF'
Usage: scripts/deploy-all.sh [--canary host] [--skip-build] [--canary-only]

Deploy order:
  1. Build Release locally, unless --skip-build is passed.
  2. Deploy only the canary host.
  3. Smoke canary: binary version, /health, /admin/activity, menubar status.
  4. If canary passed, deploy every remaining inventory host.

Hosts come from scripts/telemak-hosts.local.sh.
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --canary)
      CANARY="${2:?missing --canary value}"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --canary-only)
      CANARY_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

PRODUCTS="$ROOT/.xcbuild/Build/Products/$CONFIGURATION"
LOCAL_TELEMAK="$PRODUCTS/telemak"
LOCAL_MENUBAR="$PRODUCTS/telemak-menubar"
LOCAL_APP="$ROOT/dist/Telemak.app"

if [ "$SKIP_BUILD" = "0" ]; then
  "$ROOT/scripts/build.sh" "$CONFIGURATION"
  "$ROOT/scripts/build-menubar-app.sh" "$CONFIGURATION"
fi

for required in "$LOCAL_TELEMAK" "$LOCAL_MENUBAR" "$PRODUCTS/mlx-swift_Cmlx.bundle"; do
  if [ ! -e "$required" ]; then
    echo "missing build artifact: $required" >&2
    exit 1
  fi
done

CANARY_ENTRY="$(telemak_host_lookup "$CANARY")" || {
  echo "unknown canary: $CANARY" >&2
  telemak_hosts >&2
  exit 1
}

deploy_one() {
  entry="$1"
  name="$(telemak_host_field "$entry" 1)"
  ip="$(telemak_host_field "$entry" 2)"
  user="$(telemak_host_field "$entry" 3)"
  active_release="$(telemak_host_field "$entry" 4)"
  port="$(telemak_host_field "$entry" 5)"
  server_label="$(telemak_host_field "$entry" 6)"
  menubar_label="$(telemak_host_field "$entry" 7)"
  ssh_target="$user@$ip"
  stamp="$(date -u +%Y%m%d-%H%M%S)"
  remote_tmp="/Users/$user/telemak/.deploy-$stamp"

  echo "deploy $name ($ip) -> $active_release"
  ssh -n -o BatchMode=yes "$ssh_target" "rm -rf '$remote_tmp' && mkdir -p '$remote_tmp'"
  scp -q "$LOCAL_TELEMAK" "$LOCAL_MENUBAR" "$ssh_target:$remote_tmp/"
  find "$PRODUCTS" -maxdepth 1 -name '*.bundle' -type d | while IFS= read -r bundle; do
    scp -q -r "$bundle" "$ssh_target:$remote_tmp/$(basename "$bundle")"
  done
  if [ -d "$LOCAL_APP" ]; then
    scp -q -r "$LOCAL_APP" "$ssh_target:$remote_tmp/Telemak.app"
  fi

  ssh -o BatchMode=yes "$ssh_target" \
    "ACTIVE_RELEASE='$active_release' REMOTE_TMP='$remote_tmp' SERVER_LABEL='$server_label' MENUBAR_LABEL='$menubar_label' sh -s" <<'REMOTE'
set -eu

UID_VALUE="$(id -u)"
ROOT_DIR="$(dirname "$ACTIVE_RELEASE")"
mkdir -p "$ROOT_DIR"

launchctl bootout "gui/$UID_VALUE/$SERVER_LABEL" 2>/dev/null || true
launchctl bootout "gui/$UID_VALUE/$MENUBAR_LABEL" 2>/dev/null || true

wait_unloaded() {
  label="$1"
  n=0
  while launchctl print "gui/$UID_VALUE/$label" >/dev/null 2>&1 && [ "$n" -lt 20 ]; do
    n=$((n + 1))
    sleep 0.5
  done
}

restart_label() {
  label="$1"
  plist="$2"
  [ -f "$plist" ] || return 0
  wait_unloaded "$label"
  if launchctl print "gui/$UID_VALUE/$label" >/dev/null 2>&1; then
    launchctl kickstart -k "gui/$UID_VALUE/$label"
    return 0
  fi
  launchctl bootstrap "gui/$UID_VALUE" "$plist"
  launchctl kickstart -k "gui/$UID_VALUE/$label"
}

if [ -d "$ACTIVE_RELEASE.prev10" ]; then
  rm -rf "$ACTIVE_RELEASE.prev10"
fi
i=9
while [ "$i" -ge 1 ]; do
  if [ -d "$ACTIVE_RELEASE.prev$i" ]; then
    next=$((i + 1))
    rm -rf "$ACTIVE_RELEASE.prev$next"
    mv "$ACTIVE_RELEASE.prev$i" "$ACTIVE_RELEASE.prev$next"
  fi
  i=$((i - 1))
done
if [ -d "$ACTIVE_RELEASE" ]; then
  mv "$ACTIVE_RELEASE" "$ACTIVE_RELEASE.prev1"
fi

mkdir -p "$ACTIVE_RELEASE"
cp -p "$REMOTE_TMP/telemak" "$ACTIVE_RELEASE/telemak"
cp -p "$REMOTE_TMP/telemak-menubar" "$ACTIVE_RELEASE/telemak-menubar"
find "$REMOTE_TMP" -maxdepth 1 -name '*.bundle' -type d | while IFS= read -r bundle; do
  name="$(basename "$bundle")"
  rm -rf "$ACTIVE_RELEASE/$name"
  cp -pR "$bundle" "$ACTIVE_RELEASE/$name"
done
if [ -d "$REMOTE_TMP/Telemak.app" ]; then
  cp -pR "$REMOTE_TMP/Telemak.app" "$ACTIVE_RELEASE/Telemak.app"
fi
chmod 755 "$ACTIVE_RELEASE/telemak" "$ACTIVE_RELEASE/telemak-menubar"
rm -rf "$REMOTE_TMP"

SERVER_PLIST="$HOME/Library/LaunchAgents/$SERVER_LABEL.plist"
MENUBAR_PLIST="$HOME/Library/LaunchAgents/$MENUBAR_LABEL.plist"
if [ -n "$MENUBAR_LABEL" ] && [ -x "$ACTIVE_RELEASE/Telemak.app/Contents/MacOS/Telemak" ]; then
  cat > "$MENUBAR_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$MENUBAR_LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$ACTIVE_RELEASE/Telemak.app/Contents/MacOS/Telemak</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
  <key>StandardOutPath</key>
  <string>$HOME/telemak/menubar.out</string>
  <key>StandardErrorPath</key>
  <string>$HOME/telemak/menubar.err</string>
</dict>
</plist>
EOF
fi
restart_label "$SERVER_LABEL" "$SERVER_PLIST"
restart_label "$MENUBAR_LABEL" "$MENUBAR_PLIST"

"$ACTIVE_RELEASE/telemak" --version
REMOTE

  smoke_one "$name" "$ip" "$port" "$ssh_target" "$menubar_label"
}

smoke_one() {
  name="$1"
  ip="$2"
  port="$3"
  ssh_target="$4"
  menubar_label="$5"
  health=""
  activity=""
  tries=0

  # Read API key from the remote host. Written by the installer to
  # ~/telemak/api-key.txt (0600). Empty on pre-security-fix installs —
  # the curl calls below still work because the middleware passes through
  # when TELEMAK_API_KEY is unset in the plist.
  api_key="$(ssh -n -o BatchMode=yes "$ssh_target" \
    'cat ~/telemak/api-key.txt 2>/dev/null || echo ""' 2>/dev/null || echo "")"

  # Smoke from the remote host so the same path works whether the server
  # is bound on 0.0.0.0 for LAN traffic or loopback during local dev.
  health_cmd="curl -fsS --max-time 5 http://127.0.0.1:${port}/health"
  if [ -n "$api_key" ]; then
    activity_cmd="curl -fsS --max-time 5 -H \"Authorization: Bearer ${api_key}\" http://127.0.0.1:${port}/admin/activity"
  else
    activity_cmd="curl -fsS --max-time 5 http://127.0.0.1:${port}/admin/activity"
  fi

  while [ "$tries" -lt 15 ]; do
    if health="$(ssh -n -o BatchMode=yes "$ssh_target" "$health_cmd" 2>/dev/null)" \
      && activity="$(ssh -n -o BatchMode=yes "$ssh_target" "$activity_cmd" 2>/dev/null)"; then
      break
    fi
    tries=$((tries + 1))
    sleep 2
  done

  if [ -z "$health" ] || [ -z "$activity" ]; then
    echo "smoke failed: $name ($ip)" >&2
    return 1
  fi

  version="$(printf '%s\n' "$health" | sed -n 's/.*"version":"\([^"]*\)".*/\1/p')"
  phase="$(printf '%s\n' "$activity" | sed -n 's/.*"current_phase":"\([^"]*\)".*/\1/p')"
  menubar="$(ssh -n -o BatchMode=yes "$ssh_target" "MENUBAR_LABEL='$menubar_label' sh -c 'if launchctl print gui/\$(id -u)/\"\$MENUBAR_LABEL\" >/dev/null 2>&1; then printf running; elif pgrep -f \"[T]elemak.app/Contents/MacOS/Telemak\" >/dev/null 2>&1; then printf running-app; else printf not-running-or-not-configured; fi'")"

  printf 'ok %-12s version=%s phase=%s menubar=%s\n' "$name" "${version:-unknown}" "${phase:-unknown}" "$menubar"
}

echo "canary: $(telemak_host_field "$CANARY_ENTRY" 1)"
deploy_one "$CANARY_ENTRY"

if [ "$CANARY_ONLY" = "1" ]; then
  echo "canary-only ok"
  exit 0
fi

telemak_hosts | while IFS= read -r entry; do
  [ "$entry" = "$CANARY_ENTRY" ] && continue
  deploy_one "$entry"
done

echo "deploy-all ok"
