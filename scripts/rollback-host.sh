#!/bin/sh
# Roll one Telemak host back to a bronze/stable or previous Release directory.

set -eu

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
# shellcheck source=scripts/telemak-hosts.sh
. "$ROOT/scripts/telemak-hosts.sh"

usage() {
  cat <<'EOF'
Usage: scripts/rollback-host.sh <host> [target] [--force]

host:
  inventory name, IP, or last octet, e.g. ultra-256c, 192.168.86.32, .32

target:
  bronze-0.6.15       -> ~/telemak/Release.bronze-0.6.15 (default)
  prev1               -> Release.prev1, or Release.deepseek.prev1 on ultra-512
  Release.prev1       -> explicit sibling of active release
  /absolute/path      -> explicit remote target path

The script refuses a target without BRONZE-MANIFEST.txt or a runnable
telemak --version, unless --force is passed.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ $# -lt 1 ]; then
  usage
  exit 0
fi

HOST_ARG="$1"
TARGET_ARG="${2:-bronze-0.6.15}"
FORCE=0
if [ "${2:-}" = "--force" ] || [ "${3:-}" = "--force" ]; then
  FORCE=1
  if [ "${2:-}" = "--force" ]; then
    TARGET_ARG="bronze-0.6.15"
  fi
fi

ENTRY="$(telemak_host_lookup "$HOST_ARG")" || {
  echo "unknown host: $HOST_ARG" >&2
  telemak_hosts >&2
  exit 1
}

NAME="$(telemak_host_field "$ENTRY" 1)"
IP="$(telemak_host_field "$ENTRY" 2)"
USER="$(telemak_host_field "$ENTRY" 3)"
ACTIVE_RELEASE="$(telemak_host_field "$ENTRY" 4)"
PORT="$(telemak_host_field "$ENTRY" 5)"
SERVER_LABEL="$(telemak_host_field "$ENTRY" 6)"
MENUBAR_LABEL="$(telemak_host_field "$ENTRY" 7)"
SSH_TARGET="$USER@$IP"

case "$TARGET_ARG" in
  /*)
    TARGET_RELEASE="$TARGET_ARG"
    ;;
  bronze-*)
    TARGET_RELEASE="/Users/$USER/telemak/Release.$TARGET_ARG"
    ;;
  Release.*)
    TARGET_RELEASE="$(dirname "$ACTIVE_RELEASE")/$TARGET_ARG"
    ;;
  prev*)
    TARGET_RELEASE="$ACTIVE_RELEASE.$TARGET_ARG"
    ;;
  *)
    echo "unsupported target: $TARGET_ARG" >&2
    usage >&2
    exit 1
    ;;
esac

echo "rollback $NAME ($IP): $ACTIVE_RELEASE <- $TARGET_RELEASE"

ssh -o BatchMode=yes "$SSH_TARGET" \
  "ACTIVE_RELEASE='$ACTIVE_RELEASE' TARGET_RELEASE='$TARGET_RELEASE' FORCE='$FORCE' SERVER_LABEL='$SERVER_LABEL' MENUBAR_LABEL='$MENUBAR_LABEL' sh -s" <<'REMOTE'
set -eu

UID_VALUE="$(id -u)"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
BACKUP_RELEASE="$ACTIVE_RELEASE.rollback-$STAMP"

if [ ! -d "$TARGET_RELEASE" ]; then
  echo "target directory missing: $TARGET_RELEASE" >&2
  exit 1
fi

if [ "$FORCE" != "1" ]; then
  if [ -f "$TARGET_RELEASE/BRONZE-MANIFEST.txt" ]; then
    :
  elif [ -x "$TARGET_RELEASE/telemak" ] && "$TARGET_RELEASE/telemak" --version >/dev/null 2>&1; then
    :
  else
    echo "target has no BRONZE-MANIFEST.txt and no runnable telemak --version: $TARGET_RELEASE" >&2
    exit 1
  fi
fi

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

if [ -d "$ACTIVE_RELEASE" ]; then
  mv "$ACTIVE_RELEASE" "$BACKUP_RELEASE"
fi

TMP_RELEASE="$ACTIVE_RELEASE.tmp-$STAMP"
rm -rf "$TMP_RELEASE"
mkdir -p "$(dirname "$ACTIVE_RELEASE")"
cp -pR "$TARGET_RELEASE" "$TMP_RELEASE"
mv "$TMP_RELEASE" "$ACTIVE_RELEASE"

SERVER_PLIST="$HOME/Library/LaunchAgents/$SERVER_LABEL.plist"
MENUBAR_PLIST="$HOME/Library/LaunchAgents/$MENUBAR_LABEL.plist"
restart_label "$SERVER_LABEL" "$SERVER_PLIST"
restart_label "$MENUBAR_LABEL" "$MENUBAR_PLIST"

"$ACTIVE_RELEASE/telemak" --version
echo "backup=$BACKUP_RELEASE"
REMOTE

HEALTH=""
ACTIVITY=""
TRIES=0
while [ "$TRIES" -lt 15 ]; do
  if HEALTH="$(curl -fsS --max-time 5 "http://$IP:$PORT/health" 2>/dev/null)" \
    && ACTIVITY="$(curl -fsS --max-time 5 "http://$IP:$PORT/admin/activity" 2>/dev/null)"; then
    break
  fi
  TRIES=$((TRIES + 1))
  sleep 2
done
if [ -z "$HEALTH" ] || [ -z "$ACTIVITY" ]; then
  echo "smoke failed after rollback: $NAME ($IP:$PORT)" >&2
  exit 1
fi
printf 'health=%s\n' "$HEALTH"
printf 'activity=%s\n' "$ACTIVITY"

ssh -n -o BatchMode=yes "$SSH_TARGET" \
  "MENUBAR_LABEL='$MENUBAR_LABEL' sh -c 'if launchctl print gui/\$(id -u)/\"\$MENUBAR_LABEL\" >/dev/null 2>&1; then echo \"menubar=running\"; elif pgrep -f \"[T]elemak.app/Contents/MacOS/Telemak\" >/dev/null 2>&1; then echo \"menubar=running-app\"; else echo \"menubar=not-running-or-not-configured\"; fi'"

echo "rollback ok: $NAME"
