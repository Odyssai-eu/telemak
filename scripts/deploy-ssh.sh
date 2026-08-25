#!/bin/sh
# deploy-ssh.sh — automate the manual Telemak deploy to a remote host.
#
# Runs the documented manual procedure (see repo MEMORY.md / AGENTS.md §4),
# which deploy-all.sh cannot do headlessly because `launchctl bootstrap`
# fails over non-interactive SSH. This script instead restarts the server
# with pkill + nohup and only ships the runtime binaries + Metal bundles.
#
# Procedure:
#   1. Build Release locally unless already built.
#   2. Tar the binaries (telemak, telemak-menubar, telemak-monitor) + the
#      5 required Metal bundles (without them the runtime exits with
#      "Failed to load the default metallib").
#   3. scp the tarball to the remote host.
#   4. On the host: extract into ~/telemak/Release/, strip quarantine,
#      re-sign ad-hoc (the local "Telemak Developer" cert is not on hosts).
#   5. pkill + nohup restart of `telemak serve`.
#   6. Curl /health to verify.
#
# Exits non-zero as soon as any step fails.
#
# Usage: scripts/deploy-ssh.sh [host] [user] [ssh-port]
#   host      remote host (default 192.168.86.33)
#   user      remote user  (default admin)
#   ssh-port  ssh/scp port (default 22)
#
# Env overrides: TELEMAK_HOST, TELEMAK_USER, TELEMAK_SSH_PORT,
#                TELEMAK_SERVER_PORT (default 8003)

set -eu

CONFIGURATION="Release"
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
PRODUCTS="$ROOT/.xcbuild/Build/Products/$CONFIGURATION"

HOST="${TELEMAK_HOST:-${1:-192.168.86.33}}"
USER="${TELEMAK_USER:-${2:-admin}}"
SSH_PORT="${TELEMAK_SSH_PORT:-${3:-22}}"
SERVER_PORT="${TELEMAK_SERVER_PORT:-8003}"

SSH_TARGET="$USER@$HOST"
SSH_ARGS="-p $SSH_PORT -o BatchMode=yes"
# NOTE: $HOME must NOT expand locally — the remote HOME may differ (e.g.
# local /Users/sophie vs remote /Users/admin). Resolve it in the remote
# shell below via the REMOTE_DIR placeholder.
REMOTE_DIR='$HOME/telemak/Release'

stamp="$(date -u +%Y%m%d-%H%M%S)"
TARBALL="$(mktemp "${TMPDIR:-/tmp}/telemak-deploy-$stamp.XXXXXX.tar.gz")"
trap 'rm -f "$TARBALL"' EXIT

say() { printf '==> %s\n' "$*"; }
die() { printf '!! %s\n' "$*" >&2; exit 1; }

# --- 1. Build Release if a binary is missing ---------------------------------
TELEMAK_BIN="$PRODUCTS/telemak"
MENUBAR_BIN="$PRODUCTS/telemak-menubar"
MONITOR_BIN="$PRODUCTS/telemak-monitor"
BUNDLES="
  mlx-swift_Cmlx.bundle
  swift-crypto_Crypto.bundle
  swift-nio_NIOPosix.bundle
  swift-nio__NIOFileSystem.bundle
  swift-transformers_Hub.bundle
"

if [ ! -x "$TELEMAK_BIN" ] || [ ! -x "$MENUBAR_BIN" ] || [ ! -x "$MONITOR_BIN" ]; then
  say "Release binaries missing; building..."
  "$ROOT/scripts/build.sh" "$CONFIGURATION"
fi

for f in "$TELEMAK_BIN" "$MENUBAR_BIN" "$MONITOR_BIN"; do
  [ -x "$f" ] || die "missing build artifact: $f"
done
for b in $BUNDLES; do
  [ -d "$PRODUCTS/$b" ] || die "missing build artifact (bundle): $PRODUCTS/$b"
done

# --- 2. Tar the runtime binaries + bundles ------------------------------
say "packing ${CONFIGURATION} artifacts into $(basename "$TARBALL")"
tar -C "$PRODUCTS" -czf "$TARBALL" \
  telemak telemak-menubar telemak-monitor \
  $BUNDLES

# --- 3. scp to the remote host ------------------------------------------
say "uploading to $SSH_TARGET (port $SSH_PORT)..."
scp $SSH_ARGS "$TARBALL" "$SSH_TARGET:/tmp/$(basename "$TARBALL")" \
  || die "scp failed"

# --- 4-5. Extract, strip quarantine, sign, restart on the host ------------
say "extracting + signing + restarting on $HOST..."
ssh $SSH_ARGS "$SSH_TARGET" "REMOTE_TARBALL='$(basename "$TARBALL")' sh -s" <<'REMOTE'
set -eu
# Remote HOME resolves here (e.g. /Users/admin), not the local one.
REMOTE_DIR="$HOME/telemak/Release"
mkdir -p "$REMOTE_DIR"
tar -xzf "/tmp/$REMOTE_TARBALL" -C "$REMOTE_DIR"
rm -f "/tmp/$REMOTE_TARBALL"

cd "$REMOTE_DIR"
xattr -dr com.apple.quarantine * 2>/dev/null || true
codesign -s - --force telemak telemak-menubar telemak-monitor
for b in mlx-swift_Cmlx.bundle swift-crypto_Crypto.bundle swift-nio_NIOPosix.bundle swift-nio__NIOFileSystem.bundle swift-transformers_Hub.bundle; do
  codesign -s - --force "$b"
done

pkill -f "telemak serve" 2>/dev/null || true
sleep 1
nohup ./telemak serve --host 0.0.0.0 --port "${TELEMAK_SERVER_PORT:-8003}" > launchd.out 2> launchd.err &
echo "server started (pid $!)"
REMOTE

# --- 6. Verify health ----------------------------------------------------
say "verifying http://$HOST:$SERVER_PORT/health..."
tries=0
while [ "$tries" -lt 15 ]; do
  if body="$(curl -fsS --max-time 5 "http://$HOST:$SERVER_PORT/health" 2>/dev/null)"; then
    echo "ok $(printf '%s' "$body")"
    exit 0
  fi
  tries=$((tries + 1))
  sleep 2
done

die "health check failed on $HOST:$SERVER_PORT (see ~/telemak/Release/launchd.err on the host)"