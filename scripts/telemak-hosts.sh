#!/bin/sh
# Shared host inventory loader for Telemak operator scripts.
#
# The real fleet inventory is intentionally local-only. Copy
# scripts/telemak-hosts.example.sh to scripts/telemak-hosts.local.sh and edit
# it for your LAN before running deploy or rollback scripts.

set -eu

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
LOCAL_HOSTS="$ROOT/scripts/telemak-hosts.local.sh"

if [ ! -f "$LOCAL_HOSTS" ]; then
  cat >&2 <<'EOF'
missing local Telemak host inventory: scripts/telemak-hosts.local.sh

Create it from the public template:
  cp scripts/telemak-hosts.example.sh scripts/telemak-hosts.local.sh

Then replace the placeholder hosts, IPs, users, release paths, ports, and
LaunchAgent labels with your own LAN inventory.
EOF
  return 1 2>/dev/null || exit 1
fi

# shellcheck source=scripts/telemak-hosts.local.sh
. "$LOCAL_HOSTS"
