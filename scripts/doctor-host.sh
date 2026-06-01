#!/bin/sh
# Diagnose one Telemak host from the local operator inventory.

set -eu

ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
# shellcheck source=scripts/telemak-hosts.sh
. "$ROOT/scripts/telemak-hosts.sh"

usage() {
  cat <<'EOF'
Usage: scripts/doctor-host.sh <host|--all>

Runs operator diagnostics against a Telemak inventory host:
  - SSH reachability
  - release artifacts and binary version
  - codesign authority
  - server and menubar LaunchAgents
  - /health and /admin/activity via loopback
  - /v1/models via LAN without bearer token
  - LaunchAgent bind flag sanity

Hosts come from scripts/telemak-hosts.local.sh.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ] || [ $# -lt 1 ]; then
  usage
  exit 0
fi

FAILED=0
WARNED=0

pass() { printf 'ok   %s\n' "$1"; }
warn() { printf 'warn %s\n' "$1"; WARNED=1; }
fail() { printf 'fail %s\n' "$1"; FAILED=1; }

json_value() {
  key="$1"
  sed -n "s/.*\"$key\":\"\\([^\"]*\\)\".*/\\1/p"
}

doctor_entry() {
  entry="$1"
  name="$(telemak_host_field "$entry" 1)"
  ip="$(telemak_host_field "$entry" 2)"
  user="$(telemak_host_field "$entry" 3)"
  active_release="$(telemak_host_field "$entry" 4)"
  port="$(telemak_host_field "$entry" 5)"
  server_label="$(telemak_host_field "$entry" 6)"
  menubar_label="$(telemak_host_field "$entry" 7)"
  ssh_target="$user@$ip"

  printf '\n== %s (%s:%s) ==\n' "$name" "$ip" "$port"

  if ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$ssh_target" 'printf ok' >/dev/null 2>&1; then
    pass "ssh reachable"
  else
    fail "ssh unreachable: $ssh_target"
    return
  fi

  remote_report="$(ssh -o BatchMode=yes "$ssh_target" \
    "ACTIVE_RELEASE='$active_release' SERVER_LABEL='$server_label' MENUBAR_LABEL='$menubar_label' PORT='$port' sh -s" <<'REMOTE'
set -eu
uid_value="$(id -u)"
api_key="$(cat "$HOME/telemak/api-key.txt" 2>/dev/null || true)"

field() {
  printf '%s=%s\n' "$1" "$2"
}

field release_dir "$(test -d "$ACTIVE_RELEASE" && echo ok || echo missing)"
field binary "$(test -x "$ACTIVE_RELEASE/telemak" && echo ok || echo missing)"
field menubar_binary "$(test -x "$ACTIVE_RELEASE/telemak-menubar" && echo ok || echo missing)"
field mlx_bundle "$(test -d "$ACTIVE_RELEASE/mlx-swift_Cmlx.bundle" && echo ok || echo missing)"
field app_bundle "$(test -x "$ACTIVE_RELEASE/Telemak.app/Contents/MacOS/Telemak" && echo ok || echo missing)"

if [ -x "$ACTIVE_RELEASE/telemak" ]; then
  field version "$("$ACTIVE_RELEASE/telemak" --version 2>/dev/null || echo error)"
  field codesign "$(codesign -dv --verbose=4 "$ACTIVE_RELEASE/telemak" 2>&1 | sed -n 's/^Authority=//p' | head -n 1)"
else
  field version missing
  field codesign missing
fi

if launchctl print "gui/$uid_value/$SERVER_LABEL" >/dev/null 2>&1; then
  field server_launchagent running
else
  field server_launchagent missing
fi
if launchctl print "gui/$uid_value/$MENUBAR_LABEL" >/dev/null 2>&1; then
  field menubar_launchagent running
else
  field menubar_launchagent missing
fi

server_plist="$HOME/Library/LaunchAgents/$SERVER_LABEL.plist"
if [ -f "$server_plist" ]; then
  if grep -A1 '<string>--host</string>' "$server_plist" | grep -q '<string>0.0.0.0</string>'; then
    field bind 0.0.0.0
  elif grep -A1 '<string>--host</string>' "$server_plist" | grep -q '<string>127.0.0.1</string>'; then
    field bind 127.0.0.1
  else
    field bind unknown
  fi
else
  field bind missing-plist
fi

if [ -n "$api_key" ]; then
  auth_header="Authorization: Bearer $api_key"
  field api_key configured
else
  auth_header=""
  field api_key open
fi

if curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/tmp/telemak-doctor-health.$$ 2>/dev/null; then
  field health "$(cat /tmp/telemak-doctor-health.$$)"
else
  field health failed
fi
rm -f /tmp/telemak-doctor-health.$$

if [ -n "$auth_header" ]; then
  curl -fsS --max-time 5 -H "$auth_header" "http://127.0.0.1:$PORT/admin/activity" >/tmp/telemak-doctor-activity.$$ 2>/dev/null || true
else
  curl -fsS --max-time 5 "http://127.0.0.1:$PORT/admin/activity" >/tmp/telemak-doctor-activity.$$ 2>/dev/null || true
fi
if [ -s /tmp/telemak-doctor-activity.$$ ]; then
  field activity "$(cat /tmp/telemak-doctor-activity.$$)"
else
  field activity failed
fi
rm -f /tmp/telemak-doctor-activity.$$
REMOTE
)"

  get_field() {
    printf '%s\n' "$remote_report" | sed -n "s/^$1=//p" | tail -n 1
  }

  for item in release_dir binary menubar_binary mlx_bundle; do
    value="$(get_field "$item")"
    if [ "$value" = "ok" ]; then pass "$item=$value"; else fail "$item=$value"; fi
  done

  app_bundle="$(get_field app_bundle)"
  if [ "$app_bundle" = "ok" ]; then pass "app_bundle=ok"; else warn "app_bundle=missing"; fi

  version="$(get_field version)"
  if [ -n "$version" ] && [ "$version" != "missing" ] && [ "$version" != "error" ]; then
    pass "version=$version"
  else
    fail "version=$version"
  fi

  codesign_authority="$(get_field codesign)"
  case "$codesign_authority" in
    *Telemak*Developer*) pass "codesign=$codesign_authority" ;;
    "") warn "codesign authority missing" ;;
    *) warn "codesign=$codesign_authority" ;;
  esac

  server_state="$(get_field server_launchagent)"
  if [ "$server_state" = "running" ]; then pass "server_launchagent=running"; else fail "server_launchagent=$server_state"; fi

  menubar_state="$(get_field menubar_launchagent)"
  if [ "$menubar_state" = "running" ]; then pass "menubar_launchagent=running"; else warn "menubar_launchagent=$menubar_state"; fi

  bind_value="$(get_field bind)"
  case "$bind_value" in
    0.0.0.0) pass "bind=0.0.0.0" ;;
    127.0.0.1) fail "bind=127.0.0.1 would cut LAN routing" ;;
    *) warn "bind=$bind_value" ;;
  esac

  api_key_state="$(get_field api_key)"
  pass "admin_api_key=$api_key_state"

  health="$(get_field health)"
  if [ "$health" != "failed" ] && printf '%s' "$health" | grep -q '"status":"ok"'; then
    health_version="$(printf '%s' "$health" | json_value version)"
    pass "health=ok version=${health_version:-unknown}"
  else
    fail "health=$health"
  fi

  activity="$(get_field activity)"
  if [ "$activity" != "failed" ] && printf '%s' "$activity" | grep -q '"current_phase"'; then
    phase="$(printf '%s' "$activity" | json_value current_phase)"
    pass "activity=ok phase=${phase:-unknown}"
  else
    fail "activity=$activity"
  fi

  if curl -fsS --max-time 5 "http://$ip:$port/v1/models" >/tmp/telemak-doctor-lan.$$ 2>/dev/null; then
    pass "lan /v1/models reachable without bearer"
  else
    fail "lan /v1/models unreachable without bearer"
  fi
  rm -f /tmp/telemak-doctor-lan.$$
}

case "$1" in
  --all)
    HOSTS="$(telemak_hosts)"
    while IFS= read -r entry; do
      [ -z "$entry" ] && continue
      doctor_entry "$entry"
    done <<EOF
$HOSTS
EOF
    ;;
  *)
    ENTRY="$(telemak_host_lookup "$1")" || {
      echo "unknown host: $1" >&2
      telemak_hosts >&2
      exit 1
    }
    doctor_entry "$ENTRY"
    ;;
esac

if [ "$FAILED" -ne 0 ]; then
  exit 1
fi
if [ "$WARNED" -ne 0 ]; then
  exit 2
fi
