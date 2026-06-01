#!/bin/sh
# Example host inventory for Telemak operator scripts.
#
# Copy this file to scripts/telemak-hosts.local.sh and replace the placeholder
# values with your own LAN inventory. Keep the field order:
# name|host|ssh-user|active-release-dir|port|server-launchagent|menubar-launchagent

set -eu

telemak_hosts() {
  cat <<'EOF'
node-a|10.0.0.10|admin|/Users/admin/telemak/Release|8003|eu.odyssai.telemak|eu.odyssai.telemak.menubar
node-b|10.0.0.11|admin|/Users/admin/telemak/Release|8003|eu.odyssai.telemak|eu.odyssai.telemak.menubar
EOF
}

telemak_host_lookup() {
  needle="$1"
  telemak_hosts | awk -F'|' -v needle="$needle" '
    $1 == needle || $2 == needle {
      print
      found = 1
      exit
    }
    END {
      if (!found) {
        exit 1
      }
    }
  '
}

telemak_host_field() {
  entry="$1"
  index="$2"
  printf '%s\n' "$entry" | awk -F'|' -v index="$index" '{ print $index }'
}
