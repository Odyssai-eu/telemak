#!/bin/sh
# Shared host inventory for Telemak operator scripts.

set -eu

telemak_hosts() {
  cat <<'EOF'
ultra-512|192.168.86.29|admin|/Users/admin/telemak/Release.deepseek|8013|eu.odyssai.telemak.deepseek|eu.odyssai.telemak.deepseek.menubar
ultra-256a|192.168.86.30|admin|/Users/admin/telemak/Release|8003|eu.odyssai.telemak|eu.odyssai.telemak.menubar
ultra-256b|192.168.86.31|admin|/Users/admin/telemak/Release|8003|eu.odyssai.telemak|eu.odyssai.telemak.menubar
ultra-256c|192.168.86.32|admin|/Users/admin/telemak/Release|8003|eu.odyssai.telemak|eu.odyssai.telemak.menubar
ultra-96|192.168.86.49|admin|/Users/admin/telemak/Release|8003|eu.odyssai.telemak|eu.odyssai.telemak.menubar
max-64|192.168.86.50|admin|/Users/admin/telemak/Release|8003|eu.odyssai.telemak|eu.odyssai.telemak.menubar
EOF
}

telemak_host_lookup() {
  needle="$1"
  telemak_hosts | awk -F'|' -v needle="$needle" '
    $1 == needle || $2 == needle || ("." substr($2, 12)) == needle || substr($2, 12) == needle {
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
