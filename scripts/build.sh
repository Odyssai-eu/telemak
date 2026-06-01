#!/bin/sh
# Build Telemak via xcodebuild (required — see telemak-build-system memory).
# `swift build` doesn't compile the Metal kernels mlx-swift needs at runtime.

set -eu

CONFIGURATION="${1:-Debug}"
ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd)"
DERIVED="${ROOT}/.xcbuild"

cd "$ROOT"

LOG_FILE="$(mktemp "${TMPDIR:-/tmp}/telemak-build.XXXXXX.log")"
trap 'rm -f "$LOG_FILE"' EXIT

if xcodebuild \
  -scheme Telemak-Package \
  -configuration "$CONFIGURATION" \
  -derivedDataPath "$DERIVED" \
  -destination 'platform=macOS' \
  -skipMacroValidation \
  ENABLE_CODE_COVERAGE=NO \
  SWIFT_ENABLE_CODE_COVERAGE=NO \
  CLANG_ENABLE_CODE_COVERAGE=NO \
  GCC_GENERATE_TEST_COVERAGE_FILES=NO \
  GCC_INSTRUMENT_PROGRAM_FLOW_ARCS=NO \
  build >"$LOG_FILE" 2>&1; then
  BUILD_STATUS=0
else
  BUILD_STATUS=$?
fi

grep -vE '^(2026|note: |Note:|\s+[A-Z][a-z]+ \(in target)' "$LOG_FILE" || true

if [ "$BUILD_STATUS" -ne 0 ]; then
  echo "✗ xcodebuild failed with status $BUILD_STATUS" >&2
  exit "$BUILD_STATUS"
fi

BINARY="$DERIVED/Build/Products/$CONFIGURATION/telemak"
if [ -x "$BINARY" ]; then
  echo "✓ built $BINARY"
else
  echo "✗ build failed (no binary at $BINARY)" >&2
  exit 1
fi

if [ "$CONFIGURATION" = "Release" ] && strings "$BINARY" | grep -Eq 'LLVM_PROFILE|__llvm_prf|__LLVM_PROFILE'; then
  echo "✗ Release binary contains LLVM coverage/profiling runtime symbols" >&2
  exit 1
fi

MENUBAR="$DERIVED/Build/Products/$CONFIGURATION/telemak-menubar"
if [ -x "$MENUBAR" ]; then
  echo "✓ built $MENUBAR"
fi

# Sign Release builds with a stable code-signing identity so macOS TCC
# (Full Disk Access / removable disks) remembers the grant across
# rebuilds. Without this, every Release ships with a fresh ad-hoc
# signature → new cdhash → TCC re-prompts the operator on every
# deploy, blocking the LaunchAgent if nobody is at the screen to click.
#
# One-time setup per dev machine (see docs/CODESIGNING.md) :
#   1. Generate self-signed cert via openssl (Code Signing EKU).
#   2. Import into login or dedicated keychain.
#   3. Trust as root for codeSign policy in System.keychain (sudo +
#      Screen Sharing session on the build host — Sequoia requires a
#      console user session for the trust mutation).
#
# CODESIGN_IDENTITY env var overrides which identity to use. Defaults
# to "Telemak Developer (Odyssai-eu)" — the identity the operator minted
# 2026-05-24. Debug builds stay ad-hoc (fast inner loop, no perf
# difference, TCC ad-hoc cdhash churn doesn't matter for dev).
if [ "$CONFIGURATION" = "Release" ]; then
  IDENTITY="${CODESIGN_IDENTITY:-Telemak Developer (Odyssai-eu)}"
  if security find-identity -v -p codesigning 2>/dev/null | grep -q "$IDENTITY"; then
    echo "↳ signing Release binaries with identity '$IDENTITY'"
    codesign --sign "$IDENTITY" --force --options runtime "$BINARY"
    if [ -x "$MENUBAR" ]; then
      codesign --sign "$IDENTITY" --force --options runtime "$MENUBAR"
    fi
    echo "✓ signed (cdhash will stay stable across rebuilds)"
  else
    echo "⚠ codesign identity '$IDENTITY' not found — Release will be ad-hoc-signed"
    echo "  → TCC will re-prompt on every deploy. See docs/CODESIGNING.md to fix."
  fi
fi
