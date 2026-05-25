# Telemak — stable code-signing for TCC persistence

> Without a stable signing identity, every Release rebuild produces a new
> `cdhash` and macOS TCC re-prompts on first launch for Full Disk Access /
> Removable Volumes. If nobody is at the screen to click OK, the LaunchAgent
> can't read `/Volumes/models/odysseus/` and Telemak serves nothing.
>
> This doc captures the one-time setup that fixes that : a self-signed
> certificate trusted as code-signing root on the build host, used by
> `scripts/build.sh` for every Release. Once set up, all future rebuilds
> inherit the same signing identity → TCC remembers its grants.

## One-time setup (per build host)

The build host is whichever machine runs `./scripts/build.sh Release` —
historically Sophie's workstation, but max-64 itself works too.

### 1. Generate self-signed certificate

```bash
mkdir -p ~/.telemak-codesign
cd ~/.telemak-codesign

cat > openssl.conf << 'EOF'
[ req ]
distinguished_name = req_distinguished_name
prompt = no
x509_extensions = v3_req

[ req_distinguished_name ]
CN = Telemak Developer (Odyssai-eu)
O = Odyssai-eu
OU = Telemak

[ v3_req ]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature
extendedKeyUsage = critical, codeSigning
EOF

openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout telemak.key -out telemak.cert \
  -days 3650 -config openssl.conf

openssl pkcs12 -export -inkey telemak.key -in telemak.cert \
  -out telemak.p12 -password pass:telemak \
  -name "Telemak Developer (Odyssai-eu)"
```

### 2. Import cert + key into System keychain (with sudo)

```bash
sudo security import ~/.telemak-codesign/telemak.p12 -P telemak \
  -T /usr/bin/codesign -T /usr/bin/security \
  -k /Library/Keychains/System.keychain
```

This step works fine over SSH — it only requires `sudo`, no GUI prompt.

### 3. Trust the cert as a code-signing root (Sequoia gotcha)

```bash
sudo security add-trusted-cert -d -r trustRoot -p codeSign \
  -k /Library/Keychains/System.keychain ~/.telemak-codesign/telemak.cert
```

**This step MUST run in a user-session context on the build host** —
SSH non-interactive (even with sudo) is rejected by macOS Sequoia with
`SecTrustSettingsSetTrustSettings: The authorization was denied since
no user interaction was possible`.

The fix : connect to the host via **Screen Sharing**
(`vnc://192.168.86.50` in Finder → Connect) so the SSH session inherits
a console user session. Open Terminal on max-64 and run the command
there. Sudo prompts for the admin password ; trust takes.

To verify the trust took :

```bash
security find-identity -v -p codesigning
# expected output :
#   1) 2300608340DBE83E67C0E83386BD4D69C0005073 "Telemak Developer (Odyssai-eu)"
#      1 valid identity found
```

If you see `0 valid identities found` after the command — the trust
prompt was suppressed (no console session). Re-do via Screen Sharing.

### 4. Verify `scripts/build.sh` picks it up

```bash
./scripts/build.sh Release
# expected tail :
#   ↳ signing Release binaries with identity 'Telemak Developer (Odyssai-eu)'
#   ✓ signed (cdhash will stay stable across rebuilds)
```

If you see the `⚠ codesign identity 'Telemak Developer (Odyssai-eu)' not
found` warning, the trust step didn't land — re-do step 3.

## What happens on first deploy after signing kicks in

The first time the new signed binary launches, TCC sees a binary signed
by an unknown root (Telemak Developer, never seen before) and prompts
for Full Disk Access / Removable Volumes — same popup as ad-hoc, but
THIS one persists across rebuilds because all subsequent rebuilds use
the same identity.

Click OK / approve in System Settings → Privacy & Security. Done.
Every future `./scripts/build.sh Release && ./scripts/deploy-release.sh`
sails through with no popup.

## Overriding the identity

Set `CODESIGN_IDENTITY=<some name>` in the build environment to use a
different identity. Useful if a contributor mints their own cert with
a different CN — they don't need to share secrets, just set the env
var.

```bash
CODESIGN_IDENTITY="Acme Build (Acme)" ./scripts/build.sh Release
```

## Debug builds stay ad-hoc

`scripts/build.sh Debug` doesn't sign — fast inner loop matters more
than TCC stability in dev. The TCC re-prompt churn only hurts when
deploying to a long-running LaunchAgent.

## Rotating the cert (5-year expiry, 2031-05-25)

The cert validity is `-days 3650` (~10 years). When it expires :

1. Repeat steps 1-3 with a new cert (different CN to avoid clash, e.g.
   `Telemak Developer 2031`).
2. Update the default `CODESIGN_IDENTITY` in `scripts/build.sh` or
   override via env.
3. Next deploy : one final TCC click on the new signature, persistence
   resumes.

## References

- Apple Tech Note TN3179 (Trust Settings) — explains the Sequoia
  authorization model.
- `man security` — `add-trusted-cert`, `import`, `find-identity`.
