# Runbook — manual deployment of Telemak to a host

Telemak ships as binaries, not source. `deploy-all.sh` cannot complete a
headless deploy because `launchctl bootstrap` fails over non-interactive SSH,
so the reference procedure is the manual one below: build, tar, scp, extract,
strip quarantine, re-sign ad-hoc, restart, health-check.

`scripts/deploy-ssh.sh` automates exactly this procedure — see the note at
the bottom.

## Prerequisites

- macOS workstation with `xcodebuild` (do **not** use `swift build`: it does
  not compile the Metal kernels mlx-swift needs at runtime).
- SSH access to the host (default: `admin@192.168.86.33`, port 8003 is the
  server port; `.29` uses 8013).
- Runtime layout on the host: `~/telemak/Release/`.

## 1. Build (workstation)

```bash
./scripts/build.sh Release
```

Artifacts land in `.xcbuild/Build/Products/Release/`: the binaries
`telemak`, `telemak-menubar`, `telemak-monitor` and the Metal/resource
bundles. Release builds are signed locally with the "Telemak Developer
(Odyssai-eu)" identity; that certificate does not exist on the hosts, hence
the ad-hoc re-sign in step 4.

## 2. Tar the runtime artifacts (workstation)

The 5 bundles are mandatory — without them the runtime exits with
"Failed to load the default metallib".

```bash
cd .xcbuild/Build/Products/Release
tar -czf /tmp/telemak-deploy.tar.gz \
  telemak telemak-menubar telemak-monitor \
  mlx-swift_Cmlx.bundle \
  swift-crypto_Crypto.bundle \
  swift-nio_NIOPosix.bundle \
  swift-nio__NIOFileSystem.bundle \
  swift-transformers_Hub.bundle
```

## 3. Upload (workstation)

```bash
scp /tmp/telemak-deploy.tar.gz admin@192.168.86.33:/tmp/
```

## 4. Extract, strip quarantine, re-sign (host)

scp/shared transfer adds `com.apple.quarantine`, which breaks execution.
Strip it, then re-sign ad-hoc (the local signing identity is absent from
the hosts).

```bash
ssh admin@192.168.86.33
mkdir -p ~/telemak/Release
tar -xzf /tmp/telemak-deploy.tar.gz -C ~/telemak/Release
cd ~/telemak/Release
xattr -dr com.apple.quarantine *
codesign -s - --force telemak telemak-menubar telemak-monitor
for b in mlx-swift_Cmlx.bundle swift-crypto_Crypto.bundle \
         swift-nio_NIOPosix.bundle swift-nio__NIOFileSystem.bundle \
         swift-transformers_Hub.bundle; do
  codesign -s - --force "$b"
done
```

Gotcha: do not chain `rm -rf * && tar …` in one remote command — under zsh
the glob can fail with "no matches found" and the chain silently stops. Run
extract and cleanup as separate commands.

## 5. Restart the server (host)

`launchctl bootstrap` is unreliable over non-interactive SSH, so restart via
pkill + nohup:

```bash
pkill -f "telemak serve" || true
sleep 1
nohup ./telemak serve --host 0.0.0.0 --port 8003 > launchd.out 2> launchd.err &
```

## 6. Health check and smoke

```bash
curl -s http://192.168.86.33:8003/health
# → {"status":"ok","version":"…", …}

curl -s -X POST http://192.168.86.33:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<loaded-id>","messages":[{"role":"user","content":"hi"}],"stream":false,"max_tokens":20}'
```

Gotchas:

- A `/health` that answers with a very high uptime may be a phantom process
  on a freed port — confirm the real process exists and check
  `~/telemak/Release/launchd.err` ("No such file or directory" = binary
  missing).
- Model loading can take several minutes; `/health` may report loaded while
  chat still answers `model_not_loaded` during warm-up. First inference is
  slow (Metal kernel compilation).

## Automation

`scripts/deploy-ssh.sh` performs the exact same procedure end to end:

```bash
scripts/deploy-ssh.sh [host] [user] [ssh-port]
# defaults: 192.168.86.33 admin 22
# env overrides: TELEMAK_HOST, TELEMAK_USER, TELEMAK_SSH_PORT, TELEMAK_SERVER_PORT
```

It builds Release if artifacts are missing, packs, uploads, extracts, signs,
restarts via pkill + nohup, and polls `/health` until it answers.
