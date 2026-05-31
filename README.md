# Telemak

> *Mobilis in Mobile — pars natif.*

**Native Mac single-node runtime for the [Odysseus](https://github.com/Odyssai-eu/Odysseus) inference engine.** Built directly on Apple's `mlx-swift-lm` — no Python venv, no Docker, no orchestrator container. A single `.app` (or CLI binary) that loads MLX-quantized models from the local Hugging Face cache and exposes the same OpenAI- and Anthropic-compatible HTTP surface as Odysseus.

Part of [**OdyssAI**](https://odyssai.eu) — the open-source local AI ecosystem.

```
┌──────────────────────────────────────────────────────────┐
│  Clients  (Companion · IDE agents · any HTTP client)     │
│         ↓  HTTP  ─  /v1/chat/completions                 │
│         ↓        ─  /v1/messages                         │
├──────────────────────────────────────────────────────────┤
│  Telemak  (Swift, mlx-swift-lm, Hummingbird, single-node)│
│         ↓  Metal direct (no Python overhead)             │
├──────────────────────────────────────────────────────────┤
│  Apple Silicon (this Mac)                                │
└──────────────────────────────────────────────────────────┘
```

Or, when running inside an Odysseus cluster :

```
┌─────────────────────────────────────────────────────────┐
│  Odysseus orchestrator                                  │
│   ├── cluster "argo"  backend=jaccl  → 4 Macs distrib.  │
│   └── cluster "chat"  backend=http-proxy  → Telemak     │
└─────────────────────────────────────────────────────────┘
```

## Why Telemak

The Odysseus distributed engine (Python + JACCL + Docker orchestrator) is the right tool for **frontier MoEs sharded across 2-4 Macs** (GLM-5.1, DeepSeek V3.1, Qwen3.5-397B, etc.). It's overkill for the 80% of Apple Silicon AI use that fits on a single station.

For that 80%, Telemak gives you :

- **Zero install friction.** No Python 3.11. No `mlx + mlx-lm` venv. No `bootstrap-node.sh`. No Docker. A binary you drop and run.
- **Native performance.** mlx-swift uses Metal directly. No Python/asyncio/multiprocessing overhead. Lower TTFT for small models.
- **One process, full control.** We own the codebase — no inherited defaults to fight (the oMLX `hot_cache_max_size:"0"` / `enable_thinking:true` gotchas don't exist here).
- **Same API surface.** OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`. Any client that speaks those (Companion, IDE agents, the SDKs) hits Telemak unchanged.

## The lineage

```
Odysseus    — the voyager (engine, distributed)
Telemak     — the son (engine, single-node native)        ← this repo
Companion   — Athena disguised as Mentor (client)
Ulysse      — memory of the journeys (RAG, future)
```

## Status

**V1 — ready for use.** V0 shipped 2026-05-23 (commit `55ec3f6`); V1 (commits `60c9b9f`+) adds multi-model, KV cache reuse, capability contract, Anthropic `/v1/messages`, tool calls, CORS + bearer auth + CLI + menu-bar app.

License : Apache 2.0 (planned, matches the rest of the OdyssAI stack).

## Quick start

```bash
# Build (requires Metal Toolchain: xcodebuild -downloadComponent MetalToolchain)
./scripts/build.sh Release

# Run
TELEMAK_MODELS_DIR=/Volumes/models/odysseus \
  ./scripts/run.sh serve --host 0.0.0.0 --port 8003

# Load + chat (in another terminal)
./scripts/run.sh load mlx-community/Qwen3-0.6B-4bit
./scripts/run.sh chat "Hello" --model mlx-community/Qwen3-0.6B-4bit
./scripts/run.sh models   # what's available on disk
./scripts/run.sh unload mlx-community/Qwen3-0.6B-4bit
```

## CLI subcommands

- `telemak serve` — run the HTTP server (default `127.0.0.1:8003`).
- `telemak smoke <prompt> --model <id>` — offline load+generate, no server.
- `telemak models [--server URL]` — list available models on disk.
- `telemak load <id> [--server URL]` — POST `/admin/load`.
- `telemak unload <id> | --all [--server URL]` — POST `/admin/unload`.
- `telemak chat <prompt> --model <id> [--server URL]` — one-shot chat.

## Menu bar app

```bash
./scripts/build.sh Release
./scripts/build-menubar-app.sh Release   # wraps into dist/Telemak.app
open dist/Telemak.app                     # first launch — see Gatekeeper below
```

A SwiftUI `MenuBarExtra` app. Click the status icon to:

- See **status** (running / stopped / unreachable), **loaded models**, **recent tok/s**, **request count**, **MLX wired memory**, **uptime**.
- **Start / Stop / Restart** the local `telemak serve` LaunchAgent (`eu.odyssai.telemak`). Disabled when the endpoint is remote — start/stop only controls the local agent.
- **Open Dashboard** — launches the configured URL in your browser (default: Odysseus dashboard at `http://192.168.86.141:8000/`).
- **Settings** — change the endpoint URL, dashboard URL, and poll interval (persisted in `defaults` under `eu.odyssai.telemak.menubar`).
- **Quit** — terminate the menu-bar app (telemak serve itself keeps running via launchd).

Install: drag `dist/Telemak.app` to `/Applications`. To auto-launch at login: System Settings → General → Login Items → `+` → pick Telemak.

### First run — Gatekeeper

Telemak isn't notarized (open-source, no Apple Developer ID). First time you double-click the `.app`, macOS will say *"can't be opened because Apple cannot check it for malicious software"*. Workaround:

1. **Right-click** the `.app` → **Open**.
2. macOS shows the same warning but with an "Open" button. Click it.
3. Subsequent double-clicks work normally.

This is one-time per machine. The `telemak` CLI binary doesn't trip Gatekeeper (it's not a `.app`).

## Configuration

| Env var | Purpose | Default |
|---|---|---|
| `TELEMAK_MODELS_DIR` | Odysseus-style models directory (`<root>/<org>/<name>/snapshots/<hash>/`) | unset → HF cache only |
| `HF_HUB_CACHE` | HuggingFace cache path | `~/.cache/huggingface/hub/` |
| `TELEMAK_API_KEY` | Optional bearer token for `/admin/*`; inference endpoints stay open for LAN routing | unset → open |
| `TELEMAK_CORS_ORIGIN` | `Access-Control-Allow-Origin` value | `*` |
| `TELEMAK_MAX_SESSIONS` | Max KV-cached sessions in memory | `32` |
| `TELEMAK_LOG_LEVEL` | `trace`, `debug`, `info`, `notice`, `warning`, `error`, `critical` | `info` |
| `TELEMAK_LOAD_DEBUG` | Non-empty → emit per-step timing of `ModelRegistry.load` to stderr | unset (silent) |

## Deploy gotcha — TCC permission per binary

macOS's Transparency, Consent, Control (TCC) framework grants Privacy &
Security permissions **per code-signature hash**, not per path. Telemak's
ad-hoc-signed binary gets a fresh hash on every Release build, which
invalidates the previous grant.

Symptom after a redeploy: `/admin/load` hangs indefinitely with the
process stuck in an `open()` syscall on `/Volumes/models/...`. The smoke
binary works fine from an SSH shell (Terminal already has Full Disk
Access) but the LaunchAgent-spawned `telemak serve` is denied silently.

Fix after every deploy:
1. **System Settings → Privacy & Security → Full Disk Access** on the
   target machine.
2. Find `telemak` — if listed with `✗`, remove with `−`.
3. Add `+` → navigate to `/Users/admin/telemak/Release/telemak` → toggle ON.
4. `launchctl kickstart -k gui/$(id -u)/eu.odyssai.telemak`.

Permanent fix (V1.5): codesign with a stable Developer ID so the
signature hash persists between builds.

## Documentation

- [`AGENTS.md`](AGENTS.md) — agent runbook : start here if you are a Claude Code / Codex / Cursor agent in this folder
- Upstream library : [`ml-explore/mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm)
- Reference application : [`ml-explore/mlx-swift-examples` Applications/MLXChatExample/](https://github.com/ml-explore/mlx-swift-examples/tree/main/Applications/MLXChatExample)
- Sibling engine : [`Odyssai-eu/Odysseus`](https://github.com/Odyssai-eu/Odysseus)
