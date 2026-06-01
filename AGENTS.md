# AGENTS.md — Telemak

> Runbook for a coding agent (Claude, Codex, Cursor, Aider…) picking up
> Telemak development with a fresh context. Read this top to bottom **before**
> opening a file — it explains where Telemak fits in OdyssAI's stack, how the
> build/deploy works (it has a few gotchas), what's already shipped, what
> issues are open right now, and the workflow rules.
>
> If anything in this doc contradicts what you observe in the code, the code
> is the source of truth — note the discrepancy and update this doc in your
> commit.

## 0. What Telemak is, today

Telemak is a **native macOS HTTP runtime for MLX inference, on one Mac**.
Current version : **v0.2.0** (running in production on `<telemak-host>.local` since
2026-05-23).

- Single Swift package, two binaries : `telemak` (CLI server) and
  `telemak-menubar` (status menubar app).
- Loads MLX-quantized models from `~/.cache/huggingface/hub/` (or any
  models_dir you point it at via `--models-dir`).
- Exposes an OpenAI-compatible HTTP API on `:8003` (default). Streaming SSE,
  `/v1/chat/completions`, `/v1/models`, `/admin/load`, `/admin/unload`,
  `/admin/api/global-settings`, `/health`, capability contract at
  `/.well-known/inference-engine.json`.
- Built on the **`Odyssai-eu/mlx-swift-lm`** fork (the operator's customised fork
  of `ml-explore/mlx-swift-lm`). The fork carries MTP / hidden-states work
  that upstream hasn't accepted yet.
- Registered in Odysseus' `topology.yaml` as a `backend: http-proxy`
  cluster, so Companion → Odysseus → Telemak routing works transparently.

**What Telemak is NOT** :

- A distributed engine (Odysseus does that — `backend: jaccl` / `ring` across
  N Macs)
- A chat client (Companion does that)
- A model converter (use `mlx_lm.convert` upstream)

If a task drifts toward those, stop and re-read this section.

## 1. Where Telemak fits in the OdyssAI stack

```
┌─────────────────────┐       ┌──────────────────────┐      ┌────────────────────┐
│   Companion (React) │  HTTP │  Odysseus (FastAPI)  │ HTTP │  Telemak (Swift)   │
│   thecompai/app     │ ────► │  scripts/api.py      │ ───► │  this repo         │
│   user UI + memory  │       │  orchestrator + LB   │      │  one-Mac inference │
└─────────────────────┘       └──────────────────────┘      └────────────────────┘
                                       │                              │
                                       │ jaccl / ring (TB5)            │ direct MLX
                                       ▼                              ▼
                              ┌──────────────────┐         ┌──────────────────────┐
                              │ Argo cluster     │         │ ~/.cache/huggingface │
                              │ LAN node group   │         │ + ~/Library/...      │
                              │ operator-owned   │         │                      │
                              └──────────────────┘         └──────────────────────┘
```

Telemak is the leaf : one Mac, one (or a few) loaded models, HTTP in/out.
Everything else (multi-cluster routing, conversation persistence, RAG,
embeddings semantic router, …) lives upstream.

## 2. Repo layout

```
telemak/
├── Package.swift                 # Swift 6.1 toolchain, mlx-swift 0.31.3, hummingbird 2.x
├── Package.resolved              # Pins mlx-swift-lm to fork branch feat/v2-mtp-hidden-states
├── Sources/
│   ├── Telemak/                  # The CLI server binary
│   │   ├── Telemak.swift         # ArgumentParser entry + version constant
│   │   ├── Engine/               # MLX wiring (model loading, generation)
│   │   │   ├── ModelRegistry.swift     # id → loaded ModelContainer
│   │   │   ├── Generation.swift        # the actual mlx-swift-lm calls
│   │   │   ├── Streaming.swift         # AsyncSequence<Token> → SSE
│   │   │   └── MTP/                    # MTP speculative decoding (V2 work in progress)
│   │   │       └── MTPSpeculativeIterator.swift
│   │   └── Server/               # HTTP routes (Hummingbird)
│   │       ├── Router.swift
│   │       ├── ChatCompletions.swift
│   │       ├── Models.swift
│   │       ├── Admin.swift
│   │       └── Capabilities.swift  # /.well-known/inference-engine.json
│   └── TelemakMenuBar/           # The status menubar `.app` (separate target)
├── Tests/                        # SwiftPM tests + integration helpers
├── docs/
│   ├── CODESIGNING.md            # The TCC re-prompt gotcha + how to sign stably
│   ├── V1-BUG-stream-usage-chunk-missing.md
│   ├── V1-BUG-thinking-and-reasoning-routing.md
│   ├── V1-TODO.md
│   ├── V1-UI-multi-model-dashboard.md
│   └── V2-MTP-DRAFT-PORT.md      # The MTP port architecture (read for issue #34)
├── scripts/
│   ├── build.sh                  # Release build via xcodebuild
│   ├── build-menubar-app.sh      # Wraps menubar into .app
│   └── run.sh                    # Local dev
├── dist/Telemak.app/             # Last menubar .app bundle (528K, just the wrapper)
└── .xcbuild/                     # xcodebuild derived data (gitignored)
```

## 3. Build system — IMPORTANT

### Use xcodebuild, not `swift build`

This is the single most important gotcha. **`swift build` produces a binary
that crashes at runtime** because it doesn't compile the Metal kernels
mlx-swift needs. Use the script :

```bash
./scripts/build.sh Release         # produces .xcbuild/Build/Products/Release/telemak
./scripts/build.sh Debug           # for development
```

The script wraps `xcodebuild -scheme Telemak-Package`. Output binaries land
in `.xcbuild/Build/Products/{Release,Debug}/` :

- `telemak` (~92 MB) — the CLI server. Statically embeds all mlx-swift libs.
- `telemak-menubar` (~530 K) — the menubar app body.
- `mlx-swift_Cmlx.bundle/` — Metal kernels. **MUST sit next to `telemak`
  at runtime**, otherwise the runtime exits with a Metal load error.

### Codesigning — TCC stability

Release builds **must** be codesigned with a stable identity. macOS TCC
(Full Disk Access, Removable Disks) keys grants by codesign cdhash. Without
a stable identity, every rebuild → fresh cdhash → TCC re-prompts → the
LaunchAgent boots before anyone clicks → service breaks.

The signing identity used today is documented in
[`docs/CODESIGNING.md`](docs/CODESIGNING.md). The build script signs
automatically when `CONFIGURATION=Release`. Do NOT remove or alter that
step.

### Cross-repo : the `mlx-swift-lm` fork

`Package.swift` pins **`Odyssai-eu/mlx-swift-lm` branch
`feat/v2-mtp-hidden-states`**, not the upstream `ml-explore/mlx-swift-lm`.
The fork lives at `<mlx-swift-lm-fork>/` on the operator's
workstation. It carries :

- MTP speculative decoding scaffolding (the hidden-states surface needed
  for MTP)
- Patches to `Qwen35TextModel` / `Qwen35GatedDeltaNet` for SSM rollback
  (in progress — see issue #34)
- (Otherwise tracks upstream)

When an issue says *"in the fork"*, it means a change to
`mlx-swift-lm-odyssai`, not Telemak itself. After the fork change lands,
Telemak's only follow-up is bumping the commit pin in `Package.swift` and
`Package.resolved`.

### Available mlx-swift-lm libraries

`mlx-swift-lm-odyssai/Libraries/` provides :

| Library | What it does | Telemak uses today ? |
|---|---|---|
| MLXLLM | Causal LMs (Qwen, Gemma, Llama, GLM, MiniMax, DeepSeek, …) | ✅ yes — primary |
| MLXLMCommon | Shared types | ✅ yes |
| MLXEmbedders | Embedding models (bert/roberta/nomic/qwen3/gemma3) | ❌ not yet — issue #37 |
| MLXVLM | Vision-language models (16 archis : Qwen35MoE, Qwen3VL, FastVLM, Gemma3, …) | ❌ not yet — issue #36 |
| MLXHuggingFace + Macros | HF Hub fetch | ✅ yes (transitive) |

The plan is to consolidate `mlx-coder` (Python), `mlx-embed` (Python) and
`mlx-vlm` (Python) onto Telemak, so a single Swift process owns local
inference on 64 GB node. Issues #37 and #36 track that.

## 4. Deploy — how Telemak runs in production

Production host : **`<telemak-host>.local`** (<telemak-host>, M3 Max 64 GB). Currently
also planned for `<telemak-host>.local` (<telemak-host>, M3 Ultra 96 GB) — issue
#38 tracks turning the deploy into a one-click DMG installer.

### Layout on the host

```
~/telemak/Release/
├── telemak                      # CLI binary
├── telemak-menubar              # menubar binary
└── mlx-swift_Cmlx.bundle/       # Metal kernels (REQUIRED)

~/Library/LaunchAgents/
├── eu.odyssai.telemak.plist
└── application.eu.odyssai.telemak.menubar.<...>.plist
```

### Deploy from workstation

```bash
./scripts/build.sh Release                                        # builds locally
scp -r .xcbuild/Build/Products/Release/{telemak,telemak-menubar,mlx-swift_Cmlx.bundle} \
    admin@<host>:telemak/Release/                                 # ship
ssh admin@<host> 'launchctl kickstart -k gui/$(id -u)/eu.odyssai.telemak'   # bounce
curl -s http://<host>:8003/health                                 # smoke
```

The current LaunchAgent definition is implicit (already on the host).
Issue #38 will formalise this with a proper installer DMG.

### What's NOT deployed

- Tests (`Tests/`) — local-only.
- Source (`Sources/`, `Package.swift`) — local-only ; binary-only deploy.
- `.xcbuild/` — gitignored, regenerated per build.

## 5. HTTP API — what Telemak exposes

The full contract is at `Sources/Telemak/Server/`. Highlights :

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat. Supports `stream:true` for SSE. |
| `POST /v1/messages` | Anthropic-compatible chat (basic). |
| `GET /v1/models` | Lists currently loaded models with capabilities. |
| `POST /admin/load` `{"model":"hf-id"}` | Load a model into memory. One concurrent at a time today. |
| `POST /admin/unload` `{"model":"hf-id"}` | Unload + free wired memory. |
| `GET /admin/api/global-settings` | Runtime settings (KV cache size, enable_thinking defaults, …). |
| `POST /admin/api/global-settings` | Update runtime settings. Some settings are runtime-applied, some need restart — check the response `runtime_applied` field. |
| `GET /.well-known/inference-engine.json` | Capability contract. Tells Odysseus what tools/vision/json-mode this engine supports. |
| `GET /health` | Liveness probe. Returns 200 + version. |

The shapes are kept byte-compatible with Odysseus' `scripts/api.py`. When
in doubt, **Odysseus is the source of truth for what Companion expects** —
match it.

## Security audit triage — OdyssAI posture

Directive as of **2026-05-31** : OdyssAI products are **self-hosted,
LAN-first, mono-operator**. We secure the code; we do **not** manage the
client's network security policy. Sort every security-audit finding into one
of these buckets before changing behavior.

### Bucket A — Always fix (code hygiene, not policy)

- **Exploitable code** : injection (SSH `ProxyCommand`, path traversal,
  process spawn from request bodies), RCE, SSRF, unsafe deserialization.
  Network topology is irrelevant — fix it.
- **No undocumented shared hardcoded secret/credential** in source, docs, or
  templates. However, a documented generic default password that must be
  changed at first login is acceptable and intentional for products that use
  that model. Do **not** replace it with a silent per-install random secret.
  Keep environment overrides.

### Bucket B — Operator choice (option + docs, never forced)

- Bind interface (`0.0.0.0` LAN vs `127.0.0.1`), mandatory API key, WAN
  exposure.
- Default must stay LAN-friendly and usable without configuration.
  Hardening (API key, localhost-only, admin auth) is documented opt-in, not
  default.
- WAN exposure belongs to the operator: tunnel, firewall, reverse-proxy auth,
  IP/MAC allowlist. We can recommend Cloudflare Tunnel, allowlists, or
  reverse-proxy auth in docs; we do not impose a policy in Telemak.
- Cross-repo constraint : Odysseus (`.39`) proxies Telemak nodes over the LAN
  without sending a key. Therefore Telemak must not force `127.0.0.1` on
  orchestrated nodes. If a Telemak API key ever becomes mandatory, Odysseus
  must first learn to forward `Authorization: Bearer ...` on all upstream
  calls, otherwise production breaks.

### Concrete re-triage

- Odysseus #20 (admin open by default) → **Bucket B** : keep open on LAN,
  document opt-in hardening. Not a bug.
- Telemak #53 (bind/auth) → **Bucket B** : LAN bind + optional key; never
  force `127.0.0.1` on nodes.
- Injection / path traversal / SSRF findings → **Bucket A** : fix.
- Companion #4 (password) → default documented `itak1234`, no random
  silent lockout.

Golden rule : **secure the code, yes. Impose a network posture on the client,
no — provide options and guidance.**

## 6. Current state — what works, what's in flight

### What works in production today

- ✅ Chat completion (streaming + non-streaming) on Qwen3, Qwen3MoE,
  Qwen3.5, Qwen3.5MoE, Qwen3.6, Gemma3/4, GLM4MoE, DeepSeek, MiniMax,
  Mistral3 — basically every model architecture in `MLXLLM`.
- ✅ `/admin/load` + `/admin/unload` runtime model swap.
- ✅ Capability contract at `/.well-known/inference-engine.json` (used by
  Odysseus to negotiate features).
- ✅ Codesigning + LaunchAgent on 64 GB node — service survives reboots.
- ✅ KV cache (per-conversation prefix reuse) — sessions stay warm across
  turns.
- ✅ MTP V1 iterator (`MTPSpeculativeIterator.swift`) shipped in issue #29
  / PR #32. **But the acceptance rate collapses on Qwen3.5/3.6 because SSM
  rollback is missing** — see issue #34.

### What's in flight (open issues, ordered by where dev should start)

| # | Title | Labels | Difficulty |
|---|---|---|---|
| **#34** | V2 Step 2 — `rollback_speculative_cache` in mlx-swift-lm fork (SSM state) | `ready` | 8 |
| **#37** | feat(embeddings) : `/v1/embeddings` endpoint backed by `MLXEmbedders` | `ready` | 3 |
| **#36** | feat(vlm) : real image input on `/v1/messages` + `/v1/chat/completions` | `ready` | 5 |
| **#35** | V2 Step 4 — wire `MTPSpeculativeIterator` into `/v1/chat/completions` + `/v1/messages` | `blocked` (by #34) | 3 |
| **#38** | feat(installer) : one-click DMG installer for non-dev machines | `enhancement` (not ready yet) | 8 |

**Pick the lowest-numbered `ready` issue first**, unless the priority label
disagrees (`bug` > `enhancement`). #34 is the highest-impact ready item
(unblocks MTP perf, unblocks #35).

### What's deferred (out of scope until further notice)

- Bonjour LAN auto-discovery (operator still writes IP in `topology.yaml`)
- Apple notarization (still ad-hoc signed)
- App Store distribution
- Multi-request concurrent inference (single-flight today)
- Auto-update mechanism (Sparkle)
- Multi-version side-by-side install

## 7. Workflow — how to ship

### Direct push to `main`, no PR

As of **2026-05-25 night** (see `~/.claude/CLAUDE.md`), the project uses
direct push to main. No feature branches, no PR review. Rationale : on a
solo-operator-plus-one-agent stack, PR review added friction without
catching bugs, and divergence between deployed code and `main` became a
recurring incident.

The new invariant : **"ce qui est sur le serveur = ce qui est sur main."**
Deploy follows the commit in the same session.

### Per-issue cycle

```
1. gh issue list --label ready → pick lowest open
2. Read the issue + Reading order docs cold
3. git checkout main; git pull
4. Implement on main directly (no branch)
5. Smoke / test the deployed runtime
6. Commit + push (one coherent commit per issue when possible)
7. Deploy in the same session (scp + launchctl kickstart, see §4)
8. Comment on the issue with the recap + smoke output
9. Issue auto-closes via `Closes #N` in the commit body
```

### Commit conventions — strict

Commits ARE the audit trail now that PRs are gone. The format is enforced :

```
$kind($scope): $short imperative title

Closes #$N.

$2-4 lines of body describing what landed and why.

Difficulty: $N delivered (issue estimated $original-N).
Smoke: $command → $short observation

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
```

Rules :

- **Conventional Commits prefix** : `feat:`, `fix:`, `chore:`, `docs:`,
  `perf:`, `refactor:`. The optional `($scope)` is `mtp`, `vlm`, `embed`,
  `server`, `engine`, `build`, etc.
- **HEREDOC** for the body (multi-line commit messages).
- **`Closes #N`** so the issue auto-closes on push to main.
- **`Difficulty: N delivered`** line — used for velocity tracking.
- **`Co-Authored-By:`** footer — adapt to which model you're running on.
  For Codex : `Co-Authored-By: Codex (GPT-X) <noreply@openai.com>`.
- **NEVER `--no-verify`** — pre-commit hooks exist for a reason.

### Difficulty estimates — Fibonacci, never time

Per `~/.claude/CLAUDE.md`. Time estimates were wrong systematically.
Estimate in Fibonacci scrum points : 1 / 2 / 3 / 5 / 8 / 13 / 21.

| Points | Sense |
|---|---|
| 1 | one-line patch, rename, toggle. |
| 2 | small — 1-2 loops, one file mostly. |
| 3 | medium — clear scope, a few files. |
| 5 | bigger — multi-file, design decisions to make. |
| 8 | large — full feature, several modules. Usually break down further. |
| 13 | very large — break into sub-issues before committing. |
| 21 | epic — almost never. If you reach 21, the scope is wrong. |

The commit's `Difficulty: N delivered` line is what counts for velocity.

## 8. Smoke / verify — how to know it works

After every deploy, run these against the live host :

```bash
HOST=<telemak-host>.local   # or <telemak-host>.local, or wherever you deployed

# 1. Health
curl -s http://$HOST:8003/health
# → {"status":"ok","version":"0.2.0"}

# 2. Loaded models
curl -s http://$HOST:8003/v1/models | jq '.data[].id'

# 3. Capability contract
curl -s http://$HOST:8003/.well-known/inference-engine.json | jq

# 4. Non-streaming chat
curl -s -X POST http://$HOST:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<loaded-id>","messages":[{"role":"user","content":"hi"}],"stream":false,"max_tokens":20}'

# 5. Streaming chat
curl -N -X POST http://$HOST:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<loaded-id>","messages":[{"role":"user","content":"count to 5"}],"stream":true,"max_tokens":40}'

# 6. Companion → Odysseus → Telemak round-trip (full integration smoke)
# the operator validates from the Companion UI ; mention it in your commit body.
```

If any of these fail after your deploy, **revert the commit on main and
redeploy the prior build** — don't leave the service broken. The
invariant must hold.

## 9. When to escalate — `needs-human` label

Add the `needs-human` label and **stop** when :

- An issue depends on a change in `mlx-swift-lm-odyssai` AND the fork
  doesn't have an obvious bump path (e.g. needs the operator's call on
  upstream rebase).
- A model's chat template breaks `<|im_end|>` parsing or similar
  template ambiguity — chat-template work is fiddly and the operator wants
  to see those.
- A change would alter the HTTP API surface (`/v1/chat/completions`
  shape) — Companion + Odysseus need to know.
- A change would touch macOS TCC / codesigning / LaunchAgent setup —
  these are physical-presence-required changes on the host.
- Two valid implementations exist and you can't tell which the operator
  prefers.
- Cost / runtime ceiling hit on a long-running task.

Tag, comment with the specific question, stop. PO will pick it up and
either resolve or dispatch to the operator.

## 10. Cross-repo links — when something points elsewhere

- **Odysseus** : `~/Claude/code/MLX Distributed/` (or
  `https://github.com/Odyssai-eu/Odysseus`). The orchestrator. Read
  `scripts/api.py` if you need the exact response shape Companion expects.
- **Companion** : `~/Claude/code/thecompai/app/` (private repo, currently
  `https://github.com/thecompai/app`). The chat UI. Read
  `server/routes/chat.ts` if you change how tools/reasoning interleave.
- **mlx-swift-lm-odyssai** : `<mlx-swift-lm-fork>/` (fork
  at `https://github.com/Odyssai-eu/mlx-swift-lm`). the operator's fork of
  `ml-explore/mlx-swift-lm`. Issue #34 touches this fork, not Telemak.
- **odyssai-services** : sibling cockpit container, `<odyssai-services-host>:8001`. Hosts
  the bench tool. Use it to measure tok/s changes after MTP work lands.
- **Obsidian wiki** : `~/Claude/code/odyssai-wiki/` — cross-repo concept
  notes. Articles relevant to Telemak : `[[telemak-runtime]]`,
  `[[http-api-contract]]`, `[[topology-yaml]]`, `[[http-proxy]]`,
  `[[mtp-speculative-decoding]]`.

## 11. Concrete onboarding steps for a fresh agent

If you've never seen this repo before, do this in order :

1. **Read this file** (you're here) — top to bottom.
2. **Read `docs/V2-MTP-DRAFT-PORT.md`** — the active design doc for MTP.
3. **Skim `Sources/Telemak/Server/Router.swift`** — see all routes.
4. **Skim `Sources/Telemak/Engine/ModelRegistry.swift`** — see how models
   are loaded.
5. **Read `Package.resolved`** — verify the fork pin matches what
   `scripts/build.sh` builds against.
6. **Read the open `ready` issues** : `gh issue list --label ready` then
   `gh issue view <N>` for each.
7. **Build locally** : `./scripts/build.sh Debug && .xcbuild/Build/Products/Debug/telemak --version`. Confirm `0.2.0` (or whatever the current source says).
8. **Curl the live host** (`<telemak-host>.local:8003`) — see §8 — confirm the
   prod runtime matches what you just built. If it doesn't, that's your
   first task : align deploy with main.

After that, you're ready to pick an issue.

## 12. Don't

- Don't `swift build` — use `./scripts/build.sh`.
- Don't deploy without codesigning (Release builds, see §3).
- Don't open a PR (workflow is direct push to main, see §7).
- Don't add new HTTP endpoints without checking Companion + Odysseus
  consumers first.
- Don't bump the `mlx-swift-lm` fork without testing locally — runtime
  Metal errors don't always show up at build time.
- Don't write time estimates — Fibonacci points only.
- Don't leave the deployed runtime broken — revert the commit if smoke
  fails, ship a follow-up.

## 13. When you're done with an issue

Comment on the issue :

```markdown
Shipped on main as $commit-hash, deployed to $host.

$brief description of what landed.

Smoke : $command → $observation
Difficulty: $N delivered.
```

That seals the issue (which auto-closed via `Closes #N` on push). Move on
to the next `ready` issue. If the backlog is empty, idle until the PO
files more or the operator dispatches something.
