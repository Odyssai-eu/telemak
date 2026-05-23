# AGENTS.md — Telemak

> Runbook for an AI coding agent (Claude Code, Codex, Cursor, Aider…) building
> Telemak's first MVP. The repo is empty. Your job is to scaffold a Swift
> package, wire up `mlx-swift-lm`, expose an HTTP `/v1/chat/completions` with
> token streaming, and prove that a small Qwen or Gemma model responds end-to-end.
>
> Read [README.md](README.md) first for the product framing. Then read this
> file top-to-bottom — the order matters.

> **Permissions** : ce projet a `bypassPermissions` activé dans
> `.claude/settings.local.json`. L'agent peut écrire/exécuter/installer sans
> prompter — conçu pour des sessions overnight unattended. Le scope est
> verrouillé sur `~/Claude/code/telemak/` + (lecture) `~/Claude/code/odyssai-wiki/`
> + (lecture) `~/Claude/code/MLX Distributed/`.
>
> **Knowledge base** : `/Users/sophie/Claude/code/odyssai-wiki/` est la vault
> Obsidian centrale d'OdyssAI. Consulte-la pour le contexte cross-projet
> (HTTP API contract, topology.yaml, http-proxy backend, glossary). Notes
> atomiques avec wikilinks. Articles clés pour Telemak :
> [[telemak-runtime]], [[http-api-contract]], [[topology-yaml]], [[http-proxy]].

## 0. What Telemak is (and isn't)

**Telemak is a native macOS HTTP runtime for MLX inference, on one Mac.**
- One Swift binary (CLI for MVP, `.app` later)
- Loads MLX-quantized models from the local Hugging Face cache (`~/.cache/huggingface/hub/`)
- Exposes OpenAI- and Anthropic-compatible HTTP API on a configurable port (default `:8002`)
- Built on Apple's [`ml-explore/mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm) (released as a standalone library — separate from `mlx-swift-examples`)

**Telemak is NOT :**
- A distributed engine (Odysseus does that — `backend: jaccl` / `ring` across N Macs)
- A chat client (Companion does that)
- A model converter (use mlx-lm's `mlx_lm.convert` upstream)
- A RAG system (Ulysse will do that, future)

If your task drifts toward any of the above, STOP and ask the user.

## 1. Context you should have read

Before writing any Swift code, skim these :

| Resource | Why |
|---|---|
| [`ml-explore/mlx-swift-lm` README](https://github.com/ml-explore/mlx-swift-lm) | The library you're using. Look at the `ChatSession` example and the `LLMRegistry`. |
| [`ml-explore/mlx-swift-examples` `Applications/MLXChatExample/`](https://github.com/ml-explore/mlx-swift-examples/tree/main/Applications/MLXChatExample) | A full Swift chat UI built on the same library. **Best concrete reference for how to load a model, stream tokens, manage state.** Skim Services/ and ViewModels/. |
| `Libraries/MLXLLM/Models/*.swift` in `mlx-swift-lm` | The supported model architectures. Pre-verified : Qwen2 / Qwen3 / Qwen3MoE / Qwen3Next / Qwen35 / Qwen35MoE / Llama / Gemma / Gemma2 / Gemma3Text / Gemma4 / DeepseekV3 / GLM4 / GLM4MOE / MiniMax / Mistral3Text / Phi / Phi3 / PhiMoE / Starcoder2 / Olmo / SmolLM3 / OpenELM / Granite / Cohere / and ~20 more. |
| [`Odyssai-eu/Odysseus` topology.example.yaml](https://github.com/Odyssai-eu/Odysseus/blob/main/config/topology.example.yaml) | How Telemak will be registered in an Odysseus cluster (`backend: http-proxy`, `upstream: http://this-mac:8002`). |
| [`Odyssai-eu/Odysseus` API surface](https://github.com/Odyssai-eu/Odysseus/blob/main/docs/API.md) | The exact `/v1/chat/completions` request/response shape you must match (Odysseus uses the OpenAI spec verbatim). |

## 2. Architecture decision — locked

These choices are decided. Don't relitigate them in MVP :

| Choice | Decision | Rationale |
|---|---|---|
| Language | **Swift 6** (toolchain available with Xcode 15+) | Required by mlx-swift-lm |
| Inference library | **`mlx-swift-lm`** | Has every architecture in the Odysseus catalog (Qwen3MoE, Qwen3Next, Qwen35MoE, GLM4MOE, MiniMax, DeepseekV3, Gemma4, etc.) |
| HTTP framework | **Hummingbird 2.x** | Lighter than Vapor, async-first, designed for server-side tasks ; minimal deps (no ORM, no auth-bundled). Vapor was considered and rejected — too much overhead for a single-purpose inference daemon. |
| Build target | **CLI executable** for MVP (`swift build`) ; `.app` is V1 | Easier to test, debug, iterate. Menu-bar UI is a polish layer. |
| Default port | **`:8002`** | `:8000` is taken by Odysseus orchestrator, `:8001` by odyssai-services. |
| Tokenizer | What `mlx-swift-lm` ships (`huggingface/swift-transformers` under the hood) | No need to re-implement. |
| Streaming protocol | **SSE** (`data: {...}\n\ndata: [DONE]\n\n`), OpenAI-compatible | Companion and the SDKs already speak this. |
| Streaming primitive in Swift | `AsyncSequence<Token>` from `ChatSession` → Hummingbird streaming `ResponseBody` | mlx-swift-lm exposes this natively. |
| Concurrency | One request at a time in MVP (single inference at any moment, serialize with an actor or single async task) | Multi-request queueing is V1+. |
| Config file | TOML or YAML, at `~/.telemak/config.yaml` | Single file ; deferred-decision : whichever is easier with Swift std/Foundation. Default to TOML if no preference, smaller dep footprint. |
| Model loading | On-demand via `/admin/load`, NO startup-load | Mirror Odysseus admin API — operators bring models up explicitly. |

If you find yourself wanting to deviate, write a one-paragraph proposal and ask the user first.

## 3. MVP scope — six phases

The MVP target is **"`curl localhost:8002/v1/chat/completions` returns a streaming SSE response from a Qwen3 dense model."** Everything past that is V1.

| # | Phase | Deliverable | Time est. | Done when |
|---|---|---|---|---|
| 0 | **Scaffolding** | Swift package `Telemak`, deps `mlx-swift-lm` + `Hummingbird`, builds `swift run telemak --version` | ½ day | The binary prints a version string |
| 1 | **Load + generate** | Load `mlx-community/Qwen3-7B-MLX-8bit` (or smaller for first iteration) from the local HF cache, generate 50 tokens, print them | ½ day | `swift run telemak smoke "hello world"` prints a coherent completion to stdout |
| 2 | **HTTP non-streamed** | Hummingbird app on `:8002`, `POST /v1/chat/completions` returns a single complete JSON response | 1 day | `curl -d '{"model":"...", "messages":[...]}' http://localhost:8002/v1/chat/completions` returns OpenAI-shaped JSON |
| 3 | **Streaming SSE** | Same endpoint with `"stream": true`, token-by-token deltas | ½ day | `curl --no-buffer ...` shows tokens arriving progressively |
| 4 | **Odysseus integration** | Register Telemak as a cluster in `topology.yaml` (`backend: http-proxy`, `upstream: http://<this-mac>:8002`). Verify Companion → Odysseus → Telemak path works. | ½ day | A Companion message routed to this cluster gets answered by Telemak |
| 5 | **/v1/models + /admin/load + /admin/unload** | List loaded models ; load/unload at runtime without restart | 1 day | A second model can be loaded after the first, then unloaded, then a third loaded — all without restart |

**Total V0 (phases 0-3) : ~2.5 days.** Total integrated MVP (0-5) : ~4 days.

Phases run sequentially. Do not start phase N+1 until phase N has a passing smoke test that the user has seen.

## 4. Phase 0 — Scaffolding (start here)

### 4.1 Create the Swift package

```bash
cd /Users/sophie/Claude/code/telemak
swift package init --type executable --name Telemak
```

This creates `Package.swift`, `Sources/Telemak/Telemak.swift`, `Tests/`.

### 4.2 Add dependencies to Package.swift

Edit `Package.swift` so it looks roughly like :

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Telemak",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "telemak", targets: ["Telemak"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "0.30.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.0.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "Telemak",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "TelemakTests",
            dependencies: ["Telemak"]
        ),
    ]
)
```

> **Version pinning** : check the latest tagged release of `mlx-swift-lm` at https://github.com/ml-explore/mlx-swift-lm/releases before committing the version range above — these numbers are approximate. The exact version may have moved by the time you read this.

### 4.3 Smoke build

```bash
swift build
swift run telemak --version
```

If `swift build` fails on a missing platform or Swift version, raise to the user — don't downgrade dependencies silently.

### 4.4 Project layout (recommended)

```
Telemak/
├── Package.swift
├── Sources/Telemak/
│   ├── main.swift             # ArgumentParser entry, dispatches subcommands
│   ├── CLI/
│   │   ├── ServeCommand.swift # `telemak serve` — runs the HTTP server
│   │   └── SmokeCommand.swift # `telemak smoke "<prompt>"` — phase 1 offline test
│   ├── Server/
│   │   ├── Router.swift       # Hummingbird routes wiring
│   │   ├── ChatCompletions.swift   # /v1/chat/completions
│   │   ├── Models.swift            # /v1/models
│   │   └── Admin.swift             # /admin/load, /admin/unload
│   ├── Engine/
│   │   ├── ModelRegistry.swift     # Maps requested model id → loaded ModelContainer
│   │   ├── Generation.swift        # The actual mlx-swift-lm calls (load, generate)
│   │   └── Streaming.swift         # AsyncSequence<Token> → SSE bytes
│   └── Types/
│       ├── ChatRequest.swift       # Codable OpenAI request shape
│       └── ChatResponse.swift      # Codable OpenAI response shape (incl. delta)
└── Tests/TelemakTests/
    └── ChatCompletionsTests.swift  # Hummingbird test client + golden responses
```

This layout is a *recommendation*, not law. Keep it flat enough that the code is readable.

## 5. Phase 1 — Load + generate offline

Goal : prove the library works on this Mac, with a real model, before involving HTTP.

```bash
# Pre-flight: ensure a model is in the local HF cache
huggingface-cli download mlx-community/Qwen3-7B-MLX-8bit \
  --local-dir ~/mlx-models/mlx-community/Qwen3-7B-MLX-8bit
# (or whatever cache strategy mlx-swift-lm uses — check its docs)

# Test
swift run telemak smoke "Hello, who are you?"
# → expect a coherent completion of 30-100 tokens on stdout
```

`SmokeCommand.swift` should use the simplest possible path :

```swift
import MLXLLM
import MLXLMCommon

let model = try await loadModelContainer(...)   // see mlx-swift-lm docs for exact API
let session = ChatSession(model)
let response = try await session.respond(to: prompt)
print(response)
```

If the load fails because the model isn't in the cache, print a clear error pointing at `huggingface-cli download`. Do not auto-download in phase 1 — keep failure modes obvious.

## 6. Phase 2 — HTTP non-streamed

Implement `POST /v1/chat/completions` returning a single JSON response (no streaming yet). Reference shape :

**Request** :
```json
{
  "model": "mlx-community/Qwen3-7B-MLX-8bit",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Say hello."}
  ],
  "stream": false,
  "max_tokens": 256,
  "temperature": 0.7
}
```

**Response** (OpenAI shape, abbreviated) :
```json
{
  "id": "chatcmpl-<uuid>",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "mlx-community/Qwen3-7B-MLX-8bit",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "Hello!"},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
}
```

The Odysseus repo's `scripts/api.py` has the canonical response shape — match it byte-for-byte where you can. Companion is strict about the shape.

Test :
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen3-7B-MLX-8bit","messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

## 7. Phase 3 — Streaming SSE

Same endpoint, with `"stream": true` in the request. Emit :

```
data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":..., "model":"...", "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":..., "model":"...", "choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":..., "model":"...", "choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-x","object":"chat.completion.chunk","created":..., "model":"...", "choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Hummingbird supports streaming responses via `ResponseBody` (chunked). `ChatSession` in `mlx-swift-lm` exposes the per-token stream as an `AsyncSequence` — wire one into the other.

Test :
```bash
curl --no-buffer -N -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"...","messages":[...],"stream":true}'
# Tokens should appear progressively, not all at once.
```

## 8. Phase 4 — Odysseus integration

You don't modify Odysseus. You add a cluster entry to the operator's `~/.odysseus/topology.yaml` :

```yaml
clusters:
  default:
    label: "Argo (distributed)"
    backend: jaccl
    # ... existing nodes …

  telemak-test:               # ← new
    label: "Telemak (single-node)"
    backend: http-proxy
    upstream: http://<this-mac-host>:8002
    pools:
      - size: 1
        nodes:
          - rank: 0
            id: telemak
            ssh: admin@<this-mac-host>     # SSH is still required by Odysseus for health probes
```

> Verify the exact field names with the live `config/topology.example.yaml` in the Odysseus repo — the schema may have evolved since this doc was written.

Then test :
```bash
curl -X POST http://<odysseus-host>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"telemak-test","messages":[{"role":"user","content":"Hello"}]}'
# Should be proxied through Odysseus to Telemak and return.
```

If Companion is configured to point at Odysseus, the `telemak-test` cluster will appear in its cluster list automatically.

## 9. Phase 5 — /v1/models + /admin/load + /admin/unload

Mirror the Odysseus admin pattern :

- `GET /v1/models` → list of `{id, object: "model", created, owned_by}` entries for currently loaded models
- `POST /admin/load` `{"model": "<hf-repo>"}` → load a model into the engine (the requested model becomes available for `/v1/chat/completions`)
- `POST /admin/unload` `{"model": "<hf-repo>"}` → unload a model (frees memory)

A single Telemak process holds **one** loaded model at a time in MVP. Loading a different model unloads the previous one. (Multi-model concurrent is V1.)

Operators address Telemak directly via these endpoints OR via Odysseus's `/admin/clusters/<id>/load` (which proxies through).

## 10. Out of scope for MVP — do not implement these

| Item | Why deferred |
|---|---|
| Anthropic `/v1/messages` API | Not required to integrate with Odysseus (which converts at its layer). V1. |
| KV prefix cache (session_id reuse across turns) | Significant work, mlx-swift-lm exposes KVCache but the cross-request session pattern is non-trivial. V2. |
| Capability contract (`/.well-known/inference-engine.json`) | Odysseus can hardcode the assumed capabilities of Telemak for V0. V1. |
| Multi-request concurrency | Single-flight is fine for V0. V1. |
| Tools / function calling | Requires chat-template-level work. V1. |
| Bonjour LAN discovery | Operator writes the IP in topology.yaml. V2. |
| Menu bar `.app` UI | CLI is sufficient for V0 demonstration. V1. |
| Apple notarization / codesigning | Required for distribution but not for local dev / testing. V1. |
| `.app` installer / drag-and-drop | Same as above. V1. |
| Sampling params beyond temperature / top_p / max_tokens | Stop sequences, repetition_penalty, etc. are V1. |
| Token usage / billing metrics | V1. |
| Logging to file with rotation | Stdout is fine for V0. V1. |
| App Store distribution | Far future. |

If the user explicitly asks for one of these mid-MVP, stop and confirm — they're scope creep.

## 11. Done criteria for the MVP

Telemak V0 is **done** when, on a single Mac with a model in `~/.cache/huggingface/hub/` :

1. `swift run telemak serve` starts the HTTP server on `:8002` in under 5 seconds (cold start, before any model load)
2. `POST /admin/load` with a valid model id loads the model (cold load ~30 s for a 7B-8bit) and returns success
3. `POST /v1/chat/completions` (non-streaming) returns a coherent OpenAI-shaped JSON response in under 5 s for a 100-token completion
4. `POST /v1/chat/completions` (streaming) emits the first token in under 1 s
5. Adding `telemak-test` as a `backend: http-proxy` cluster in Odysseus' `topology.yaml` makes the model reachable via Odysseus' OpenAI surface
6. `GET /v1/models` lists the loaded model
7. `POST /admin/unload` frees the model and unallocates the wired memory

All seven, end-to-end, observed by the user.

## 12. When you get blocked

- **`swift build` fails with cryptic linker errors** → check Xcode CLT version, ensure Swift 5.9+, check that mlx-swift-lm has a tagged release compatible with your Swift toolchain
- **Model load fails with "not found"** → check the cache path mlx-swift-lm expects. Look at MLXChatExample's `Services/ModelService.swift` (or wherever it loads from) for the canonical pattern
- **Streaming chunks arrive in batches, not per-token** → likely a Hummingbird response-body flush issue. Check that you're calling the equivalent of `Flush()` after each chunk, or that the framework's streaming body type is the right one (not `ByteBuffer`-collected)
- **Response shape diverges from OpenAI by a small field** → match Odysseus byte-for-byte. The `scripts/api.py` in the Odysseus repo is the source of truth for what Companion expects
- **The model architecture you want isn't in `Libraries/MLXLLM/Models/`** → it probably is, double-check the architecture name in the model's `config.json`. If genuinely missing, raise to user — porting an architecture is V1+, not MVP

## 13. Tell the user when you're done

When the 7 done-criteria above all pass, post this to the user :

> Telemak MVP V0 is up. The binary at `<path>` serves `/v1/chat/completions`
> (streaming and non-streaming) on `:8002`. A loaded model responds in
> `<X>` seconds first-token, `<Y>` tok/s steady-state on this Mac
> (`<host>`, `<chip>`, `<RAM>`). Registered as cluster `telemak-test` in
> Odysseus' topology — Companion sees it and chats successfully.
>
> Next : V1 should pick up [list your top 2-3 follow-ups : Anthropic API,
> KV cache, .app bundle, …].

Then stop. The user takes the next decision.
