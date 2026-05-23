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

**Pre-MVP — scaffolding stage.** See [`AGENTS.md`](AGENTS.md) for the runbook to build the V0.

License : Apache 2.0 (planned, matches the rest of the OdyssAI stack).

## Documentation

- [`AGENTS.md`](AGENTS.md) — agent runbook : start here if you are a Claude Code / Codex / Cursor agent in this folder
- Upstream library : [`ml-explore/mlx-swift-lm`](https://github.com/ml-explore/mlx-swift-lm)
- Reference application : [`ml-explore/mlx-swift-examples` Applications/MLXChatExample/](https://github.com/ml-explore/mlx-swift-examples/tree/main/Applications/MLXChatExample)
- Sibling engine : [`Odyssai-eu/Odysseus`](https://github.com/Odyssai-eu/Odysseus)
