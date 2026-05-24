# Telemak V2 — Port `qwen3_5_mtp` draft architecture

> **Goal** : enable **MTP speculative decoding** in Telemak so
> Qwen3.6-35B-A3B serves at ~76 tok/s instead of ~47 tok/s on max-64
> (1.6× speedup demonstrated by Inferencer.app on the same model
> with the same draft repo).
>
> Pick this up **after V1 is shipped** (multi-model UI + KV cache work
> already landed in V1 ; the `SpeculativeTokenIterator` wiring V1's
> classic-spec exploration touched is reused for V2).

## Status — blocker LIFTED 2026-05-24

Earlier note (2026-05-24 morning agent session) claimed V2 was blocked
by `Qwen35DecoderLayer` / `Qwen35Attention` / `Qwen35SparseMoeBlock`
being `final class` non-public in mlx-swift-lm. That analysis was
incomplete : **there's an MIT-licensed, MLX-native reference
implementation in Python** that covers the entire stack. Porting
Python → Swift is straightforward since the MLX APIs are nearly 1:1
across the two language bindings.

The new plan is the "Inferencer route" — port the Python reference to
Swift and own the model layer end-to-end. Independence from
mlx-swift-lm's visibility constraints on the affected classes.

## Status — Units 1 + 3 + Bonus SHIPPED (2026-05-24 evening)

Three merged PRs : [#25](https://github.com/Odyssai-eu/telemak/pull/25)
(Unit 1 architecture port), [#26](https://github.com/Odyssai-eu/telemak/pull/26)
(intermediate docs), [#27](https://github.com/Odyssai-eu/telemak/pull/27)
(Unit 3 API surface + draft loader + split-mtp endpoint + capability
contract bump to v0.3.0).

What's live on `inferencerlabs/Qwen3.6-35B-A3B-MTP-MLX-9bit` :

- Build : xcodebuild green (Debug + Release).
- API : `POST /admin/load` accepts `draft_model`. Engine
  capability advertises `speculative_decoding: {modes: ["mtp_adapter"],
  active_pairs: []}` and flips `supported: true` automatically when
  a pair is loaded.
- Loader path : `MTPModelLoader.load(identifier:)` short-circuits
  `LLMTypeRegistry` (our draft is `Module + BaseLanguageModel`, not
  `LanguageModel`), stages config, runs `MLXLMCommon.loadWeights`
  with the quantization settings the safetensors carry.
- Registry pairing : `loadDraft(_:pairedWith:)` + `unload(_:)` work
  in either direction (unload main drops paired draft ; unload draft
  clears back-reference).
- Bonus : `POST /admin/models/split-mtp` shells out to
  `mlx_vlm.speculative.drafters.qwen3_5_mtp.split` so an operator
  can pop an MTP drafter out of any Qwen3.5/3.6 source from the
  dashboard.

Smoke status : engine v0.3.0 deployed to max-64. Capability endpoint
serves the new shape ; load with a draft was blocked by **TCC Full
Disk Access** — each new binary signature invalidates the prior
grant, the operator (Sophie) must approve the new binary in System
Settings → Privacy & Security → Full Disk Access before the loader
can read `/Volumes/models`. Not a code bug.

Once TCC is regranted, the load should complete : verified all 47
safetensor keys in `Qwen3.6-35B-A3B-MTP-MLX-9bit/model.safetensors`
match the ModuleInfo paths in `Qwen35MTPDraftModel` +
`MTPDecoderLayer` + `MTPAttention` + `MTPSparseMoeBlock` exactly.

## Unit 2 — the speculative loop

### Step 1 (SHIPPED) — mlx-swift-lm fork wired

[Odyssai-eu/mlx-swift-lm](https://github.com/Odyssai-eu/mlx-swift-lm)
branch `feat/v2-mtp-hidden-states` adds three thin public helpers
on `Qwen35TextModel` / `Qwen35Model` :

```swift
public func forwardWithHidden(_ inputs: MLXArray, cache: [KVCache]?) -> (logits, hidden)
public func embed(_ inputs: MLXArray) -> MLXArray              // borrow target's embed_tokens
public func applyLMHead(_ hidden: MLXArray) -> MLXArray        // borrow target's lm_head
```

Telemak's `Package.swift` now pins the fork (Odysseus-eu URL +
`feat/v2-mtp-hidden-states` branch). Build green against the fork ;
upstream main is otherwise unchanged.

### Step 2 (NOT YET) — `rollback_speculative_cache` on the main model

When the target rejects a draft proposal, the main model's KV caches
have to roll back N positions AND — for Qwen3.5/3.6 — the SSM /
linear-attention caches need their intermediate state restored from
a per-step buffer. Roughly 40% of layers in Qwen3.6-35B-A3B are
linear-attention, so a "trim only KV" simplification doesn't work
— the model's internal state diverges immediately.

The Blaizzy Python implementation (
`mlx_vlm/models/qwen3_5/language.py:690 — rollback_speculative_cache`
) does this with a `gdn_states` buffer captured during the verify
forward. Porting to Swift requires :

- Modifying `Qwen35GatedDeltaNet.callAsFunction` to optionally
  capture per-step intermediate states (~30 LOC, in the fork).
- Implementing the rollback logic on `Qwen35TextModel` (~100 LOC,
  in the fork).
- A `target_verify` mode that takes a multi-token block in one
  forward pass and returns per-position logits + hidden states +
  the rollback buffer (~80 LOC across attention + MLP).

This is ~250 LOC of *upstream* work in the fork, plus testing
against reference outputs.

### Step 3 (NOT YET) — `MTPSpeculativeTokenIterator`

Parallel iterator following the `TokenIteratorProtocol` contract,
implementing the speculative loop :

1. Prefill target with prompt ; sample first bonus token.
2. Bind draft to target (borrow embed + lm_head).
3. Prefill draft from the target's hidden + (input_ids[1:] +
   first_bonus). Seed the draft.
4. Loop until max_tokens :
   - Draft proposes `block_size - 1` candidate tokens (autoregressive,
     `block_size - 1` small forwards through the draft).
   - Target verifies all candidates in one batched forward
     (`forwardWithHidden` on `[bonus, draft₀, draft₁, …]`), gets
     per-position logits + hidden + rollback buffer.
   - Acceptance walk : compare target's argmax at each position to
     the draft's proposal. Find first reject ; accept everything
     before.
   - `rollback_speculative_cache` trims caches by the rejected
     count.
   - Draft `accept_verified_tokens` updates its own cache + seed
     for the next round.

~500 LOC of Swift. Same algorithmic structure as Blaizzy's
`_mtp_rounds` in `mlx_vlm/speculative/mtp.py:364`.

### Step 4 (NOT YET) — Wire into /v1/chat/completions

When `registry.draftId(for: req.model)` is set, switch from the
normal `TokenIterator` to `MTPSpeculativeTokenIterator`. Surface
acceptance rate + per-round token-count in `/admin/sessions`
telemetry.

### Status summary

| Step | LOC | Status |
|---|---|---|
| 1 — Fork + expose hidden states | ~50 | ✅ shipped |
| 2 — `rollback_speculative_cache` upstream | ~250 | TODO |
| 3 — `MTPSpeculativeTokenIterator` | ~500 | TODO |
| 4 — Wire into chat completions | ~50 | TODO |

Steps 2-4 together are realistically one focused 2-day sprint — the
hard part isn't LOC count, it's the math correctness (acceptance
walk + SSM rollback) which needs reference outputs to validate
against. Inferencer.app demonstrates the speedup so the approach is
proven ; we just have to land it in Swift.

## Historical runbook (now stale — kept for context)

The plan below was written before Units 1+3+Bonus shipped. The Unit 2
section above supersedes the inline Unit 2 here. Kept because the
prose still maps the Python →
Swift work units cleanly.

### Unit 3 — Telemak API + ModelRegistry wiring

`/admin/load` accepts an optional `draft_model` field + a
`num_draft_tokens` knob :

```jsonc
POST /admin/load
{
  "model": "inferencerlabs/Qwen3.6-35B-A3B-MLX-9bit",
  "draft_model": "inferencerlabs/Qwen3.6-35B-A3B-MTP-MLX-9bit",
  "num_draft_tokens": 3
}
```

The tricky parts that aren't just JSON plumbing :

- **Loader path** : mlx-swift-lm's `LLMTypeRegistry.shared` returns
  `LanguageModel`, but our `Qwen35MTPDraftModel` is plain `Module`
  (no `lm_head`, no `embed_tokens` — those are borrowed from the
  target at bind time). The shared registry rejects "qwen3_5_mtp".
  We need a parallel loader path in `ModelLoader` that :
  1. Detects `model_type == "qwen3_5_mtp"` in the config.
  2. Instantiates `Qwen35MTPDraftModel(config)` directly (no
     factory dispatch).
  3. Loads safetensors via `loadWeights(model:weights:)` —
     `MLXLMCommon`'s public weight loader — after running the
     wrapper's `sanitize()`.
  4. Returns a wrapper container (not the standard
     `ModelContainer`, since the draft has no tokenizer of its own
     — it borrows from the target).
- **ModelRegistry pairing** : add an optional
  `draftId: String?` to `Loaded` plus a `pairing` map
  `[String: String]` (main_id → draft_id). Drafts shouldn't appear
  in `/v1/models` (they're internal to spec decoding) but they DO
  consume RAM — update the memory accounting accordingly.
- **/v1/chat/completions** auto-uses the draft when one is paired.
  Reads `pairing[req.model]` and switches to
  `MTPSpeculativeIterator` instead of `TokenIterator`. No client
  API change.
- **/.well-known/inference-engine.json** advertises :
  ```jsonc
  "speculative_decoding": {
    "supported": true,
    "modes": ["mtp_adapter"],
    "active_pairs": [
      {"main": "<id>", "draft": "<id>", "block_size": 3,
       "acceptance_rate_recent": 0.87}
    ]
  }
  ```

### Bonus — `split-mtp` admin endpoint

`POST /admin/models/split-mtp { source, output }` invokes Blaizzy's
`split.py` via subprocess. Lets a Telemak operator pop an MTP
drafter out of any Qwen3.5 / Qwen3.6 source from the dashboard.

Prereq : the host has `mlx-vlm` installed (`pip install mlx-vlm`).
The endpoint shells out to `python3 -m
mlx_vlm.speculative.drafters.qwen3_5_mtp.split <source> --output
<output>` and surfaces stdout / stderr to the response. If
`mlx-vlm` isn't installed, return a clear 503 with the install
command.

### Order of operations next session

1. Unit 3 loader path (Telemak's parallel `Qwen35MTPDraftModel`
   loader from local-only directory).
2. Smoke test : load `Qwen3.6-35B-A3B-MTP-MLX-9bit`, instantiate
   the model, verify weights load without shape mismatch.
3. Unit 2 speculative loop.
4. Unit 3 final wiring + capability advertise.
5. Bonus split-mtp endpoint.

## The canonical reference — Blaizzy's `mlx-vlm`

Prince Canuma's [`Blaizzy/mlx-vlm`](https://github.com/Blaizzy/mlx-vlm)
project (MIT, MLX-native Python, maintained alongside ml-explore's
own work) ships a **complete** Qwen3.5 / Qwen3.6 MTP drafter stack :

```
mlx_vlm/speculative/
├── mtp.py                                            # spec decoding loop adapted for MTP
├── common.py                                         # shared speculative primitives
└── drafters/
    └── qwen3_5_mtp/
        ├── __init__.py
        ├── config.py                       (45 LOC)  # Qwen3_5MTPConfig dataclass
        ├── qwen3_5_mtp.py                 (422 LOC)  # ★ Qwen3_5MTPDraftModel
        ├── split.py                       (161 LOC)  # ★ extract MTP weights from any Qwen3.5/3.6 model
        └── README.md
```

The `Qwen3_5MTPDraftModel` class is structured exactly as our
architecture analysis (below) suggested :

- One `Qwen3_5DecoderLayer` (or `Qwen3_5MoeDecoderLayer` for the 35B MoE),
  reused from the main model's layer class.
- `pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `fc` (Linear concat
  projection), `norm`.
- Forward signature : `(embedded_token, hidden_state) → next_N_tokens_logits`.
- A small `block_size` knob (default 3) propagated via the config.

It also publishes :

- The **speculative-decoding loop** (`mtp.py`) that wraps the
  main-model verify pass with the draft-model proposal pass — this is
  the V2 counterpart of mlx-swift-lm's `SpeculativeTokenIterator` (in
  `Libraries/MLXLMCommon/Evaluate.swift`), already wired for classic
  drafters in V1.
- A **splitter** (`split.py`) that takes any Qwen3.5 / Qwen3.6 source
  checkpoint with `mtp.*` weights and emits a standalone drafter
  folder. **This unblocks Sophie's Argo target** — extracting an MTP
  draft from Qwen3.5-397B-A17B becomes a Python one-liner instead of
  R&D.

## Other public references (sanity / cross-check)

| Repo | Language | What for |
|---|---|---|
| `InternLM/lmdeploy/pytorch/spec_decode/proposers/qwen3_5_mtp.py` | PyTorch | PyTorch reference — useful when the MLX→Swift translation hits a math ambiguity |
| `jd-opensource/xllm/models/llm/qwen3_5_mtp.h` | C++ | Lowest-level reference, helpful for understanding the KV cache reuse pattern between main + draft |
| `NVIDIA/TensorRT-Edge-LLM/.../modeling_qwen3_5_mtp.py` | PyTorch | NVIDIA's reference — typically the cleanest of the bunch |
| `sgl-project/sglang/models/qwen3_next_mtp.py` | PyTorch | SGLang impl, useful for batch-verify semantics (V3 distributed) |
| `vllm-project/vllm/config/speculative.py` | Python | Speculative-decoding config conventions used across the open-source serving world |

## Architecture `qwen3_5_mtp` (cross-checked against config.json + safetensors)

```jsonc
{
  "model_type": "qwen3_5_mtp",
  "block_size": 3,                          // draft predicts 3 tokens / round
  "text_config": {
    "model_type": "qwen3_5_moe_text",       // same arch as main model layers
    "mtp_num_hidden_layers": 1,             // ⚡ ONE single transformer layer
    "mtp_use_dedicated_embeddings": false,  // ⚡ reuses main model embeddings
    "tie_word_embeddings": false,           // BUT has its own LM head
    "hidden_size": 2048,
    "num_experts": 256, "num_experts_per_tok": 8,
    "shared_expert_intermediate_size": 512,
    "attn_output_gate": true,
    "head_dim": 256,
    "num_attention_heads": 16, "num_key_value_heads": 2,
    "vocab_size": 248320,
    "layer_types": ["linear_attention", ..., "full_attention", ...],
    // ... full MoE config matching main model layer 0
  }
}
```

### Safetensors content (46 weights total, 906 MB on Qwen3.6-35B-A3B-MTP-MLX-9bit)

```
pre_fc_norm_embedding.weight            ← normalize incoming embedding (from main model)
pre_fc_norm_hidden.weight               ← normalize hidden state (from main model)
layers.0.input_layernorm.weight
layers.0.self_attn.{q,k,v}_proj.{weight,scales,biases}
layers.0.self_attn.{q,k}_norm.weight
layers.0.mlp.gate.{weight,scales,biases}              ← MoE router
layers.0.mlp.switch_mlp.{down,gate,up}_proj.*         ← 256 experts
layers.0.mlp.shared_expert.{down,gate,up}_proj.*      ← shared expert
layers.0.mlp.shared_expert_gate.{weight,scales,biases}
norm.weight                             ← final norm
fc.{weight,scales,biases}               ← LM head (predicts next-N tokens)
```

**Key observation** : there's **no `embed_tokens.weight`** in the file.
Confirming `mtp_use_dedicated_embeddings: false` — the draft is loaded
WITH the main model and uses the main model's embedding lookup table.
This means the draft model holds 906 MB but its actual run-time RAM
includes the main embeddings borrowed from the main model.

## V2 plan — Python → Swift port

The work is split into three coherent units, each shippable
independently :

### Unit 1 — Architecture port (`Qwen3_5MTPDraftModel`)

Target file : `Sources/Telemak/Engine/MTP/Qwen35MTPDraftModel.swift`

Port `mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py` (422 LOC)
to Swift. The MLX Python → MLX Swift API mapping :

| Python | Swift | Notes |
|---|---|---|
| `mx.array` | `MLXArray` | identical semantics |
| `nn.Linear(a, b, bias=False)` | `Linear(a, b, bias: false)` | `MLXNN.Linear` |
| `nn.RMSNorm(hidden, eps=…)` | `RMSNorm(dimensions: hidden, eps: …)` | `MLXNN.RMSNorm` |
| `nn.Module` | `Module` | inherit |
| `mx.concatenate([a, b], axis=-1)` | `concatenated([a, b], axis: -1)` | top-level fn |
| `mx.softmax(x, axis=-1)` | `softmax(x, axis: -1)` | top-level fn |
| `@property def foo(self)` | computed `var foo: …` | |
| weight loading via `self.update_modules(...)` | conform `Module` + standard sanitize | mlx-swift-lm pattern |

The decoder layer dependency : two options to weigh once the bulk of
the port is in place.

- **Option A — vendor `Qwen35DecoderLayer` from mlx-swift-lm**
  (~400 LOC copied into `Sources/Telemak/Engine/MTP/Vendored/`).
  Lower porting cost, slight maintenance debt on each mlx-swift-lm
  bump (rebase the vendored copy).
- **Option B — port `Qwen3_5DecoderLayer` from Blaizzy mlx-vlm**
  (~200 LOC Python → Swift). Higher porting cost but full
  independence — we own the layer math.

Decision : pick during the work. Both are documented escape hatches.
Default to Option A (vendor) since mlx-swift-lm's class is
production-tested by ml-explore.

### Unit 2 — Speculative loop port

Target file : `Sources/Telemak/Engine/MTP/MTPSpeculativeIterator.swift`
(or extend the existing `SpeculativeTokenIterator` if mlx-swift-lm's
class is already public-extensible).

Port `mlx_vlm/speculative/mtp.py`. The semantics differ from classic
spec decoding in two ways :

- The draft model is called with `(embedded_token, hidden_state)` and
  the hidden state comes from the main model's verify pass — there's
  a tight coupling between the two models per spec round.
- The draft emits **block_size** candidate tokens per call (3 by
  default for Qwen3.6-35B-MTP), not 1. Acceptance/rejection sweeps
  over the candidate window.

The KV cache is **shared between main and draft on linear-attention
layers** — that's a non-trivial optimisation referenced in xllm.h
and lmdeploy. First Telemak iteration can skip this (correctness
first, perf later) ; the cache-sharing perf bump is a follow-up.

### Unit 3 — Telemak API + ModelRegistry wiring

Target files : `Sources/Telemak/Server/Models.swift`,
`Sources/Telemak/Engine/ModelRegistry.swift`,
`Sources/Telemak/Server/ChatCompletions.swift`,
`Sources/Telemak/Server/WellKnown.swift`.

API surface :

```jsonc
POST /admin/load
{
  "model": "inferencerlabs/Qwen3.6-35B-A3B-MLX-9bit",
  "draft_model": "inferencerlabs/Qwen3.6-35B-A3B-MTP-MLX-9bit",   // ← new
  "num_draft_tokens": 3                                            // ← new, defaults to block_size from draft config
}
```

The `ModelRegistry` (already multi-model in V1) holds BOTH models,
paired explicitly. `POST /v1/chat/completions` with this main model id
auto-uses the paired draft model.

Capability contract :

```jsonc
"speculative_decoding": {
  "supported": true,
  "modes": ["classic_drafter", "mtp_adapter"],
  "active_pairs": [
    {"main": "<id>", "draft": "<id>", "block_size": 3, "acceptance_rate_recent": 0.87}
  ]
}
```

### Bonus — Wrap Blaizzy's `split.py` as a Telemak admin endpoint

Target file : `Sources/Telemak/Server/Models.swift` +
`scripts/split-mtp.py` (Python helper called via subprocess from the
endpoint).

`split.py` runs in seconds on a downloaded checkpoint. Wrapping it
as `POST /admin/models/split-mtp` lets a Telemak operator pop an MTP
drafter out of any Qwen3.5/3.6 source without leaving the dashboard.

This **also unblocks the V3 Argo path** : split a Qwen3.5-397B-A17B
checkpoint into a drafter folder, then have the orchestrator pair
them on the Argo cluster.

## Done criteria V2

1. `Qwen35MTPDraftModel` class registered, loads
   `inferencerlabs/Qwen3.6-35B-A3B-MTP-MLX-9bit` without crashing.
2. Pair load via `POST /admin/load` succeeds, both models in memory.
3. `POST /v1/chat/completions` produces a coherent completion at
   **measurably faster tok/s** than the main model alone — target
   **>1.5×** on a 300-word reply benchmark on Qwen3.6-35B-A3B / max-64.
4. Acceptance rate per spec round exposed in `/admin/sessions` or
   similar telemetry (debug + tune).
5. No regression on non-MTP main models (regression test : load a
   Gemma or Llama, run chat completion, no draft involved → same
   speed as before).
6. Companion sees no breaking change in `/v1/models` or
   `/v1/chat/completions` API shape.

## Out of scope V2 (defer to V3)

- **Distributed MTP on Argo cluster (Python)** — needs `mlx-lm` +
  `mlx-distributed` to support MTP draft AND pipeline-parallel batch
  verification. Real R&D, but Blaizzy's `split.py` removes the
  "extract the draft from the source model" half of the problem.
- **Auto-pairing main↔draft** — Telemak V2 requires explicit pairing
  in `/admin/load`. Auto-discovery (heuristic : if
  `<repo>-MTP-MLX-9bit` exists, suggest it) is V3 polish. The V2 UI
  surface for this is already half-prepped via the V1 multi-model
  card.
- **KV cache sharing main↔draft on linear-attention layers** — perf
  optimisation referenced by lmdeploy / xllm. Skip for V2 (correctness
  first), benchmark, then add as a V2.x perf round.

## References

- [`Blaizzy/mlx-vlm`](https://github.com/Blaizzy/mlx-vlm) —
  MIT-licensed canonical MLX Python reference (port target).
  - `mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py` —
    model class.
  - `mlx_vlm/speculative/drafters/qwen3_5_mtp/split.py` — weight
    splitter.
  - `mlx_vlm/speculative/mtp.py` — speculative decoding loop.
- [`InternLM/lmdeploy/.../qwen3_5_mtp.py`](https://github.com/InternLM/lmdeploy/tree/main/lmdeploy/pytorch/spec_decode/proposers) —
  PyTorch reference.
- [`NVIDIA/TensorRT-Edge-LLM/.../modeling_qwen3_5_mtp.py`](https://github.com/NVIDIA/TensorRT-Edge-LLM) —
  NVIDIA reference.
- mlx-swift-lm `SpeculativeTokenIterator` :
  `Libraries/MLXLMCommon/Evaluate.swift` — classic-drafter wiring,
  starting point for the MTP extension.
- mlx-swift-lm `Qwen35.swift` : `Libraries/MLXLLM/Models/Qwen35.swift`
  — the candidate to vendor for `Qwen35DecoderLayer` (Option A above).
- HF model repos :
  [`inferencerlabs/Qwen3.6-35B-A3B-MLX-9bit`](https://huggingface.co/inferencerlabs/Qwen3.6-35B-A3B-MLX-9bit) +
  [`inferencerlabs/Qwen3.6-35B-A3B-MTP-MLX-9bit`](https://huggingface.co/inferencerlabs/Qwen3.6-35B-A3B-MTP-MLX-9bit).
