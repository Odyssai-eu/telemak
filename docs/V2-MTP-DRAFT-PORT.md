# Telemak V2 — Port `qwen3_5_mtp` draft architecture

> Goal : enable **MTP speculative decoding** in Telemak so Qwen3.6-35B-A3B
> serves at ~76 tok/s instead of ~47 tok/s on max-64 (1.6× speedup
> demonstrated by Inferencer.app on the same model with the same draft
> repo).
>
> Pick this up **after V1 is shipped** (the multi-model + KV cache work in
> [V1-TODO.md](V1-TODO.md) finishes first; V2 builds on the
> SpeculativeTokenIterator wiring that V1's classic-spec exploration may
> already touch).

## Background — what we know

The MTP draft is **not** a standalone LanguageModel — it's an adapter
that piggybacks on the main model's embeddings. Recon done 2026-05-24
on these HF repos :

- Main model : [`inferencerlabs/Qwen3.6-35B-A3B-MLX-9bit`](https://huggingface.co/inferencerlabs/Qwen3.6-35B-A3B-MLX-9bit) — standard mlx-swift-lm-loadable Qwen3.5MoE arch
- Draft model : [`inferencerlabs/Qwen3.6-35B-A3B-MTP-MLX-9bit`](https://huggingface.co/inferencerlabs/Qwen3.6-35B-A3B-MTP-MLX-9bit) — 906 MB, **architecture `qwen3_5_mtp` not registered in mlx-swift-lm**

Inferencer.app loads both together and uses `SpeculativeTokenIterator`
under the hood (mlx-swift native, no fork required for the core spec
loop — only the architecture registration is missing).

## Architecture `qwen3_5_mtp` (extracted from config.json + safetensors header)

```jsonc
{
  "model_type": "qwen3_5_mtp",
  "block_size": 3,                       // draft predicts 3 tokens / round
  "text_config": {
    "model_type": "qwen3_5_moe_text",    // same arch as main model layers
    "mtp_num_hidden_layers": 1,          // ⚡ ONE single transformer layer
    "mtp_use_dedicated_embeddings": false, // ⚡ reuses main model embeddings
    "tie_word_embeddings": false,        // BUT has its own LM head
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

### Safetensors content (46 weights total, 906 MB)

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

## What needs to change in mlx-swift-lm (or Telemak's local fork)

### 1. New model class `Qwen35MTP` in `Libraries/MLXLLM/Models/`

A struct conforming to `LanguageModel` that :
- Defines 1 `Qwen3_5MoeLayer` (same as the main model's layer class — already exists in `Qwen35.swift`)
- Holds `preFcNormEmbedding`, `preFcNormHidden`, `norm`, `fc` (LM head)
- Forward pass : `(embedded_token, hidden_state) → next_N_tokens_logits`
- `block_size` config field exposed so the spec iterator knows how many tokens to expect per call

### 2. Modify `SpeculativeTokenIterator` (in `Libraries/MLXLMCommon/Evaluate.swift`)

Currently the iterator assumes the draft model is a standalone `LanguageModel` that owns its embeddings. For MTP-style :
- Pass `mainModel.embedTokens(...)` output into the draft's forward
- Pass `mainModel`'s last hidden state into the draft's forward
- Accept `block_size` candidate tokens per draft call (not 1)

Easiest path : add a `SpeculativeDraftModel` protocol with an alternate
`forward(embedded:hidden:)` method, and let MTP drafts implement it.
Classic drafts (standalone Qwen) keep using the existing `forward(input:)`.

### 3. Model registry detection

Add `qwen3_5_mtp` to the recognized `model_type` list. The factory loads
the new `Qwen35MTP` class when it sees this type.

### 4. Telemak API surface

```jsonc
POST /admin/load
{
  "model": "inferencerlabs/Qwen3.6-35B-A3B-MLX-9bit",
  "draft_model": "inferencerlabs/Qwen3.6-35B-A3B-MTP-MLX-9bit",   // ← new
  "num_draft_tokens": 3                                            // ← new, defaults to block_size from draft config
}
```

The `ModelRegistry` (already multi-model in V1) holds BOTH models, paired
explicitly. `POST /v1/chat/completions` with this main model id auto-uses
the paired draft model.

### 5. Capability contract

`/.well-known/inference-engine.json` exposes :
```jsonc
"speculative_decoding": {
  "supported": true,
  "modes": ["classic_drafter", "mtp_adapter"],
  "active_pairs": [
    {"main": "<id>", "draft": "<id>", "block_size": 3, "acceptance_rate_recent": 0.87}
  ]
}
```

## Done criteria V2

1. `Qwen35MTP` class registered, loads `inferencerlabs/Qwen3.6-35B-A3B-MTP-MLX-9bit` without crashing
2. Pair load via `POST /admin/load` succeeds, both models in memory
3. `POST /v1/chat/completions` produces a coherent completion at **measurably faster tok/s** than the main model alone — target **>1.5×** on a 300-word reply benchmark
4. Acceptance rate per spec round exposed in `/admin/sessions` or similar telemetry (debug + tune)
5. No regression on non-MTP main models (regression test : load a Gemma or Llama, run chat completion, no draft involved → same speed as before)
6. Companion sees no breaking change in `/v1/models` or `/v1/chat/completions` API shape

## Out of scope V2 (defer to V3)

- **Distributed MTP on Argo cluster (Python)** — needs `mlx-lm` + `mlx-distributed` to support MTP draft AND pipeline-parallel batch verification. Real R&D, possibly upstream PR territory.
- **Extracting MTP weights from Qwen3.5-397B-A17B ourselves** — Sophie's Argo target. Would require Python script to slice the original Qwen3.5-397B model weights into a separate MTP draft repo. Out of scope unless Inferencer publishes the draft.
- **Auto-pairing main↔draft** — Telemak V2 requires explicit pairing in `/admin/load`. Auto-discovery (heuristic : if `<repo>-MTP-MLX-9bit` exists, suggest it) is V3 polish.

## References

- mlx-swift-lm SpeculativeTokenIterator : `Libraries/MLXLMCommon/Evaluate.swift` (already exists, just needs the new draft-model protocol)
- mlx-swift-lm Qwen3.5 main model : `Libraries/MLXLLM/Models/Qwen35.swift` (re-use its layer class for the MTP draft's single layer)
- Inferencer's video demo : [youtube.com/xcreate](https://youtube.com/xcreate) — search "MTP speculative decoding"
- Original Qwen MTP paper / architecture writeup : check Qwen team's tech reports on Qwen3-Next / Qwen3.6 series
