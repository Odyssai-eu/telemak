# Telemak Model Compatibility

Telemak is a native `mlx-swift-lm` runtime. This ledger records model families
that need operator attention, so support decisions stay boring and reversible.

## Stable Text Paths

| Family | Status | Notes |
|---|---|---|
| Gemma 3/4 text | Supported | Baseline chat path. MTP is suspended; run without speculative decoding unless a future issue reopens it. |
| Qwen 3 / Qwen 3 Coder / Qwen 3 Next | Supported | Good default for code-agent workloads. |
| MiniMax M2 | Supported when present in the fork | Reasoning output may be wrapped in `<think>`; upstream proxy should route it to `reasoning_content`. |
| Mistral Medium 3.5 | Supported, slow on large checkpoints | Prefer validated external conversions before blaming Telemak. Large dense/MoE models need a high-memory node. |
| Step 3.7 Flash | Supported, text-only | Requires the Step port in the OdyssAI `mlx-swift-lm` fork. `reasoning_effort=minimal/low` gets an extra Step-only guard because the template always opens `<think>`. |

## Non-Goals / Suspended Paths

| Family | Status | Notes |
|---|---|---|
| MTP speculative decoding | Suspended | Gemma/Qwen MTP work remains blocked until there is a clear perf win over the already-fast baseline. |
| DeepSeek V4 Flash | Not accepted as Telemak work yet | Large multi-node candidates belong in Odysseus unless a single-node MLX path is proven stable. |
| Vision for Step 3.7 | Not supported | Current Step integration is text-only. |

## Validation Checklist

For a new model family:

1. Load via `/admin/load` and watch `/admin/activity` for `current_phase=loading`.
2. Smoke a short non-streaming chat.
3. Smoke a streaming chat and check chunk batching/usage.
4. Check for reasoning leakage (`<think>`, `</think>`, `</thinking>`).
5. Check a longer context prompt if the model uses sliding-window attention.
6. Record quirks here before deploying broadly.
