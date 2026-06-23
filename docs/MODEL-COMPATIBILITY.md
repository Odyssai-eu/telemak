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

## Preflight failure codes

`POST /admin/load` runs a structured preflight (issue #64) before the heavy
MLX load. The preflight resolves the canonical local directory and checks
that `config.json` parses, the safetensors shards are all on disk, and the
`model_type` is at least dispatchable. It does **not** fetch from the hub —
an identifier that is not on disk falls through to the load path unchanged.

The HTTP response is shaped like every other Telemak error
(`{"error": {"type", "code", "message", …}}`). Codes:

| Code | HTTP | Meaning | Operator action |
|---|---|---|---|
| `model_dir_missing` | 400 | The local path the id resolved to (absolute path or `<models_dir>/<org>/<name>/snapshots/<hash>`) does not exist. | Confirm the id is spelled correctly; confirm `models_dir` points at the right root (`GET /admin/models-dir`). |
| `config_missing` | 400 | The resolved dir has no `config.json`. Snapshot is incomplete or the id points at the wrong level. | Re-run the HF download (or move the snapshot under `<id>/snapshots/<hash>/` if it landed flat). |
| `config_parse_failed` | 400 | `config.json` exists but is not valid JSON. | Open the file, look for a truncated download or a sidecar merge gone wrong. |
| `shards_incomplete` | **503** | `model.safetensors.index.json` references shard files that are not all on disk. The response payload also includes `missing_shards: [...]` and `retryable: true`. | Transient — likely a partial / aborted download. Wait for the download to finish, then retry. |
| `unsupported_model_type` | 400 | The config has no `model_type` and no `architectures`, and the weight files are not in a format mlx-swift-lm knows how to load (no `.safetensors` or `.gguf`). The response payload includes the `model_type` mlx-swift saw (`null` when absent). | Usually a broken HF repo or a non-MLX checkpoint. Check the upstream release notes. |

Sharded downloads are the most common source of preflight failures: HF
downloads the index file first and the shards after, so a load that hits
mid-download will get `shards_incomplete` — wait a few seconds and retry.
The preflight is deliberately cheap (a handful of `stat` calls + one JSON
parse on `config.json` + one on the index when present) so it never
materially slows down a load that is about to succeed.
