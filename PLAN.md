# Plan: Fix the ~30× perf regression in Hy3 MoE-MTP speculative decoding
_Locked via grill-with-docs — by Claude + Sophie. Terms per CONTEXT.md._

## Goal
The Hy3 MoE-MTP speculative path is **correct** (acceptance 0.476, exactness holds) but **~30× slower
than autoregressive** (0.57 tok/s vs 17-19 AR; 139.6s/80 tokens). Root cause confirmed by
minimax-review reading the code: the extracted **draft head ships bf16/unquantized**, so its
192-expert `SwitchGLU` runs on bf16 `gatherMM` (~2× memory bandwidth, no fused dequant) — vs the
working Qwen3.5 MTP path whose head loads **quantized** (`QuantizedSwitchLinear`/`gatherQuantizedMM`).
The head runs ~3-4 forwards/round, so this dominates. **Fix it to reach ≥1.3× sustained without
touching the verified hidden-state contract or the acceptance/exactness math.**

## Approach
1. **[THE FIX] Quantize the draft head to 4-bit (Q4), group_size 32.** Per CONTEXT.md, head
   precision is a speed↔acceptance knob, NOT a quality knob (the trunk *verifies* every proposed
   token → exactness holds regardless), so go aggressive: 4-bit gives the lightest gather (½ the
   bytes of 8-bit, ¼ of bf16). In `scripts/hy3_mtp_extract.py`, after building the head dict,
   **`mx.quantize(w, bits=4, group_size=32)`** each Linear/SwitchGLU weight (`eh_proj`,
   `self_attn.{q,k,v,o}_proj`, `mlp.router.gate`, `mlp.shared_mlp.*`, and the 3-D
   `mlp.switch_mlp.{gate,up,down}_proj`) → emit `.weight`(packed)/`.scales`/`.biases`. **Keep full
   precision**: all RMSNorm weights (`enorm`/`hnorm`/`final_layernorm`/`input_layernorm`/
   `post_attention_layernorm`/`q_norm`/`k_norm`) and `mlp.router.expert_bias`. Add
   `"quantization": {"bits":4,"group_size":32,"mode":"affine"}` to the sidecar `config.json` so
   `loadHYV3MTP` → `baseConfig.perLayerQuantization` → `loadWeights` builds `QuantizedLinear`/
   `QuantizedSwitchLinear` that bind the pre-quantized weights (same path the 8-bit trunk loads by).
2. **[SAFE PERF] Add explicit `eval()` points in `HYV3MTPSpeculativeIterator`** so the lazy MLX graph
   stays bounded (minimax-review #2): `eval(hidden)` after `main.forwardWithHidden` in `runRound`,
   and `eval(hPrev)` after each `forwardToken` in `draftBlock` (or `asyncEval`). Perf-only — does not
   change the math, so acceptance stays attributable to Q4 alone.
3. **[SHAPE — likely no-op] Verify the `acceptVerifiedTokens` re-forward shapes are bounded.** With
   `blockSize=3`, the re-forward chunk is {1,2,3} tokens — only **3 distinct shapes**, so MLX
   compiles 3 kernels once and caches them; this is NOT a per-round recompile (minimax-review #3
   slightly overstated it). **No code change** unless the post-Q4 profile shows shape churn — avoids
   the correctness risk of padding the draft cache.
4. **[MEASURE — one cycle]** Clear the server backlog first (the smoke endpoint serializes; my test
   smokes queued): `POST /admin/unload` then reload the pair, or reboot .29. Run **one warmup smoke**
   (pays the one-time Q4-kernel Metal compile), then **one measured greedy smoke** (80 tok, temp 0).
   Compare vs the bf16 baseline (0.57 tok/s, acceptance 0.476): success = **≥1.3× decode AND
   acceptance ≥0.3 AND exact**. Do NOT fire concurrent smokes.

## Key decisions & tradeoffs
- **Q4 (not 8-bit), because the head is a drafter** — its precision only costs acceptance (= speed),
  never output quality (the trunk verifies). So we take the most aggressive quant that keeps
  acceptance >0.3. Q4 reuses mlx's fast `gatherQuantizedMM`.
- **Bundle all 3, measure once** — #2/#3 are perf-only (don't change acceptance), so acceptance
  stays cleanly attributable to Q4; saves an expensive measurement cycle (server serializes smokes).
- **Don't touch** the PRE-norm contract, embed-first concat, or the acceptance walk (verified at 0.476).

## Risks / open questions
- **Acceptance may drop below the 0.3 kill threshold at Q4.** Fallback ladder if so: **Q4 → 6-bit →
  8-bit** head (each a script re-run + config bump; the Swift loads any of them via the quantization
  block). Measure and dial back.
- **`mx.quantize` on the 3-D `switch_mlp` weight `[E, out, in]`** — must quantize per-expert along the
  last axis so `QuantizedSwitchLinear` binds; verify the produced shapes match what mlx-swift expects.
- **The unexplained ~115s "per-call fixed cost"** (16-tok smoke also slow). Hypothesis: the bf16
  192-expert kernel's first-forward Metal compile. Q4 should shrink/replace it with the fast
  quantized kernel; if a large fixed cost survives Q4, profile the prefill path specifically.
- The head shares `embed_tokens` + `lm_head` from the **8-bit trunk** (not in the sidecar) — unchanged
  by Q4; only the head's own weights are re-quantized.

## Out of scope
- The acceptance/exactness logic, the hidden-state contract, draft-depth (D2/D3) tuning.
- Generalizing the iterator to the protocol form (the parallel Hy3 path stays).
- The bf16→quantized question for any other model (Qwen35 already loads quantized).
