# Plan: Hy3 MoE-MTP speculative decoding on Telemak
_Locked via grill-with-docs — by Claude + Sophie. Terms per CONTEXT.md. Rev 2 (post MiniMax round 1)._

## Goal
Make Telemak serve **Hy3** (`inferencerlabs/Hy3-preview-MLX-9bit`, on ultra-512 / .29) with native
**MTP** speculative decoding, using Hy3's own embedded **MoE MTP head** as the **drafter** — no
external draft model. First **MoE-MTP** in the stack (Telemak #70 class), directly on Hy3 per
Sophie's call. **Success: ≥1.3× sustained decode + exactness. Kill: acceptance <0.3 after the
hidden-state contract is verified.** Sizing: this is a **13-point epic** (matches #70), not a set of
line edits — see Approach for the real new-code surface.

## Approach
0. **Read the SOURCE first — resolve the hidden-state contract (gates step 3).** Read the
   HunYuan-3 / DeepSeek-V3 MTP-module reference (`enorm`/`hnorm`/`eh_proj` + nextn-layer) to decide
   whether the head consumes the trunk's **PRE**- or **POST**-final-norm hidden. The boundary in our
   code is `HYV3ModelInner`'s final `norm` (`mlx-swift-lm-odyssai/Libraries/MLXLLM/Models/HYV3.swift`,
   the `norm(h)` at the end of `HYV3ModelInner.callAsFunction`). Telemak's working Qwen3.5 path feeds
   **POST**-norm (`Qwen35.swift:698`, surfaced by `forwardWithHidden:772`). Until this is resolved,
   step 3's `forwardWithHidden` is undefined. This is the acceptance-determining unknown (the EAGLE
   failure). **RESOLVED (vLLM `hy_v3_mtp.py`, the canonical impl; cross-checked vLLM/SGLang DeepSeek +
   Qwen3-Next + the DSV3 paper):**
   - **PRE-norm hidden** — the head consumes the trunk's last-decoder-layer hidden BEFORE `HYV3ModelInner.norm`
     (the head's own `hnorm` re-norms it; `model.norm` is only the shared-head logit-time norm). So Hy3's
     `forwardWithHidden`/`targetVerify` must return the **pre-`norm`** hidden (NOT post, unlike the Qwen35 path).
   - **Concat embedding-FIRST**: `eh_proj(cat[enorm(embed), hnorm(hidden)], axis=-1)`. Reversing it loads
     fine but silently tanks acceptance (wrong weight columns).
   - **The MTP head has its OWN `final_layernorm`** before the shared (no-op-norm) `lm_head` — block → `final_layernorm` → shared `lm_head`. Do NOT add a second norm.
   - **`embed_tokens` + `lm_head` are SHARED with the trunk**; the MTP input is the NEXT token's embedding.
1. **Extract + convert the MTP head → sidecar.** The head at `model.layers.80` in `tencent/Hy3-preview`
   (VERIFIED present: HTTP 200, full MoE layer — `eh_proj`, `enorm`, `hnorm`, `self_attn.*`+`q/k_norm`,
   `mlp.{router.gate, experts.0..191.{gate,up,down}_proj, shared_mlp.*, expert_bias}`, the three norms)
   is **full-precision bf16 with experts UNSTACKED** (per-expert tensors). Conversion script (Odysseus
   `scripts/`, precedent `mistral_eagle_convert.py`): download only the layer-80 shards (resolved from
   the index `weight_map`), **stack the 192 per-expert tensors into the `switch_mlp` `[E,…]` layout**
   that `HYV3MoE` expects (the trunk's InferencerLabs quant is already fused — the *head* is not, so it
   needs the same stacking the trunk conversion did), **quantize to 8-bit group_size 32** to match the
   trunk, write `mtp.safetensors` + a staged config with `mlx_lm_extra_tensors.mtp_file` **and a
   recognized `model_type: "hy_v3_mtp"`** (the loader's `switch modelType` at `MTPModelLoader.swift:81`
   dispatches on it — an unknown type throws `LoadError.wrongModelType`). **Verify the
   stacked expert shapes match the trunk's `switch_mlp` before quantizing.** Pair with the existing
   9-bit trunk on .29 — no full re-conversion.
2. **Swift Hy3 MTP head class** (fork, new file) — modeled on `Qwen35MTPDraftModel` *structure*
   (`eh_proj`→`fc`, `enorm`→`pre_fc_norm_embedding`, `hnorm`→`pre_fc_norm_hidden`, `final_layernorm`→
   `norm`, borrow `embed`+`lm_head` from the trunk), but with TWO Hy3-specific divergences the review
   flagged:
   - its decoder layer's MoE uses **`HYV3MoE`/`HYV3Router`** (sigmoid + `expert_bias` correction),
     **NOT** Qwen35's `MTPSparseMoeBlock` (`Qwen35MTPLayer.swift:167`, which is softmax + `shared_expert_gate` — wrong router for Hy3);
   - its `sanitize` **must NOT add the `+1.0` RMSNorm offset** that `Qwen35MTPDraftModel.sanitize`
     (`:157–175`) applies — that's a Qwen3.5/3.6 pre-fused-residual quirk; Hy3 uses plain `RMSNorm`.
3. **Generalize the speculative iterator + add the HYV3 ABI.** `MTPSpeculativeIterator` is **type-locked
   to Qwen3.5** (`main: any Qwen35HiddenStateProvider`, `draft: Qwen35MTPDraftModel`,
   `MTPSpeculativeIterator.swift:50–51`) — it is **NOT** reusable as-is (the rev-1 plan was wrong on
   this). Extract a generic `HiddenStateProvider` protocol (the 5 ABI methods) + an `MTPDraft` protocol
   (`draftBlock`/`acceptVerifiedTokens`/`prefillFromTargetHidden`), make `MTPSpeculativeIterator<Main,
   Draft>` generic over both, and conform BOTH the existing Qwen35 pair AND the new Hy3 pair (so the one
   verified accept/rollback loop serves both — don't duplicate it). Then add the 5 ABI methods to
   `HYV3Model` (`forwardWithHidden`/`targetVerify`/`rollbackSpeculativeCache`/`embed`/`applyLMHead`,
   mirroring `Qwen35.swift:772–1003`), honoring the step-0 norm contract. Hy3 = GQA full-attention, no
   SSM/linear-attn → `rollbackSpeculativeCache` is a plain KV trim (no GDN replay).
4. **Telemak loader + dispatch wiring (explicit-sidecar path).** We pair via the **explicit** sidecar
   API (`draft_model` + `allow_unverified_mtp:true` + `force_llm:true`), so `MTPCompatibility.inspect`
   on the **trunk** correctly returns `.noMTP` (the 9-bit trunk has its head stripped) — we do NOT rely
   on auto-detection and do NOT need to bypass `isMoEMTPHead` (`MTPCompatibility.swift:174`) for an
   embedded path. Required seams: a `"hy_v3_mtp"` case in the `switch modelType` + load path in
   `MTPModelLoader.load` (`:81`) + a `LoadedDraftModel` case (`:26`); the iterator-dispatch case in
   `ChatCompletionsMTP.runMTPIteratorCollectingPieces` (`:380`); add `hy_v3` to the `force_llm`
   auto-gate (`Models.swift:191–194`) so a self-paired Hy3 sidecar doesn't silently fall back to
   non-LLM. (Sidecar-aware auto-pairing in `inspect` is out of scope for v1 — explicit load only.)
5. **Build → deploy → validate, gated.** `./scripts/build.sh Release` (xcodebuild) → `deploy-all.sh
   --canary-only --skip-build` to .29 `Release.deepseek`:8013. Load:
   `POST /admin/load {model, draft_model, allow_unverified_mtp:true, force_llm:true}`. **Milestone
   gates, in order:** (a) head loads + emits one sane draft token (infra smoke — the whole MTP
   machinery is currently unexercised); (b) **exactness** (MTP == AR byte-identical) via
   `/admin/mtp/smoke`; (c) **acceptance ≥0.3** after the norm contract; (d) **≥1.3× sustained**. Stop
   at the kill threshold if (c) fails post-norm-fix.

## Key decisions & tradeoffs
- **Target = Hy3 direct** (not dense-27B, not 397B). Sophie's call — go straight at the MoE-MTP
  problem on the target model. Tradeoff: all blockers at once, no warm-up on a known-good head.
- **Sidecar head extraction** (layer 80 only, ~2GB quantized), not full re-conversion of the ~600GB
  original. The 9-bit trunk already works and is loaded.
- **Head from the original (unstacked bf16) → stacked + quantized to the trunk's switch_mlp / 8-bit-g32
  layout.** The trunk was pre-fused by InferencerLabs; the original head is per-expert, so the
  conversion does that stacking.
- **Iterator generalized via protocols + generics, NOT duplicated.** The accept/rollback/residual
  logic is the subtle, EAGLE-class-risk core — one verified copy, two conformers.
- **Pre-committed success/kill** (≥1.3× / acceptance<0.3 post-norm-fix) to avoid the EAGLE sink.

## Risks / open questions
- **Hidden-state PRE/POST-norm contract** — THE acceptance-determining risk (same class that killed
  EAGLE). Front-loaded as step 0; gated by the exactness/acceptance smoke. **Open until the source read
  resolves it.**
- **Surface area is a 13-point epic, not line edits** (the rev-1 plan undercounted): conversion script
  + Hy3 MTP head class + sanitize + 5 HYV3 ABI methods + **iterator generalization** (protocols +
  generics, touching the existing Qwen35 conformance — regression-test the Qwen35 MTP path still works)
  + loader case + dispatch case + force_llm gate.
- **MoE-MTP has no public reference impl** for this family. Head structure ref = HunYuan-3/DeepSeek-V3;
  speculative-cycle ref = Telemak's working Qwen35 path + #67's MTPLX analysis.
- **Quant compatibility** of the 8-bit-g32 head with the trunk's hidden space — verify via exactness on
  a short gen.
- **Expert layout** — confirm the stacked layer-80 experts match `HYV3MoE`'s `switch_mlp` shapes before
  quantizing (a mismatch = head won't load).
- ~~tencent/Hy3-preview availability~~ — **VERIFIED live** (HTTP 200 on its `model.safetensors.index.json`
  + `config.json`; MTP head present at `model.layers.80`). Not an open risk.

## Out of scope
- 397B / 80B MoE-MTP (#67/#68) — the Hy3 head class + the generalized iterator may generalize later.
- Dense 27B MTP (#69) — skipped per Sophie.
- Sidecar-aware auto-pairing in `MTPCompatibility.inspect` — v1 is explicit-load only.
- Draft-depth (D2/D3) tuning beyond confirming ≥1.3×.
- Full MTP-preserving conversion of the whole Hy3 model — sidecar only.
