# Plan Review Log: Hy3 MoE-MTP speculative decoding on Telemak

Act 1 (grill-with-docs) complete — plan locked with Sophie, CONTEXT.md glossary written. MAX_ROUNDS=5.

Decision tree resolved during the grill (each branch verified against the real cluster/checkpoints,
not assumed — config-lies discipline):
- Target: **Hy3 direct** (Sophie chose over dense-first and 397B-first).
- Verified: NO model on the cluster has an MTP head on disk (all mlx_lm-stripped); the proven dense
  27B head is gone. The premise of "397B has its head" was false (9-bit stripped, original incomplete).
- Verified: the Hy3 MTP head **does exist** in `tencent/Hy3-preview` as `model.layers.80` (a full MoE
  layer with `eh_proj`/`enorm`/`hnorm`) — feasible via sidecar extraction.
- Success ≥1.3× + exactness; kill if acceptance <0.3 after the hidden-state contract is verified.

---

## Round 1 — MiniMax-M3
Read (its tools): MTPModelLoader, MTPSpeculativeIterator, Qwen35MTPDraftModel, MTPCompatibility,
HYV3.swift, Qwen35.swift, Qwen35MTPLayer, ChatCompletionsMTP, Models.swift, Qwen35MTPConfig + greps
(LoadedDraft, force_llm, model_type, experts/.mtp, mtp_file/extra_tensors). **VERDICT: REVISE.**

Findings:
1. (critical) `MTPSpeculativeIterator` is type-locked to Qwen35 (`main: any Qwen35HiddenStateProvider`,
   `draft: Qwen35MTPDraftModel`, :50-51) → "reuse unchanged" is false.
2. (critical) `HYV3Model` has none of the 5 ABI methods; norm-contract (step 0) gates them.
3. (critical) Compat `inspect` runs on the trunk, which has the head stripped → returns `.noMTP`;
   widening `isMoEMTPHead` is the wrong lever for the sidecar path.
4. (critical) `force_llm` auto-gate (Models.swift:191-194) omits hy_v3 → silent non-LLM fallback risk.
5. (important) Step-1 "stack experts" contradicts HYV3 trunk being pre-fused — clarify the *head* (from
   the original) is unstacked and needs stacking; verify shapes.
6. (important) Qwen35MTPDraftModel.sanitize adds +1.0 to RMSNorm (Qwen quirk) — Hy3 must NOT.
7. (minor) `MTPSparseMoeBlock` is softmax+shared_expert_gate (Qwen), not Hy3's sigmoid+bias → use HYV3MoE.
8. (minor) Doubts `tencent/Hy3-preview` exists on HF.
9. (note) Streaming buffers all pieces before SSE → ~30s first-token silence on a big model (existing).
10. (meta) Plan systematically undercounts the new-code surface.

### Claude's response (final arbiter)
ACCEPTED 1,2,3,4,5,6,7,10 → revised PLAN.md rev 2:
- Step 3 rewritten: iterator is NOT reused — generalize via `HiddenStateProvider`+`MTPDraft` protocols
  + generic `MTPSpeculativeIterator<Main,Draft>`, conform both Qwen35 and Hy3; regression-test Qwen35.
- Step 4 rewritten: explicit-sidecar path → trunk `inspect` rightly `.noMTP`; we don't fight
  `isMoEMTPHead`; real seams = loader case + dispatch case + force_llm gate. Auto-pairing out of scope v1.
- Step 1 clarified: head = unstacked bf16 from the original → stack to switch_mlp + quantize 8-bit g32;
  verify shapes.
- Step 2: use HYV3MoE (sigmoid+bias), NOT MTPSparseMoeBlock; sanitize must NOT add the +1.0 norm offset.
- Sizing bumped to a 13-point epic; new-code surface enumerated in Risks.
REJECTED 8 (logged reason): `tencent/Hy3-preview` VERIFIED live this session — HTTP 200 on its index.json
+ config.json, MTP head found at `model.layers.80`. MiniMax can't see HF/cluster, it speculated.
NOTED 9: existing Telemak streaming behavior, not a plan change.

---

## Round 2 — MiniMax-M3
Re-read (incl. ModelRegistry.swift + `isMTPWeightKey|.mtp` grep). Confirmed all 7 round-1 findings
addressed; independently verified the sidecar path: `MTPCompatibility.inspect(isDraft:true)` →
`.sidecarOnly` + `overrideRequired` (MTPCompatibility.swift:86-95), unblocked by `allow_unverified_mtp`
via `canRun(allowUnverified:)` (:49) — `isMoEMTPHead` (:107) never reached. **VERDICT: APPROVED.**

One new refinement (incorporated, not a blocker): the staged sidecar config must carry a recognized
`model_type` or `MTPModelLoader.load`'s `switch` (:81) throws `wrongModelType`. → PLAN.md rev 3: pin
`model_type: "hy_v3_mtp"` in the staged config (step 1) + a `"hy_v3_mtp"` switch case (step 4).

**Converged at round 2/5. Plan APPROVED. No code written during either act — awaiting Sophie's sign-off.**
