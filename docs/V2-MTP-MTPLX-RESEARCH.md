# MTPLX Research Notes

Issue: #41
Date: 2026-05-26

## Verdict

Port MTPLX's probability-ratio acceptance before enabling MTP for normal
non-zero-temperature chat.

Telemak's current `MTPSpeculativeIterator` is greedy-only: draft tokens are
argmax proposals, target verification accepts only when target argmax matches
draft argmax, and rejection emits the target argmax. That is fine for
`temperature <= 0`, but it is not sampling-correct for `temperature > 0`.

MTPLX implements the Leviathan/Chen speculative sampling rule:

- sample draft token from draft distribution `q`
- accept with `min(1, p(token) / q(token))`, where `p` is the target
  distribution under the same sampler
- on rejection, sample from normalized residual `(p - q)+`

So the clear answer is: yes, prob-ratio acceptance plus residual correction is
needed to preserve the target model distribution at non-zero temperature.

## What MTPLX Does That Telemak Does Not

- Uses the target model's native MTP heads; no external sidecar drafter.
- Keeps draft and target probability distributions for every proposed token.
- Applies exact speculative sampling at `temperature > 0` instead of argmax
  equality.
- Gates models through a compatibility contract: verified, architecture
  compatible but unverified, incompatible architecture, or no MTP.
- Includes per-machine depth tuning (`AR`, `D1`, `D2`, `D3`) and only saves a
  speculative depth when it beats AR on that Mac.
- Adds custom Metal/kernel work for hot paths such as GDN replay and fused
  sampling.

## Telemak Inventory

Current Telemak state:

- `MTPSpeculativeIterator` explicitly says greedy-only and uses
  `MLX.argMax(logits, axis: -1)` for target verification.
- `Qwen35MTPDraftModel.draftBlock` returns only token ids. Draft logits and
  draft probabilities are not retained.
- The iterator does not receive `GenerateParameters`, so it cannot mirror the
  request sampler (`temperature`, `top_p`, `top_k`) yet.
- Rejection correction is target argmax, not residual sampling.
- SSM rollback exists in the fork-facing protocol, which is the hard
  state-management piece MTPLX also has to solve.

## What To Port

Port first:

1. Add sampler-aware MTP acceptance in Swift for `temperature > 0`.
2. Extend the draft path to return proposed token ids plus draft logits or
   sparse draft distributions.
3. Build target distributions from verify logits using exactly the same
   sampler semantics as chat generation.
4. On rejection, sample from residual `(p - q)+`.
5. Keep greedy argmax as the fast path for `temperature <= 0`.

Then benchmark:

1. Add an admin MTP bench comparing AR, D1, D2, D3 on the same loaded model.
2. Persist only an explicit operator setting or a measured local winner.

Defer:

- MTPLX custom Metal kernels. Telemak should first land correctness and
  integrated chat wiring, then measure whether MLX Swift's current kernels are
  actually the bottleneck.
- Fan-control modes. They are operationally useful for MTPLX but not required
  for Telemak's runtime contract.

## Follow-Up Issues

Recommended follow-ups:

- #42: implement sampler-correct MTP acceptance in
  `MTPSpeculativeIterator`.
- #43: add an MTPLX-style verified/unverified compatibility gate for embedded
  MTP checkpoints before auto-enabling MTP.

## Sources

- https://github.com/youssofal/MTPLX
- https://github.com/youssofal/MTPLX/blob/main/mtplx/sampling.py
- https://github.com/youssofal/MTPLX/blob/main/mtplx/generation.py
- https://github.com/youssofal/MTPLX/blob/main/docs/model-compatibility.md
- https://huggingface.co/Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed
