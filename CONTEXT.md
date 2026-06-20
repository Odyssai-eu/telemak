# CONTEXT — MTP speculative decoding (glossary)

Domain glossary for the native MTP (Multi-Token Prediction) speculative-decoding work in
Telemak. Glossary only — no implementation details (those live in PLAN.md / the code).

- **MTP head (draft head)** — a small module bundled *inside* a model's own checkpoint that
  predicts the next token from the main model's hidden state. Used as the **drafter** in
  speculative decoding, so no separate/external draft model is needed. **Its numeric precision
  (quantization) is a speed↔acceptance knob, NOT a quality knob:** a lower-precision head proposes
  less accurately → lower *acceptance* (less speedup) → but never wrong output, because the trunk
  *verifies* every proposed token. So the head can be quantized aggressively for speed; the only
  cost is acceptance.

- **Trunk** — the main model (the full Hy3), as opposed to its MTP head.

- **Draft / verify** — one speculative round: the draft head proposes K candidate tokens
  autoregressively; the trunk verifies all K in a single batched forward; the longest matching
  prefix is accepted, the rest discarded.

- **Acceptance rate** — fraction of drafted tokens the verify step accepts. The metric of value:
  decode speedup is roughly proportional to it. A pipeline can run flawlessly with acceptance ≈ 0
  (see *the EAGLE failure*) — "it runs" proves nothing.

- **Exactness** — the MTP output distribution must equal the plain autoregressive (AR) output.
  Speculative decoding is exact *by construction* (the trunk verifies every token), so exactness
  is the cheap gate; acceptance is the real one.

- **Hidden-state contract** — *which* hidden state the draft head consumes: the trunk's hidden
  **before** its final norm (PRE) or **after** (POST). Getting this wrong gives acceptance ≈ 0.

- **The EAGLE failure** — the reference cautionary tale: a prior speculative drafter (Mistral
  EAGLE) had a complete, coherent pipeline yet acceptance ≈ 0 → 0.52× baseline, abandoned. Prime
  suspect: the hidden-state contract (PRE vs POST norm).

- **Head source (Hy3)** — `tencent/Hy3-preview`, tensor block `model.layers.80` (the MTP layer,
  stored as one extra layer beyond the 80 trunk layers; *not* named "mtp"/"nextn").

- **Sidecar** — the MTP head shipped as a separate `mtp.safetensors`, paired with an already-loaded
  trunk at load time (as opposed to re-converting the whole model).

- **Kill threshold** — the pre-agreed point at which the approach is declared dead and rolled back
  (not patched): acceptance < 0.3 *after* the hidden-state contract has been correctly applied and
  verified.

- **Success bar** — ≥ 1.3× sustained decode speedup with exactness held.
