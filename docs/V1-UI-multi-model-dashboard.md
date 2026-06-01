# Telemak V1 — Multi-model UI in Odysseus dashboard

> Backend supports multi-model concurrent loading since V1 Block 5
> (`Sources/Telemak/Server/Models.swift` — `/admin/load` + `/admin/unload`,
> `models_loaded` list in `/health`). The dashboard UI in Odysseus still
> assumes ONE model at a time. Finish the UI so the operator can run a chat
> model + an embedder side-by-side on the same Mac without ssh-curl
> gymnastics.
>
> Pure dashboard work. No Telemak Swift changes required (the data is
> already exposed at `/health`). Small surface in Odysseus :
> `scripts/api.py` for cluster status enrichment, and
> `scripts/dashboard.html` for the rendering refactor.

## Why this matters

The Telemak Home card on the Odysseus dashboard today behaves as if
Telemak were Argo : single model, single Unload button. But Telemak's
unique selling point vs Argo is exactly that you can park multiple
models in wired memory at once (e.g. a 35B chat MoE + a 0.6B embedder
+ a small TTS model). Until the UI shows them all and lets you
add/remove individually, that capability is invisible and unusable.

Mirror Argo's pool list pattern (`renderArgoPoolsList` +
`renderArgoAddPoolForm`), adapted to "models in one process" instead
of "pools across nodes".

## Current state — what's already there

### Telemak backend (no change required)

```bash
GET http://<telemak>:8003/health
→ {
    "status": "ok",
    "models_loaded": ["inferencerlabs/Qwen3.6-35B-A3B-MLX-9bit"],  # ← LIST
    "wired_memory_used_gb": 37.3,
    "wired_memory_free_gb": 16.7,
    "avg_tok_s_recent": 24.1,                                       # aggregate, not per-model
    ...
  }

POST http://<telemak>:8003/admin/load
{ "model": "<hf-id>" }
→ loads in addition to anything already loaded; does not unload others

POST http://<telemak>:8003/admin/unload
{ "model": "<hf-id>" }   # specific model
{ "all": true }          # everything
→ removes from the in-memory registry, frees wired memory
```

### Odysseus dashboard today

`renderTelemakHomeCard` in `scripts/dashboard.html` (~ line 2909) :

- Reads `a.model` (single string) + `a.loaded` (bool)
- Shows ONE row "Loaded model" with one Unload button
- Load form only visible when nothing loaded (`!loaded && reachable`)
- `cluster_status[telemak-max64]` returned by `scripts/api.py` exposes
  `model` (string) — picks `models_loaded[0]` and drops the rest

This works fine when there's one model. It hides the rest when there
are several, and blocks the user from loading an additional model
without first unloading what's there.

## What to build

### 1. Odysseus `scripts/api.py` — enrich cluster status for kind=telemak

Find the `cluster_status` builder for telemak clusters (search for
where `model = loaded[0]` or similar single-model assignment is done
for `kind == "telemak"`).

Replace with :

```python
# For telemak clusters, surface the full list of loaded models +
# memory budget. Keep `model` (singular) populated with loaded[0] for
# backwards compatibility with the simpler "any model loaded?" check
# used elsewhere in the dashboard, but the multi-model view reads
# `models_loaded`.
status["models_loaded"] = health.get("models_loaded", []) or []
status["wired_memory_used_gb"] = health.get("wired_memory_used_gb")
status["wired_memory_free_gb"] = health.get("wired_memory_free_gb")
status["avg_tok_s_recent"] = health.get("avg_tok_s_recent")
status["loaded"] = bool(status["models_loaded"])
status["model"] = status["models_loaded"][0] if status["models_loaded"] else None
```

### 2. Odysseus `scripts/dashboard.html` — refactor `renderTelemakHomeCard`

Current single-model layout becomes a list. Mirror the Argo pool list
visual but simpler (no per-row node display, no alias-rename, no
pipeline mode).

Mock :

```
┌─ Telemak (64 GB node) ──────────── ● 2 loaded · 38 / 64 GB wired ──────┐
│  telemak · native swift · single-node · http://<telemak-host>:8003   │
│  [Refresh] [Cluster settings]                                       │
│                                                                     │
│  Loaded models                                                      │
│  ┌────────────────────────────────────────────────────────────────┐│
│  │  Qwen3.6-35B-A3B-MLX-9bit             alias: max64:35b    [⌫] ││
│  │  Qwen3-Embedding-0.6B-mxfp8           alias: max64:emb    [⌫] ││
│  └────────────────────────────────────────────────────────────────┘│
│                                                                     │
│  [ + Load another model ]                                           │
│                                                                     │
│  ▾ form when clicked :                                              │
│    [ inferencerlabs/Qwen3.6-... ]                                   │
│    [Load]                          (existing form, lightly reused)  │
└─────────────────────────────────────────────────────────────────────┘
```

Empty-state (no models loaded) : skip the "Loaded models" section
entirely, show the Load form directly (same look as today's
empty-state).

Implementation notes :

- Iterate `a.models_loaded` (the new list) instead of `a.model`
- Alias display : reuse the `_telemak_short_id` slugging logic that
  the proxy uses for multi-model routing (`cluster_id:short_id`).
  If only one model is loaded, the bare `cluster_id` also routes to
  it ; show that as a hint
- Unload button per-row : `data-act="telemak-unload"
  data-cluster="<cid>" data-model="<full-hf-id>"`. The existing
  unload handler (`actions["telemak-unload"]`) gets a `data-model`
  param ; if absent (back-compat for the old single-Unload button),
  fall back to "all"
- Memory pill in the header : `${used.toFixed(1)} / ${(used +
  free).toFixed(1)} GB wired`. Light-rust if `free < 5 GB` (memory
  warning).
- "+ Load another model" button always visible when `reachable` —
  click toggles `state.telemak_load_form[cid].open` between
  visible/hidden. The form itself is the same as today's empty-state
  load form, just framed as an addition rather than an only-option

### 3. Telemak Swift — small touch (optional, V1.x)

The Telemak `/health` already returns `models_loaded`. The aggregate
`avg_tok_s_recent` is reasonable for the header pill, but the list
rows would benefit from per-model recent tok/s and per-model wired
memory. This is polish, not blocking V1 ship.

If you want to add it cleanly (~30 lines in
`Sources/Telemak/Engine/ModelRegistry.swift` to track per-model
stats, expose in `/health` as `models_loaded_detail` with the same
keys as currently aggregated), do it. Otherwise leave a TODO and
keep the dashboard rows showing just the model id + alias.

## Test plan

1. Empty Telemak (no model loaded) → Home card shows the simple
   "Load a model" form, same as today.
2. Load one model via the form → it appears in the new "Loaded
   models" list with its alias, "+ Load another model" button below
   stays visible.
3. Click "+ Load another model" → form opens inline (does not unload
   the first one). Submit a second model (e.g. an embedder).
4. Both models appear in the list. Unloading one via its ⌫ button
   does NOT touch the other.
5. Companion can chat with `cluster_id` when one model loaded ; must
   use `cluster_id:short_id` form when ≥ 2 loaded (already the V1
   proxy contract, just surface this in the alias column).
6. Refresh the dashboard, kill Telemak, restart → list rebuilds from
   `/health` correctly. No stale local state.
7. Cluster Settings → Models tab matrix card (`renderTelemakModelsCard`,
   line ~2706) — adapt to also show the list, with same per-row
   Unload affordance, so the user has parity between the Home tab
   and the Models tab.

## Done criteria

- [ ] `cluster_status[telemak-*]` returns `models_loaded` (list),
  `wired_memory_used_gb`, `wired_memory_free_gb` in
  `scripts/api.py`
- [ ] `renderTelemakHomeCard` shows N models with per-row Unload
  buttons + "+ Load another model" form
- [ ] `renderTelemakModelsCard` (Models tab) mirrors the same list
- [ ] Memory pill in the card header shows wired used/total
- [ ] Companion smoke : load 2 models on 64 GB node, chat with one,
  embed with the other via `cluster_id:short_id` aliases — both
  work concurrently
- [ ] No regression on single-model Telemak clusters

## Out of scope

- Per-model tok/s metrics in the list rows (deferred to V1.x once
  `Sources/Telemak/Engine/ModelRegistry.swift` tracks per-model
  stats).
- Mac menu-bar load/unload (the operator's decision : the menubar already
  has a "Open dashboard" link, all interactive work happens in the
  Odysseus dashboard).
- MTP speculative decoding (V2, gated on
  `V2-MTP-DRAFT-PORT.md` blocker resolution).
