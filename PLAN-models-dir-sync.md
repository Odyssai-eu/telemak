# Plan: models_dir master/slave sync (OdyssAI-X ⟶ Telemak)
_Locked via grill-with-docs — by Claude + Sophie, 2026-06-21. Rev 3 (post MiniMax rounds 1-2)._

## Goal
Make a Telemak cluster's **models directory** a single, configurable source of truth owned by **OdyssAI-X (master)**, so a **paired Telemak (slave)** discovers / lists / loads / serves models from the directory OdyssAI-X dictates — with **no hardcoded path anywhere in code** (depersonalization-clean). Fixes the empty model picker on nodes whose models are nested one level deeper than the scan root (server scans `/Volumes/models`; models live at `/Volumes/models/odysseus/<org>/<name>/`) and removes the silent-empty-picker failure mode.

## Terms (this feature's glossary — the repo's existing `CONTEXT.md`/`PLAN.md` are the SEPARATE MTP context, ignore them)
- **models directory** — the root the Telemak server scans for available models, at depth `<org>/<name>[/snapshots/<hash>]/config.json` (2 levels). "Effective" = the value after resolution.
- **master / OdyssAI-X** — the Python engine; sole authority on a *paired* cluster's models directory (`cd["models_dir"]`).
- **slave / Telemak** — the native Swift app (HTTP server :8003 + menubar); reads + adapts to the directory the master sets.
- **paired** — registered as a `kind=telemak` cluster in OdyssAI-X. **standalone** — no OdyssAI-X registration.
- **reconcile** — OdyssAI-X idempotently (re)asserting its directory onto the Telemak server.

## Approach

### Telemak server (Swift — `Sources/Telemak/`)
1. **Persisted runtime config** — new `config.json` in the **same data dir as `state.json`/`api-key.txt`** (verify exact path at impl — `~/.telemak/` vs `~/telemak/`; match `StateStore`). Keys `models_dir`, `managed` bool. New `Server/Config.swift`, **atomic write** (`Data.write(options:.atomic)` = temp+rename; mirror the `StateStore` actor pattern).
2. **One CACHED resolver, applied at EVERY read site.** `ModelsConfig.effectiveDir() -> String?` = `config.json.models_dir` (master) **>** env `TELEMAK_MODELS_DIR` (legacy/migration) **>** `nil`. **Computed at boot + recomputed on POST — NOT a per-call env read** (MiniMax r2-C). **Route ALL 9 confirmed reads through it** (a 2-site patch is the bug): `ModelLoader.swift:53,92,105,466`, `EmbedderLoader.swift:75`, `RamBudget.swift:29`, `MTPModelLoader.swift:265`, `Server/Models.swift:328`, `AvailableModels.swift:31`. **Update those sites' `TELEMAK_MODELS_DIR`-named error strings** to "set your models directory (OdyssAI-X or the menubar)". When `nil`, `/admin/models/available` returns `{models: [], models_dir_unset: true}`.
3. **`POST /admin/models-dir {dir, create?, managed?}`** (route `Server/Models.swift` after L35 + handler; register `App.swift:28`): validate; not-exists + `create:true` → `mkdir -p` then accept; not-exists + no-create → `409 not_found`; on accept → atomic-write `config.json` (+ `managed`), **recompute the cached effectiveDir**. **No explicit re-scan** — `available()` scans the FS per request (`Models.swift:106`). **Loaded models untouched.** (Consolidate the create logic with the installer path — same helper.)
4. **`GET /admin/models-dir`** → `{ dir, source: "config"|"env"|"unset", managed: bool }`.
5. **Dir-aware registry + state replay** (MiniMax r1-#4, r2-A/B/D):
   - The model registry entry **records the effective dir at load time**; **unload re-canonicalizes the id against that recorded dir, not the current one** (else a runtime dir-change orphans the loaded entry — r2-A).
   - `state.json` gains a per-entry `dir`. On boot (`ServeCommand.swift:74-88` `replayState`): **skip + log** any entry whose recorded dir ≠ current `effectiveDir()`. **Absent dir (old-schema entry) → skip** (untrusted — r2-D).
   - **Schema migration**: the old `state.json` (plain `[String]`) won't decode → on first boot after upgrade it is **reset** (acceptable; the master repopulates via the load proxies — r2-B). State this explicitly.

### OdyssAI-X (Python — `scripts/api.py`, sibling `MLX Distributed` repo)
6. **Master push on save** — `kind=telemak` cluster saved with `models_dir` → `POST {upstream}/admin/models-dir {dir: cd["models_dir"], managed: true}`. On `409`, surface "create it?" → re-POST `create:true`.
7. **Lazy reconcile-on-mismatch** — before the available-models proxy (`api.py:8974`, today no dir param + `models_dir: None`) and the load proxy (`_telemak_proxy_load` ~`9124`): compare Telemak's effective dir (`GET /admin/models-dir`) vs `cd["models_dir"]`; **POST only on mismatch**. Self-heals. Only `cd["models_dir"]` — never hardcoded.

### Telemak menubar (Swift — `Sources/TelemakMenuBar/`)
8. **Replace the "Dashboard" button** (`TelemakMenuBarApp.swift:648-650`) **with a "Models" view** (after ~L520): model **picker** (`/admin/models/available`), **effective models path**, **local Load** (`POST /admin/load`).
9. **Add `/admin/models-dir` to the HealthPoller** (the menubar polls `/health`+`/admin/activity` at `:315,352`; extend the `refreshActivity()` pattern ~`:351-380`) **WITH the Bearer auth header** (`Settings.apiKey`, r2-F) — the admin routes are bearer-protected. **`managed:true` ⇒ read-only ALWAYS** (precedence over `source` — r2-I), labelled "managed by OdyssAI-X". Else (**standalone**, `managed:false`) editable → `POST /admin/models-dir {managed:false, create?}` (create-prompt). `Settings.swift`.

### Depersonalization (`Sources/TelemakMenuBar/Installer.swift`)
10. **Kill the hardcoded `/Volumes/models/odysseus`** in `defaultModelsDir()` (197-203) **AND remove the `~/Telemak-Models` fallback + the `install()` auto-create (L217)** (Q2-B). First-run picker (119-131) **requires an explicit choice** + **writes it to `config.json`** (managed:false), **not** the LaunchAgent plist env (L263; plist env stays a legacy read only). **`--provision` (L21-23) without `--models-dir` → abort** with "models-dir required when provisioning from OdyssAI-X" (r2-G — the Configurator MUST pass it). No `odysseus`/personalized path in code.

### Docs + tests
11. **README** (`:65,116`): `TELEMAK_MODELS_DIR` = legacy/fallback; `config.json` is the source. **CLI** (`ClientCommands.swift:26-30`): print the "set your models directory" hint when `models_dir_unset` (r2-E, minor).
12. **Tests** (`Tests/TelemakTests/` has zero coverage): unit-test `ModelsConfig.effectiveDir()` resolution order (config > env > nil); assert the 9 sites consume it; test create-on-missing on **both** the install picker and the POST path (r2-H).

### Deploy
13. Rebuild Telemak (server + menubar = one binary) → redeploy the fleet (8 nodes). Post-deploy, nodes have no `config.json` → fall to legacy env (`/Volumes/models`, the bug) **until OdyssAI-X's reconcile pushes `cd["models_dir"]`** (`/Volumes/models/odysseus`) → `.30/.31` fixed. Release-aligned.

## Key decisions & tradeoffs
- **Single CACHED `effectiveDir()` at all 9 sites** (r1-#1, r2-C) — no per-call env reads, no source-of-truth drift.
- **Installer writes `config.json`, not the plist env** (r1-#2) — unifies the runtime source; env = pure legacy.
- **No neutral default; no auto-create** (r1-#3, Q2-B) — explicit choice or the prompt.
- **Read-only-when-paired; `managed` always wins** (Q1-A, r2-I).
- **Reconcile push-on-save + lazy-on-mismatch** — master wins + self-heals.
- **Validate + offer-create (Q3)**; **non-destructive dir change**; **dir-aware registry/replay** with recorded-dir unload canonicalization + old-schema reset (r1-#4, r2-A/B/D).
- **No explicit re-scan** (r2-#9) — per-request scan.

## Risks / open questions
- **Data-dir path** (`~/.telemak/` vs `~/telemak/`) — `config.json` must sit where `state.json`/`api-key.txt` live; verify at impl.
- **Per-request FS scan cost** — `available()` scans each call; a short TTL cache is a separate, noted optimization.
- **state.json one-time reset** on first upgrade is acceptable (master repopulates).
- **`/Volumes/models/odysseus` on-disk path** stays as Sophie's data (not renamed).

## Implementation notes (MiniMax round 3 — pre-approved clarifications)
- **`ModelsConfig` must be a SHARED target** — `TelemakMenuBar` does NOT depend on `Telemak` (separate executables, see `Package.swift`). Put `ModelsConfig` (pure config-file reader, no Hummingbird/MLX deps) in `TelemakVersion` (already shared by both) or a new `TelemakShared` lib — do NOT duplicate. (r3-ii)
- **Registry carries the load-time dir** — add `recordedDir: String?` to `Loaded`/`LoadedEmbedder`/`LoadedDraft` (`ModelRegistry.swift` ~22-50), thread into `persistState()` (~415) and into a new `canonicalIdentifier(_:against dir:)` overload used by **unload** (not the current dir). (r3-i/iii)
- **Canonicalize on write** — `URL(fileURLWithPath:).standardized.path` the dir before persisting to `config.json`, so the resolver compares apples-to-apples. (r3-vi)
- **Thread `ModelsConfig` through** — init in `ServeCommand.swift` (~43-54); add a 3rd param to `ModelsHandler(registry:activity:)` (`Models.swift:10-12`).
- **PREREQUISITE / SCOPE** — the OdyssAI-X `api.py` push+reconcile (steps 6-7) is a SEPARATE repo/PR. **`.30/.31` is fixed only once BOTH** the Telemak build AND the api.py change ship. Sequence: Telemak server+menubar build → deploy fleet → api.py push/reconcile → OdyssAI-X cluster config drives the dir. (r3-vii)
- **Noted (bounded / pre-existing)** — a wiredMemory reservation can leak one entry on unload-by-absolute-path after a dir change (r3-iv, bounded); the `~/.telemak/` (state) vs `~/telemak/` (api-key) split is pre-existing tech debt → align to `~/.telemak/` (r3-v, separate).

## Out of scope
- Renaming the on-disk `/Volumes/models/odysseus` directory.
- Changing the scan depth/layout (`<org>/<name>` 2-level).
- HF auto-download into the dir; a TTL scan cache (noted, separate).
- The MTP context (`PLAN.md`/`CONTEXT.md`).
