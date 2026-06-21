# Plan Review Log: models_dir master/slave sync (OdyssAI-X ⟶ Telemak)

Act 1 (grill-with-docs) complete — plan locked with Sophie:
- **Q1-A** — menubar read-only when paired, editable only standalone; reconcile = push-on-save + lazy idempotent-on-mismatch.
- **Q2-B** — no hardcoded default; resolution `config.json > env > unset→explicit-config prompt`.
- **Q3** — validate path + offer-to-create if missing; re-scan on change; loaded models untouched.

MAX_ROUNDS=5 · MAX_TOOL_CALLS=25 · MODEL=MiniMax-M3 · PLAN_FILE=PLAN-models-dir-sync.md

## Round 1 — MiniMax (REVISE)
Read: App.swift, Models.swift, AvailableModels.swift, ModelLoader.swift, Installer.swift, TelemakMenuBarApp.swift, Settings.swift, StateStore.swift, ServeCommand.swift, EmbedderLoader.swift, RamBudget.swift, MTPModelLoader.swift + searches. Verified the file/line/symbol claims.
1. **(critical)** `TELEMAK_MODELS_DIR` read at **9 sites**, not 2 → need one resolver.
2. **(critical)** Installer bakes env into the plist (L263) → should write `config.json`.
3. **(important)** `~/Telemak-Models` fallback + `install()` auto-create (L217) violate Q2-B.
4. **(important)** `state.json` replay (`ServeCommand.swift:74-88`) races a dir change → must be dir-aware.
5. **(important)** `models_dir_unset` flag + a menubar `/admin/models-dir` poller are missing.
6. **(important)** `TELEMAK_MODELS_DIR`-named error strings become stale.
7. **(minor)** README documents the env var.
8. **(minor)** `RamBudget` nil-silent (orthogonal).
9. **(minor)** "re-scan" is implicit — `available()` scans per request.
10. **(meta)** no tests for the resolution order.

### Claude's response (Rev 2)
ACCEPTED 1,2,3,4,5,6,7,9,10 — all folded into Rev 2: single `ModelsConfig.effectiveDir()` at all 9 sites + error strings (1,6); installer writes `config.json` not the plist env (2); removed `~/Telemak-Models` fallback + auto-create (3); dir-aware state replay (4); `models_dir_unset` payload + `/admin/models-dir` menubar poller (5); README legacy note (7); clarified no-explicit-rescan (9); resolution-order tests (10). NOTED 8 — orthogonal, covered by the resolver refactor.

## Round 2 — MiniMax (REVISE)
Re-read: Middleware.swift, ModelRegistry.swift, ClientCommands.swift + searches (StateStore actor, api-key path, state.json consumers). Confirmed all 10 round-1 findings addressed. New (Rev-2 vs code):
- A. `canonicalIdentifier` re-canonicalizes against the NEW dir on unload → record dir-at-load in the registry.
- B. old `state.json` (`[String]`) won't decode the new schema → reset on first upgrade.
- C. `effectiveDir()` per-call env read → cache (boot + on-POST).
- D. replay vs absent recorded dir → absent ⇒ skip.
- E. CLI prints `(empty)` → surface unset hint (minor).
- F. menubar poll/POST need the Bearer auth header; api-key path `~/telemak` vs `~/.telemak` → verify.
- G. `--provision` without `--models-dir` falls to the removed default → abort; Configurator must pass dir.
- H. create dup (install vs POST) → consolidate + test both.
- I. `source:env` AND `managed:true` conflict → `managed` always wins ⇒ read-only.

### Claude's response (Rev 3)
ACCEPTED A,B,C,D,F,G,I + folded E,H into Rev 3: dir-aware registry w/ recorded-dir unload canonicalization (A) + old-schema reset (B) + absent→skip (D); cached `effectiveDir` boot+on-POST (C); menubar Bearer auth + data-dir path verify (F); `--provision` aborts without `--models-dir` (G); `managed:true`→read-only-always (I); CLI unset hint (E); consolidate create + test both paths (H).

## Round 3 — MiniMax (APPROVED)
Re-read: `Package.swift` (target deps), ModelRegistry/WiredMemory searches. Confirmed all 9 round-2 findings addressed; core logic sound. New (non-blocking, impl-time), all folded into Rev 3 "Implementation notes":
- i. `canonicalIdentifier(_:against dir:)` overload for unload.
- ii. **`ModelsConfig` must be a SHARED target** — `TelemakMenuBar` can't import `Telemak`.
- iii. `recordedDir` field on `Loaded`/`LoadedEmbedder`/`LoadedDraft` + `persistState()`.
- iv. wiredMemory one-entry leak on unload-by-path-after-change (bounded, noted).
- v. `~/.telemak` vs `~/telemak` path split (pre-existing tech debt, noted).
- vi. canonicalize path (`.standardized.path`) on config write.
- vii. api.py changes are a prerequisite PR — `.30/.31` fixed only after both ship.

**VERDICT: APPROVED** — "Core logic is sound: single cached resolver at all 9 sites, manager-wins precedence, dir-aware registry/replay with schema migration, no per-call env reads, consolidated installer flow, bearer auth on menubar, no hardcoded paths."

## Resolution — CONVERGED (3 rounds)
Plan locked + approved. No code written during grill/review. Awaiting Sophie's sign-off to implement.
