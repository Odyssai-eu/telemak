# Telemak V1 — TODO

> Runbook V1 pour Telemak. V0 (commit `55ec3f6` + post-V0 fixes) est shippé
> et tourne en prod sur `192.168.86.50:8003` via LaunchAgent.
> Voir [`SESSION-2026-05-23-2000-telemak-mvp-v0.md`](SESSION-2026-05-23-2000-telemak-mvp-v0.md)
> pour l'historique V0.
>
> Cette V1 ship en un seul bloc — pas de V1.5 / V2. Tout est en scope.

## Contexte rapide

**V0 livre** :
- `/v1/chat/completions` stream + non-stream
- `/v1/models`, `/admin/load`, `/admin/unload`, `/health`
- SSE usage chunk avant `[DONE]` (commit `2d3c926`)
- Release build deployed sur max-64.lan via LaunchAgent (`eu.odyssai.telemak`)
- Integration Odysseus en first-class `kind: telemak` cluster (commits `518e2fa`, `687a137`, `edc64f6` côté Odysseus)
- Filter `<think>` dans Odysseus proxy pour Qwen3.5/3.6 auto-open think models

**V1 cible** : multi-modèle concurrent, KV prefix cache, capability contract,
multi-request concurrency, sampling params complets, tokens accurate, tools,
Anthropic `/v1/messages`, CORS, logging, optional auth, CLI subcommands, menu
bar `.app`, doc Gatekeeper.

## Design decisions (verrouillées)

| # | Décision | Choix |
|---|---|---|
| D1 | Use case multi-modèle principal | Petits modèles + chat sur même Mac. Ex `max-64` : Qwen3.5-35B-A3B (chat) + Qwen3-Embedding-0.6B (embeddings) en parallèle. Sophie veut ce pattern explicitement. |
| D2 | Eviction policy quand `load` dépasse RAM | **REFUSE** avec error explicite + memory breakdown. Pas d'auto-LRU. Opérateur unload explicitement ce qu'il veut libérer. Aligné avec la philo Odysseus "no auto-swap". |
| D3 | `POST /admin/unload` sans `{model}` | **REFUSE 400** ("specify model id"). Pour unload all : `POST /admin/unload?all=true` avec confirmation explicite. |
| D4 | Alias scheme dans Odysseus `/v1/models` | Quand 1 seul modèle chargé : `telemak-max64` (back-compat). Quand N modèles : émettre N entries `telemak-max64:<short-model-id>` (colon, URL-safe). L'alias `telemak-max64` seul disparaît dès qu'il y a N>1 modèles. |
| D5 | Distribution `.app` | Pas d'App Store, pas de notarization Apple. README explique le Gatekeeper bypass : *"right-click `.app` → Open la première fois"*. Open-source acceptable. |
| D6 | Bonjour / mDNS discovery | **Cut**. Manual IP entry dans Odysseus dashboard suffit. |
| D7 | KV cache scope | session_id-based per-session KVCache (header `X-Session-Id` ou body field). LRU eviction quand session count dépasse 32 (config). Drop cache si session change de modèle. |
| D8 | Concurrence multi-request | Queue derrière per-model lock (FIFO). Pas de batching mlx-swift V1 — c'est V2 si besoin. |

## Cross-repo work flag

Certaines tâches V1 nécessitent des changes côté **Odysseus** (`~/Claude/code/MLX Distributed/`).
L'agent Telemak a R/W access là-bas via `additionalDirectories` dans
`.claude/settings.local.json`. Touche aux fichiers Odysseus uniquement pour les
items marqués `[ODYSSEUS]` ci-dessous. Tout le reste est Telemak-only.

## Phase 5 — Model lifecycle finalize (V0 polish before V1 starts)

- **5a** Persist last-loaded model(s) à `~/.telemak/state.json`. Au startup, rejoue les `load` automatiquement. Skip si fail (log + continue).
- **5b** `GET /admin/models/available` : scan `TELEMAK_MODELS_DIR` + `~/.cache/huggingface/hub/` → `[{id, source, size_gb, last_modified}]`. Sans ça, l'opérateur doit deviner les IDs.
- **5c** `/health` enrichi : `{status, uptime_s, models_loaded: [...], wired_memory_used_gb, wired_memory_free_gb, requests_served, avg_tok_s_recent}`.

## V1 — Block 1 : Multi-model + core inference

- **v1.1** `ModelRegistry` actor : remplacer `current: ModelContainer?` par `loaded: [String: ModelContainer]`. Read-side : `get(id) -> ModelContainer?`. Write : `add(id, container)`, `remove(id)`, `removeAll()`. Single-flight lock per model (different models can serve concurrent requests, same model serializes — V1 keeps it simple, batching is V2).
- **v1.2** `POST /admin/load {model}` : si déjà chargé → 200 no-op. Sinon : check RAM budget (avoir helper `estimateRamFor(model) -> Int64`), si fit → load. Si insuffisant → **400** avec body `{error: "insufficient_memory", needed_gb: X, available_gb: Y, currently_loaded: [...]}`. Pas d'auto-evict.
- **v1.3** `POST /admin/unload {model}` : unload spécifique. Sans `{model}` ni `?all=true` → **400 "specify model id or pass ?all=true"**. Avec `?all=true` → unload tout (utile pour reset).
- **v1.4** `GET /admin/memory` : `{used_gb, free_gb, total_gb, per_model: {id: gb}}`. Utilisé par dashboard + future eviction policy.
- **v1.5** `GET /v1/models` retourne maintenant N entries (un par modèle chargé), pas juste 1.
- **v1.6** `POST /v1/chat/completions` route le request sur le bon container par `req.model`. Si `req.model` non chargé → **404 model_not_loaded** avec `ready_models: [...]`.

### [ODYSSEUS] côté `scripts/api.py` :

- **v1.7** `_telemak_loaded_models` retourne déjà `list[str]` — bon. Modifier `_v1_models` du Telemak block (lignes ~4690) pour émettre **N entries** au lieu d'un seul `cluster_id`. Schema : si 1 modèle → garde `cluster_id` (back-compat) ; si N → `<cluster_id>:<short-id>` par modèle (short-id = part après le `/` final du HF id, ex `qwen3.5-35b-a3b-mlx-9bit`).
- **v1.8** `_telemak_proxy_chat_completion` : parser `req.model`, si `:` → split cluster + model suffix, rewrite `forward_body.model` au full HF id correspondant. Si pas de `:` et 1 seul chargé → comportement actuel (rewrite au seul loaded). Si pas de `:` et N chargés → 400 "ambiguous model id, use cluster:model form".

## V1 — Block 2 : KV prefix cache + capability

- **v1.9** KV prefix cache per session. Accepter `session_id` dans body field OU header `X-Session-Id`. Maintenir `[String: KVCache]` dans `ModelRegistry` keyed par `(model_id, session_id)`. mlx-swift-lm expose `KVCache` — adapter pour passer une cache existante à `ChatSession`. Drop cache si session bascule sur autre modèle.
- **v1.10** LRU eviction sur sessions : max 32 sessions cached par défaut (config `TELEMAK_MAX_SESSIONS` env var). Evict la plus vieille quand on dépasse.
- **v1.11** `GET /admin/sessions` : `{sessions: [{id, model, last_used_s, kv_size_mb}]}`. Pour debug + dashboard observability.
- **v1.12** `POST /admin/sessions/clear` (avec `?session_id=X` ou `?all=true`) : drop cache manuellement.
- **v1.13** `GET /.well-known/inference-engine.json` : capability contract OpenAI-compat. Schema :
  ```json
  {
    "engine": "telemak",
    "version": "0.2.0",
    "capabilities": {
      "stream": true,
      "tools": true,
      "vision": false,
      "max_context": 32768,
      "session_cache": true,
      "openai_compat": "v1",
      "anthropic_compat": "v1"
    },
    "models": [{"id": "<hf-repo>", "x_telemak": {"size_gb": N, "context": ...}}, ...]
  }
  ```
- **v1.14** [ODYSSEUS] Le `/v1/models` Telemak section déjà émet `x_odyssai` block — V1 doit aussi auto-discover via `/.well-known/inference-engine.json` du Telemak upstream au lieu d'hardcoder `stream: true, tools: false`. Cache ce contract (TTL 60s).

## V1 — Block 3 : Inference robustness

- **v1.15** Multi-request concurrency : per-model FIFO queue (Swift actor sérialise déjà les requests sur le même container). Vérifier qu'inférences sur **modèles différents** s'exécutent en vrai parallel (deux actors différents, deux task groups). Test : two concurrent curl streams to two different loaded models, both progress.
- **v1.16** Sampling params : étendre `ChatCompletionRequest` Codable schema avec :
  - `stop: [String]?`
  - `repetition_penalty: Float?`
  - `top_k: Int?`
  - `min_p: Float?`
  - `seed: UInt64?`
  Câbler dans `mlx-swift-lm`'s `GenerateParameters` + `SamplerParameters`.
- **v1.17** Token usage accurate : remplacer `completionChars / 4` par `tokenizer.encode(piece).count` pendant le stream + `tokenizer.encode(userPrompt).count` pour prompt_tokens. Le tokenizer est accessible via `ModelContainer`.
- **v1.18** Audit que `mlx-swift-lm`'s `streamResponse` expose vraiment le token count, pas juste les chunks de texte. Si oui, utiliser cette source directement.

## V1 — Block 4 : Tooling

- **v1.19** `enable_thinking` au chat-template level : accepter `enable_thinking: Bool?` dans request (ou body field `thinking: Bool`, alias). Pass to `ChatSession` via instructions OR via chat_template kwargs (mlx-swift-lm doit exposer). Si oui → le filter `<think>` côté Odysseus devient redondant (toujours OK, c'est un belt-and-suspenders).
- **v1.20** Tools / function calling :
  - Accepter `tools: [Tool]?` et `tool_choice: ToolChoice?` dans request (schema OpenAI standard).
  - Rendre les tools dans le chat_template (mlx-swift-lm + tokenizer supportent `apply_chat_template(messages, tools=...)`).
  - Parser le model output pour `<tool_call>...</tool_call>` ou `<|tool_call|>...` (model-specific — voir Qwen3 / Llama 3 docs).
  - Émettre `delta.tool_calls: [{id, type:"function", function:{name, arguments}}]` dans stream.
  - Non-stream : `message.tool_calls`, `finish_reason: "tool_calls"` au lieu de "stop".
  - Capability `tools: true` dans `/.well-known/inference-engine.json`.
  - Test minimum : Companion peut envoyer un tool call (ex Tavily search) et le voir parser correctement.
- **v1.21** Anthropic `/v1/messages` parity :
  - Accepter le request shape Anthropic (`system: String?`, `messages: [{role, content: ContentBlock[]}]`, `max_tokens` required, etc).
  - Convertir en interne au format OpenAI pour l'inférence.
  - Reconvertir la réponse en shape Anthropic (`content: [{type, text}]`, `stop_reason`, `usage`).
  - Streaming : émettre les events Anthropic — `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`.
  - [ODYSSEUS] Le `/v1/messages` côté Odysseus existe déjà — verify que kind=telemak routing passe par `_telemak_proxy_chat_completion` adapté pour le format Anthropic, ou par un `_telemak_proxy_messages` nouveau.

## V1 — Block 5 : Ops

- **v1.22** CORS headers : `Access-Control-Allow-Origin: *` (config `TELEMAK_CORS_ORIGIN` env var pour override), `Allow-Methods: GET,POST,OPTIONS`, `Allow-Headers: Content-Type, Authorization, X-Session-Id`. Handler OPTIONS pour le preflight.
- **v1.23** Logging : remplacer `print` par `swift-log`. Output JSON-formatted, niveau configurable via `TELEMAK_LOG_LEVEL`. Rotating file handler avec daily rotation, max 7 files, à `~/.telemak/logs/telemak-YYYY-MM-DD.log`.
- **v1.24** Optional admin API key auth : si `TELEMAK_API_KEY` env var est set → exiger header `Authorization: Bearer <key>` sur `/admin/*`. Les endpoints d'inférence restent ouverts pour le routage LAN Odysseus. Sinon → open (current behavior, LAN-trusted install).
- **v1.25** CLI subcommands :
  - `telemak models` : liste local cache + TELEMAK_MODELS_DIR (réutilise logique de `/admin/models/available`)
  - `telemak load <id>` : load via HTTP (assume `serve` is running) ou direct si standalone
  - `telemak unload <id>` ou `telemak unload --all`
  - `telemak chat <prompt> [--model <id>]` : test offline d'inférence, useful pour smoke
  - `telemak version` (déjà existant)
- **v1.26** Menu bar `.app` (SwiftUI cible séparée dans Package.swift) :
  - Icone dans menu bar (logo OdyssAI)
  - Click → popover montrant :
    - Status (running / starting / error)
    - Loaded models (avec wired memory used per model)
    - tok/s récent
    - Boutons : Load (avec model picker depuis `/admin/models/available`), Unload per-model, Open Dashboard, Quit
    - Toggle "Auto-load at boot"
  - Connect au backend Telemak via `127.0.0.1:8003` (assume running via LaunchAgent)
- **v1.27** Doc Gatekeeper : section README "First run" :
  - Download `.app` ou build localement
  - First open : "App can't be opened because Apple cannot check it for malicious software"
  - Solution : right-click `.app` → Open → confirm dialog
  - Subsequent opens : normal double-click

## Done criteria V1 launch

Telemak V1 est **done** quand, sur max-64.lan :

1. `telemak serve` charge auto les modèles persistés au boot (5a)
2. Deux modèles chargés simultanément (ex Qwen3.5-35B-A3B + Qwen3-Embedding-0.6B), visibles dans `/v1/models` (v1.5)
3. `POST /v1/chat/completions {model: "qwen3.5-..."}` route correctement, parallèle à un `POST /v1/embeddings {model: "qwen3-emb..."}` (v1.6, v1.15)
4. Sur conversation multi-tour avec `session_id`, tour 2+ a TTFT << tour 1 (KV cache effectif, v1.9)
5. `GET /.well-known/inference-engine.json` retourne capabilities complètes (v1.13)
6. Tool call test : Companion envoie un Tavily search, Telemak parse + retourne correctement (v1.20)
7. `/v1/messages` (Anthropic) reçoit + répond avec shape Anthropic (v1.21)
8. Bearer auth fonctionne si `TELEMAK_API_KEY` set, ouvert sinon (v1.24)
9. CLI : `telemak models | grep qwen` listent les modèles dispos (v1.25)
10. Menu bar `.app` montre les modèles chargés + tok/s live (v1.26)
11. [ODYSSEUS] Dashboard Home Telemak card liste N modèles avec Unload per-model (v1.7, v1.8)

## Commands utiles pendant le dev

```bash
# Build Release
cd /Users/sophie/Claude/code/telemak && ./scripts/build.sh Release

# Deploy on max-64.lan
cd .xcbuild/Build/Products/Release && tar czf /tmp/telemak.tgz telemak mlx-swift_Cmlx.bundle
scp /tmp/telemak.tgz admin@192.168.86.50:/tmp/
ssh admin@192.168.86.50 'cd ~/telemak/Release && tar xzf /tmp/telemak.tgz && launchctl kickstart -k gui/$(id -u)/eu.odyssai.telemak'

# Verify
curl http://192.168.86.50:8003/health
curl http://192.168.86.50:8003/v1/models

# Smoke via Odysseus
curl http://192.168.86.141:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"telemak-max64","messages":[{"role":"user","content":"hi"}],"stream":false,"max_tokens":50}'
```

## Workflow git

- Travailler sur `feat/v1-<block>` branches, PR vers main
- Auto-mode classifier refuse `git push origin main` direct — c'est intentionnel
- Sophie merge les PRs après review (ou si elle te dit `merge`)
- Commits Conventional + HEREDOC + Co-Authored-By footer
