# Telemak — qu'est-ce que c'est ?

> **odyssai.eu · le runtime mono-Mac.** Un moteur d'inférence IA **natif
> Swift** qui tourne sur **un seul Mac Apple Silicon**, expose les APIs
> OpenAI **et** Anthropic en local, et garde plusieurs modèles
> co-chargés en mémoire wired simultanément. Pensé comme le pendant
> "single-Mac" d'Odysseus : pas de cluster, pas de container Docker,
> pas de Python — juste un binaire et un menu-bar.
>
> *État au 2026-05-29 · version 0.6.15 · baseline stable `v0.6.15-stable` · son grand frère côté distribué : **Odysseus**.*

---

## En une phrase

> Tu poses un `.app` sur un Mac Studio (ou un MacBook Pro), tu le lances
> depuis le menu-bar, et tu as `/v1/chat/completions` + `/v1/messages` qui
> servent Gemma 4, Qwen Coder Next, MiniMax ou d'autres modèles MLX natifs —
> utilisable directement par Companion ou par n'importe quel client
> OpenAI/Anthropic, ou enrôlable dans un cluster Odysseus comme provider
> HTTP en deux clics.

---

## Le problème qu'on résout

Pour un usage IA local **sur un seul Mac**, le paysage 2026 n'est
satisfaisant nulle part :

| Option | Limites |
|---|---|
| **Ollama** | Excellent pour démarrer mais : pas de multi-modèle concurrent en mémoire wired, pas de KV cache cross-turn typé OpenAI, API standard partielle (Anthropic absent), pas d'integration cluster native, pas pensé MLX |
| **LM Studio** | UI chat embarquée mais pas d'API headless propre pour servir d'autres clients, pas multi-modèle co-loadé, ferme le hardware sur sa UI |
| **mlx-vlm / mlx-lm Python** | Stack native MLX mais Python : dépendances lourdes, venv à maintenir, pas de menu-bar, pas de `.app`, pas d'integration LAN simple |
| **Inferencer.app** | Excellent perf MLX sur 1 Mac, mais propriétaire, payant, fermé, focus single-user chat, pas d'API ouverte pour autres clients |
| **Odysseus seul** | Génial pour distribué multi-Mac (≥ 200 B paramètres) mais lourd pour un seul Mac : Docker + orchestrateur Python + runner SSH pour servir Qwen3.6-35B sur un MacBook → overkill |
| **Serveur HTTP custom MLX** | Faut tout réécrire à chaque rebuild : tokenizer, streaming SSE, sessions, tools, Anthropic translation, multi-modèle, KV cache, wired memory policy |

Côté hardware, un Mac Studio M4 Max 64 GB ou un M3 Ultra 96 GB fait
tourner sans problème Gemma 4 26B-A4B autour de 70 tok/s, Qwen Coder
Next autour de 40-50 tok/s, plus un embedder à côté — **si** le runtime
sait gérer la mémoire wired correctement et garder les modèles chauds.

Ce qui manquait : un **runtime mono-Mac**, **natif Swift**, **headless
sauf un menu-bar discret**, qui parle les APIs standard et qui s'intègre
dans le LAN Odysseus sans bricolage.

---

## Ce que fait Telemak

### Concrètement, c'est :

1. **Un binaire Swift natif** qui tourne directement sur Apple Silicon
   sans interpréteur, sans container, sans dépendances système. Pas de
   Python à installer. Distribué en `.app` bundle (`Telemak.app`) avec
   `LSUIElement=true` (pas d'icône Dock).

2. **Un serveur HTTP** (Hummingbird) sur un port configurable
   (par défaut `8003`) qui expose :
   - `POST /v1/chat/completions` — OpenAI-compatible (streaming SSE + non-stream)
   - `POST /v1/messages` — Anthropic-compatible (event sequence complète)
   - `POST /v1/embeddings` — OpenAI-compatible, backé par `MLXEmbedders`
   - `GET /v1/models` — catalog des modèles actuellement chargés
   - `POST /admin/load` — charger un modèle par HuggingFace id
   - `POST /admin/unload` — décharger un modèle (par id, ou tous)
   - `GET /admin/models/available` — inventaire local des modèles disponibles
   - `GET /admin/memory` — usage RAM par modèle + ceiling wired
   - `GET /admin/sessions` — sessions KV-cache actives
   - `GET /admin/activity` — activité live : phase, modèle courant, tokens, tok/s, erreur
   - `GET /.well-known/inference-engine.json` — capability contract auto-découvrable
   - `GET /health` — état + métriques + tok/s récents

3. **Un menu-bar SwiftUI** (`Telemak Menu Bar.app`) qui affiche
   l'état du daemon, la version runtime, les modèles chargés, la phase
   courante (`prefill`, `decode`, `streaming`, `idle`), les tokens générés,
   le tok/s live, et permet Start / Stop / Restart sans terminal.
   Lien direct vers le dashboard Odysseus quand un est configuré.

4. **Multi-modèle concurrent** : charger un chat-MoE 35B + un embedder
   0.6B + un small TTS simultanément, chacun servable par son endpoint
   sans déchargement automatique. La mémoire wired est réservée par
   modèle via une politique explicite (`WiredMemoryCoordinator`),
   pas par chance.

5. **KV cache cross-turn** : les sessions identifiées par
   `X-Session-Id` ou `session_id` dans le body persistent le prompt
   cache sur disque (`~/.telemak/sessions/`) avec éviction LRU et
   reportent `cached_tokens` dans la réponse `usage` block —
   compatible avec ce que Companion/Odysseus consomment.

6. **Tools + thinking** : support natif des `tools` OpenAI et du
   `enable_thinking` Qwen, avec routing correct du `reasoning_content`
   vers le bon channel selon le client.

7. **Integration cluster Odysseus** : se déclare via
   `kind=telemak` + `upstream=http://<mac>:8003` dans le dashboard
   Odysseus. Une fois enrôlé, Telemak apparaît comme un cluster
   single-node, ses modèles co-chargés sont routables via aliases
   `telemak-max64:35b`, `telemak-max64:embedder`, etc. Les capabilities
   sont découvertes live : LLM, embeddings, vision si chargé, sessions,
   tools, MTP si une paire est explicitement active.

---

## Comment ça marche — vue d'ensemble

```
┌───────────────────────────────────────────────────────────────────┐
│  Client (Companion, Claude Code, curl, SDK OpenAI/Anthropic)      │
│  POST /v1/chat/completions  OR  POST /v1/messages                 │
└─────────────────────────────┬─────────────────────────────────────┘
                              │ HTTP (localhost ou LAN)
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  Telemak binary natif (Swift, mlx-swift-lm, Hummingbird)          │
│  ─────────────────────────────────────────────────────────────    │
│  Hummingbird router + Sendable handlers                           │
│                                                                   │
│   • ModelRegistry (multi-modèle) — N modèles co-chargés           │
│   • WiredMemoryCoordinator — réservation explicite par modèle     │
│   • SessionStore — KV cache cross-turn par sessionId              │
│   • StatsTracker — tok/s récents, requests served                 │
│   • ActivityTracker — phase live, modèle courant, erreurs         │
│   • Tokenizer (swift-transformers) + chat template Jinja          │
│   • Streaming SSE chunks (delta + usage finale + [DONE])          │
│   • Anthropic event sequence (message_start → … → message_stop)   │
└──────┬──────────────────────────────────────────────────────┬─────┘
       │ direct MLX call                                      │
       │ (pas de SSH, pas de container, in-process)           │
       ▼                                                      ▼
┌───────────────────────────────────┐         ┌────────────────────────┐
│  mlx-swift-lm sur Apple Neural    │         │  Menu-bar Telemak.app  │
│  Engine + GPU via Metal           │         │  Start/Stop/Restart    │
│  - Gemma 4 / Qwen / MiniMax       │         │  Version + activité    │
│  - Embedders MLXEmbedders         │         │  Lien dashboard        │
│  - (autres modèles co-chargés)    │         └────────────────────────┘
│  - Wired memory ticket par modèle │
└───────────────────────────────────┘
              + LaunchAgent (autostart au login)
              + Daily-rotating JSON logs (~/.telemak/logs/)
              + State persistance (~/.telemak/state.json)
```

### Le cycle d'une requête `/v1/chat/completions`

1. Le client envoie `POST /v1/chat/completions` avec un `model: "<hf-id>"`.
2. Telemak vérifie que le modèle est dans `ModelRegistry`. Si non → 404.
3. Le tokenizer (swift-transformers) applique le chat template Jinja
   (système + history + user) → liste de tokens.
4. Si un `session_id` (header ou body) matche un cache disque dispo,
   le KV cache est rechargé (mlx-swift-lm `loadPromptCache`) et seul
   le dernier user message est tokenisé pour le prefill.
5. mlx-swift-lm streame les tokens : chacun est encodé en
   `ChatCompletionChunk`, envoyé en SSE.
6. À la fin, un chunk `usage` standard OpenAI est émis (depuis le
   bug-fix V1 — `prompt_tokens` / `completion_tokens` / `total_tokens` /
   `prompt_tokens_details.cached_tokens`), puis `[DONE]`.
7. Le KV cache final est sauvé sur disque (si `session_id` présent)
   et le StatsTracker enregistre la durée + tokens pour les métriques.

### Le cycle d'une requête `/v1/messages`

Même chose côté MLX, mais le wrapper Anthropic émet la séquence
d'événements complète : `message_start` → `content_block_start` →
`content_block_delta` (par token) → `content_block_stop` →
`message_delta` (avec `usage` final) → `message_stop`. Compatible
direct avec Claude Code, l'Anthropic SDK, Companion.

### Le cycle d'un enrôlement dans Odysseus

1. Sur le dashboard Odysseus, "+ Add Telemak" avec l'URL upstream.
2. Odysseus polle `GET /.well-known/inference-engine.json` →
   apprend les capabilities (modèles loaded, tools, vision, speculative
   decoding, etc.).
3. Le Telemak apparaît dans la matrice des clusters comme un node
   single-Mac. Ses modèles co-chargés sont listés.
4. Companion appelle `model: "telemak-max64:35b"` → Odysseus route
   vers `<telemak-mac>:8003` avec le model id complet via la table
   d'alias multi-modèle (`cluster_id:short_id`).
5. Le user voit dans Companion `Cached: N tok (XX%)` quand un
   prefix-cache hit a lieu côté Telemak.

---

## Pourquoi on a fait ça

### Le déclencheur

the operator utilise Odysseus distribué (3-4 Mac Ultra) pour les modèles
≥ 200 B. Mais pour les modèles courants 30-70 B — un Qwen3.6-35B-A3B,
un Gemma 4 31 B, un Qwen3-Coder-Next — un seul Mac Studio M4 Max ou
M3 Ultra suffit largement, et le surcoût d'Odysseus distribué n'a pas
de sens : pas besoin de SSH inter-nodes, pas besoin de
pipeline/tensor parallel, pas besoin du runner.py + container Docker.

Les options testées en avril 2026 :

- **Ollama** : marche mais pas multi-modèle concurrent en mémoire wired,
  pas d'API Anthropic, pas pensé MLX (utilise llama.cpp Metal backend
  plus lent que mlx-swift-lm en pratique).
- **LM Studio en mode "API server"** : OK pour OpenAI compat seulement,
  pas de KV cache typé, pas d'integration cluster.
- **mlx-vlm Python en mode serveur** : marche, mais Python à maintenir
  sur le Mac, venv qui pourrit, pas de menu-bar, pas de `.app`.
- **Inferencer.app** : excellent perf MLX mais propriétaire/fermé, focus
  single-user, pas d'API ouverte.

### La décision

Écrire **un runtime mono-Mac natif Swift** qui :

- Réutilise mlx-swift-lm pour le compute (le moteur MLX-Swift maintenu
  par Apple en open source) — pas de réimplémentation MLX.
- Ajoute par-dessus tout ce que les solutions ci-dessus n'avaient pas :
  multi-modèle concurrent, KV cache typé, API standards
  (OpenAI **et** Anthropic), tools natifs, capability contract auto-
  découvrable, integration cluster Odysseus, menu-bar discret.
- Distribué en **un seul binaire** + `.app` bundle. Pas de Python à
  installer, pas de container, pas de SSH config. `brew install` ou
  drag-drop `.app` dans `/Applications`.
- Écrit en Swift parce que (1) on évite le venv hell, (2) on a un menu-bar
  natif SwiftUI gratuit, (3) le packaging Gatekeeper-compliant existe
  déjà, (4) l'embedding ABI avec mlx-swift est plus propre qu'avec
  mlx-python.

### Le positionnement par rapport à Odysseus

Pas de concurrence — **complémentarité claire** :

| | Telemak | Odysseus |
|---|---|---|
| Cible | Mono-Mac, 30-70 B | Cluster, 200-700 B |
| Compute | In-process mlx-swift-lm | runners.py via SSH |
| Déploiement | `.app` bundle + LaunchAgent | Docker container |
| Stack | 1 binaire Swift | Python FastAPI + runners MLX-LM |
| API exposée | OpenAI + Anthropic | OpenAI + Anthropic |
| Multi-modèle | Oui (in-process) | Oui (multi-pool sur nodes disjoints) |
| Cluster providers | Devient un node `kind=telemak` | Centralise N providers |
| RAM exploitable | RAM d'un seul Mac | Somme cluster (jusqu'à 2 TB) |
| Distribué | Non | Oui (pipeline/tensor parallel) |

Un user typique installe **Odysseus sur un Mac Mini/Studio comme
orchestrateur**, puis **Telemak sur 1-N Macs Studio comme providers
locaux**, et a une stack cohérente du laptop perso au cluster.

---

## Avantages

### 1. Multi-modèle concurrent en mémoire wired

Sur les hosts de production :

| Modèle | Taille (9-bit MLX) | Wired |
|---|---|---|
| Gemma 4 26B-A4B 9-bit | ~30 GB | 64 GB node / 96 GB node |
| Qwen3-Coder-Next 9-bit | ~60-90 GB selon build | 96 GB node / 256 GB node |
| MiniMax-M2.7 8-bit | gros MoE | 512 GB node |
| Mistral Medium 3.5 8/9-bit | gros dense/MoE | 512 GB node |
| Embedder MLXEmbedders | petit modèle | co-chargeable |

`/admin/load` charge LLM ou embedder selon la config. Le ceiling wired
est configurable via `iogpu.wired_limit_mb` ; Telemak respecte la limite
et refuse poliment un load qui ferait dépasser (`400 insufficient_memory`
avec breakdown). Le multi-modèle reste supporté, mais la production
actuelle privilégie la stabilité : un gros modèle chaud par host, plus
des petits modèles quand la marge mémoire est claire.

### 2. APIs standard (OpenAI + Anthropic)

Pas de SDK propriétaire. N'importe quel client compatible marche :
- Companion (le client de référence de la stack odyssai.eu)
- Claude Code via `odyssai-launch`
- Continue.dev, Aider, Codex
- Python `openai`, `anthropic`, ou raw `httpx`
- Swift `OpenAIKit`, `swift-anthropic`

### 3. KV cache cross-turn typé

Pas un truc opaque. Les sessions sont scoped par `session_id` (header
`X-Session-Id` ou body field), persistées sur disque (`~/.telemak/
sessions/`) avec éviction LRU (`TELEMAK_MAX_SESSIONS`, défaut 32),
et le hit-rate est reporté dans le `usage` block OpenAI standard
(`prompt_tokens_details.cached_tokens`). Compatible direct avec le
StatsRow "Cached: N tok (XX%)" de Companion.

### 4. Menu-bar natif, pas d'icône Dock

`LSUIElement=true` dans l'Info.plist : Telemak vit en menu-bar
silencieusement. Affiche en temps réel :
- L'état du daemon (running / unreachable)
- La version runtime (`0.6.15`, visible dans le menu)
- Les modèles chargés
- Le wired memory used / free
- L'activité live : active requests, modèle courant, phase, tokens, tok/s,
  dernière erreur
- Boutons Start / Stop / Restart (via `launchctl bootstrap`/`bootout`)
- Lien direct vers le dashboard Odysseus

Pas d'app à minimiser, pas de fenêtre orpheline. Un click pour
l'overlay, un click pour le fermer.

### 5. LaunchAgent intégré

`eu.odyssai.telemak.plist` installé dans `~/Library/LaunchAgents/`
permet :
- Auto-start au login
- Restart automatique si le binaire crash
- Logs stdout/stderr capturés vers `~/telemak/launchd.out|err`
- Stable face aux mises à jour de macOS

`KeepAlive=true` pour le daemon ; le menu-bar est une app Aqua
séparée qui ne dépend pas du daemon (on peut Stop le daemon depuis
le menu sans tuer le menu).

### 6. Logs JSON rotatifs

Pas de logs ASCII fragiles. Chaque ligne est un JSON typé
(`{"ts","level","msg","metadata","source"}`), parsé directement par
n'importe quel tooling (jq, Loki, journalctl-style). Rotation
quotidienne (`~/.telemak/logs/telemak-YYYY-MM-DD.log`), rétention 7
jours par défaut.

### 7. Integration cluster Odysseus en 30 secondes

Telemak expose `/.well-known/inference-engine.json` que Odysseus
auto-découvre. "+ Add Telemak" dans le dashboard Odysseus → URL
upstream → 30 s plus tard les modèles co-chargés sont routables
depuis Companion via aliases `telemak-<id>:<short-model>`. La stack
Odysseus + Telemak + Companion forme un assistant IA privé complet
sur le LAN.

### 8. Capability contract observable

`GET /.well-known/inference-engine.json` retourne :

```jsonc
{
  "engine": "telemak",
  "version": "0.6.15",
  "capabilities": {
    "stream": true,
    "tools": true,
    "vision": false,
    "embeddings": true,          // true quand un embedder est loaded
    "max_context": 32768,
    "session_cache": true,
    "openai_compat": "v1",
    "anthropic_compat": "v1",
    "speculative_decoding": {
      "supported": false,
      "modes": ["mtp_adapter", "embedded_head"],
      "active_pairs": []
    }
  },
  "models": [...]
}
```

Companion et Odysseus consomment ça pour décider quoi proposer dans
le UI. Pas de feature detection hardcodée — la capability vient du
runtime, mise à jour live.

### 9. Multi-API translation héritée

Quand un Telemak est enrôlé dans Odysseus, la translation OpenAI ↔
Anthropic d'Odysseus s'applique automatiquement : un client OpenAI
peut appeler les modèles Telemak via `/v1/chat/completions` (passe
direct), un client Anthropic via `/v1/messages` (passe direct), et
les clients qui parlent l'un peuvent appeler les modèles servis par
l'autre via Odysseus.

### 10. MTP speculative decoding : suspendu

Telemak contient du code de recherche MTP : loader draft, itérateurs Qwen
3.5/3.6 et Gemma 4 assistant, acceptance sampler-correcte, endpoint
`/admin/mtp/smoke`. Mais **le chantier est suspendu depuis le 2026-05-27**.

Pourquoi :

- Gemma 4 26B-A4B atteint déjà ~72 tok/s sans MTP sur 64 GB node.
- Le port Gemma/MTP a introduit des régressions de vitesse catastrophiques
  sur certains chemins de chargement.
- Le gain attendu ne justifie pas le risque tant que Telemak doit rester
  stable pour Companion/Odysseus.

Règle actuelle : ne pas activer MTP par défaut, ne pas reprendre sans spike
dédié avec baseline externe, mêmes modèles, même host, mêmes prompts.

---

## Différences vs alternatives

### vs Ollama

| | Ollama | Telemak |
|---|---|---|
| Backend MLX natif | non (llama.cpp Metal) | oui (mlx-swift-lm) |
| API OpenAI | partielle | complète + streaming |
| API Anthropic | non | oui + event sequence complète |
| Multi-modèle concurrent | non (un seul à la fois) | oui, wired memory dédiée par modèle |
| KV cache cross-turn typé | non | oui, OpenAI usage compliant |
| Menu-bar natif | non | oui |
| `.app` bundle Gatekeeper | non | oui |
| Capability contract | non | oui (auto-discovery) |
| Speculative decoding MTP | non | suspendu |
| Tools + thinking routing | partial | natif Qwen-aware |

### vs LM Studio

| | LM Studio | Telemak |
|---|---|---|
| UI chat embarquée | oui (focus principal) | non (UI dans Companion/Odysseus) |
| Mode serveur HTTP | oui (OpenAI partial) | oui (OpenAI + Anthropic) |
| Multi-modèle concurrent | non | oui |
| Headless (sans fenêtre) | difficile | natif (menu-bar only) |
| Open source | non | oui (Apache 2.0) |
| LaunchAgent autostart | non | oui |
| Integration cluster | non | oui (kind=telemak) |

**Pour qui** : LM Studio si tu veux UN client desktop qui fait tout.
Telemak si tu veux un **runtime headless** qui sert N clients (web,
mobile, CLI agents) sans fenêtre orpheline.

### vs Inferencer.app

| | Inferencer.app | Telemak |
|---|---|---|
| Backend MLX natif | oui | oui |
| Speculative decoding MTP | oui (excellent) | code expérimental, suspendu |
| Open source | non (propriétaire payant) | oui (Apache 2.0) |
| API standard publique | non (interne) | OpenAI + Anthropic |
| Multi-client externe | non (own UI) | oui (n'importe quel client compatible) |
| Self-host + custom build | non | oui |
| Integration cluster | non | oui (Odysseus) |

Inferencer.app est techniquement excellent et a démontré la viabilité
du MTP speculative decoding sur Mac. Telemak vise le même perf
plafond avec une approche **ouverte, intégrable, scriptable**.

### vs mlx-vlm / mlx-lm Python en mode serveur

| | mlx-vlm Python serveur | Telemak |
|---|---|---|
| Langage | Python | Swift natif |
| Distribution | venv + dépendances | `.app` bundle / binaire |
| Démarrage | `python -m mlx_vlm.server` | `launchctl bootstrap` ou click `.app` |
| Menu-bar | non | oui |
| API Anthropic | non | oui |
| Multi-modèle concurrent | partial | natif |
| Capability auto-discovery | non | oui |
| Integration cluster Odysseus | difficile | natif |

mlx-vlm Python reste **la reference** d'inférence MLX côté
Python — Telemak ne réécrit pas MLX, il s'appuie sur mlx-swift-lm
(la même équipe Apple) et ajoute le couche service propre par-dessus.

### vs Odysseus (le grand frère)

| | Odysseus | Telemak |
|---|---|---|
| Topologie | Cluster (1-N Mac via SSH) | Mono-Mac (in-process) |
| Compute layer | Python `runner.py` spawné via SSH | mlx-swift-lm in-process |
| Distribution | Container Docker | `.app` bundle |
| Modèle cible | 200-700 B distribués | 30-70 B mono-Mac |
| RAM max | ~2 TB (cluster) | RAM d'un seul Mac (96-192 GB) |
| Multi-pool nodes disjoints | oui (Argo) | non (single node) |
| Cloud providers passthrough | oui (templates intégrés) | non (delégué à Odysseus si enrôlé) |
| Translation OpenAI↔Anthropic | oui | implicite (les deux APIs natives) |
| Cible utilisateur | sysadmin avec ≥ 2 Mac | dev avec 1 Mac |
| Complexité install | Docker + SSH + topology config | drag-drop `.app` |

C'est **complémentaire**, pas concurrent. Un Telemak qui sert
Qwen3.6-35B en local peut s'enrôler comme provider d'un Odysseus
qui orchestre par-dessus Qwen3.5-397B sur un cluster Apple — les
deux modèles coexistent dans le même catalog `/v1/models` côté
Companion.

---

## Ce que ce n'est PAS

- **Pas un client de chat**. Telemak n'a pas d'UI conversationnelle.
  Pour ça, il y a Companion ou n'importe quel client OpenAI/Anthropic-
  compatible.
- **Pas un cluster distribué**. Un seul Mac. Pour distribué, il y a
  Odysseus.
- **Pas une box magique pour n'importe quel modèle**. Limite : la
  RAM physique du Mac où il tourne. Qwen3.5-397B sur un MacBook
  Pro 36 GB n'est pas possible — c'est pour ça qu'Odysseus existe.
- **Pas un produit pour grand public**. Cible 2026 : power user
  technique qui a déjà mlx-swift-lm installé ou qui sait ce qu'est
  un LaunchAgent. Brew formula + `.app` signing viendront quand la
  V2 sera stabilisée.
- **Pas une dépendance optionnelle d'Odysseus**. Odysseus existait
  avant et continue de fonctionner sans Telemak (avec ses runners
  Python). Telemak est une **alternative locale** quand le compute
  tient dans un seul Mac.

---

## Architecture résumée

```
                ┌──────────────────────────────────────────────┐
                │  Menu-bar Telemak (SwiftUI)                  │
                │  - HealthPoller (live status + tok/s)        │
                │  - Modèles chargés + RAM wired               │
                │  - Start / Stop / Restart                    │
                │  - Lien dashboard Odysseus                   │
                │  - Settings (endpoint, dashboard URL)        │
                └──────────────────────┬───────────────────────┘
                                       │ HTTP localhost
                                       │
                ┌──────────────────────▼───────────────────────┐
                │  Telemak daemon binary (Swift)               │
                │  Hummingbird + mlx-swift-lm                  │
                │                                              │
                │  • Router (Sendable handlers)                │
                │  • OpenAI /v1/chat/completions               │
                │  • Anthropic /v1/messages                    │
                │  • /v1/models, /admin/*, /health             │
                │  • /.well-known/inference-engine.json        │
                │  • CORS + Bearer auth middleware             │
                │                                              │
                │  • ModelRegistry (actor, multi-modèle)       │
                │  • WiredMemoryCoordinator (actor)            │
                │  • SessionStore (actor, KV cache disque)     │
                │  • StatsTracker (actor)                      │
                │  • ModelLoader (HF / local resolver)         │
                │  • Tokenizer (swift-transformers)            │
                │  • StopChecker (streaming stop-sequences)    │
                │                                              │
                │  Persist : ~/.telemak/{state,sessions,logs}/ │
                └──────────────────────┬───────────────────────┘
                                       │ in-process MLX calls
                                       │
                ┌──────────────────────▼───────────────────────┐
                │  mlx-swift-lm                                │
                │  (Apple open source, MLX-Swift bindings)     │
                │  Apple Neural Engine + GPU via Metal         │
                │  Unified memory wired par ticket             │
                └──────────────────────────────────────────────┘

  + LaunchAgent eu.odyssai.telemak.plist (autostart, restart, logs)
  + ~/.telemak/state.json (modèles persistés au restart)
  + JSON logs daily-rotated (~/.telemak/logs/telemak-YYYY-MM-DD.log)
```

### Composants

| Composant | Rôle | Tech |
|---|---|---|
| **Daemon** | Serveur HTTP + inférence in-process | Swift 6, Hummingbird 2.24, mlx-swift-lm 3.31 |
| **Menu-bar** | UI de contrôle native macOS | SwiftUI, `MenuBarExtra`, `LSUIElement` |
| **ModelRegistry** | Multi-modèle actor + accounting | Swift actor |
| **WiredMemoryCoordinator** | Réservation wired par modèle | Swift actor + sysctl `iogpu.wired_limit_mb` |
| **SessionStore** | KV cache disque + LRU eviction | Swift actor + mlx-swift-lm `loadPromptCache` |
| **ActivityTracker** | État live : phase, modèle, tokens, tok/s, last_error | Swift actor + `/admin/activity` |
| **ModelLoader** | HF id → local dir + config staging | Swift + JSONSerialization |
| **JSON logger** | Daily-rotated structured logs | `swift-log` custom handler |
| **LaunchAgent** | Autostart + restart + log capture | macOS launchd plist |
| **Anthropic adapter** | OpenAI flow → Anthropic event sequence | Swift + Hummingbird SSE |

### Endpoints exposés

| Method | Path | Rôle |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI streaming + non-stream |
| `POST` | `/v1/messages` | Anthropic event sequence |
| `POST` | `/v1/embeddings` | OpenAI embeddings via MLXEmbedders |
| `GET` | `/v1/models` | Modèles loaded |
| `POST` | `/admin/load` | Charger un modèle (main + draft optionnel) |
| `POST` | `/admin/unload` | Décharger un modèle (par id ou tous) |
| `GET` | `/admin/memory` | RAM par modèle + ceiling |
| `GET` | `/admin/activity` | Activité runtime live |
| `GET` | `/admin/sessions` | Sessions KV cache actives |
| `POST` | `/admin/sessions/clear` | Vider les sessions |
| `GET` | `/admin/models/available` | Inventaire local (non-loaded) |
| `POST` | `/admin/models/split-mtp` | Extraire un draft MTP (subprocess vers mlx-vlm) |
| `POST` | `/admin/mtp/smoke` | Smoke MTP expérimental, non-prod |
| `GET` | `/health` | Liveness + métriques live |
| `GET` | `/.well-known/inference-engine.json` | Capability contract |

---

## État actuel (2026-05-29)

- **Version stable** : `0.6.15`
- **Tag rollback** : `v0.6.15-stable`
- **Bronze rollback** : `~/telemak/Release.bronze-0.6.15` sur les hosts
- **Déploiements actifs** : inventaire local opérateur dans
  `scripts/telemak-hosts.local.sh` (gitignored). Le dépôt public ne shippe
  que `scripts/telemak-hosts.example.sh`.

### Modèles testés récemment

- **Gemma 4 26B-A4B 9-bit** : ~72 tok/s sans MTP sur 64 GB node.
- **Qwen3-Coder-Next 9-bit** : ~41 tok/s sur 512 GB node, ~52 tok/s sur 96 GB node.
- **MiniMax-M2.7 8-bit** : ~33 tok/s end-to-end, decode ~43 tok/s sur 512 GB node après chunk coalescing.
- **Mistral Medium 3.5 8/9-bit** : charge, mais ~4 tok/s ; pas un bon fit Telemak aujourd'hui.
- **DeepSeek V4 Flash/Pro** : non supporté côté Telemak Swift (`deepseek_v4` absent de `mlx-swift-lm`) ; réservé à un spike dédié.

### Améliorations récentes

- ✅ **Version visible** dans le menu-bar pour suivre exactement le build déployé.
- ✅ **`/admin/activity`** : `active_requests`, `current_model`,
  `current_request_started_at`, `current_generated_tokens`,
  `current_tok_s`, `current_phase`, `last_error`.
- ✅ **Menu-bar activity** : affichage direct de l'activité runtime.
- ✅ **Chunk coalescing** : streaming groupé pour éviter le token-par-token
  coûteux dans Companion/Odysseus/browser.
- ✅ **Embeddings** : `/v1/embeddings` via `MLXEmbedders`.
- ✅ **Loader fixes** : paths canonicalisés, Mistral Inferencer config normalisée,
  alias `minimax_m2`, support text-only load.
- ✅ **Clean release builds** : Release sans runtime coverage/profiling.
- ✅ **Codesigning stable** : identité `Telemak Developer (Odyssai-eu)`.
- ✅ **Déploiement opérateur** : `scripts/deploy-all.sh` canary-first.
- ✅ **Rollback opérateur** : `scripts/rollback-host.sh` bronze/prevN-aware.
- ✅ **Installer base** : `scripts/package-dmg.sh` produit un DMG avec
  `Telemak.app` + alias `/Applications`.

### Politique actuelle

- **Stabilisation > features**. Telemak 0.6.15 est la baseline stable.
- **MTP suspendu** officiellement. Code conservé, non prioritaire.
- **Déploiement canary obligatoire** : un host cobaye d'abord, rollout ensuite.
- **Rollback toujours disponible** : bronze + `Release.prevN` sur les hosts.

### Prochaines actions

- 📋 Finaliser l'installeur user-friendly : drag-and-drop vers `/Applications`,
  first-run install/update, LaunchAgents serveur + menu-bar.
- 📋 Normaliser le menubar de `64 GB node` qui tourne encore via
  `/Applications/Telemak.app` hors LaunchAgent standard.
- 📋 Durcir les health/activity smokes dans Odysseus et les scripts opérateur.
- 📋 Garder un registre de compatibilité modèles : bons fits, lents, non supportés.

---

## Positionnement résumé

> **Pour les power users Apple Silicon** qui veulent un runtime
> d'inférence **headless natif** sur un seul Mac, sans Docker, sans
> Python, sans UI orpheline :
>
> - Sert localement les APIs OpenAI **et** Anthropic
> - Garde N modèles co-chargés en mémoire wired
> - Vit en menu-bar discret avec autostart au login
> - S'enrôle dans un cluster Odysseus en 30 secondes
> - MTP présent en code expérimental, suspendu tant que la perf de base suffit
> - Open source (Apache 2.0), un seul binaire, `.app` bundle
>
> Telemak est le runtime. Odysseus est l'orchestrateur quand on
> distribue. Companion est le client par-dessus.

---

## Le triplet Odysseus + Telemak + Companion

Les trois projets sont distincts mais conçus pour s'emboîter :

| | Odysseus | Telemak | Companion |
|---|---|---|---|
| Rôle | Orchestrateur cluster + cloud passthrough | Runtime mono-Mac | Client UX |
| Compute | Non (pilote les runners Python) | Oui (in-process mlx-swift-lm) | Non |
| Distribution | Container Docker | `.app` bundle Swift | Container Docker |
| API exposée | OpenAI + Anthropic | OpenAI + Anthropic | REST + SSE métier |
| Multi-Mac | Oui (1-N nodes) | Non (1 Mac) | N/A |
| Cible hardware | Mac Mini/Studio (host orchestrateur) | Mac Studio / MacBook Pro | n'importe quel host Docker |
| Mémoire compilée user | Non | Non | Oui (Karpathy wiki) |
| Cloud passthrough | Oui (templates) | Non (delégué à Odysseus) | Non (delégué) |

### Combinaisons valides

- **Telemak seul** : un Mac, un binaire, deux APIs exposées. Companion
  ou Claude Code branchés directement. Mode "indie dev".
- **Odysseus + Telemak** : Odysseus orchestre, Telemak fournit le compute
  local sur un node. Avantage : intégration cluster sans Docker sur le
  Mac d'inférence.
- **Odysseus + Telemak + Companion** : la stack complète odyssai.eu.
  Odysseus centralise les modèles locaux (Telemak) + les clouds
  (Anthropic / OpenAI / OpenRouter), Companion ajoute la UX + mémoire +
  projets + partage. Aucun autre service requis.

---

## Référence rapide

- **Repo** : `Odyssai-eu/telemak.git` (`~/Claude/code/telemak/`)
- **Fork mlx-swift-lm** : `Odyssai-eu/mlx-swift-lm` (fork runtime ; MTP branch conservée mais chantier suspendu)
- **Binary** : `Telemak.app` (LaunchAgent `eu.odyssai.telemak.plist`)
- **Version actuelle** : 0.6.15 (2026-05-29)
- **Baseline stable** : tag `v0.6.15-stable`, bronze `Release.bronze-0.6.15`
- **Port par défaut** : `8003`
- **Modèles** : `/Volumes/models/odysseus/` (layout HF `<org>/<name>/snapshots/<hash>/`)
- **Sessions** : `~/.telemak/sessions/`
- **Logs** : `~/.telemak/logs/telemak-YYYY-MM-DD.log` (JSON, rotation 7 jours)
- **State** : `~/.telemak/state.json` (modèles persistés)
- **Endpoints exposés** : `/v1/chat/completions`, `/v1/messages`, `/v1/embeddings`, `/v1/models`, `/admin/*`, `/health`, `/.well-known/inference-engine.json`
- **Build** : `./scripts/build.sh [Debug|Release]` (xcodebuild — Metal Toolchain requis)
- **Deploy parc** : `scripts/deploy-all.sh --canary node-a`
- **Rollback host** : `scripts/rollback-host.sh <host> [bronze-0.6.15|prevN]`
- **Container parent** : aucun — c'est ça l'intérêt
- **Issues queue** : `gh issue list --repo Odyssai-eu/telemak --label ready`
- **Site web** : https://telemak.odyssai.eu (en cours)
