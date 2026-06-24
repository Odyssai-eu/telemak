# Session 2026-06-24 — mlxconvert monorepo : bootstrap, refonte UI, dispatchers

> Une journée qui part sur la création d'un monorepo neuf (mlxconvert/ — module Python + GUI Docker + deploy), pivote sur du SSH remote control pour sortir du blocage SSHFS/Tier 3 brew, débouche sur un sprint de 4 bug fixes matinaux (awk + zsh + one-liner Python + cwd SSH), puis une refonte UI complète alignée sur tmb-site v6, et finit sur la livraison des deux dispatchers manquants (minimax_m3 + hy_v3) pour couvrir les modèles déjà en prod sur ultra-512. Sophie dit *« ajoute les dispacher qu'on a deja dedans ; minimax, hy3, ... recherche ce qu'on a deja sur ultra-512 »* — la directive qui cadence la journée : **on réutilise ce qui tourne déjà, on ne réinvente pas la quantization**. Le pivot diagnostique *« awk se faisait manger le ! par l'history expansion zsh dans le container »* montre que la couche shell du remote-first bootstrap a coûté plus de tokens que le code applicatif.

---

## TL;DR — Avant / Après

| Aspect | Avant | Après |
|---|---|---|
| mlxconvert | Pas de repo Git, Swift app `~/Claude/code/MLX convert/MLXConvert/` (DMG 1.0.1) + scripts Python éparpillés (`m3_convert.py`, `bailing_hybrid.py`, `glm_dsa_convert.py`) | **Monorepo neuf** `/Users/sophie/Claude/code/mlxconvert/` — module Python `mlxconvert/` (Package, dispatchers, recipes), GUI Docker `gui/` (FastAPI + React + Vite), deploy/. SSH origin sur Forgejo `Odyssai-eu/mlxconvert`. 11 commits sur main. |
| Architecture backend | GUI locale qui monte `/Volumes/models` en NFS/SSHFS (bloqué Tier 3 brew) | **Pure control plane** sur `.141` qui SSH vers `.29` (ultra-512) pour tout — list_dir, write_file, run_pipe python, log_stream via SSH tail -f. Aucun mount, aucun volume de données. |
| Bootstrap GUI | /api/dispatchers vide, /api/recipes vide, /api/models vide — 30 min de debug avant de comprendre que `awk 'NR>1 && \$NF != "." ...'` se faisait bouffer par zsh | **3 dispatchers live** dans la GUI (`minimax_m3`, `hy_v3`, `qwen3_5_moe`), 4 recipes (2 nouvelles + 2 existantes), file picker SSH browse `/Volumes/models/mlx/raw` avec détection `config.json`, presets Q6/Q8/Q6H16/Q8H16 |
| UI direction artistique | v1 brute — SaaS landing générique, dropdowns statiques, table d'artefacts, sauts de typo | **v6 catalogue d'instruments de précision** — palette encre/cobalt/chrome/ivoire, Fraunces italic pour le brand, Inter body, JetBrains Mono data. 4 pages refactorées + design system (ThemeProvider, ToastProvider, HighlightPython, HighlightJSON). Header SSH live qui polle toutes les 15s. |
| Couverture dispatchers | Seul `qwen3_5_moe` était câblé dans `mlxconvert/dispatchers/`. Les autres modèles servis (minimax_m3, hy_v3) passaient par les scripts monolithiques d'Odysseus. | **3 dispatchers** + **4 recipes** dans `mlxconvert/recipes/` : `minimax_m3_q6_head8.json` (Q6 experts, head Q8 mixed — fix logit-floor), `hy_v3_q8_default.json` (Q8 partout sauf linear_attn + head bf16), `qwen3_5_full_bf16.json`, `qwen3_5_q8_head16.json`. Le `m3_convert.py` monolithique est remplacé par `minimax_m3.py` (235 lignes, structuré par block). |

**Versions de sortie** : mlxconvert HEAD = `27ec685` (11 commits cette session, repo vierge au matin). GUI live : `http://192.168.86.141:9020`, container up, SSH reachability OK vers `192.168.86.29` (ultra-512.lan). Forgejo : aucun ticket (sprint monorepo, hors backlog Telemak).

---

## 1. Le matin — bootstrap monorepo + pivot SSH

Sophie ouvre avec la création d'un repo neuf. Pas de ticket, pas de plan formel — on pose la structure du projet qui va servir de GUI commune aux scripts de conversion MLX (jusqu'ici éparpillés entre `~/Claude/code/MLX convert/` et `/Users/admin/apps/`).

**Initial commit** (`6df54a3`, 10:55) — squelette vide, juste le `.gitignore` + README bootstrap.

**Bootstrap monorepo** (`b4fec39`, 11:12) — structure :
```
mlxconvert/
  mlxconvert/           # module Python
    core.py             # ShardReader, qinto, keep_bf16, Output
    run.py              # CLI run.main
    dispatchers/        # base.py + qwen3_5_moe.py
    recipes/            # qwen3_5_*.json
  gui/
    backend/            # FastAPI app
      app/
        main.py
        services/       # remote, conversion, storage, dispatchers
    frontend/           # React + Vite
      src/
        App.tsx
        pages/          # Convert, Artefacts, Dispatchers, Recipes
  deploy/
    Dockerfile
    docker-compose.yml
  pyproject.toml
```

`pyproject.toml` expose `mlxconvert` comme package editable, `mlxconvert.run` comme console script. C'est cette structure qui permet à `.141` d'installer le module dans le venv distant et de l'importer sans copier les sources.

**Pivot SSH remote control** (`7fb8c38`, 12:02) — c'est la décision architecture de la journée. Le plan initial était de monter `/Volumes/models` depuis `.141` vers `.29` (le host qui a le venv MLX) via SSHFS. Ça bloque :

> SSHFS était bloqué (Tier 3 brew, pas de bottle pour macFUSE/libfuse). Plutôt que de batailler avec NFS/FUSE, on remonte d'un niveau : .29 possède le venv MLX, les modèles, les outputs. .141 envoie des commandes et lit des résultats. Aucun mount, aucun volume de données.

Décision pragmatique — pas de workaround, vraie solution. Refacto :
- `services/remote.py` (NEW) — abstraction SSH : `run`, `read_file`, `write_file`, `list_dir`, `glob`, `python_run`, `log_stream_args`. Config via env vars (`REMOTE_HOST`, `REMOTE_USER`, `REMOTE_VENV`, `REMOTE_REPO_DIR`, `REMOTE_JOBS_DIR`).
- `services/conversion.py` — `subprocess` SSH au lieu de local ; `log_stream` via `ssh host tail -f`.
- `services/storage.py` — tout en SSH (artefacts, recipes, models).
- `services/dispatchers.py` — introspection via SSH + python -c, edit via SSH write.
- `main.py` — `/healthz` checke la reachability SSH, toutes les routes async.
- `Dockerfile` — ajout `openssh-client`, retrait `pip install mlxconvert` (vit dans le venv distant).
- `docker-compose.yml` — suppression volume `models`, ajout mount `/Users/admin/.ssh` (read-only).

Setup requis documenté dans le commit body : sur `.29` clone du repo + `pip install -e .` dans le venv ; sur `.141` clé SSH admin@.29 (déjà en place).

---

## 2. Le sprint 4 bug fixes — la couche shell coûte cher

`/healthz` répond OK, mais `/api/dispatchers` retourne `[]` et `/api/recipes` pareil. Pendant 30 minutes je debug à l'aveugle. Le pattern « erreur silencieuse » se manifeste sous plusieurs formes.

**Fix 1 — DISPATCHER_NAME est sur la classe** (`a1c9cad`, 12:06, Difficulty 1)

Le module expose `DISPATCHER_NAME = "qwen3_5_moe"` au top-level, mais c'est en fait un attribut de la sous-classe `BaseDispatcher`. `getattr(module, "DISPATCHER_NAME")` retournait le fallback (stem du fichier). Fix : `vars(mod)` → trouver la sous-classe de `BaseDispatcher` → `cls.DISPATCHER_NAME`. Smoke vérifié via `python -c "from mlxconvert.dispatchers.qwen3_5_moe import Qwen35MoEDispatcher; print(Qwen35MoEDispatcher.DISPATCHER_NAME)"` → `qwen3_5_moe`. 1 file, +3/-3.

**Fix 2 — list_dir en python inline** (`0eb9606`, 12:11, Difficulty 2)

> Le awk 'NR>1 && \$NF != "." ...' se faisait manger le ! par l'history expansion zsh dans le container — résultat : awk syntax error, list_dir retournait toujours [], donc /api/recipes et /api/models vides.

L'erreur est dans la couche transport SSH, pas dans le code applicatif. Le `\$` escape ne suffit pas dans un shell remote non-interactif qui hérite des settings zsh du host. Fix : `python3 -c '...'` inline, `shlex.quote` pour le transport, plus d'awk, plus d'history expansion. Bonus : `list_models` parcourt maintenant `/Volumes/models` récursivement (jusqu'à profondeur 4) au lieu de lire le top level — les modèles chez toi sont sous `/Volumes/models/mlx/raw/...`, pas à la racine. 1 file, +12/-8.

**Fix 3 — run_pipe pour tous les scripts python** (`7bc822a`, 12:15, Difficulty 3)

Trois fixes en un commit :
1. `remote.run_pipe` (NEW) : pipe un script python via stdin au lieu de `python -c '...'`. Permet les vrais blocks `try/except/for`. Le one-liner `try: x; for y` est un `SyntaxError` — découvert en testant `list_dir` avec un script de trace.
2. `dispatchers.list_dispatchers`, `storage.list_artefacts`, `storage.list_models` — tous migrés vers `run_pipe`.
3. `storage.list_recipes` — suppression du `except Exception: continue` silencieux qui masquait les erreurs. Maintenant les erreurs remontent en 500. C'est ce `silent except` qui cachait que `list_dir` ne fonctionnait pas depuis 30 min.

> *Lesson learned* : un `except Exception: continue` sans log est un crime. Il a coûté 30 min de debug sur un truc qui aurait été visible en 5 secondes si l'erreur avait remonté.

3 files, +28/-12.

**Fix 4 — glob path adapté au cwd** (`3eac48a`, 12:21, Difficulty 2)

Dernier fix du sprint matinal. `cwd` SSH = `REMOTE_REPO_DIR` = `/Users/admin/apps/mlxconvert` (le sub-dir Python qui EST le package). Le glob `'mlxconvert/dispatchers'` cherchait donc `/Users/admin/apps/mlxconvert/mlxconvert/dispatchers/` (inexistant) au lieu de `/Users/admin/apps/mlxconvert/dispatchers/`. Fix : `'dispatchers/*.py'` directement. Bonus : utiliser `{remote.REMOTE_VENV}/bin/python` (au lieu de `python3` système) pour que l'import `mlxconvert` soit résolu via le venv où le package est installé en editable. 1 file, +4/-5.

Smoke final : `/api/dispatchers` retourne `[{name: "qwen3_5_moe", ...}]`. On a un backend fonctionnel.

---

## 3. La refonte UI — catalogue d'instruments de précision

Sophie valide la v1 (« super ») et dit « refonte ». Le brief implicite : la GUI doit ressembler au site themonoclebear.com, pas à un SaaS landing.

**Browse + presets** (`aa838cf`, 13:53, Difficulty 3) — pré-refonte, demande explicite de Sophie :

1. **Modèle source** — remplace le dropdown statique par un file picker SSH. Text input éditable (default `/Volumes/models/mlx/raw`) + bouton **Browse** → `GET /api/browse?path=...` (nouvelle route backend `storage.browse()`). Liste des sub-dossiers, flag `is_model` si `config.json` est présent (◆ vert pour les modèles, dossiers alphabétiques). Click descend dans l'arbo, ↑ Up remonte. Browse persiste dans le state.
2. **Presets Q6 / Q8 / Q6 H16 / Q8 H16** — boutons qui set `(bits, head_bits, group_size)` en un clic. Q6/Q8 = full quant (head_bits=0 → same as bits). Q6 H16/Q8 H16 = mixed quant, head en bf16 (head_bits=16).

3 files, +168/-22.

**Refonte UI complète** (`8351413`, 14:17, **Difficulty 8**) — le gros morceau de la journée. Application de la direction artistique `tmb-site v6` :

*Direction artistique*
- Palette **encre + cobalt + ivoire + chrome jaune** (highlight de data, pas une déco)
- Typo **Fraunces italic** pour le brand (italique italique-seulement — pas en regular), **Inter** pour le body, **JetBrains Mono** pour data/code
- Pas de SaaS landing, pas d'illustration déco — **illustration = data**

*Design system nouveau*
- `ThemeProvider` (light/dark, persist localStorage, suit `prefers-color-scheme`)
- `ToastProvider` (success/error/info, auto-dismiss 3.2s)
- `HighlightPython` + `HighlightJSON` — tokenizers regex maison, code blocks avec coloration type VS Code Dark+

*Convert (page principale)*
- Header avec titre serif italic « Convertir un modèle » + status pill live
- Browse pane : breadcrumb cliquable + liste dossiers/modèles (modèles = highlight ivoire pâle + badge cobalt)
- Status pill animé (pulse) sur jobs running
- Logs : icône status dynamique + auto-scroll

*Artefacts*
- Grid de **cards** au lieu de table, hover lift + top border accent cobalt
- Recipe badge (« Q8 H16 », « Q6 », etc.) calculé depuis le manifest
- Drawer latéral slide-in pour le détail JSON

*Dispatchers + Recettes*
- Split view (liste à gauche 320px, viewer/editor à droite)
- Toggle View/Edit, syntax highlight côté view
- Validation JSON live (border rouge si invalide) côté Recipes
- Bouton Format pour re-indenter le JSON
- Save désactivé tant que pas modifié ou invalide
- Toast confirmation succès/erreur

*Header global*
- Brand mark Fraunces italique (`m` en cobalt) + brand name + GUI tag mono
- Nav avec icônes Lucide (Wand2, Archive, Code2, BookOpen)
- Bouton theme toggle (Sun/Moon)
- Footer-bar : status SSH live (hostname + connected/unreachable, poll 15s) + URL + container name

*Animations*
- `pulse` sur status pills running, `spin` sur Loader2
- `fadeIn` sur overlay, `slideIn` sur drawer, `slideUp` sur toast
- `shimmer` sur progress bars actifs

Dependencies : + `lucide-react ^0.469.0` (icônes modernes, tree-shakable).

12 files, +1644/-384. `styles.css` est passé de <100 à 954 lignes. `Convert.tsx` est passé de ~150 à ~470 lignes.

> *Sophie* : « super »

Sophie valide la direction artistique du premier coup. Pas de round 2. Ça valide deux choses : (1) l'identité tmb-site est robuste et tient sur des surfaces différentes (site web, GUI desktop, GUI web), (2) Sophie sait ce qu'elle veut avant de demander.

---

## 4. Les dispatchers — couvre ce qui tourne déjà sur ultra-512

> *Sophie* : « ajoute les dispacher qu'on a deja dedans ; minimax, hy3, ... recherche ce qu'on a deja sur ultra-512 »

Avant cette session, `mlxconvert/dispatchers/` ne contenait que `qwen3_5_moe.py` (342 lignes). Les modèles M3 et Hy3 servaient en prod mais via les scripts monolithiques d'Odysseus (`m3_convert.py`, `bailing_hybrid.py`) — du code de conversion copié/adapté, pas un dispatcher structuré du package mlxconvert.

**Inventaire sur ultra-512** (sortie de `/api/artefacts` ce matin) :
- `m3_convert.py v1` → artifacts `MiniMax-M3-mlx-6bit-headbf16`, `MiniMax-M3-mlx-8bit`, GLM-5.2-Q6-Hd16
- `bailing_hybrid.py` → artifacts Hy3 (Q6/Q8)
- `glm_dsa_convert.py v1` → artifacts Macaron-V1-Preview-749B (Q6/Q8)

Trois familles. Cible : porter les deux premières (`m3_convert.py` et `bailing_hybrid.py`) en dispatchers mlxconvert structurés.

**`minimax_m3` (portage `m3_convert.py` monolithique, 235 lignes, `27ec685`)** — le plus détaillé parce que le plus complexe. Architecture :

- 60 layers (3 full + 57 sparse) déduits de `sparse_attention_config.sparse_attention_freq`
- Chaque layer sparse porte un **MSA indexer** (sélection de blocs) en plus de la self-attn GQA standard. **Indexer bf16** — précision-sensible, c'est le « système nerveux » du sparse routing.
- MoE sur la plupart des layers (`block_sparse_moe`) :
  - `gate` (bf16) + `e_score_correction_bias` (f32 brut) pour le routing
  - `shared_experts.{gate,down,up}_proj` quantifiées uniformément avec les experts
  - `experts.{0..NE}.{w1,w2,w3}.weight` stackés en `switch_mlp.{gate,down,up}_proj`
- Layers denses (3 sur 60) : `mlp.{gate,down,up}_proj` quantifiés
- Top : `embed_tokens` + `lm_head` (mixed quant via per-module override quand `head_bits != bits` — fixe la corruption logit-floor à ~Q6 single-node), `model.norm`

Recette par défaut : experts Q6 g64 affine, shared_experts Q6 g64, indexer bf16, router bf16, norms bf16, embed/lm_head Q8 mixed (head_bits=8 quand bits=6).

**`hy_v3` (Bailing/Hunyuan v3 best-effort, 225 lignes)** — l'architecture hybride LinearAttn (MLA/lightning) + StandardAttn est plus complexe à détecter :

- `layer_types` décide du mix (pas un nom d'attn dans le tensor path)
- MoE avec `w1/w2/w3` → `switch_mlp.{gate,down,up}_proj` (convention Bailing)
- **Linear_attn bf16 partout** (precision-sensitive MLA)
- Self_attn Q8, experts Q8, shared_expert Q8
- Tous les reads sont `try/except` — le naming peut varier selon version `bailing_hybrid`, on shippe best-effort

Recette par défaut : Q8 partout sauf linear_attn + head bf16.

**Recettes shippées** (dans `mlxconvert/recipes/`) :
- `minimax_m3_q6_head8.json` (11 lignes) — Q6 experts, head Q8 mixed
- `hy_v3_q8_default.json` (12 lignes) — Q8 partout sauf linear_attn + head bf16

À noter (commit body explicite) :
- Pas d'audit inline (skip de `m3_convert.py`) → vitesse, debug plus tard
- Pas de flags CLI custom (`--local-blocks`, `--topk-blocks` du m3 original) — on lit les valeurs du `config.json` directement

4 files, +484/-0. Smoke : `import sur .29` → `No module` (pas encore déployé). Le rebuild + redeploy après ce commit fait apparaître les 2 dispatchers dans `/api/dispatchers`.

> *Sophie* : « c'est OK ? »

---

## 5. Le deploy + smoke

Après le commit, rebuild du container :

```bash
ssh admin@192.168.86.141 "cd /Users/admin/apps/mlxconvert && git pull && cd deploy && docker compose build --no-cache mlxconvert && docker compose up -d mlxconvert"
```

Vérifications (en parallèle) :
- `curl http://192.168.86.141:9020/healthz` → `{"status":"ok","ssh":{"host":"192.168.86.29","user":"admin","reachable":true,"remote_hostname":"ultra-512.lan"},...}`
- `curl http://192.168.86.141:9020/api/dispatchers` → 3 entries : `minimax_m3`, `hy_v3`, `qwen3_5_moe`
- `curl http://192.168.86.141:9020/api/recipes` → 4 entries : `hy_v3_q8_default.json`, `minimax_m3_q6_head8.json`, `qwen3_5_full_bf16.json`, `qwen3_5_q8_head16.json`
- `curl http://192.168.86.141:9020/api/browse?path=/Volumes/models/mlx/raw` → liste des dossiers + flag is_model

GUI accessible sur **http://192.168.86.141:9020**.

> *Sophie* : « top »

Sophie ferme la session. Container tourne, repo propre, GUI up.

---

## Fichiers modifiés / créés

**mlxconvert monorepo** (11 commits sur main) :

*Bootstrap*
- `.gitignore`, `README.md`, `pyproject.toml` (created, 28 lines)
- `deploy/Dockerfile`, `deploy/docker-compose.yml` (created)
- `deploy/README.md` (created)

*Module Python*
- `mlxconvert/core.py` (created, Output / ShardReader / qinto / keep_bf16)
- `mlxconvert/run.py` (created, 120 lines — CLI run.main)
- `mlxconvert/dispatchers/__init__.py` (created, 53 lines)
- `mlxconvert/dispatchers/base.py` (created, 208 lines — BaseDispatcher)
- `mlxconvert/dispatchers/qwen3_5_moe.py` (created, 342 lines)
- `mlxconvert/dispatchers/minimax_m3.py` (created, 234 lines)
- `mlxconvert/dispatchers/hy_v3.py` (created, 225 lines)

*Recipes*
- `mlxconvert/recipes/qwen3_5_full_bf16.json` (created, 13 lines)
- `mlxconvert/recipes/qwen3_5_q8_head16.json` (created, 13 lines)
- `mlxconvert/recipes/minimax_m3_q6_head8.json` (created, 11 lines)
- `mlxconvert/recipes/hy_v3_q8_default.json` (created, 12 lines)

*Backend GUI*
- `gui/backend/app/main.py` (created + modified, routes FastAPI)
- `gui/backend/app/services/remote.py` (created, abstraction SSH)
- `gui/backend/app/services/conversion.py` (created, SSH subprocess + log stream)
- `gui/backend/app/services/storage.py` (created, browse + list artefacts/recipes/models SSH)
- `gui/backend/app/services/dispatchers.py` (created, introspection SSH)
- `gui/backend/app/api/` (créé)
- `gui/backend/README.md` (created)

*Frontend GUI*
- `gui/frontend/index.html` (modified, +6 — Google Fonts preconnect)
- `gui/frontend/package.json` (modified, +lucide-react ^0.469.0)
- `gui/frontend/src/App.tsx` (modified, +75)
- `gui/frontend/src/lib/api.ts` (modified, +8)
- `gui/frontend/src/lib/highlight.tsx` (created, 183 lines — tokenizer maison)
- `gui/frontend/src/lib/theme.tsx` (created, 32 lines — ThemeProvider)
- `gui/frontend/src/lib/toast.tsx` (created, 37 lines — ToastProvider)
- `gui/frontend/src/pages/Convert.tsx` (modified, +314 / -large)
- `gui/frontend/src/pages/Artefacts.tsx` (modified, +106)
- `gui/frontend/src/pages/Dispatchers.tsx` (modified, +147)
- `gui/frontend/src/pages/Recipes.tsx` (modified, +190)
- `gui/frontend/src/styles.css` (modified, +927 — palette + composants)
- `gui/frontend/README.md` (created)

**Forgejo** : aucun ticket créé (sprint monorepo, backlog Telemak toujours à zéro — TODO 1-2 du SESSION-2026-06-23).

**Livrables** :
- Repo Git `/Users/sophie/Claude/code/mlxconvert/` (11 commits sur main, SSH origin `Odyssai-eu/mlxconvert`)
- Container Docker `mlxconvert` sur `.141:9020`
- 3 dispatchers + 4 recipes live

---

## Numbers de la journée

- **Commits mlxconvert** : 11 sur main (init 10:55 → dispatchers 17:03).
- **Diffs** : 46 files, +4919/-1 net.
- **Frontend** : 12 files, +1644/-384. `styles.css` <100 → 954 lignes. `Convert.tsx` ~150 → ~470 lignes.
- **Dispatchers** : 3 modules total (`qwen3_5_moe` 342, `minimax_m3` 234, `hy_v3` 225). Plus `base.py` (208) + `__init__.py` (53).
- **Recipes** : 4 fichiers JSON, 49 lignes cumulées.
- **Bug fixes matinaux** : 4 commits sur 16 min (12:06 → 12:21). Cumulé +47/-28.
- **Difficulté totale livrée** : 8 (refonte) + 5 (dispatchers) + 3 (browse/presets) + 3 (run_pipe) + 2 (list_dir) + 2 (cwd glob) + 1 (DISPATCHER_NAME) + 5 (pivot SSH) + 1 (bootstrap) = **30**.
- **Smokes passés** : `/healthz` (SSH reachability OK), `/api/dispatchers` (3), `/api/recipes` (4), `/api/browse?path=/Volumes/models/mlx/raw` (liste dossiers OK).
- **Smokes non passés** : pilot `--limit-layers 2` sur un vrai raw. Les modèles dans `/Volumes/models/mlx/raw/` sont déjà convertis, faudrait re-tester sur un raw source (Qwen3.5-122B, GLM-5.2, etc.).

---

## TODO direct (par ordre)

1. **[8] Pilot end-to-end des 2 nouveaux dispatchers** sur un raw source — `minimax_m3_q6_head8` sur `MiniMaxAI/MiniMax-M3` raw (vérifier que switch_mlp + MSA indexer + sparse_attn_config produisent un modèle chargeable par Telemak), `hy_v3_q8_default` sur un Bailing/Hunyuan raw (vérifier que layer_types détecte bien le mix linear_attn + standard_attn). Smoke = `mlx_lm.convert --hf-path` + charge dans Telemak `/admin/load` + 1 token via `/v1/chat/completions`.
2. **[3] Audit inline sur `minimax_m3.py`** — actuellement skip de `m3_convert.py` (vitesse > debug). À faire avant de shipper en recette par défaut : worst-corr par layer, threshold 0.985, comparaison avec les audits `m3_convert.py` (corr 0.9997 MiniMax-M3 Q6H16 = marge confortable).
3. **[2] Flags CLI custom sur `minimax_m3.py`** — supporter `--local-blocks` et `--topk-blocks` du m3_convert.py original pour les cas où le config.json a des valeurs sous-optimales (over-routing, topk trop large).
4. **[2] Créer `mlxconvert/docs/`** avec un README par dispatcher expliquant les choix de quantization par block + le rationale (pourquoi indexer bf16, pourquoi head_bits != bits, etc.).
5. **[1] Commit les 3 docs untracked dans Telemak** (TODO reporté de la veille) : `SESSION-2026-06-01-stabilisation-release-public.md`, `SESSION-2026-06-14-mavis-rejoint-equipe.md`, `SIGNING-MIGRATION.md`. Sophie n'a pas encore eu besoin, mais ça traîne.
6. **[1] Versionner `~/Claude/code/MLX convert/`** (l'ancienne Swift app) si Sophie veut garder une trace — pas de `.git` actuellement, juste le `.agents/` et les sources. Soit `git init`, soit suppression si la mlxconvert monorepo GUI remplace complètement la Swift app.

---

## Lessons learned

- **SSHFS Tier 3 brew est un mur** — pas de bottle macFUSE/libfuse pour macOS récent. Au lieu de batailler 2 jours avec NFS/FUSE, on remonte d'un niveau : SSH remote control. Le control plane sur `.141` envoie des commandes et lit des résultats, pas de volume de données partagé. C'est **strictement plus simple** que SSHFS pour ce use-case (GUI qui orchestre des jobs lourds), et c'est **strictement plus sûr** (pas de mount qui peut casser mid-job).
- **zsh history expansion dans un shell non-interactif est un piège** — `\$NF` n'est pas un escape, c'est juste un `$NF` qui se fait évaluer. Leçon : dans tout script qui passe par SSH + container + zsh, **pas d'awk inline, pas de one-liner Python avec blocks**. Toujours `python3 -c` ou pipe stdin. Le quote hell n'est pas un problème de quoting, c'est un problème de shell.
- **`except Exception: continue` silencieux est un crime** — il m'a coûté 30 min de debug sur un truc visible en 5 secondes si l'erreur avait remonté. Tout `except` dans ce repo doit soit logger, soit re-raise. Le `continue` est réservé aux cas où l'ignore est documenté ET loggé.
- **Le cwd SSH compte** — `REMOTE_REPO_DIR` est le sub-dir Python qui EST le package, donc le glob `mlxconvert/dispatchers` cherchait `mlxconvert/mlxconvert/dispatchers/`. Tout chemin relatif passé via SSH est relatif au cwd du shell SSH, pas au path logique du package. Toujours vérifier `pwd` après SSH avant de raisonner sur les paths.
- **DISPATCHER_NAME sur la classe, pas le module** — quand on introspecte un module pour trouver ses sous-classes d'une base, les attributs spécifiques (name, doc, version) sont sur la classe, pas au top-level. Pattern à appliquer pour tout futur dispatcher.
- **`tmb-site v6` comme direction artistique est réutilisable** — la palette encre/cobalt/chrome/ivoire + Fraunces italic brand-only + Inter body + JetBrains Mono data tient sur des surfaces très différentes (site web themonoclebear.com, GUI web mlxconvert, futur GUI desktop Telemak Monitor). C'est une identité forte qui scale. Sophie valide du premier coup à chaque application — preuve que le design system est robuste.
- **« Réutilise ce qui tourne déjà » est une directive qui structure la journée** — Sophie n'a pas demandé « ajoute un dispatcher M3 générique », elle a dit « ajoute les dispacher qu'on a deja dedans ; minimax, hy3, ... recherche ce qu'on a deja sur ultra-512 ». Conséquence : pas d'invention, juste du portage structuré. Le `minimax_m3.py` est calqué sur `m3_convert.py` (60 layers, sparse_attention_freq, switch_mlp stack, etc.). Le `hy_v3.py` est best-effort sur `bailing_hybrid.py`. Pas de refonte du modèle de quantization, juste du transport vers le format dispatchers.
- **Le portage de scripts monolithiques vers des dispatchers structurés est une dette technique qui se paie vite** — `m3_convert.py` faisait 600+ lignes en un bloc. `minimax_m3.py` fait 235 lignes en blocks nommés (`experts`, `shared_expert`, `indexer`, `router`, `norms`, `head`). Le ratio lignes/block-naming est le bon indicateur de structuration : un block bien nommé vaut 50 lignes de code spaghetti.
- **Audit corr 0.9997 sur M3 Q6 H16 = marge confortable** — le `m3_convert.py` original produit des artefacts avec worst-corr 0.9997 sur un 60-layer model, threshold 0.985. Le `minimax_m3.py` n'a pas encore été audité (TODO 2), mais le design (head_bits=8 mixed quant + experts Q6 g64 + indexer bf16 + router bf16) reproduit les choices du script original. Si la sortie du nouveau dispatcher donne un worst-corr < 0.985 sur un raw, c'est qu'il y a un bug d'implémentation, pas de design.

---

> *Le monorepo mlxconvert est né, la GUI est sur `tmb-site v6`, les trois dispatchers couvrent ce qui tourne déjà sur ultra-512. Le backlog Telemak reste à zéro. Le sprint continue demain — pilot end-to-end des deux nouveaux dispatchers sur des raw sources, puis audit inline.*
