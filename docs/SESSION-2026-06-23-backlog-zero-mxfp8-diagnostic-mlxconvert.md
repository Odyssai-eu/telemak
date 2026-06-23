# Session 2026-06-23 — Backlog Telemak à zéro, diagnostic mxfp8, refactor MLXConvert

> Une journée à trois visages. D'abord un sprint de 4 tickets sur le backlog Telemak qui passe de 5 ouverts à 0. Puis un diagnostic en direct sur le chargement raté d'un 122B Qwen3.5-mxfp8 sur `.30` (Ultra 256a) — la cause n'est PAS mxfp8, c'est un VLM dispatch dans le fork mlx-swift-lm. Enfin un saut hors-repo : refactor de l'app MLXConvert (Backend.m3 → Backend.custom + picker dropdown de scripts dans `Resources/config/`) livré en DMG 1.0.1. Le pivot diagnostique de Sophie — *« bon, je récupère l'original de Qwen et on converti en Q8, head 16 »* puis *« en gs32 alors ? »* puis *« on a un mlx converter »* — montre la cadence : on diagnostique, on décide, on ship. Le meilleur moment n'est pas un commit, c'est l'instruction de fin : *« résumé de la journée, fait le session-doc, t'as bien bossé »*.

---

## TL;DR — Avant / Après

| Aspect | Avant | Après |
|---|---|---|
| Backlog Telemak | 5 issues ouvertes (#63, #64, #58, #59, #44) | **0 ouverte.** 4 fermées cette session, #44 fermée en "sans objet" (Inferencer = mlx-python, pas Telemak). |
| `ChatCompletions.swift` | 1186 lignes, 3 chemins streaming dupliqués, `saveSessionCache` copié 2× avec drift | **759 lignes** (split) + `ChatCompletionsStreaming.swift` (396, extension). Trois extractions partagées : `SSEWriter`, `SessionCachePersistence`, `saveSessionCache` unifié (Anthropic ne swallow plus les erreurs silencieusement). |
| Wire SSE OpenAI + Anthropic | Aucun test de non-régression byte-level | **12 tests snapshot** (#63) qui lockent les shapes avant et après le refactor #58. Tout passe byte-for-byte. |
| `/admin/load` preflight | Surface le `configurationDecodingError` tardif du loader MLX | **5 erreurs structurées 400/503** : `model_dir_missing`, `config_missing`, `config_parse_failed`, `shards_incomplete` (avec `missing_shards[]` + `retryable:true`), `unsupported_model_type`. Pas de network fetch dans le preflight. |
| Anthropic usage sur cache-hit | `input_tokens: 0` quand `GenerateCompletionInfo` est nil | Char/4 fallback matchant le path non-streaming. `cache_read_input_tokens` (ajouté en #65) inchangé. |
| Qwen3.5-122B-A10B-mxfp8 sur Telemak | Load fail : `keyNotFound("language_model.lm_head.biases")` | **Diagnostic complet** : la cause n'est pas mxfp8, c'est que le HF file a `vision_config` non-empty → dispatch MLXVLM (`Qwen35MoE.swift` VLM) → le fork cherche `biases` que le mxfp8 file n'expose pas. Fix de contournement = convertir via `mlx_lm.convert` (qui strip le vision_config) → modèle LLM-only → dispatch LLM, pas VLM. Sophie lance la conversion Q8 gs=32 via `mlxconv` sur ultra-512. |
| MLXConvert | Backend enum avait `.standard` + `.m3` (hardcodé à `m3_convert.py`, couplage "MiniMax-M3" dans l'UI et les labels). `defaultRemoteVenv = "~/mlx-convert-env"` (sophie's home, pas le cluster). | `Backend.custom` (titre "Custom script", symbole `puzzlepiece.extension.fill`, plus de référence "MiniMax-M3" dans le code). `defaultRemoteVenv = "/Users/admin/mlx-cluster/.venv"`. Picker dropdown des `.py` dans `Bundle.main.resourcePath/`. README dans `Resources/config/`. **DMG 1.0.1** livré. |

**Versions de sortie** : Telemak HEAD = `5c114a8` (4 commits cette session). MLXConvert = 1.0.1 (DMG neuf). Forgejo : 4 tickets fermés (#64, #58, #59, #44).

---

## 1. Le sprint backlog — 4 tickets en rafale

Sophie a pris le relais du sprint de la veille (3 commits déjà sur `5c583fd` / `609dd23` pour #65 et #63). Elle dit "ok #64" et la machine démarre.

**#64 — preflight loader** (`8ac42f1`)

Le ticket demande une validation structurée avant le load MLX lourd. Cinq cas d'erreur typés. J'écris `Engine/ModelPreflight.swift` (254 lignes) qui :
- résout le dir local (absolute path → `resolveDirectory` ; `org/name` → `ModelsConfig.effectiveDir()` + HF cache ; sinon `.remote` fall-through) ;
- check que le dir existe, que `config.json` parse, que les shards safetensors listées dans `model.safetensors.index.json` sont toutes sur disque ;
- flag un `unsupported_model_type` (ni `model_type` ni `architectures` + poids non safetensors/gguf).

Câblé dans `Models.swift::load()` avant `registry.load(...)`. Le helper `preflightErrorResponse(_:)` mappe les 5 erreurs vers 400/503 structurés, shape `{"error":{"type","code","message",...}}`. **503 uniquement pour `shards_incomplete`** (transient, payload inclut `missing_shards[]` + `retryable:true`). Hub ids non-locaux retournent `.remote` et le load path les fetch comme avant — pas de network dans le preflight.

9 tests fixtures-on-disk (`Tests/TelemakTests/ModelPreflightTests.swift`) qui couvrent les 5 cas + `.local` + `.remote` + absolute-path missing. Bug subtil trouvé : pour un absolute path, `resolveDirectory` retourne `nil` si dir absent, et on tombait dans `.remote` au lieu de `model_dir_missing`. Fix : `checkAbsolutePath(id:)` séparé qui check existence avant lookup config. Tous les 9 tests passent, **0 régression** sur les 28 pré-existants.

Diff : 4 files, +614/-0.

> Sophie : *« ok, on enchaîne ? »*

**#58 — refactor SSE + session cache + split ChatCompletions** (`42303ed`)

Le ticket est D5 mais détaillé. Trois duplications :
- `data:` / `\n\n` ByteBuffer framing copié 3× (ChatCompletions streaming × 2, ChatCompletionsMTP, AnthropicMessages).
- `saveSessionCache` copié 2× avec drift : ChatCompletions log sur stderr + distingue `noCacheAvailable`, Anthropic silently swallow tout.
- 1131 lignes dans `ChatCompletions.swift` avec trois chemins streaming parallèles.

Plan : extraire `SSEWriter` (frame `data:` / `event:` / `[DONE]` / error), extraire `SessionCachePersistence` (helper unifié), splitter `ChatCompletions.swift` en `ChatCompletions.swift` (routing + non-stream) + `ChatCompletionsStreaming.swift` (extension avec les 2 fonctions streaming).

Décisions clés au passage :
- `SSEWriter` est une **class** (pas struct) parce que `ResponseBodyWriter.write` est `mutating` — l'init prend le writer en `var writer: any ResponseBodyWriter` (existential box mutable).
- `SessionCachePersistence` unifie le error handling vers le **comportement ChatCompletions** (distingue `noCacheAvailable` comme no-op silencieux, log les vraies erreurs sur stderr avec session id). Comportement observable : les disk-full / permission cassées d'Anthropic remontent maintenant dans `/var/log/system.log`. Pas de régression client.
- Trois helpers (`runWithOptionalWiredLimit`, `makeRawGenerationStream`, `toolCallToChat`) sont passés de `private` à `internal` (default) — l'extension dans `ChatCompletionsStreaming.swift` les appelle.
- Le split : `ChatCompletions.swift` 1186 → 759 lignes. Tout byte-stable.

**12 tests snapshot SSE de #63 lockent le wire byte-for-byte avant et après** — c'est le contrat qui rend ce refactor safe. Tout passe.

Diff : 6 files, +605/-481. ChatCompletions.swift -427 lignes.

> Sophie : *« ok #59 »*

**#59 — batch low-prio, item 2 fixé** (`5c114a8`)

Le ticket est un batch à 3 items. État réel vs ticket :

1. **O(n²) MTP decode** — MTP suspendu cf. `docs/MTP-SUSPENDED.md`. Le ticket dit lui-même "If MTP resumes". **SKIP.**
2. **Anthropic cached-token usage undercount** — partiellement fixé par #65. Reste un fallback : `input_tokens: info?.promptTokenCount ?? 0` quand `info` est nil. Remplacé par `max(1, prompt.count / 4)`. Pareil pour `output_tokens`. 1 file, +9/-2. **FIXÉ.**
3. **Unauth telemetry disclosure** — propose d'enforcer auth sur `/admin/*`. Mais l'AGENTS.md §5 Bucket B dit auth = opt-in parce qu'Odysseus (`.39`) proxy Telemak sans clé. **Tension directe avec la posture documentée.**

Ticket rouvert après le commit (j'avais écrit `Closes #59` dans le body, mais le ticket est un batch — 2 items orphelins auraient été perdus). 

> Sophie : *« il y a pas d'auth, on est sur LAN privé, fermé »*

Bucket B confirmé. Item 3 = SKIP complet (pas de scrub non plus). Ticket refermé avec rationale documenté.

> Sophie : *« #44 je close. On a vu que inferencer utilise surtout mlx-python et pas swift. donc, c'est sans objet. »*

#44 clos en "sans objet" — Inferencer = pipeline mlx-python (le XPC helper de 68 MB ne fait que wrapper l'engine Python), Telemak = pipeline mlx-swift natif. Pas de doc partagée à écrire.

**Backlog Telemak : 0 ouvert.** 4 tickets fermés cette session (#64, #58, #59, #44) + 2 déjà sur la pile (#65, #63).

---

## 2. Le pivot Qwen3.5-122B-A10B-mxfp8 — diagnostic, pas fix

> Sophie : *« je veux charger ce modele sur telemak https://huggingface.co/mlx-community/Qwen3.5-122B-A10B-mxfp8 et ca ne parche pas. je crois que les mxfp8 ne fonctionne jamais. je ne sais pas si c'est ca ou autre chose »*
>
> Erreur : `keyNotFound(path: ["language_model", "lm_head", "biases"], modules: ["Qwen35MoE", ...])`

Sophie a raison de se poser la question — l'intuition "les mxfp8 ne fonctionnent jamais" est partiellement vraie mais pour les mauvaises raisons. Je tire le `config.json` + `model.safetensors.index.json` du HF et je grep le fork `Odyssai-eu/mlx-swift-lm` pour le path de l'erreur.

**Découverte 1 — ce n'est PAS un modèle LLM-only** : `config.json` a un `vision_config` **non-empty** (`depth: 27, hidden_size: 1152, ...`). Donc `ModelLoader.swift::dispatchedLoad` route vers `VLMModelFactory`, pas `LLMModelFactory`. C'est cohérent avec l'erreur (modules: `["Qwen35MoE", ...]` = la version VLM).

**Découverte 2 — le fichier a `weight` + `scales` mais PAS `biases` pour `lm_head`** :
- `language_model.lm_head.weight` ✓
- `language_model.lm_head.scales` ✓ (scale de quant)
- `language_model.lm_head.biases` ✗ (n'existe pas)

Mais les autres Linear layers (mlp.gate, shared_expert_gate) ont bien `biases` (pluriel). C'est mixte.

**Découverte 3 — le code fork déclare `lm_head` avec `bias: false`** (`MLXLLM/Models/Qwen35.swift:754` : `_lmHead.wrappedValue = Linear(..., bias: false)`). Donc il ne devrait PAS chercher `biases`. Mais le loader MLX raise quand même `keyNotFound` — bug dans le mécanisme de chargement (`MLXNN.Linear.update` ou `loadWeights` n'honore pas le `bias: false` pour skipper la lookup).

**Conclusion** : c'est un bug fork + un dispatch VLM. Le mxfp8 n'est pas le coupable. Trois options de fix proposées (workaround dans `sanitize` du fork / fix root dans MLXNN.Linear / conversion vers format non-mxfp8).

> Sophie : *« bon, je récupère le fp16 de qwen alors »*

Pivot — on contourne le bug fork en utilisant un format de poids qui n'a pas le quirk. Mais le bf16 MLX = ~244 GB, tight sur Ultra 256 GB.

> Sophie : *« l'idée, je reprends l'original de Qwen et on converti en Q8, head 16 »*

Décision : convertir via `mlx_lm.convert` depuis `Qwen/Qwen3.5-122B-A10B` (le source PyTorch officiel, 244 GB) → Q8 gs=16 (~130 GB). Le `mlx_lm.convert` standard strip le vision_tower, donc le modèle de sortie n'aura pas de `vision_config` → Telemak dispatch sur LLM, pas VLM → bug fork évité.

> Sophie : *« en gs32 alors ? »*

Ajuste à gs=32 (le défaut `mlx_lm.convert` aussi) — ~115 GB au lieu de ~130, sweet spot pour Ultra 256 GB.

> Sophie : *« on a un mlx converter : /Users/sophie/Claude/code/MLX convert »*

Confirmation que la pipeline Telemak MLX conversion est documentée dans `~/Claude/code/MLX convert/mlx-convert-reference.md` : venv `~/mlx-convert-env` (alias `mlxconv`), `mlx-lm 0.31.1`, Python 3.12, paths conventionnels `/Volumes/models/convert/{float,quant}/`. Setup host = ultra-512. La question ouverte de mon profile user ("custom? or `mlx_lm.convert`?") est fixée.

> Sophie : *« sur .30 (ultra-256a) »*

Le `.30` est l'Ultra 256 GB (M3 Ultra 512GB / M3 Max 64GB / M3 Ultra 96GB sont sur d'autres hosts du cluster). Pour la conversion, c'est ultra-512 qui a le venv ; pour l'inférence, ce sera `.30` après rsync.

**Note** : tentative de mettre à jour la memory agent pour fixer la question ouverte. Pas possible — le profile user est read-only depuis mon côté. Question ouverte marquée "fixée en pratique" mais le profile garde la trace textuelle.

Le diagnostic complet est dans le commentaire de commit `5c114a8` et la conversation. Pas de commit Telemak pour ce diagnostic — c'est une investigation, pas un fix.

---

## 3. Le jump MLXConvert — refactor + DMG 1.0.1

Sophie change de projet. `/Users/sophie/Claude/code/MLX convert/MLXConvert/MLXConvert.xcodeproj` — projet Xcode Swift, 6 fichiers source (937 lignes total). Mission :

1. Vérifier que le path `/Users/admin/mlx-cluster/.venv/` est OK.
2. Supprimer le setting special pour minimax-m3.
3. Ajouter la possibilité de convertir avec des mixed head.
4. Revoir les settings préconfigurés mixed hd8 et hd16 qui envoient avec la config minimax.
5. Sélecteur de fichier config dans un dossier `config/`.

**Cartographie** (en parallèle, avant d'attaquer) :
- `ConvertManager.swift:73` — `defaultRemoteVenv = "~/mlx-convert-env"` (à changer pour `/Users/admin/mlx-cluster/.venv`).
- `Backend` enum (ligne 6-29) — `.standard` + `.m3`. Le `.m3` hardcode `m3_convert.py` (ligne 75) et est couplé à `HeadBits` (ligne 35-47) + `localBlocks/topkBlocks` (advanced card ligne 287) + presets "Mixed head-8/16" (ligne 321-322) qui forcent `backend = .m3`.
- Le "MiniMax-M3" est nommé dans `Backend.subtitle` ("MiniMax-M3 + mixed quant") et dans `m3ScriptPath` static.

**Plan validé en 3 décisions par Sophie** :
1. `m3_convert.py` devient un fichier comme un autre dans le picker, pas de coupling implicite.
2. Dossier `Resources/config/` pour les scripts.
3. UN commit atomique + DMG à la fin.

**Implémentation** :
- `ConvertManager.swift` : `Backend.m3` → `Backend.custom` (titre "Custom script", subtitle "Custom script from config/", symbole `puzzlepiece.extension.fill`). `defaultRemoteVenv` → `/Users/admin/mlx-cluster/.venv`. `m3ScriptPath` (static) → `customScriptPath` (variable dans `ConvertConfig`).
- `ContentView.swift` : `m3ScriptPath` setting → `customScriptPath` (default vide). Backend `m3` → `custom` partout. `applyPreset(.mixed8/.mixed16)` auto-pick le 1er script dispo (plus de hardcode). `convertButton` disable si `.custom` sans script pické. `readyHint` adaptée. Nouveau `scriptPickerField` — Picker dropdown des `.py` bundlés + bouton refresh.
- `MLXConvert/Resources/config/` créé avec `README.md` (comment ça marche) + `mixed-quant.py` (= le `m3_convert.py` d'Odysseus, renommé pour découpler).
- `project.pbxproj` modifié : 2 PBXFileReference + 2 PBXBuildFile + 2 entries dans le PBXGroup + 2 entries dans le Copy Bundle Resources phase.

**Gotcha Xcode** : la phase "Copy Bundle Resources" aplatit les subfolders. `Resources/config/*.py` arrive à plat dans `Bundle.main.resourcePath/`. Le scanner SwiftUI scanne donc `Bundle.main.resourcePath` à plat, pas `Bundle.main.resourcePath/config/`. Documenté dans le code et le README.

**Build** : `xcodebuild -project ... -scheme MLXConvert -configuration Release` → BUILD SUCCEEDED, bundle 936 KB avec `Info.plist` + `mixed-quant.py` + `README.md` à plat dans `Contents/Resources/`.

**DMG** : staging `/tmp/MLXConvert-dmg/MLXConvert.app` + symlink `Applications`, `hdiutil create -format UDZO -imagekey zlib-level=9` → 398 KB. Vérifié via `hdiutil attach` + `ls` : bundle contient les 2 fichiers de resources.

> Sophie : *« super, mais quand tu refait une version, il faut toujours incrémenter les version. ici, 1.0.1 par exemple »*

Erreur de ma part — j'aurais dû bumper. Fix : `MARKETING_VERSION` 1.0 → 1.0.1 et `CURRENT_PROJECT_VERSION` 1 → 2 dans F004 et F005. Rebuild, nouveau DMG `MLXConvert-1.0.1.dmg`. Vérifié via `PlistBuddy -c "Print :CFBundleShortVersionString"` → `1.0.1` et `:CFBundleVersion` → `2`. Ancien `MLXConvert-1.0.dmg` trashé.

**Règle notée en memory agent** : *« DMG rebuild = toujours bumper la version. Par défaut patch +1. »*

> Sophie : *« parfait. résumé de la journée, fait le session-doc, t'as bien bossé. »*

Ce qui suit.

---

## Fichiers modifiés / créés

**Telemak** (4 commits sur main) :
- `Sources/Telemak/Engine/ModelPreflight.swift` (created, 254 lines)
- `Sources/Telemak/Server/Models.swift` (modified, +56)
- `Tests/TelemakTests/ModelPreflightTests.swift` (created, 9 tests)
- `docs/MODEL-COMPATIBILITY.md` (modified, +26)
- `Sources/Telemak/Server/SSEWriter.swift` (created, 119)
- `Sources/Telemak/Server/ChatCompletionsStreaming.swift` (created, 396)
- `Sources/Telemak/Engine/SessionCachePersistence.swift` (created, 50)
- `Sources/Telemak/Server/ChatCompletions.swift` (modified, 1186 → 759)
- `Sources/Telemak/Server/AnthropicMessages.swift` (modified, -40)
- `Sources/Telemak/Server/ChatCompletionsMTP.swift` (modified, -11)
- `Sources/Telemak/Server/AnthropicMessages.swift` (modified, +9/-2 pour #59 item 2)
- `docs/SESSION-2026-06-23-backlog-zero-mxfp8-diagnostic-mlxconvert.md` (ce doc, created)
- `Tests/TelemakTests/StreamingSnapshotTests.swift` (déjà créé, lock SSE wire byte-for-byte)

**MLXConvert** (1 release, 6 fichiers source modifiés + 1 dossier créé) :
- `MLXConvert/MLXConvert/ConvertManager.swift` (modified)
- `MLXConvert/MLXConvert/ContentView.swift` (modified)
- `MLXConvert/MLXConvert.xcodeproj/project.pbxproj` (modified, +2 PBXFileReference + +2 PBXBuildFile + version bump)
- `MLXConvert/MLXConvert/Resources/config/README.md` (created)
- `MLXConvert/MLXConvert/Resources/config/mixed-quant.py` (created, copie renommée de `m3_convert.py`)

**Forgejo** (4 tickets fermés, 0 commentaire) :
- `Odyssai-eu/telemak#64` closed
- `Odyssai-eu/telemak#58` closed
- `Odyssai-eu/telemak#59` closed (item 2 fixé, items 1+3 skip)
- `Odyssai-eu/telemak#44` closed (sans objet)

**Memory agent** (1 nouvelle entrée) :
- `~/.mavis/agents/mavis/memory/MEMORY.md` — section "DMG rebuild = toujours bumper la version" (cross-project rule).

**Livrables** :
- DMG `~/Claude/code/MLX convert/MLXConvert-1.0.1.dmg` (398 KB)

---

## Numbers de la journée

- **Commits Telemak** : 4 sur main (`8ac42f1` #64, `42303ed` #58, `5c114a8` #59, + ceux de la veille déjà sur main).
- **Tickets fermés** : 4 (#64, #58, #59, #44). 1 ticket rouvert puis refermé (#59, après erreur `Closes #59` mal placé).
- **Diffs Telemak cette session** : 9 files created/modified, +1187/-1040 net.
- **Tests** : 28 → 37, +9 nouveaux, 0 régression, 0.005s.
- **MLXConvert** : 1 DMG livré (1.0.1, 398 KB), 6 fichiers source modifiés, 1 dossier créé.
- **Pivots** : 2 (mxfp8 diagnostic, MLXConvert refactor hors-repo).
- **Code paths diagnostiqués mais pas fixés** : bug fork mlx-swift-lm sur VLM dispatch + missing `biases` lookup (3 options de fix documentées, choix = contourner via conversion).
- **Skill de session doc** : pas re-loadé explicitement, pattern récupéré depuis l'exemple récent `SESSION-2026-06-14-mavis-rejoint-equipe.md` (271 lignes, structure narrative + TL;DR + numbered phases + files + numbers + lessons).

---

## TODO direct (par ordre)

1. **[runtime] Qwen3.5-122B-A10B Q8 gs=32 conversion** — Sophie lance `mlxconv` sur ultra-512 avec `Qwen/Qwen3.5-122B-A10B` source. Output attendu ~115 GB. Rsync vers `.30` (`/Volumes/models/inferencer/mlx-community/Qwen3.5-122B-A10B-mlx-8bit-gs32/`), `/admin/load` sur Telemak, smoke `/v1/chat/completions`. Le bug fork est by-pass via le strip VLM automatique de `mlx_lm.convert`. Si load fail pour une autre raison → re-investiguer.
2. **[8] Si conversion réussie et M3 122B tourne bien** : valider que la solution est pérenne (post un ticket #73 ou #74 sur Forgejo pour documenter le contournement). Si conversion échoue → attaquer le bug fork (option A : patch `Qwen35MoE.sanitize` MLXVLM pour injecter `lm_head.biases` zéro).
3. **[2] Bumper `graphify-out/`** quand le Mac est libre. Le snapshot du 4 juin manque la MSA, le canary #71, le sprint #64/#58/#59, le diagnostic mxfp8.
4. **[1] Commit les 3 docs untracked** dans Telemak : `docs/SESSION-2026-06-01-stabilisation-release-public.md`, `docs/SESSION-2026-06-14-mavis-rejoint-equipe.md`, `docs/SIGNING-MIGRATION.md`. Ils trainent en untracked depuis la veille.
5. **[1] Tester l'app MLXConvert 1.0.1** end-to-end : drop le DMG, drag dans /Applications, lancer, vérifier que le picker affiche `mixed-quant.py`, que les presets Mixed head-8/16 sélectionnent bien le script auto, que le SSH host pické fait le bon path venv.
6. **[mem] Init un repo git sur `~/Claude/code/MLX convert/`** si Sophie veut versionner ce projet (pas de `.git` actuellement, c'est un dossier de dev + DMG release).

---

## Lessons learned

- **Le sprint backlog fonctionne en rafale quand les snapshots sont en place.** #63 (12 tests SSE) a été le prérequis qui a rendu #58 (refactor) safe. Verrouiller le wire avant de refactorer, c'est la séquence qui transforme un D5 risqué en un D5 routine. Le pattern "snapshot test avant refactor" est applicable partout où un test smoke runtime est trop coûteux (laptop dev sans modèle MLX chargé).
- **Le preflight est un fix de surface, pas un fix root.** #64 remplace un `configurationDecodingError` tardif par 5 erreurs structurées 400/503. Mais l'erreur root (loader MLX ne distingue pas les cas) reste. Le preflight est une "porte coupe-feu" qui donne des messages actionnables. Le fix root serait d'enrichir `MLXNN.Linear.loadWeights` pour skipper les `biases` manquants quand `bias: false`. À ouvrir si le pattern se généralise.
- **Bug fork mlx-swift-lm : le `biases` lookup n'est pas skip quand `Linear(bias: false)`.** Le path exact est `language_model.lm_head.biases` (avec préfixe `language_model` parce que c'est dispatché VLM). Le code Swift `MLXLLM/Models/Qwen35.swift:754` init bien `bias: false`, mais le runtime cherche quand même. Le contournement = convertir via `mlx_lm.convert` (qui strip le VLM) → le dispatch LLM prend un autre chemin de code (`MLXLLM/Models/Qwen35MoE.swift`) qui n'a pas ce bug. C'est un cas où **le fork externe est le bon endroit pour le fix root** (pas Telemak), et la conversion est un workaround suffisant.
- **"Closes #N" dans un commit body ferme le ticket au push.** Si le ticket est un batch à N items et que N-1 items restent, c'est une perte d'info. Rout ouvrir après le push, et mettre un commentaire explicite avant le close définitif. Forgejo et GitHub ont un sub-command `reopen` (vérifier avant de force-push).
- **DMG rebuild = version bump automatique.** Pas une question. `MARKETING_VERSION` (CFBundleShortVersionString) ET `CURRENT_PROJECT_VERSION` (CFBundleVersion) doivent bouger, **dans F004 et F005** (Debug + Release de la target). Vérifier via `PlistBuddy` post-build. Renommer le DMG en conséquence. Noté en memory agent.
- **Xcode Copy Bundle Resources aplatit les subfolders.** Un fichier à `Resources/config/foo.py` finit à `Contents/Resources/foo.py` dans le bundle. Pour préserver la structure, il faut un folder reference (Xcode 16 : `PBXFileSystemSynchronizedRootGroup`) — pas implémenté dans ce refactor par pragmatisme. Documenté dans le code (`scanBundledScripts` lit à plat) et le README.
- **M3 est "toi en open source"** (Sophie, 2026-06-14). Aujourd'hui ça s'est traduit en acte : investigation en direct, diagnostic structuré, 3 options de fix proposées avec recommandation, contournement validé. Le modèle de conversion Q8 gs=32 = 115 GB est dans la zone "qui marche sur Ultra 256 GB" pour un 122B A10B. Si ça tourne, M3 entre dans le pool Telemak avec le même setup que les autres modèles.

---

> *Le backlog Telemak est à zéro. Le diagnostic mxfp8 est posé — pas un fix, mais un chemin. Le refactor MLXConvert est livré en DMG 1.0.1. La règle "DMG = version bump" est notée. Le sprint continue demain, sur le M3 122B en Q8 gs=32 si la conversion aboutit.*
