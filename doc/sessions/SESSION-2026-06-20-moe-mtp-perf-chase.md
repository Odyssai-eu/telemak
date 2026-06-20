# Session 2026-06-20 — MoE-MTP : ça marche, mais le mur perf

> Journée marathon. Le matin on livre le **premier MoE-MTP du stack qui MARCHE** —
> Hy3, acceptance 0.581, sortie exacte, là où EAGLE avait échoué. Puis le mur : ~30×
> trop lent. La chasse occupe tout le reste — Q4, profiling par-op, MiniMax partenaire,
> Inferencer, GitHub. Le tournant n'est pas un fix : c'est un **repro isolé** qui
> **disculpe mlx-swift** et transforme un « file une issue » confiant en « ne file rien,
> c'est pas notre bug ». Deux leçons de Sophie scellent la journée : *« tu devrais
> t'arrêter et dire : c'est pas normal »* et *« on n'est pas avec des amateurs »*. La
> meilleure sortie du jour est un **négatif rigoureux** + une discipline, pas un tok/s.

---

## TL;DR — la journée en 4 axes

| Axe | Livré |
|---|---|
| **Build** | Hy3 MoE-MTP implémenté end-to-end (tête draft + iterator + layer vendoré + câblage loader/dispatch). **Acceptance 0.476 → 0.581, exact, 1er MoE-MTP du stack.** |
| **Chasse perf** | ~30× trop lent. Tout éliminé par preuve : tête, cache, mask, build CPU, dtype, tri, version MLX, **et le gather/SwitchGLU eux-mêmes**. Cause = effet forward-complet non-isolable. |
| **Nettoyage .29** | Dépotoir telemak (28 slots) wipé → 1 slot neuf ; trou `deploy-all` (ne crée pas le plist serveur) comblé ; verdict M3+Telemak (pas de MSA + pas de cjk_lock → inférieur à OdyssAI-X). |
| **Recadrage** | mlx-swift disculpé → **pas d'issue upstream** (pas de repro = ticket d'amateur). 1 issue filée : OdyssAI-X #62 (alias `default` opaque). |

Version de sortie : aucune. Telemak reste **`0.6.47`** (le perf n'a pas été fixé, le MTP reste sur la branche `feat/hy3-mtp`).

---

## 1. Le matin — l'épopée MoE-MTP (le 1er du stack qui marche)

Suite du port Hy3 de la veille. On déroule l'épopée MTP : contrat hidden PRE-norm
résolu depuis le vLLM `hy_v3_mtp.py`, script d'extraction de la tête (`model.layers.80`
→ sidecar `mtp.safetensors`, commits `68daa4b` + `6dc7613` Q4 côté Odysseus), classe
draft Swift + iterator spéculatif + decoder layer MoE vendoré (telemak `596a0e9`,
`f0f1474`, `b86a9a0`), ABI hidden-state sur `HYV3Model` dans le fork (`d80cc80`).

Le résultat est **réel et acquis** : le MoE-MTP **fonctionne** — acceptance **0.476 →
0.581**, sortie **exacte**, cohérente. **C'est le premier MoE-MTP du stack**, et il bat
le fiasco EAGLE (acceptance ≈ 0, abandonné). Le `CONTEXT.md` du repo pose le vocabulaire
(la précision de la tête est un **levier vitesse↔acceptance, PAS un levier qualité** —
le trunk vérifie chaque token).

Mais le smoke crache le mur : **~0.5 tok/s vs 17-19 en AR. ~30× trop lent.**

---

## 2. Le mur — MiniMax partenaire, et les fausses pistes

Sophie cadre le mode de travail :

> Sophie : *"minimax, est ton partenaire"*

On lance `/minimax-review` puis `/grill-with-docs-minimax`. Sophie tranche le fix tête :

> Sophie : *"on peut meme quantifier plus les head, un Q4, c'est ça qui accélère. et
> comme le modele vérifie les tokens suggérés, la qualité reste OK"*

Le raisonnement est juste (la tête est un drafter → sa précision ne coûte que de
l'acceptance). On quantize la tête en **Q4** (`6dc7613`). Résultat : **ça ne bouge quasi
rien** (0.57 → 0.67 tok/s). Un `eval()` ajouté pour borner le graphe lazy (`0ac2bce`)
révèle la vérité : **verify_s = 115 s** contre **draft_s = 0.089 s**. La tête n'a JAMAIS
été le goulot ; le timing lazy le masquait.

Les diagnostics de MiniMax (tête bf16, puis mask + KV non-quant + scaling-contexte)
sont tous **réfutés par ma preuve empirique** (un smoke 4-token montre un verify
**constant** ~3.15 s indépendant du contexte). On remonte le désaccord, MiniMax accepte
les réfutations. Discipline codex_double_review : on ne papier-mâche pas un désaccord
cross-modèle.

---

## 3. Le profiling — du "c'est la tête" au "c'est le switchMLP"

On instrumente le forward du verify, par paliers (5 cycles build→deploy→smoke sur `.29`) :

- **Par-couche** : les couches MoE coûtent **~40 ms** chacune (×79 ≈ 3 s) ; la couche
  dense (layer 0) = 3 ms. → c'est le bloc MoE.
- **Build vs compute** (split forcé) : **6 ms de build CPU (0%) vs 3173 ms de compute
  GPU (100%)**. → pas le dispatch, c'est du GPU.
- **Sous-op MoE** : `router 4 ms · switchMLP 32 ms · combine 0.9 ms · shared 0.5 ms`.
  → **le `switchMLP` (la SwitchGLU) est le coupable, 32 ms**.

En parallèle, Sophie pointe Inferencer (*« comment inferencer a résolu le problème ?
t'as le code source ici »*). Lecture du source : `Sources/*.swift` sont des
**reconstructions reverse-engineerées** (placeholders, `fatalError`), pas le vrai code ;
le moteur réel = **MLX stock** (libmlx 0.31.1, zéro kernel Metal custom). Inferencer
n'a **rien résolu** — son 1.21× était sur un petit MoE. Recherche GitHub : `mlx` #3632
(« gather_qmm NAX kernel », mergé le 16 juin, post-date ma 0.31.2) semble coller au
symptôme « matrix-vector rapide / matrix-matrix lent ».

---

## 4. Le repro isolé — "on n'est pas avec des amateurs"

Avant de filer #3632 chez ml-explore, Sophie pose la barre :

> Sophie : *"soit hyper précis, on n'est pas avec des amateurs. ne soit pas trop AI
> status, on est humain, on est pro."*

Donc : **repro isolé minimal** (package SwiftPM, deps publiques mlx-swift) du
`gatherQuantizedMM`, lancé **sur `.29` (M3 Ultra, le hardware du bug), modèle 309 GB
chargé**. Le résultat **renverse tout** :

| Test (M3 Ultra .29, 8-bit g32, seq=3) | Temps |
|---|---|
| `gatherQuantizedMM` isolé | **0.47 ms** |
| SwitchGLU complète (3 gathers) isolée | **1.08 ms** |
| mlx **Python** 0.31.2, même op | 1.03 ms |
| **dans le modèle Hy3 chargé** | **32 ms** |

Le primitif est **sain** sur M3 — identique au Python, même avec le 309 GB chargé. Les
32 ms n'existent **que dans le forward complet**, **non minimalement reproductibles**.
On élimine aussi le dtype (bf16/f32/f16 = 1 ms) et le seuil `doSort` (le forcer **empire**
à 117 ms — reverté `688d55e`).

**Conclusion, contre ma propre hypothèse de départ : il n'y a PAS de bug mlx-swift
fileable.** Filer un perf non-reproductible chez ml-explore = le ticket d'amateur qu'un
mainteneur ferme en deux lignes. Sophie d'accord : **on ne file pas.** Le repro a
**évité une fausse issue** et économisé une journée d'upgrade MLX inutile. Nettoyage du
profiling temp (fork `58aab09`, diff vs base ABI = vide).

---

## 5. Le nettoyage .29 — "tu devrais t'arrêter et dire : c'est pas normal"

Sophie : *« est-ce que mlx-swift de .29 est remis à l'état original ? opérationnel ? »*

Je trouve **deux instances Telemak** (8003 + 8013) et, au lieu de m'arrêter, je continue
à fouiller. Sophie coupe net — **la leçon de la journée** :

> Sophie : *"ca va pas si tu as 8003 et 8013. tu devrais t'arreter ici et dire, c'est
> pas normal"*

Recadrage. Le défaut à tuer : barreler quand quelque chose cloche au lieu de
**s'arrêter et remonter**. On déroule proprement :
- `kill le 8013` → tué. Découverte : **8003 était un zombie déjà cassé** (vieux build,
  flappe, échoue à auto-recharger un modèle depuis HF en offline) — **pas mon kill**.
- Fausse alerte de fuite mémoire que je signale… puis re-vérifie : le 85 GB libre était
  **transitoire** (la libération de 325 GB prend qq secondes) → 350 GB libres, **rien
  ne fuit**. Re-vérifier avant d'affirmer.
- Nettoyage : dépotoir `~/telemak` (28 slots, 4.8 GB) wipé → deploy neuf. **Trou réel
  découvert** : `deploy-all.sh` n'écrit QUE le plist menubar, **jamais le serveur** →
  install from-scratch cassée. Comblé en reconstruisant le plist depuis le gabarit de `.30`.

---

## 6. Le verdict M3 sur Telemak — pas de MSA + pas de cjk_lock

Sophie soupçonne un souci M3 + Telemak + la MSA. Vérifié **par le code** :
- **Pas de MSA** : `MiniMaxM3.swift` l.5-6 — l'indexer block-sparse est **délibérément
  non porté** (« left out until 128k »), attention pleine causale. Par design, pas un bug.
- **Pas de `cjk_lock`** : le sampler Swift n'a NI cjk_lock, NI no_repeat_ngram, NI
  rep_penalty. **Toutes les corrections qualité M3 vivent dans le `runner.py` Python**
  (OdyssAI-X), jamais portées en Swift → **M3 sur Telemak leakerait du CJK** sur prose FR.

Test live tenté → load 326 GB **trop lent** :

> Sophie : *"trop long pour loader minimax, deja 3 minutes"* … puis *"kill il est bloqué à 325GB"*

Cumul **pas de MSA + pas de cjk_lock + 326 GB lent**, alors qu'**OdyssAI-X sert déjà M3
avec tous les fixes** → telemak-M3 **strictement inférieur**. Décision : telemak laissé
**installé mais arrêté** sur `.29` (prêt à charger un autre modèle, mémoire rendue).

---

## 7. Argo + l'issue #62

Sophie : *« j'ai mis sur argo Nex N2 et Minimax »* — **le bon endroit** pour M3 (OdyssAI-X,
avec les fixes), pas Telemak. Vérif : `minimax` servi en local, Argo activé (5-node jaccl,
master `.30`), **`Nex N2` = l'alias `default`**.

> Sophie : *"oui, c'est defaut. c'est chiant d'ailleurs, mets ça en issue."*

L'alias `default` est **opaque** : `/v1/models` l'affiche sans dire quel vrai modèle il
résout (la donnée existe pourtant — chaque pool porte `alias` ET `model`). Filé
**[OdyssAI-X #62](http://192.168.86.141:3001/Odyssai-eu/OdyssAI-X/issues/62)** (Forgejo,
Difficulty 2, sans label `ready`).

---

## Numbers de la journée

- **Commits** : 6 telemak (`b898585`→`0ac2bce`), 8 fork (dont 5 temp/revert → **net 2
  réels** : `e86f4ca` port Hy3 + `d80cc80` ABI), 2 Odysseus (`68daa4b` + `6dc7613`).
- **Version** : telemak **`0.6.47`** inchangée (perf non fixé).
- **Perf MoE-MTP** : acceptance **0.581** (exact), ~**0.5 tok/s** vs 17-19 AR.
- **Repro** : `gatherQuantizedMM` M3 isolé **0.47 ms** vs **32 ms** in-model.
- **Issues** : 1 filée (OdyssAI-X #62).
- **Cycles build→deploy `.29`** : ~8 (profiling + clean + redeploy).

## Fichiers clés

- `Sources/Telemak/Engine/MTP/HYV3MTP*.swift` — iterator + draft + layer + config + loader.
- fork `Libraries/MLXLLM/Models/{HYV3,HYV3MTP,MiniMaxM3}.swift`, `MLXLMCommon/SwitchLayers.swift`.
- `/tmp/moegatherbench/` — le repro isolé qui disculpe mlx-swift (gardé).
- Mémoires : `[[hy3_moe_mtp_perf_wall]]`, `[[telemak_minimax_m3_verdict]]`.

## TODO direct (par ordre)

1. **[5] Perf MoE-MTP** — chantier dédié à froid : profiler l'accès mémoire du forward
   in-model (hyp. experts-froids sur le pool 285 GB), seul moyen de cracker les 32 ms
   non-isolables. Tout le reste est éliminé.
2. **[2] OdyssAI-X #62** — dispatcher à l'agent (ajouter `ready`) : surfacer le modèle
   derrière `default` dans `/v1/models` + dashboard.
3. **[3] (si on tient à M3-Telemak)** — porter `cjk_lock` + sampling par-requête du
   `runner.py` vers le sampler Swift. Sinon laisser tomber (OdyssAI-X suffit).

## Lessons learned

- **S'arrêter quand c'est anormal** (Sophie, la leçon centrale) : deux instances, une
  fuite apparente, un smoke vide — autant de signaux à **remonter**, pas à traverser en
  pilote auto.
- **Le repro isolé avant de filer** : « on n'est pas avec des amateurs ». La rigueur a
  transformé un « bug mlx-swift confiant » en « pas notre bug » et évité une fausse issue
  + une journée d'upgrade MLX. Un négatif prouvé vaut mieux qu'un faux positif posté.
- **Le timing lazy ment** : `verify_s` semblait minuscule jusqu'à ce qu'un `eval()`
  forcé révèle 115 s. Toujours borner le graphe avant de lire un nombre MLX.
