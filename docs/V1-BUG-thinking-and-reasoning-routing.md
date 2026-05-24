# Telemak V1 — Bug : `enable_thinking` ignoré + reasoning_content over-router

> Bug bloquant pour l'usage Companion → Odysseus → Telemak.
> Symptôme côté utilisateur : "ghost" — l'utilisateur prompte, rien
> n'apparaît dans le chat. Cause réelle : Telemak met TOUT en
> `reasoning_content` et JAMAIS rien en `content`, donc Companion (qui
> affiche `content`) ne voit aucun token visible.
>
> Découvert 2026-05-24 sur `telemak-max64` avec le modèle
> `inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit` proxyé via Odysseus
> (`http://192.168.86.141:8000/v1/chat/completions` cluster
> `kind=telemak`).

## Repro

```bash
curl -s -m 15 -X POST http://192.168.86.50:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"inferencerlabs/Qwen3.5-35B-A3B-MLX-9bit",
       "stream":true,
       "max_tokens":150,
       "enable_thinking":false,
       "messages":[{"role":"user","content":"say hi in french"}]}'
```

Observé :
```
data: {... "delta": {"reasoning_content": "T"} ...}
data: {... "delta": {"reasoning_content": "hinking "} ...}
data: {... "delta": {"reasoning_content": "P"} ...}
data: {... "delta": {"reasoning_content": "ro"} ...}
...  (continue indéfiniment en reasoning_content)
```

Attendu :
```
data: {... "delta": {"content": "Bonjour"} ...}
data: {... "delta": {"content": " !"} ...}
...
```

Notes importantes :
- Aucune balise `<think>` n'est émise par le modèle. Pas de `<think>`,
  pas de `</think>`. Telemak décide tout seul que les premiers tokens
  ("Thinking Process:") ressemblent à du thinking et les route en
  `reasoning_content`.
- `enable_thinking: false` dans le body est **complètement ignoré**.
- Même comportement en stream et en non-stream. En non-stream avec
  `max_tokens: 20`, le content est rempli (le détecteur n'a pas eu le
  temps de kicker) ; avec `max_tokens >= 50`, content reste vide.

## Cause

Deux bugs distincts, à corriger ensemble :

### Bug 1 — `enable_thinking: false` non câblé

Le handler `/v1/chat/completions` côté Telemak ne lit pas le champ
`enable_thinking` du body, donc il ne le passe pas au chat template
Jinja lors du render du prompt.

Conséquence : le chat template Qwen3.5/Qwen3.6 rend par défaut avec
`enable_thinking=true`, ce qui pré-conditionne le modèle à émettre du
thinking. Le modèle ne sortira jamais de ce mode pour cette requête.

### Bug 2 — auto-routing heuristique vers `reasoning_content`

Quelque part dans la stack Telemak (probablement dans le formatter de
chunks OpenAI-compatible), il y a un détecteur qui décide qu'un token
va en `reasoning_content` plutôt qu'en `content` **sans qu'une balise
`<think>` ait été observée dans la sortie**. C'est ce détecteur qui
fait que les premiers tokens "Thinking Process:" déclenchent le
routing reasoning, et comme le modèle n'émet jamais `</think>`, tout
y reste à vie.

C'est le bug grave : même avec `enable_thinking: false` correctement
câblé (bug 1 fixé), si un jour un modèle décide d'écrire le mot
"thinking" en début de réponse pour une autre raison, on retomberait
dans le même piège.

## Fix attendu

### Côté Bug 1 — câbler `enable_thinking`

Dans le handler `/v1/chat/completions` (probablement
`Sources/Telemak/Routes/ChatCompletions.swift` ou équivalent — adapter
au layout réel du repo) :

1. Parser le champ optionnel `enable_thinking: Bool?` (default `nil` =
   laisser le template décider, ce qui revient à `true` pour Qwen3.x).
2. Le passer au render du chat template comme kwarg Jinja.

Si la couche `mlx-swift-lm` expose une signature du genre :

```swift
tokenizer.applyChatTemplate(
    messages: messages,
    addGenerationPrompt: true,
    additionalContext: ["enable_thinking": false]
)
```

…utiliser ce paramètre. Sinon (si la lib Swift ne supporte pas les
kwargs), fallback équivalent : prepender la string `/no_think ` au
contenu du dernier message user juste avant le render — c'est la
convention Qwen documentée pour désactiver le mode thinking côté
template.

Test : après fix, la repro ci-dessus avec `enable_thinking: false` doit
sortir directement du `content` sans la moindre mention de
"Thinking Process".

### Côté Bug 2 — virer l'auto-routing

Trouver l'endroit qui décide `delta.reasoning_content = token` vs
`delta.content = token`. Probablement dans le streaming converter
mlx-swift-lm-token → OpenAI-chunk.

Règle correcte (à implémenter) :

```
state = OUTSIDE_THINK            # initial state
buffer = ""                      # short buffer pour détecter <think>/</think>
                                 # qui peuvent arriver en plusieurs tokens

pour chaque token émis par le modèle :
    buffer += token
    parser_consume_tags(buffer):
        - si on voit "<think>" en entier dans buffer, state = INSIDE_THINK,
          on consomme la balise du buffer, on flushe ce qui était avant en
          content
        - si on voit "</think>" en entier dans buffer ET state == INSIDE_THINK,
          state = OUTSIDE_THINK, on flushe ce qui était avant en reasoning_content,
          on consomme la balise
    si buffer ne contient pas de balise partielle en cours :
        flush buffer dans delta.content si state == OUTSIDE_THINK
        flush buffer dans delta.reasoning_content si state == INSIDE_THINK
        buffer = ""
```

Le buffer court (quelques chars max) sert UNIQUEMENT à ne pas couper
une balise `<think>` qui arrive en plusieurs tokens (`<`, `think`,
`>`). On ne décide JAMAIS du routing sur le contenu sémantique des
tokens.

Référence d'implémentation existante côté Odysseus :
`scripts/api.py:_split_think_stream` (Python, même logique).

### Comportement attendu après les deux fix

| Cas | enable_thinking | Modèle émet | content | reasoning_content |
|---|---|---|---|---|
| 1 | `false` | "Bonjour !" | "Bonjour !" | (vide) |
| 2 | `true` (default Qwen) | "Bonjour !" | "Bonjour !" | (vide) |
| 3 | `true` | "`<think>`raisonne`</think>`Bonjour !" | "Bonjour !" | "raisonne" |
| 4 | `true` | "Thinking Process: 1." (pas de balise) | "Thinking Process: 1." | (vide) |

Le cas 4 est celui qui foire aujourd'hui — actuellement il route tout
en reasoning_content. Avec les deux fix, il restera en content (le
client décide d'afficher ou pas).

## Tests de validation

1. **stream + `enable_thinking: false`** : repro curl ci-dessus, doit
   sortir du content immédiatement, aucun token en reasoning_content.

2. **stream + `enable_thinking: true`** sur Qwen3.5 avec un prompt
   long ("write 500 words about X") : doit produire du content
   pendant tout le stream. Si le modèle émet `<think>...</think>`
   quelque part, ce bloc va dans reasoning_content, le reste en
   content.

3. **non-stream + `enable_thinking: false`** : `content` rempli,
   `reasoning_content` vide ou absent.

4. **non-régression** : un modèle qui émet vraiment `<think>...</think>`
   (genre Hy3, MiniMax) doit continuer à voir ses balises retirées de
   `content` et le bloc thinking exposé en `reasoning_content`.

5. **Companion bout-en-bout** : ouvrir un chat sur `telemak-max64`,
   prompter "hello", la réponse doit s'afficher (pas de "ghost"). Tester
   à la fois avec et sans la case "Show thinking" du settings (selon
   l'UI Companion).

## Lien Companion (pour contexte)

Le request body envoyé par Companion à Odysseus contient déjà
`enable_thinking: false` :

```jsonc
{
  "model": "telemak-max64",
  "stream": true,
  "temperature": 0.7,
  "max_tokens": 8192,
  "enable_thinking": false,        // ← déjà câblé Companion-side
  "session_id": "...",
  "messages": [...],
  "tools": [...]
}
```

Odysseus le forwarde tel quel à Telemak via le proxy
`_telemak_proxy_chat_completion` (`scripts/api.py`). Donc côté
Companion + côté Odysseus c'est OK, le câblage manquant est
strictement dans Telemak.

## Done criteria

- [ ] Field `enable_thinking` parsé dans le handler
  `/v1/chat/completions`
- [ ] Câblé au chat template (kwarg Jinja ou `/no_think` prepend, au
  choix selon ce que mlx-swift-lm expose)
- [ ] Auto-routing reasoning_content viré, remplacé par tag-based
  splitter
- [ ] Les 5 tests de validation passent
- [ ] Smoke Companion bout-en-bout sur `telemak-max64` : prompter
  "hello", réponse s'affiche

## Hors scope de ce bug

- Performance Telemak (le `avg_tok_s_recent: 7` observé en même temps
  laisse penser que le build a régressé vers Debug ou que multi-model
  perturbe le throughput — à traiter séparément si confirmé).
- Auto-pairing main↔draft model pour MTP (V2, cf
  `V2-MTP-DRAFT-PORT.md`).
