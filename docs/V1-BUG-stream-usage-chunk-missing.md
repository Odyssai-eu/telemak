# Telemak V1 — Bug : `usage` chunk jamais émis en streaming

> Bug fonctionnel : le client (Companion, dashboard Odysseus, n'importe
> quel consumer OpenAI-compatible) ne peut pas afficher
> `prompt_tokens` / `completion_tokens` / `total_tokens` / `tok/s` pour
> les requêtes streaming Telemak, parce que Telemak ne les envoie
> jamais dans le SSE stream.
>
> Découvert 2026-05-24 par the operator qui voit dans la barre meta du chat
> Companion seulement `Duration: 8.25s · Chunks: 345 · Model: ...` au
> lieu de la ligne complète avec tokens + tok/s.

## Repro

```bash
curl -s -m 30 -N -X POST http://<telemak-host>:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model":"inferencerlabs/Qwen3.6-35B-A3B-MLX-9bit",
    "stream":true,
    "enable_thinking":false,
    "max_tokens":50,
    "stream_options":{"include_usage":true},
    "messages":[{"role":"user","content":"hello"}]
  }' | tail -5
```

Observé (dernier chunk avant `[DONE]`) :
```
data: {"created":...,"object":"chat.completion.chunk","id":"...","model":"...",
       "choices":[{"delta":{},"finish_reason":"stop","index":0}]}

data: [DONE]
```

Attendu (spec OpenAI `stream_options.include_usage`) :
```
data: {"created":...,"object":"chat.completion.chunk","id":"...","model":"...",
       "choices":[{"delta":{},"finish_reason":"stop","index":0}]}

data: {"created":...,"object":"chat.completion.chunk","id":"...","model":"...",
       "choices":[],
       "usage":{"prompt_tokens":N,"completion_tokens":M,"total_tokens":N+M}}

data: [DONE]
```

Note : en mode non-stream, l'usage block est bien retourné (voir
`Sources/Telemak/Server/ChatCompletions.swift:188-214`). C'est
uniquement le path streaming qui le drop.

## Cause

`Sources/Telemak/Server/ChatCompletions.swift` ligne ~340 :

```swift
let finishReason = anyToolCalls ? "tool_calls" : "stop"
let stop = ChatCompletionChunk(
    id: id, object: "chat.completion.chunk",
    created: created, model: modelId,
    choices: [.init(index: 0, delta: .init(role: nil, content: nil), finishReason: finishReason)]
)
try await send(stop)

if cachedTokens > 0 {
    // x_telemak_usage : non-standard, seulement cached_tokens
    let usageChunk: [String: Any] = [
        ...
        "x_telemak_usage": [
            "prompt_tokens_details": ["cached_tokens": cachedTokens],
        ],
    ]
    ...
}

try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
```

Le chunk standard avec `prompt_tokens` / `completion_tokens` /
`total_tokens` n'est jamais construit ni envoyé. Le seul chunk
usage-ish est conditionnel sur `cachedTokens > 0` et n'expose qu'un
sous-champ propriétaire.

Les compteurs eux-mêmes EXISTENT : `info?.generationTokenCount` est
lu ligne ~373 pour `stats.recordRequest`, donc Telemak connaît
`completion_tokens`. Pour `prompt_tokens`, c'est `promptTokens`
calculé plus tôt (utilisé dans le path non-stream ligne 188-214).

## Fix attendu

Entre le chunk `finish_reason` et le `[DONE]`, émettre un chunk
standard OpenAI portant `usage` :

```swift
// Standard OpenAI streaming usage chunk. Sent unconditionally so
// clients can render tok/s, input/output tokens, total. The OpenAI
// spec gates this on `stream_options.include_usage: true` but it
// doesn't hurt to always emit — clients that don't read it discard
// silently. Match the field shape exactly so any OpenAI-compatible
// client (Companion, dashboard, SDKs) reads it without quirks.
let elapsed = Date().timeIntervalSince(genStart)
let observedCompletionTokens = info?.generationTokenCount ?? 0
let totalTokens = promptTokens + observedCompletionTokens
let usagePayload: [String: Any] = [
    "id": id,
    "object": "chat.completion.chunk",
    "created": created,
    "model": modelId,
    "choices": [],
    "usage": [
        "prompt_tokens": promptTokens,
        "completion_tokens": observedCompletionTokens,
        "total_tokens": totalTokens,
        // Optional: keep x_telemak_usage extension as nested for cache info
        "prompt_tokens_details": ["cached_tokens": cachedTokens],
    ],
]
if let payload = try? JSONSerialization.data(withJSONObject: usagePayload) {
    var buf = ByteBuffer()
    buf.writeString("data: ")
    buf.writeBytes(payload)
    buf.writeString("\n\n")
    try await writer.write(buf)
}

try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
```

Le fix supprime ou remplace le bloc `if cachedTokens > 0 { ... }` qui
émettait `x_telemak_usage` séparément — soit on garde ces infos en
extension à l'intérieur du `usage` standard (`prompt_tokens_details`,
shape OpenAI), soit on les laisse en `x_telemak_usage` à côté.

## Comportement attendu après fix

Repro ci-dessus doit retourner :
```
data: {"...","choices":[{"delta":{},"finish_reason":"stop","index":0}]}

data: {"...","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":9,"total_tokens":21,"prompt_tokens_details":{"cached_tokens":0}}}

data: [DONE]
```

Et côté Companion la barre meta du chat doit afficher l'équivalent de
ce qu'elle montre pour les modèles Argo/Hades : `Duration: 8.25s ·
345 tok / 8.25s = 41.8 tok/s · 12 in / 333 out · Model: ...`.

## Optionnel — honorer `stream_options.include_usage`

La spec OpenAI dit que le chunk usage est seulement émis si le client
envoie `stream_options.include_usage: true`. Le plus simple est
d'émettre TOUJOURS le chunk usage — c'est plus propre pour les clients
et inoffensif pour ceux qui ignorent ce champ. Si tu préfères honorer
strictement la spec :

```swift
let includeUsage = (body.streamOptions?.includeUsage ?? false)
if includeUsage {
    // emit usage chunk
}
```

Companion enverra `stream_options.include_usage: true` côté client
quand ce fix sera live (PR à faire dans `thecompai/app` après).

## Tests de validation

1. **Stream + Qwen3.6** : repro ci-dessus, voir un chunk
   `"usage":{"prompt_tokens":N,...}` juste avant `[DONE]`.
2. **Stream + tool call** : `finish_reason: "tool_calls"` doit aussi
   être suivi du chunk usage.
3. **Stream avec `stream_options.include_usage: false` ou absent** :
   selon décision (toujours-émettre OU honorer-spec) — documenter.
4. **Non-régression non-stream** : usage block dans la réponse JSON
   inchangé, pareil qu'avant.
5. **Smoke Companion** : la barre meta du chat affiche tokens + tok/s
   pour les modèles Telemak comme pour les autres clusters.

## Done criteria

- [ ] Chunk SSE standard `usage` émis entre `finish_reason` et
  `[DONE]` dans `ChatCompletions.swift` stream path
- [ ] Mêmes champs que la réponse non-stream :
  `prompt_tokens` / `completion_tokens` / `total_tokens`
- [ ] Garde la donnée `cached_tokens` quelque part (fold dans
  `prompt_tokens_details` ou conserver `x_telemak_usage`)
- [ ] Repro curl voit le chunk usage avant `[DONE]`
- [ ] Companion bout-en-bout : meta line complète sur chat Telemak

## Hors scope

- Anthropic `/v1/messages` stream — vérifier si le même trou existe
  dans `Sources/Telemak/Server/AnthropicMessages.swift:200` où on
  voit `"usage": ["input_tokens": 0, "output_tokens": 0]` — pas
  populated non plus. Probablement un fix jumeau à faire en même
  temps.
- TTFT (time-to-first-token) : champ supplémentaire utile mais pas
  bloquant pour V1 — viendrait en V1.x.
