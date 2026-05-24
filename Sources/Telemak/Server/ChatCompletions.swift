import Foundation
import Hummingbird
import MLXLMCommon
import MLXRandom

/// POST /v1/chat/completions
struct ChatCompletionsHandler: Sendable {
    let registry: ModelRegistry
    let stats: StatsTracker
    let sessionStore: SessionStore?

    func add(to router: Router<BasicRequestContext>) {
        router.post("/v1/chat/completions") { request, context async throws -> Response in
            try await self.handle(request, context: context)
        }
    }

    func handle(_ request: Request, context: BasicRequestContext) async throws -> Response {
        let body = try await request.body.collect(upTo: 1 << 20)   // 1 MB cap
        let payload: ChatCompletionRequest
        do {
            payload = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(buffer: body))
        } catch {
            return jsonError(.badRequest, code: "invalid_request_error", message: "JSON decode failed: \(error)")
        }

        guard let modelId = payload.model, !modelId.isEmpty else {
            return jsonError(.badRequest, code: "invalid_request_error", message: "missing 'model'")
        }

        let container: ModelContainer
        if let loaded = await registry.get(modelId) {
            container = loaded.container
        } else {
            let ready = await registry.loadedModelIds
            return modelNotLoadedError(requested: modelId, ready: ready)
        }

        if let seed = payload.seed {
            MLXRandom.seed(seed)
        }

        var params = GenerateParameters()
        if let maxTokens = payload.maxTokens { params.maxTokens = maxTokens }
        if let temperature = payload.temperature { params.temperature = temperature }
        if let topP = payload.topP { params.topP = topP }
        if let topK = payload.topK { params.topK = topK }
        if let minP = payload.minP { params.minP = minP }
        if let repetitionPenalty = payload.repetitionPenalty { params.repetitionPenalty = repetitionPenalty }
        let stopSequences = payload.stop?.asArray ?? []

        let toolSpecs: [[String: any Sendable]]? = payload.tools.map { values in
            values.compactMap { value in
                if case .object(let dict) = value {
                    return dict.mapValues { $0.toSendable() }
                }
                return nil
            }
        }

        var additionalContext: [String: any Sendable]? = nil
        if let enableThinking = payload.enableThinking {
            additionalContext = ["enable_thinking": enableThinking]
        }

        let instructions = payload.system ?? extractSystem(from: payload.messages)
        let userPrompt = renderUserPrompt(from: payload.messages)
        if userPrompt.isEmpty {
            return jsonError(.badRequest, code: "invalid_request_error",
                              message: "no user message to generate from")
        }

        // session_id from body OR X-Session-Id header. Cache hit if SessionStore
        // has an entry for (session_id, modelId) — we then prefill ONLY the
        // latest user message instead of the full history.
        let sessionId = payload.sessionId ?? request.headers[.init("X-Session-Id")!]
        let cacheHit: URL? = await {
            guard let sessionId, let sessionStore else { return nil }
            return await sessionStore.hit(sessionId: sessionId, modelId: modelId)
        }()
        let promptForGeneration: String = (cacheHit != nil) ? lastUserMessageOnly(payload.messages) : userPrompt
        let effectiveInstructions: String? = (cacheHit != nil) ? nil : instructions

        if payload.stream == true {
            return streamingResponse(
                container: container,
                instructions: effectiveInstructions,
                params: params,
                userPrompt: promptForGeneration,
                modelId: modelId,
                sessionId: sessionId,
                cacheHit: cacheHit,
                stopSequences: stopSequences,
                toolSpecs: toolSpecs,
                additionalContext: additionalContext,
                stats: stats,
                sessionStore: sessionStore
            )
        }

        // Non-streaming path. Use streamDetails to get accurate token counts
        // from GenerateCompletionInfo, and apply stop-sequence trimming on
        // the way out.
        let session: ChatSession
        var cachedTokens = 0
        if let cacheHit {
            do {
                let (loaded, _) = try loadPromptCache(url: cacheHit)
                cachedTokens = loaded.first?.offset ?? 0
                session = ChatSession(
                    container,
                    instructions: nil,
                    cache: loaded,
                    generateParameters: params,
                    additionalContext: additionalContext,
                    tools: toolSpecs
                )
            } catch {
                session = ChatSession(
                    container, instructions: instructions,
                    generateParameters: params,
                    additionalContext: additionalContext, tools: toolSpecs
                )
            }
        } else {
            session = ChatSession(
                container, instructions: instructions,
                generateParameters: params,
                additionalContext: additionalContext, tools: toolSpecs
            )
        }

        let genStart = Date()
        var completion = ""
        var stopChecker = StopChecker(stops: stopSequences)
        var stoppedEarly = false
        var info: GenerateCompletionInfo?
        var collectedToolCalls: [ChatToolCall] = []
        do {
            for try await gen in session.streamDetails(to: promptForGeneration, images: [], videos: []) {
                switch gen {
                case .chunk(let s):
                    let emitted = stopChecker.feed(s)
                    if !emitted.isEmpty { completion += emitted }
                    if stopChecker.hit {
                        stoppedEarly = true
                    }
                case .info(let i):
                    info = i
                case .toolCall(let call):
                    collectedToolCalls.append(toolCallToChat(call))
                }
            }
            if !stopChecker.hit {
                completion += stopChecker.flushRemaining()
            }
        } catch {
            return jsonError(.internalServerError, code: "generation_failed",
                              message: "model generation failed: \(error)")
        }
        let genElapsed = Date().timeIntervalSince(genStart)

        if let sessionId, let sessionStore {
            await saveSessionCache(
                session: session,
                sessionId: sessionId,
                modelId: modelId,
                sessionStore: sessionStore
            )
        }

        let promptTokens = info?.promptTokenCount ?? max(1, (promptForGeneration.count + (effectiveInstructions?.count ?? 0)) / 4)
        let completionTokens = info?.generationTokenCount ?? max(1, completion.count / 4)
        await stats.recordRequest(tokens: completionTokens, elapsedSeconds: genElapsed)

        let usage = ChatCompletionResponse.Usage(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            totalTokens: promptTokens + completionTokens,
            promptTokensDetails: cachedTokens > 0 ? .init(cachedTokens: cachedTokens) : nil
        )

        let _ = stoppedEarly
        let finishReason = collectedToolCalls.isEmpty ? "stop" : "tool_calls"

        let response = ChatCompletionResponse(
            id: "chatcmpl-\(UUID().uuidString.lowercased())",
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: modelId,
            choices: [
                .init(
                    index: 0,
                    message: ChatMessage(
                        role: "assistant",
                        content: collectedToolCalls.isEmpty ? completion : (completion.isEmpty ? nil : completion),
                        toolCalls: collectedToolCalls.isEmpty ? nil : collectedToolCalls
                    ),
                    finishReason: finishReason
                )
            ],
            usage: usage
        )

        let data = try JSONEncoder().encode(response)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }

    // MARK: - Streaming

    private func streamingResponse(
        container: ModelContainer,
        instructions: String?,
        params: GenerateParameters,
        userPrompt: String,
        modelId: String,
        sessionId: String?,
        cacheHit: URL?,
        stopSequences: [String],
        toolSpecs: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        stats: StatsTracker,
        sessionStore: SessionStore?
    ) -> Response {
        let id = "chatcmpl-\(UUID().uuidString.lowercased())"
        let created = Int(Date().timeIntervalSince1970)

        let body = ResponseBody(contentLength: nil) { writer in
            let encoder = JSONEncoder()
            let session: ChatSession
            var cachedTokens = 0
            if let cacheHit {
                do {
                    let (loaded, _) = try loadPromptCache(url: cacheHit)
                    cachedTokens = loaded.first?.offset ?? 0
                    session = ChatSession(
                        container, instructions: nil, cache: loaded,
                        generateParameters: params,
                        additionalContext: additionalContext, tools: toolSpecs
                    )
                } catch {
                    session = ChatSession(
                        container, instructions: instructions,
                        generateParameters: params,
                        additionalContext: additionalContext, tools: toolSpecs
                    )
                }
            } else {
                session = ChatSession(
                    container, instructions: instructions,
                    generateParameters: params,
                    additionalContext: additionalContext, tools: toolSpecs
                )
            }

            func send(_ chunk: ChatCompletionChunk) async throws {
                let data = try encoder.encode(chunk)
                var buffer = ByteBuffer()
                buffer.writeString("data: ")
                buffer.writeBytes(data)
                buffer.writeString("\n\n")
                try await writer.write(buffer)
            }

            let genStart = Date()
            var stopChecker = StopChecker(stops: stopSequences)
            var info: GenerateCompletionInfo?
            var anyToolCalls = false

            do {
                let role = ChatCompletionChunk(
                    id: id, object: "chat.completion.chunk",
                    created: created, model: modelId,
                    choices: [.init(index: 0, delta: .init(role: "assistant", content: nil), finishReason: nil)]
                )
                try await send(role)

                for try await gen in session.streamDetails(to: userPrompt, images: [], videos: []) {
                    switch gen {
                    case .chunk(let piece):
                        let emit = stopChecker.feed(piece)
                        if !emit.isEmpty {
                            let chunk = ChatCompletionChunk(
                                id: id, object: "chat.completion.chunk",
                                created: created, model: modelId,
                                choices: [.init(index: 0, delta: .init(role: nil, content: emit), finishReason: nil)]
                            )
                            try await send(chunk)
                        }
                    case .info(let i):
                        info = i
                    case .toolCall(let call):
                        anyToolCalls = true
                        let chatCall = self.toolCallToChat(call)
                        let chunk = ChatCompletionChunk(
                            id: id, object: "chat.completion.chunk",
                            created: created, model: modelId,
                            choices: [.init(
                                index: 0,
                                delta: .init(role: nil, content: nil, toolCalls: [chatCall]),
                                finishReason: nil
                            )]
                        )
                        try await send(chunk)
                    }
                    if stopChecker.hit { break }
                }
                if !stopChecker.hit {
                    let tail = stopChecker.flushRemaining()
                    if !tail.isEmpty {
                        let chunk = ChatCompletionChunk(
                            id: id, object: "chat.completion.chunk",
                            created: created, model: modelId,
                            choices: [.init(index: 0, delta: .init(role: nil, content: tail), finishReason: nil)]
                        )
                        try await send(chunk)
                    }
                }

                let finishReason = anyToolCalls ? "tool_calls" : "stop"
                let stop = ChatCompletionChunk(
                    id: id, object: "chat.completion.chunk",
                    created: created, model: modelId,
                    choices: [.init(index: 0, delta: .init(role: nil, content: nil), finishReason: finishReason)]
                )
                try await send(stop)

                if cachedTokens > 0 {
                    // Surface the cache-hit signal in a usage-style trailing
                    // chunk. Not standard OpenAI SSE shape but Companion's
                    // StatsRow reads x_telemak_usage when present.
                    let usageChunk: [String: Any] = [
                        "id": id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": modelId,
                        "choices": [],
                        "x_telemak_usage": [
                            "prompt_tokens_details": ["cached_tokens": cachedTokens],
                        ],
                    ]
                    if let payload = try? JSONSerialization.data(withJSONObject: usageChunk) {
                        var buf = ByteBuffer()
                        buf.writeString("data: ")
                        buf.writeBytes(payload)
                        buf.writeString("\n\n")
                        try await writer.write(buf)
                    }
                }

                try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
            } catch {
                let payload = #"{"error":{"message":"streaming aborted: \#(error)","type":"generation_failed"}}"#
                try? await writer.write(ByteBuffer(string: "data: \(payload)\n\n"))
            }
            let elapsed = Date().timeIntervalSince(genStart)
            let observedTokens = info?.generationTokenCount ?? 1
            await stats.recordRequest(tokens: observedTokens, elapsedSeconds: elapsed)

            if let sessionId, let sessionStore {
                await saveSessionCache(
                    session: session,
                    sessionId: sessionId,
                    modelId: modelId,
                    sessionStore: sessionStore
                )
            }

            try await writer.finish(nil)
        }

        return Response(
            status: .ok,
            headers: [
                .contentType: "text/event-stream",
                .cacheControl: "no-cache",
                .connection: "keep-alive",
            ],
            body: body
        )
    }

    // MARK: - Helpers

    /// Save the live cache from `session` to a fresh URL and register it in
    /// `sessionStore`. Best-effort: failures (e.g. session.saveCache throws
    /// ChatSessionError.noCacheAvailable on empty sessions) are swallowed
    /// so a save miss never breaks the response.
    private func saveSessionCache(
        session: ChatSession,
        sessionId: String,
        modelId: String,
        sessionStore: SessionStore
    ) async {
        let url = await sessionStore.nextCacheURL(for: sessionId)
        do {
            try await session.saveCache(to: url)
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
            await sessionStore.update(
                sessionId: sessionId,
                modelId: modelId,
                cacheURL: url,
                byteSize: size
            )
        } catch {
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// Convert mlx-swift-lm's `ToolCall` (`{function: {name, arguments:
    /// [String: JSONValue]}}`) to the OpenAI wire shape
    /// (`{id, type:"function", function:{name, arguments:"<json-string>"}}`).
    private func toolCallToChat(_ call: ToolCall) -> ChatToolCall {
        // `function.arguments` is `[String: MLXLMCommon.JSONValue]`, which is
        // mlx-swift-lm's local JSONValue (not ours). Serialize via its
        // `anyValue` representation.
        let argsAny = call.function.arguments.mapValues { $0.anyValue }
        let argsString: String
        if let data = try? JSONSerialization.data(withJSONObject: argsAny),
           let s = String(data: data, encoding: .utf8) {
            argsString = s
        } else {
            argsString = "{}"
        }
        return ChatToolCall(
            id: "call_\(UUID().uuidString.lowercased().prefix(24))",
            type: "function",
            function: ChatToolCallFunction(name: call.function.name, arguments: argsString)
        )
    }

    private func lastUserMessageOnly(_ messages: [ChatMessage]) -> String {
        if let last = messages.reversed().first(where: { $0.role == "user" }) {
            return last.content ?? ""
        }
        return ""
    }

    private func extractSystem(from messages: [ChatMessage]) -> String? {
        let systemParts = messages.filter { $0.role == "system" }.compactMap(\.content)
        return systemParts.isEmpty ? nil : systemParts.joined(separator: "\n\n")
    }

    private func renderUserPrompt(from messages: [ChatMessage]) -> String {
        let nonSystem = messages.filter { $0.role != "system" }
        if nonSystem.count == 1, nonSystem[0].role == "user" {
            return nonSystem[0].content ?? ""
        }
        return nonSystem.map { msg in
            let body = msg.content ?? ""
            return msg.role == "user" ? body : "[\(msg.role)] \(body)"
        }.joined(separator: "\n\n")
    }

    private func modelNotLoadedError(requested: String, ready: [String]) -> Response {
        let payload: [String: Any] = [
            "error": [
                "type": "model_not_loaded",
                "code": "model_not_loaded",
                "message": "model '\(requested)' is not loaded. POST /admin/load first.",
                "ready_models": ready,
            ]
        ]
        let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        return Response(
            status: .notFound,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }

    private func jsonError(_ status: HTTPResponse.Status, code: String, message: String) -> Response {
        struct Err: Encodable {
            struct Inner: Encodable {
                let message: String
                let type: String
                let code: String
            }
            let error: Inner
        }
        let payload = Err(error: .init(message: message, type: code, code: code))
        let data = (try? JSONEncoder().encode(payload)) ?? Data("{}".utf8)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }
}
