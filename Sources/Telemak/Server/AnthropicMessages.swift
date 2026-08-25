import Foundation
import Hummingbird
import MLXLMCommon

/// POST /v1/messages — Anthropic-compatible surface.
///
/// V1 strategy: convert the request to the OpenAI-shaped internal flow,
/// generate, then convert the response back to Anthropic shape. Streaming
/// emits the Anthropic event sequence: `message_start`, `content_block_start`,
/// `content_block_delta`, `content_block_stop`, `message_delta`,
/// `message_stop`.
struct AnthropicMessagesHandler: Sendable {
    let registry: ModelRegistry
    let stats: StatsTracker
    let activity: ActivityTracker
    let sessionStore: SessionStore?

    func add(to router: Router<BasicRequestContext>) {
        router.post("/v1/messages") { request, context async throws -> Response in
            try await self.handle(request, context: context)
        }
    }

    func handle(_ request: Request, context: BasicRequestContext) async throws -> Response {
        let body = try await request.body.collect(upTo: 40 * 1024 * 1024)
        let payload: AnthropicMessagesRequest
        do {
            payload = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(buffer: body))
        } catch {
            return jsonError(.badRequest, type: "invalid_request_error", message: "JSON decode failed: \(error)")
        }

        guard let rawModelId = payload.model, !rawModelId.isEmpty else {
            return jsonError(.badRequest, type: "invalid_request_error", message: "missing 'model'")
        }
        let modelId = ModelLoader.canonicalIdentifier(rawModelId)

        let container: ModelContainer
        if let loaded = await registry.get(modelId) {
            container = loaded.container
        } else {
            return jsonError(.notFound, type: "not_found_error",
                              message: "model '\(modelId)' is not loaded. POST /admin/load first.")
        }

        var params = GenerateParameters()
        params.maxTokens = payload.maxTokens
        // B2 — engine defaults from the environment as the base; payload
        // fields below always win.
        if let t = ServerDefaults.temperature { params.temperature = t }
        if let p = ServerDefaults.topP { params.topP = p }
        if let k = ServerDefaults.topK { params.topK = k }
        if let t = payload.temperature { params.temperature = t }
        if let p = payload.topP { params.topP = p }
        if let k = payload.topK { params.topK = k }

        // Build templateContext for enable_thinking and reasoning_effort
        var templateContext: [String: any Sendable] = [:]
        let effectiveThinking = payload.enableThinking ?? ServerDefaults.enableThinking
        if let effectiveThinking {
            templateContext["enable_thinking"] = effectiveThinking
        }
        if let reasoningEffort = payload.reasoningEffort, !reasoningEffort.isEmpty {
            templateContext["reasoning_effort"] = reasoningEffort
        }
        let additionalContext: [String: any Sendable]? = templateContext.isEmpty ? nil : templateContext
        let sessionCacheScope = Self.sessionCacheScope(additionalContext)

        let system = payload.system?.asString
        let userPrompt = anthropicRenderPrompt(payload.messages)
        if userPrompt.isEmpty {
            return jsonError(.badRequest, type: "invalid_request_error",
                              message: "no user message to generate from")
        }
        let imageBatch: VisionImageBatch
        do {
            imageBatch = try await VisionInputs.collectAnthropicImages(from: payload.messages)
        } catch {
            return jsonError(.badRequest, type: "invalid_request_error", message: "\(error.localizedDescription)")
        }

        let sessionId = payload.sessionId ?? request.headers[.init("X-Session-Id")!]
        let cacheHit: URL? = await {
            guard let sessionId, let sessionStore else { return nil }
            return await sessionStore.hit(sessionId: sessionId, modelId: modelId, cacheScope: sessionCacheScope)
        }()
        let prompt: String = cacheHit != nil ? lastUserText(payload.messages) : userPrompt
        let effectiveSystem: String? = cacheHit != nil ? nil : system

        if payload.stream == true {
            return streamingResponse(
                container: container,
                params: params,
                cacheHit: cacheHit,
                effectiveSystem: effectiveSystem,
                prompt: prompt,
                images: imageBatch,
                modelId: modelId,
                sessionId: sessionId,
                additionalContext: additionalContext,
                sessionCacheScope: sessionCacheScope,
                stats: stats,
                sessionStore: sessionStore
            )
        }

        let session: ChatSession
        var cachedTokens = 0
        if let cacheHit {
            do {
                let (loaded, _) = try loadPromptCache(url: cacheHit)
                cachedTokens = loaded.first?.offset ?? 0
                session = ChatSession(container, instructions: nil, cache: loaded, generateParameters: params, additionalContext: additionalContext)
            } catch {
                session = ChatSession(container, instructions: system, generateParameters: params, additionalContext: additionalContext)
            }
        } else {
            session = ChatSession(container, instructions: system, generateParameters: params, additionalContext: additionalContext)
        }

        let activityId = await activity.begin(model: modelId, phase: .prefill)
        let genStart = Date()
        var completion = ""
        var info: GenerateCompletionInfo?
        do {
            await activity.setPhase(activityId, .decode)
            for try await gen in session.streamDetails(to: prompt, images: imageBatch.images, videos: []) {
                switch gen {
                case .chunk(let s):
                    await activity.incrementGeneratedTokens(activityId)
                    completion += s
                case .info(let i):
                    info = i
                    await activity.setGeneratedTokens(activityId, i.generationTokenCount)
                case .toolCall: break
                }
            }
        } catch {
            await activity.fail(activityId, error: "\(error)")
            return jsonError(.internalServerError, type: "api_error",
                              message: "model generation failed: \(error)")
        }
        let elapsed = Date().timeIntervalSince(genStart)

        if let sessionId, let sessionStore {
            await SessionCachePersistence.save(
                session: session,
                sessionId: sessionId,
                modelId: modelId,
                cacheScope: sessionCacheScope,
                sessionStore: sessionStore
            )
        }

        let promptTokens = info?.promptTokenCount ?? max(1, prompt.count / 4)
        let completionTokens = info?.generationTokenCount ?? max(1, completion.count / 4)
        await activity.setGeneratedTokens(activityId, completionTokens)
        await activity.finish(activityId)
        await stats.recordRequest(tokens: completionTokens, elapsedSeconds: elapsed)

        let response = AnthropicMessageResponse(
            id: "msg_\(UUID().uuidString.lowercased().prefix(24))",
            type: "message",
            role: "assistant",
            model: modelId,
            content: [AnthropicContentBlock(type: "text", text: completion)],
            stopReason: "end_turn",
            stopSequence: nil,
            usage: .init(
                inputTokens: promptTokens,
                outputTokens: completionTokens,
                cacheReadInputTokens: cachedTokens > 0 ? cachedTokens : nil
            )
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
        params: GenerateParameters,
        cacheHit: URL?,
        effectiveSystem: String?,
        prompt: String,
        images: VisionImageBatch,
        modelId: String,
        sessionId: String?,
        additionalContext: [String: any Sendable]?,
        sessionCacheScope: String,
        stats: StatsTracker,
        sessionStore: SessionStore?
    ) -> Response {
        let messageId = "msg_\(UUID().uuidString.lowercased().prefix(24))"

        let body = ResponseBody(contentLength: nil) { writer in
            let sse = SSEWriter(writer: writer)
            let session: ChatSession
            var cachedTokens = 0
            if let cacheHit {
                do {
                    let (loaded, _) = try loadPromptCache(url: cacheHit)
                    cachedTokens = loaded.first?.offset ?? 0
                    session = ChatSession(container, instructions: nil, cache: loaded, generateParameters: params, additionalContext: additionalContext)
                } catch {
                    session = ChatSession(container, instructions: effectiveSystem, generateParameters: params, additionalContext: additionalContext)
                }
            } else {
                session = ChatSession(container, instructions: effectiveSystem, generateParameters: params, additionalContext: additionalContext)
            }

            // Event encoding: `event: <name>\ndata: {<json>}\n\n`.
            func send(event: String, payload: [String: Any]) async throws {
                try await sse.write(event: event, data: payload)
            }

            let genStart = Date()
            var completion = ""
            var info: GenerateCompletionInfo?
            let activityId = await activity.begin(model: modelId, phase: .prefill)

            do {
                // message_start
                try await send(event: "message_start", payload: [
                    "type": "message_start",
                    "message": [
                        "id": messageId,
                        "type": "message",
                        "role": "assistant",
                        "model": modelId,
                        "content": [],
                        "stop_reason": NSNull(),
                        "stop_sequence": NSNull(),
                        "usage": ["input_tokens": 0, "output_tokens": 0],
                    ],
                ])

                // content_block_start (index 0, text block)
                try await send(event: "content_block_start", payload: [
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": ["type": "text", "text": ""],
                ])

                await activity.setPhase(activityId, .decode)
                for try await gen in session.streamDetails(to: prompt, images: images.images, videos: []) {
                    await activity.setPhase(activityId, .decode)
                    switch gen {
                    case .chunk(let piece):
                        await activity.incrementGeneratedTokens(activityId)
                        completion += piece
                        await activity.setPhase(activityId, .streaming)
                        try await send(event: "content_block_delta", payload: [
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": ["type": "text_delta", "text": piece],
                        ])
                    case .info(let i):
                        info = i
                        await activity.setGeneratedTokens(activityId, i.generationTokenCount)
                    case .toolCall:
                        break
                    }
                }

                try await send(event: "content_block_stop", payload: [
                    "type": "content_block_stop",
                    "index": 0,
                ])

                // `message_delta` carries the final usage block. Anthropic
                // spec puts `output_tokens` here; we also include
                // `input_tokens` (technically belongs to `message_start`,
                // but we don't know the prompt token count until
                // `GenerateCompletionInfo` arrives — populate at delta
                // time so clients reading either event see the right
                // total). `cache_read_input_tokens` reports session-cache
                // reuse for Odysseus/Companion metrics — only present when
                // > 0 so the shape stays minimal for cache-miss streams.
                //
                // Fallback: when `GenerateCompletionInfo` is nil (no
                // `info` chunk arrived before the stream ended), use a
                // char/4 estimate of the prompt rather than `0` — clients
                // reading Anthropic usage on cache-miss / info-less
                // streams previously got 0 input tokens, an obvious
                // undercount vs the OpenAI path's `prompt_tokens_details`.
                var usageBlock: [String: Any] = [
                    "input_tokens": info?.promptTokenCount ?? max(1, prompt.count / 4),
                    "output_tokens": info?.generationTokenCount ?? max(1, completion.count / 4),
                ]
                if cachedTokens > 0 {
                    usageBlock["cache_read_input_tokens"] = cachedTokens
                }
                try await send(event: "message_delta", payload: [
                    "type": "message_delta",
                    "delta": [
                        "stop_reason": "end_turn",
                        "stop_sequence": NSNull(),
                    ],
                    "usage": usageBlock,
                ])

                try await send(event: "message_stop", payload: [
                    "type": "message_stop",
                ])
            } catch {
                await activity.fail(activityId, error: "\(error)")
                try? await send(event: "error", payload: [
                    "type": "error",
                    "error": ["type": "api_error", "message": "\(error)"],
                ])
            }

            let elapsed = Date().timeIntervalSince(genStart)
            let observed = info?.generationTokenCount ?? max(1, completion.count / 4)
            await activity.setGeneratedTokens(activityId, observed)
            await activity.finish(activityId)
            await stats.recordRequest(tokens: observed, elapsedSeconds: elapsed)

            if let sessionId, let sessionStore {
                await SessionCachePersistence.save(
                    session: session,
                    sessionId: sessionId,
                    modelId: modelId,
                    cacheScope: sessionCacheScope,
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

    private func anthropicRenderPrompt(_ messages: [AnthropicMessage]) -> String {
        if messages.count == 1, messages[0].role == "user" {
            return messages[0].content.asPlainText
        }
        return messages.map {
            let text = $0.content.asPlainText
            return $0.role == "user" ? text : "[\($0.role)] \(text)"
        }.joined(separator: "\n\n")
    }

    private func lastUserText(_ messages: [AnthropicMessage]) -> String {
        if let last = messages.reversed().first(where: { $0.role == "user" }) {
            return last.content.asPlainText
        }
        return ""
    }

    private static func sessionCacheScope(_ context: [String: any Sendable]?) -> String {
        guard let context, !context.isEmpty else { return "" }
        return context.keys.sorted().map { key in
            let value = context[key].map { "\($0)" } ?? ""
            return "\(key)=\(value)"
        }.joined(separator: ";")
    }

    private func jsonError(_ status: HTTPResponse.Status, type: String, message: String) -> Response {
        let payload: [String: Any] = [
            "type": "error",
            "error": ["type": type, "message": message],
        ]
        let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }
}
