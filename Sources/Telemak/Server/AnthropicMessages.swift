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
    let sessionStore: SessionStore?

    func add(to router: Router<BasicRequestContext>) {
        router.post("/v1/messages") { request, context async throws -> Response in
            try await self.handle(request, context: context)
        }
    }

    func handle(_ request: Request, context: BasicRequestContext) async throws -> Response {
        let body = try await request.body.collect(upTo: 1 << 20)
        let payload: AnthropicMessagesRequest
        do {
            payload = try JSONDecoder().decode(AnthropicMessagesRequest.self, from: Data(buffer: body))
        } catch {
            return jsonError(.badRequest, type: "invalid_request_error", message: "JSON decode failed: \(error)")
        }

        guard let modelId = payload.model, !modelId.isEmpty else {
            return jsonError(.badRequest, type: "invalid_request_error", message: "missing 'model'")
        }

        let container: ModelContainer
        if let loaded = await registry.get(modelId) {
            container = loaded.container
        } else {
            return jsonError(.notFound, type: "not_found_error",
                              message: "model '\(modelId)' is not loaded. POST /admin/load first.")
        }

        var params = GenerateParameters()
        params.maxTokens = payload.maxTokens
        if let t = payload.temperature { params.temperature = t }
        if let p = payload.topP { params.topP = p }
        if let k = payload.topK { params.topK = k }

        let system = payload.system?.asString
        let userPrompt = anthropicRenderPrompt(payload.messages)
        if userPrompt.isEmpty {
            return jsonError(.badRequest, type: "invalid_request_error",
                              message: "no user message to generate from")
        }

        let sessionId = payload.sessionId ?? request.headers[.init("X-Session-Id")!]
        let cacheHit: URL? = await {
            guard let sessionId, let sessionStore else { return nil }
            return await sessionStore.hit(sessionId: sessionId, modelId: modelId)
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
                modelId: modelId,
                sessionId: sessionId,
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
                session = ChatSession(container, instructions: nil, cache: loaded, generateParameters: params)
            } catch {
                session = ChatSession(container, instructions: system, generateParameters: params)
            }
        } else {
            session = ChatSession(container, instructions: system, generateParameters: params)
        }
        _ = cachedTokens // future: surface via usage extension

        let genStart = Date()
        var completion = ""
        var info: GenerateCompletionInfo?
        do {
            for try await gen in session.streamDetails(to: prompt, images: [], videos: []) {
                switch gen {
                case .chunk(let s): completion += s
                case .info(let i): info = i
                case .toolCall: break
                }
            }
        } catch {
            return jsonError(.internalServerError, type: "api_error",
                              message: "model generation failed: \(error)")
        }
        let elapsed = Date().timeIntervalSince(genStart)

        if let sessionId, let sessionStore {
            await saveSessionCache(
                session: session,
                sessionId: sessionId,
                modelId: modelId,
                sessionStore: sessionStore
            )
        }

        let promptTokens = info?.promptTokenCount ?? max(1, prompt.count / 4)
        let completionTokens = info?.generationTokenCount ?? max(1, completion.count / 4)
        await stats.recordRequest(tokens: completionTokens, elapsedSeconds: elapsed)

        let response = AnthropicMessageResponse(
            id: "msg_\(UUID().uuidString.lowercased().prefix(24))",
            type: "message",
            role: "assistant",
            model: modelId,
            content: [AnthropicContentBlock(type: "text", text: completion)],
            stopReason: "end_turn",
            stopSequence: nil,
            usage: .init(inputTokens: promptTokens, outputTokens: completionTokens)
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
        modelId: String,
        sessionId: String?,
        stats: StatsTracker,
        sessionStore: SessionStore?
    ) -> Response {
        let messageId = "msg_\(UUID().uuidString.lowercased().prefix(24))"

        let body = ResponseBody(contentLength: nil) { writer in
            let session: ChatSession
            if let cacheHit {
                do {
                    let (loaded, _) = try loadPromptCache(url: cacheHit)
                    session = ChatSession(container, instructions: nil, cache: loaded, generateParameters: params)
                } catch {
                    session = ChatSession(container, instructions: effectiveSystem, generateParameters: params)
                }
            } else {
                session = ChatSession(container, instructions: effectiveSystem, generateParameters: params)
            }

            // Event encoding: `event: <name>\ndata: {<json>}\n\n`.
            func send(event: String, payload: [String: Any]) async throws {
                let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
                var buf = ByteBuffer()
                buf.writeString("event: ")
                buf.writeString(event)
                buf.writeString("\n")
                buf.writeString("data: ")
                buf.writeBytes(data)
                buf.writeString("\n\n")
                try await writer.write(buf)
            }

            let genStart = Date()
            var completion = ""
            var info: GenerateCompletionInfo?

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

                for try await gen in session.streamDetails(to: prompt, images: [], videos: []) {
                    switch gen {
                    case .chunk(let piece):
                        completion += piece
                        try await send(event: "content_block_delta", payload: [
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": ["type": "text_delta", "text": piece],
                        ])
                    case .info(let i):
                        info = i
                    case .toolCall:
                        break
                    }
                }

                try await send(event: "content_block_stop", payload: [
                    "type": "content_block_stop",
                    "index": 0,
                ])

                try await send(event: "message_delta", payload: [
                    "type": "message_delta",
                    "delta": [
                        "stop_reason": "end_turn",
                        "stop_sequence": NSNull(),
                    ],
                    "usage": [
                        "output_tokens": info?.generationTokenCount ?? 0,
                    ],
                ])

                try await send(event: "message_stop", payload: [
                    "type": "message_stop",
                ])
            } catch {
                try? await send(event: "error", payload: [
                    "type": "error",
                    "error": ["type": "api_error", "message": "\(error)"],
                ])
            }

            let elapsed = Date().timeIntervalSince(genStart)
            let observed = info?.generationTokenCount ?? max(1, completion.count / 4)
            await stats.recordRequest(tokens: observed, elapsedSeconds: elapsed)

            if let sessionId, let sessionStore {
                await saveSessionCache(
                    session: session, sessionId: sessionId, modelId: modelId,
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
                sessionId: sessionId, modelId: modelId,
                cacheURL: url, byteSize: size
            )
        } catch {
            try? FileManager.default.removeItem(at: url)
        }
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
