import Foundation
import Hummingbird
import MLX
import MLXLMCommon
import MLXRandom

/// POST /v1/chat/completions
struct ChatCompletionsHandler: Sendable {
    let registry: ModelRegistry
    let stats: StatsTracker
    let activity: ActivityTracker
    let sessionStore: SessionStore?
    let wiredMemory: WiredMemoryCoordinator?

    func add(to router: Router<BasicRequestContext>) {
        router.post("/v1/chat/completions") { request, context async throws -> Response in
            try await self.handle(request, context: context)
        }
    }

    func handle(_ request: Request, context: BasicRequestContext) async throws -> Response {
        let body = try await request.body.collect(upTo: 40 * 1024 * 1024)
        let payload: ChatCompletionRequest
        do {
            payload = try JSONDecoder().decode(ChatCompletionRequest.self, from: Data(buffer: body))
        } catch {
            return jsonError(.badRequest, code: "invalid_request_error", message: "JSON decode failed: \(error)")
        }

        guard let rawModelId = payload.model, !rawModelId.isEmpty else {
            return jsonError(.badRequest, code: "invalid_request_error", message: "missing 'model'")
        }
        let modelId = ModelLoader.canonicalIdentifier(rawModelId)

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
        if let kvBits = payload.kvBits { params.kvBits = kvBits }
        if let kvGroupSize = payload.kvGroupSize { params.kvGroupSize = kvGroupSize }
        if let quantizedKvStart = payload.quantizedKvStart { params.quantizedKVStart = quantizedKvStart }
        let stopSequences = payload.stop?.asArray ?? []

        let toolSpecs: [[String: any Sendable]]? = payload.tools.map { values in
            values.compactMap { value in
                if case .object(let dict) = value {
                    var spec: [String: any Sendable] = [:]
                    for (key, json) in dict {
                        if let sendable = json.toTemplateSendable() {
                            spec[key] = sendable
                        }
                    }
                    return spec
                }
                return nil
            }
        }

        var templateContext: [String: any Sendable] = [:]
        if let enableThinking = payload.enableThinking {
            templateContext["enable_thinking"] = enableThinking
        }
        if let reasoningEffort = payload.reasoningEffort, !reasoningEffort.isEmpty {
            templateContext["reasoning_effort"] = reasoningEffort
        }
        let additionalContext: [String: any Sendable]? = templateContext.isEmpty ? nil : templateContext

        let instructions = payload.system ?? extractSystem(from: payload.messages)
        let userPrompt = renderUserPrompt(from: payload.messages)
        if userPrompt.isEmpty {
            return jsonError(.badRequest, code: "invalid_request_error",
                              message: "no user message to generate from")
        }
        let imageBatch: VisionImageBatch
        do {
            imageBatch = try await VisionInputs.collectOpenAIImages(from: payload.messages)
        } catch {
            return jsonError(.badRequest, code: "invalid_request_error", message: "\(error.localizedDescription)")
        }
        let hasToolHistory = payload.messages.contains { message in
            message.role == "tool" || !(message.toolCalls?.isEmpty ?? true)
        }

        if hasToolHistory {
            return try await structuredMessagesResponse(
                payload: payload,
                container: container,
                modelId: modelId,
                params: params,
                stopSequences: stopSequences,
                imageBatch: imageBatch,
                toolSpecs: toolSpecs,
                additionalContext: additionalContext
            )
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

        // MTP fast path. If this main model has a paired speculative draft
        // and the request doesn't need tool calls or vision (the draft
        // can't propose either), route through the MTP iterator. Falls
        // through to the regular ChatSession path on any unsupported
        // feature so no caller is regressed by enabling a draft.
        //
        // The session prompt-cache (cacheHit) is also incompatible with
        // the iterator's `targetVerify`/`rollback` semantics today, so
        // we treat a cache hit as a fall-through. Sessions that span
        // multiple turns will get baseline perf on follow-up turns; the
        // first turn (no cache hit) benefits from MTP. That's a known
        // limitation tracked separately.
        if cacheHit == nil,
           let draftEntry = await mtpDraftIfEligible(
               for: modelId,
               toolSpecs: toolSpecs,
               imageBatch: imageBatch
           )
        {
            return try await runMTPChat(
                container: container,
                draftEntry: draftEntry,
                modelId: modelId,
                userPrompt: promptForGeneration,
                instructions: effectiveInstructions,
                params: params,
                stopSequences: stopSequences,
                stream: payload.stream == true,
                seed: payload.seed,
                cachedTokens: 0
            )
        }

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
                images: imageBatch,
                toolSpecs: toolSpecs,
                additionalContext: additionalContext,
                stats: stats,
                activity: activity,
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
                // Cache file unreadable / format mismatch — fall back to a
                // fresh session and full prefill. Surface the underlying
                // error to the LaunchAgent's stderr log so operators can
                // see if cache corruption is happening repeatedly.
                FileHandle.standardError.write(Data("[telemak.kv] loadPromptCache failed for \(cacheHit.lastPathComponent): \(error)\n".utf8))
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

        let activityId = await activity.begin(model: modelId, phase: .prefill)
        let genStart = Date()
        var completion = ""
        var stopChecker = StopChecker(stops: stopSequences)
        var stoppedEarly = false
        var info: GenerateCompletionInfo?
        var collectedToolCalls: [ChatToolCall] = []
        do {
            await activity.setPhase(activityId, .decode)
            try await runWithOptionalWiredLimit {
                for try await gen in session.streamDetails(to: promptForGeneration, images: imageBatch.images, videos: []) {
                    switch gen {
                    case .chunk(let s):
                        await activity.incrementGeneratedTokens(activityId)
                        let emitted = stopChecker.feed(s)
                        if !emitted.isEmpty { completion += emitted }
                        if stopChecker.hit {
                            stoppedEarly = true
                        }
                    case .info(let i):
                        info = i
                        await activity.setGeneratedTokens(activityId, i.generationTokenCount)
                    case .toolCall(let call):
                        collectedToolCalls.append(toolCallToChat(call))
                    }
                }
                if !stopChecker.hit {
                    completion += stopChecker.flushRemaining()
                }
            }
        } catch {
            await activity.fail(activityId, error: "\(error)")
            return jsonError(.internalServerError, code: "generation_failed",
                              message: "model generation failed: \(error)")
        }
        let genElapsed = Date().timeIntervalSince(genStart)

        if collectedToolCalls.isEmpty {
            let recovered = recoverTaggedToolCalls(from: completion)
            if !recovered.toolCalls.isEmpty {
                completion = recovered.content
                collectedToolCalls = recovered.toolCalls
            }
        }

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
        await activity.setGeneratedTokens(activityId, completionTokens)
        await activity.finish(activityId)
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

    private func structuredMessagesResponse(
        payload: ChatCompletionRequest,
        container: ModelContainer,
        modelId: String,
        params: GenerateParameters,
        stopSequences: [String],
        imageBatch: VisionImageBatch,
        toolSpecs: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) async throws -> Response {
        if !imageBatch.images.isEmpty {
            return jsonError(
                .badRequest,
                code: "unsupported_request_error",
                message: "tool-history requests with image content are not supported yet"
            )
        }

        let rawMessages = rawTemplateMessages(from: payload.messages, system: payload.system)
        if payload.stream == true {
            return streamingStructuredMessagesResponse(
                container: container,
                rawMessages: rawMessages,
                params: params,
                modelId: modelId,
                stopSequences: stopSequences,
                toolSpecs: toolSpecs,
                additionalContext: additionalContext
            )
        }

        let input = UserInput(
            messages: rawMessages,
            tools: toolSpecs,
            additionalContext: additionalContext
        )
        let activityId = await activity.begin(model: modelId, phase: .prefill)
        let genStart = Date()
        var completion = ""
        var stopChecker = StopChecker(stops: stopSequences)
        var info: GenerateCompletionInfo?
        var collectedToolCalls: [ChatToolCall] = []

        do {
            await activity.setPhase(activityId, .decode)
            let stream = try await makeRawGenerationStream(
                container: container,
                input: input,
                params: params,
                toolSpecs: toolSpecs
            )
            try await runWithOptionalWiredLimit {
                for await gen in stream {
                    switch gen {
                    case .chunk(let s):
                        await activity.incrementGeneratedTokens(activityId)
                        let emitted = stopChecker.feed(s)
                        if !emitted.isEmpty { completion += emitted }
                        if stopChecker.hit { break }
                    case .info(let i):
                        info = i
                        await activity.setGeneratedTokens(activityId, i.generationTokenCount)
                    case .toolCall(let call):
                        collectedToolCalls.append(toolCallToChat(call))
                    }
                }
                if !stopChecker.hit {
                    completion += stopChecker.flushRemaining()
                }
            }
        } catch {
            await activity.fail(activityId, error: "\(error)")
            return jsonError(
                .internalServerError,
                code: "generation_failed",
                message: "model generation failed: \(error)"
            )
        }

        let genElapsed = Date().timeIntervalSince(genStart)
        if collectedToolCalls.isEmpty {
            let recovered = recoverTaggedToolCalls(from: completion)
            if !recovered.toolCalls.isEmpty {
                completion = recovered.content
                collectedToolCalls = recovered.toolCalls
            }
        }

        if !collectedToolCalls.isEmpty {
            completion = cleanContentBeforeToolCall(completion)
        }

        let promptTokens = info?.promptTokenCount ?? max(1, rawMessages.description.count / 4)
        let completionTokens = info?.generationTokenCount ?? max(1, completion.count / 4)
        await activity.setGeneratedTokens(activityId, completionTokens)
        await activity.finish(activityId)
        await stats.recordRequest(tokens: completionTokens, elapsedSeconds: genElapsed)

        let usage = ChatCompletionResponse.Usage(
            promptTokens: promptTokens,
            completionTokens: completionTokens,
            totalTokens: promptTokens + completionTokens,
            promptTokensDetails: nil
        )
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

    private func streamingResponse(
        container: ModelContainer,
        instructions: String?,
        params: GenerateParameters,
        userPrompt: String,
        modelId: String,
        sessionId: String?,
        cacheHit: URL?,
        stopSequences: [String],
        images: VisionImageBatch,
        toolSpecs: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        stats: StatsTracker,
        activity: ActivityTracker,
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

            let activityId = await activity.begin(model: modelId, phase: .prefill)
            func send(_ chunk: ChatCompletionChunk) async throws {
                await activity.setPhase(activityId, .streaming)
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
            var pendingContent = ""
            var pendingPieces = 0
            let maxPiecesPerChunk = 10
            let maxCharactersPerChunk = 512

            func flushPendingContent(force: Bool = false) async throws {
                guard !pendingContent.isEmpty else { return }
                if !force,
                   pendingPieces < maxPiecesPerChunk,
                   pendingContent.count < maxCharactersPerChunk
                {
                    return
                }

                let content = pendingContent
                pendingContent.removeAll(keepingCapacity: true)
                pendingPieces = 0

                let chunk = ChatCompletionChunk(
                    id: id, object: "chat.completion.chunk",
                    created: created, model: modelId,
                    choices: [.init(index: 0, delta: .init(role: nil, content: content), finishReason: nil)]
                )
                try await send(chunk)
            }

            do {
                let role = ChatCompletionChunk(
                    id: id, object: "chat.completion.chunk",
                    created: created, model: modelId,
                    choices: [.init(index: 0, delta: .init(role: "assistant", content: nil), finishReason: nil)]
                )
                try await send(role)

                await activity.setPhase(activityId, .decode)
                try await runWithOptionalWiredLimit {
                    for try await gen in session.streamDetails(to: userPrompt, images: images.images, videos: []) {
                        await activity.setPhase(activityId, .decode)
                        switch gen {
                        case .chunk(let piece):
                            await activity.incrementGeneratedTokens(activityId)
                            let emit = stopChecker.feed(piece)
                            if !emit.isEmpty {
                                pendingContent += emit
                                pendingPieces += 1
                                try await flushPendingContent()
                            }
                        case .info(let i):
                            info = i
                            await activity.setGeneratedTokens(activityId, i.generationTokenCount)
                        case .toolCall(let call):
                            try await flushPendingContent(force: true)
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
                }
                if !stopChecker.hit {
                    let tail = stopChecker.flushRemaining()
                    if !tail.isEmpty {
                        pendingContent += tail
                        pendingPieces += 1
                    }
                }
                try await flushPendingContent(force: true)

                let finishReason = anyToolCalls ? "tool_calls" : "stop"
                let stop = ChatCompletionChunk(
                    id: id, object: "chat.completion.chunk",
                    created: created, model: modelId,
                    choices: [.init(index: 0, delta: .init(role: nil, content: nil), finishReason: finishReason)]
                )
                try await send(stop)

                // Standard OpenAI streaming `usage` chunk — emitted between
                // finish_reason and [DONE] so clients can render tok/s,
                // prompt/completion tokens, total. We always emit (spec gates
                // it on `stream_options.include_usage: true`, but clients
                // that don't read it discard silently — and Companion's
                // chat-meta row depends on this signal).
                //
                // Counts come from `GenerateCompletionInfo` when the model
                // emitted one before the iterator finished; otherwise we
                // fall back to a coarse character-based estimate matching
                // the non-stream path.
                let promptTokens = info?.promptTokenCount ?? max(1, userPrompt.count / 4)
                let completionTokens = info?.generationTokenCount ?? 1
                var usageBlock: [String: Any] = [
                    "prompt_tokens": promptTokens,
                    "completion_tokens": completionTokens,
                    "total_tokens": promptTokens + completionTokens,
                ]
                if cachedTokens > 0 {
                    usageBlock["prompt_tokens_details"] = ["cached_tokens": cachedTokens]
                }
                let usageChunk: [String: Any] = [
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": modelId,
                    "choices": [],
                    "usage": usageBlock,
                ]
                if let payload = try? JSONSerialization.data(withJSONObject: usageChunk) {
                    var buf = ByteBuffer()
                    buf.writeString("data: ")
                    buf.writeBytes(payload)
                    buf.writeString("\n\n")
                    try await writer.write(buf)
                }

                try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
            } catch {
                await activity.fail(activityId, error: "\(error)")
                let payload = #"{"error":{"message":"streaming aborted: \#(error)","type":"generation_failed"}}"#
                try? await writer.write(ByteBuffer(string: "data: \(payload)\n\n"))
            }
            let elapsed = Date().timeIntervalSince(genStart)
            let observedTokens = info?.generationTokenCount ?? 1
            await activity.setGeneratedTokens(activityId, observedTokens)
            await activity.finish(activityId)
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

    private func streamingStructuredMessagesResponse(
        container: ModelContainer,
        rawMessages: [[String: any Sendable]],
        params: GenerateParameters,
        modelId: String,
        stopSequences: [String],
        toolSpecs: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) -> Response {
        let id = "chatcmpl-\(UUID().uuidString.lowercased())"
        let created = Int(Date().timeIntervalSince1970)

        let body = ResponseBody(contentLength: nil) { writer in
            let encoder = JSONEncoder()
            func send(_ chunk: ChatCompletionChunk) async throws {
                let data = try encoder.encode(chunk)
                var buffer = ByteBuffer()
                buffer.writeString("data: ")
                buffer.writeBytes(data)
                buffer.writeString("\n\n")
                try await writer.write(buffer)
            }

            let activityId = await activity.begin(model: modelId, phase: .prefill)
            let genStart = Date()
            var stopChecker = StopChecker(stops: stopSequences)
            var info: GenerateCompletionInfo?
            var anyToolCalls = false
            var pendingContent = ""
            var pendingPieces = 0
            let maxPiecesPerChunk = 10
            let maxCharactersPerChunk = 512

            func flushPendingContent(force: Bool = false) async throws {
                guard !pendingContent.isEmpty else { return }
                if !force,
                   pendingPieces < maxPiecesPerChunk,
                   pendingContent.count < maxCharactersPerChunk
                {
                    return
                }

                let content = pendingContent
                pendingContent.removeAll(keepingCapacity: true)
                pendingPieces = 0
                try await send(.init(
                    id: id,
                    object: "chat.completion.chunk",
                    created: created,
                    model: modelId,
                    choices: [.init(index: 0, delta: .init(role: nil, content: content), finishReason: nil)]
                ))
            }

            do {
                try await send(.init(
                    id: id,
                    object: "chat.completion.chunk",
                    created: created,
                    model: modelId,
                    choices: [.init(index: 0, delta: .init(role: "assistant", content: nil), finishReason: nil)]
                ))

                let input = UserInput(
                    messages: rawMessages,
                    tools: toolSpecs,
                    additionalContext: additionalContext
                )
                let stream = try await makeRawGenerationStream(
                    container: container,
                    input: input,
                    params: params,
                    toolSpecs: toolSpecs
                )

                await activity.setPhase(activityId, .decode)
                try await runWithOptionalWiredLimit {
                    for await gen in stream {
                        switch gen {
                        case .chunk(let piece):
                            await activity.incrementGeneratedTokens(activityId)
                            let emit = stopChecker.feed(piece)
                            if !emit.isEmpty {
                                pendingContent += emit
                                pendingPieces += 1
                                try await flushPendingContent()
                            }
                            if stopChecker.hit { break }
                        case .info(let i):
                            info = i
                            await activity.setGeneratedTokens(activityId, i.generationTokenCount)
                        case .toolCall(let call):
                            try await flushPendingContent(force: true)
                            anyToolCalls = true
                            try await send(.init(
                                id: id,
                                object: "chat.completion.chunk",
                                created: created,
                                model: modelId,
                                choices: [.init(
                                    index: 0,
                                    delta: .init(role: nil, content: nil, toolCalls: [toolCallToChat(call)]),
                                    finishReason: nil
                                )]
                            ))
                        }
                    }
                    if !stopChecker.hit {
                        let tail = stopChecker.flushRemaining()
                        if !tail.isEmpty {
                            pendingContent += tail
                            pendingPieces += 1
                        }
                    }
                }
                try await flushPendingContent(force: true)

                let finishReason = anyToolCalls ? "tool_calls" : "stop"
                try await send(.init(
                    id: id,
                    object: "chat.completion.chunk",
                    created: created,
                    model: modelId,
                    choices: [.init(index: 0, delta: .init(role: nil, content: nil), finishReason: finishReason)]
                ))

                let promptTokens = info?.promptTokenCount ?? max(1, rawMessages.description.count / 4)
                let completionTokens = info?.generationTokenCount ?? 1
                let usageChunk: [String: Any] = [
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": modelId,
                    "choices": [],
                    "usage": [
                        "prompt_tokens": promptTokens,
                        "completion_tokens": completionTokens,
                        "total_tokens": promptTokens + completionTokens,
                    ],
                ]
                if let payload = try? JSONSerialization.data(withJSONObject: usageChunk) {
                    var buf = ByteBuffer()
                    buf.writeString("data: ")
                    buf.writeBytes(payload)
                    buf.writeString("\n\n")
                    try await writer.write(buf)
                }
                try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
            } catch {
                await activity.fail(activityId, error: "\(error)")
                let payload = #"{"error":{"message":"streaming aborted: \#(error)","type":"generation_failed"}}"#
                try? await writer.write(ByteBuffer(string: "data: \(payload)\n\n"))
            }

            let elapsed = Date().timeIntervalSince(genStart)
            let observedTokens = info?.generationTokenCount ?? 1
            await activity.setGeneratedTokens(activityId, observedTokens)
            await activity.finish(activityId)
            await stats.recordRequest(tokens: observedTokens, elapsedSeconds: elapsed)
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
        } catch ChatSessionError.noCacheAvailable {
            // No cache to save (empty generation, tool-only response, etc.).
            // Not an error — just skip persistence.
            try? FileManager.default.removeItem(at: url)
        } catch {
            // Genuinely unexpected: log so operators can see if disk is full,
            // permissions broke, etc. Don't fail the response.
            FileHandle.standardError.write(Data("[telemak.kv] saveCache failed for session \(sessionId): \(error)\n".utf8))
            try? FileManager.default.removeItem(at: url)
        }
    }

    /// Run `body` under a wired-memory active ticket when a coordinator is
    /// configured. Pass-through otherwise — keeps the non-prod / unit-test
    /// paths uncluttered. Body is NOT @Sendable so it can capture
    /// non-Sendable state (ChatSession, mutable locals from the request
    /// handler).
    private func runWithOptionalWiredLimit<R>(_ body: () async throws -> R) async throws -> R {
        guard let wiredMemory else { return try await body() }
        let ticket = await wiredMemory.makeInferenceTicket(
            workspaceBytes: WiredMemoryCoordinator.defaultWorkspaceBytes
        )
        return try await WiredMemoryTicket.withWiredLimit(ticket, body)
    }

    private func makeRawGenerationStream(
        container: ModelContainer,
        input: UserInput,
        params: GenerateParameters,
        toolSpecs: [[String: any Sendable]]?
    ) async throws -> AsyncStream<Generation> {
        try await container.perform(nonSendable: input) { context, input in
            let lmInput = try await context.processor.prepare(input: input)
            return try MLXLMCommon.generate(
                input: lmInput,
                parameters: params,
                context: context,
                tools: toolSpecs
            )
        }
    }

    private func rawTemplateMessages(
        from messages: [ChatMessage],
        system: String?
    ) -> [[String: any Sendable]] {
        var result: [[String: any Sendable]] = []
        if let system, !messages.contains(where: { $0.role == "system" }) {
            result.append(["role": "system", "content": system])
        }
        result.append(contentsOf: messages.map(rawTemplateMessage))
        return result
    }

    private func rawTemplateMessage(_ message: ChatMessage) -> [String: any Sendable] {
        var raw: [String: any Sendable] = ["role": message.role]
        raw["content"] = message.content?.asPlainText ?? ""

        if let toolCallId = message.toolCallId {
            raw["tool_call_id"] = toolCallId
        }
        if let name = message.name {
            raw["name"] = name
        }
        if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
            raw["tool_calls"] = toolCalls.map(rawTemplateToolCall)
        }
        return raw
    }

    private func rawTemplateToolCall(_ call: ChatToolCall) -> [String: any Sendable] {
        [
            "id": call.id,
            "type": call.type,
            "function": [
                "name": call.function.name,
                "arguments": decodeToolArgumentsForTemplate(call.function.arguments),
            ] as [String: any Sendable],
        ]
    }

    private func decodeToolArgumentsForTemplate(_ arguments: String) -> any Sendable {
        guard let data = arguments.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data)
        else {
            return arguments
        }
        return sanitizeJSONObjectForTemplate(object) ?? arguments
    }

    private func sanitizeJSONObjectForTemplate(_ object: Any) -> (any Sendable)? {
        switch object {
        case let dict as [String: Any]:
            var result: [String: any Sendable] = [:]
            for (key, value) in dict {
                if let clean = sanitizeJSONObjectForTemplate(value) {
                    result[key] = clean
                }
            }
            return result
        case let array as [Any]:
            return array.compactMap { sanitizeJSONObjectForTemplate($0) }
        case let value as String:
            return value
        case let value as Bool:
            return value
        case let value as Int:
            return value
        case let value as Double:
            return value
        case let value as Float:
            return Double(value)
        case _ as NSNull:
            return nil
        default:
            return String(describing: object)
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

    private func recoverTaggedToolCalls(from text: String) -> (content: String, toolCalls: [ChatToolCall]) {
        var remaining = text
        var calls: [ChatToolCall] = []

        while let start = remaining.range(of: "<minimax:tool_call>"),
              let end = remaining.range(
                  of: "</minimax:tool_call>",
                  range: start.upperBound ..< remaining.endIndex
              )
        {
            let block = String(remaining[start.lowerBound ..< end.upperBound])
            if let call = parseMiniMaxToolCall(block) {
                calls.append(call)
                remaining.removeSubrange(start.lowerBound ..< end.upperBound)
            } else {
                break
            }
        }

        return (
            remaining.trimmingCharacters(in: .whitespacesAndNewlines),
            calls
        )
    }

    private func cleanContentBeforeToolCall(_ content: String) -> String {
        var clean = content.trimmingCharacters(in: .whitespacesAndNewlines)
        if clean == "</think>" {
            return ""
        }
        if clean.hasSuffix("</think>") {
            clean.removeLast("</think>".count)
            clean = clean.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return clean
    }

    private func parseMiniMaxToolCall(_ block: String) -> ChatToolCall? {
        guard let name = firstRegexGroup(#"<invoke\s+name="([^"]+)">"#, in: block) else {
            return nil
        }

        var args: [String: String] = [:]
        let pattern = #"<parameter\s+name="([^"]+)">([\s\S]*?)</parameter>"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let nsRange = NSRange(block.startIndex ..< block.endIndex, in: block)
        for match in regex.matches(in: block, range: nsRange) {
            guard match.numberOfRanges == 3,
                  let keyRange = Range(match.range(at: 1), in: block),
                  let valueRange = Range(match.range(at: 2), in: block)
            else { continue }
            let key = String(block[keyRange])
            let value = String(block[valueRange]).trimmingCharacters(in: .whitespacesAndNewlines)
            args[key] = value
        }

        let argsString: String
        if let data = try? JSONSerialization.data(withJSONObject: args),
           let json = String(data: data, encoding: .utf8) {
            argsString = json
        } else {
            argsString = "{}"
        }

        return ChatToolCall(
            id: "call_\(UUID().uuidString.lowercased().prefix(24))",
            type: "function",
            function: ChatToolCallFunction(name: name, arguments: argsString)
        )
    }

    private func firstRegexGroup(_ pattern: String, in text: String) -> String? {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let nsRange = NSRange(text.startIndex ..< text.endIndex, in: text)
        guard let match = regex.firstMatch(in: text, range: nsRange),
              match.numberOfRanges > 1,
              let range = Range(match.range(at: 1), in: text)
        else { return nil }
        return String(text[range])
    }

    private func lastUserMessageOnly(_ messages: [ChatMessage]) -> String {
        if let last = messages.reversed().first(where: { $0.role == "user" }) {
            return last.content?.asPlainText ?? ""
        }
        return ""
    }

    private func extractSystem(from messages: [ChatMessage]) -> String? {
        let systemParts = messages.filter { $0.role == "system" }.compactMap { $0.content?.asPlainText }
        return systemParts.isEmpty ? nil : systemParts.joined(separator: "\n\n")
    }

    private func renderUserPrompt(from messages: [ChatMessage]) -> String {
        let nonSystem = messages.filter { $0.role != "system" }
        if nonSystem.count == 1, nonSystem[0].role == "user" {
            return nonSystem[0].content?.asPlainText ?? ""
        }
        return nonSystem.map { msg in
            let body = msg.content?.asPlainText ?? ""
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
