import Foundation
import Hummingbird
import MLX
import MLXLMCommon

// MARK: - Streaming response paths for /v1/chat/completions
//
// Split out of `ChatCompletions.swift` per issue #58. Two streaming
// paths exist:
//   - `streamingResponse` — the canonical OpenAI chat stream for plain
//     text / tool-less / single-history requests. Uses `ChatSession`.
//   - `streamingStructuredMessagesResponse` — for tool-history requests
//     where the body has `tool` messages or `tool_calls`. A1 KV-bridge:
//     receives a fully assembled `ChatSession` (KV cache rehydrated by
//     the caller) and streams only the delta prompt through it.
//
// Both share `SSEWriter` for the wire framing (so the byte-level
// output is identical to the pre-refactor code) and
// `SessionCachePersistence` for the post-stream save.

/// Single-consumer handoff box for values that are Sendable in
/// practice (created by the caller, owned and used sequentially by one
/// streaming body) but not declared so. Mirrors `VisionImageBatch`.
private struct UncheckedSendableBox<T>: @unchecked Sendable {
    let value: T
}

extension ChatCompletionsHandler {

    // MARK: - plain streaming (no tool history)

    func streamingResponse(
        container: ModelContainer,
        instructions: String?,
        params: GenerateParameters,
        userPrompt: String,
        modelId: String,
        sessionId: String?,
        cacheHit: URL?,
        sessionCacheScope: String,
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
            let sse = SSEWriter(writer: writer)
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
                try await sse.write(data: chunk)
            }

            let genStart = Date()
            var stopChecker = StopChecker(stops: stopSequences)
            var thinkRepair = ThinkRepair()
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
                            let emit = thinkRepair.feed(stopChecker.feed(piece))
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
                    let tail = thinkRepair.feed(stopChecker.flushRemaining())
                    if !tail.isEmpty {
                        pendingContent += tail
                        pendingPieces += 1
                    }
                }
                try await flushPendingContent(force: true)

                let finishReason = MTPStopPolicy.finishReason(
                    hasToolCalls: anyToolCalls,
                    stoppedEarly: stopChecker.hit,
                    stopReason: info?.stopReason
                )
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
                try await sse.write(data: usageChunk)

                try await sse.writeDone()
            } catch {
                await activity.fail(activityId, error: "\(error)")
                try await sse.writeError(message: "streaming aborted: \(error)")
            }
            let elapsed = Date().timeIntervalSince(genStart)
            let observedTokens = info?.generationTokenCount ?? 1
            await activity.setGeneratedTokens(activityId, observedTokens)
            await activity.finish(activityId)
            await stats.recordRequest(tokens: observedTokens, elapsedSeconds: elapsed)

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

    // MARK: - structured-messages streaming (with tool history)
    //
    // A1 — KV-bridge: the session arrives fully assembled by the caller
    // (cache-hit KV rehydrate or history rehydrate), so this path only
    // streams the delta prompt through `session.streamDetails`, reports
    // `cached_tokens` in the usage chunk, and persists the grown cache
    // for the next tool turn. Mirrors `streamingResponse` above.

    func streamingStructuredMessagesResponse(
        container: ModelContainer,
        session: ChatSession,
        prompt: String,
        cachedTokens: Int,
        role: Chat.Message.Role,
        params: GenerateParameters,
        modelId: String,
        stopSequences: [String],
        toolSpecs: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        sessionId: String?,
        sessionStore: SessionStore?,
        sessionCacheScope: String
    ) -> Response {
        let id = "chatcmpl-\(UUID().uuidString.lowercased())"
        let created = Int(Date().timeIntervalSince1970)

        let boxedSession = UncheckedSendableBox(value: session)
        let body = ResponseBody(contentLength: nil) { writer in
            let session = boxedSession.value
            let sse = SSEWriter(writer: writer)
            func send(_ chunk: ChatCompletionChunk) async throws {
                try await sse.write(data: chunk)
            }

            let activityId = await activity.begin(model: modelId, phase: .prefill)
            let genStart = Date()
            var stopChecker = StopChecker(stops: stopSequences)
            var thinkRepair = ThinkRepair()
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

                await activity.setPhase(activityId, .decode)
                try await runWithOptionalWiredLimit {
                    for try await gen in session.streamDetails(to: prompt, role: role, images: [], videos: []) {
                        switch gen {
                        case .chunk(let piece):
                            await activity.incrementGeneratedTokens(activityId)
                            let emit = thinkRepair.feed(stopChecker.feed(piece))
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

                let finishReason = MTPStopPolicy.finishReason(
                    hasToolCalls: anyToolCalls,
                    stoppedEarly: stopChecker.hit,
                    stopReason: info?.stopReason
                )
                try await send(.init(
                    id: id,
                    object: "chat.completion.chunk",
                    created: created,
                    model: modelId,
                    choices: [.init(index: 0, delta: .init(role: nil, content: nil), finishReason: finishReason)]
                ))

                let promptTokens = info?.promptTokenCount ?? max(1, prompt.count / 4)
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
                try await sse.write(data: usageChunk)
                try await sse.writeDone()
            } catch {
                await activity.fail(activityId, error: "\(error)")
                try await sse.writeError(message: "streaming aborted: \(error)")
            }

            let elapsed = Date().timeIntervalSince(genStart)
            let observedTokens = info?.generationTokenCount ?? 1
            await activity.setGeneratedTokens(activityId, observedTokens)
            await activity.finish(activityId)
            await stats.recordRequest(tokens: observedTokens, elapsedSeconds: elapsed)

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
}
