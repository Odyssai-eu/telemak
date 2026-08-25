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
        // B2 — engine defaults from the environment (set by the app at
        // spawn) as the base; payload fields below always win.
        if let t = ServerDefaults.temperature { params.temperature = t }
        if let p = ServerDefaults.topP { params.topP = p }
        if let k = ServerDefaults.topK { params.topK = k }
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
        // B2 — payload wins; otherwise the engine default from the
        // environment; never force a value when both are absent so the
        // model's template default applies.
        let effectiveThinking = payload.enableThinking ?? ServerDefaults.enableThinking
        if let effectiveThinking {
            templateContext["enable_thinking"] = effectiveThinking
        }
        if let reasoningEffort = payload.reasoningEffort, !reasoningEffort.isEmpty {
            templateContext["reasoning_effort"] = reasoningEffort
        }
        let additionalContext: [String: any Sendable]? = templateContext.isEmpty ? nil : templateContext
        let sessionCacheScope = Self.sessionCacheScope(additionalContext)

        let instructions = Self.instructionsWithReasoningGuard(
            base: payload.system ?? extractSystem(from: payload.messages),
            modelId: modelId,
            effort: payload.reasoningEffort
        )
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
                additionalContext: additionalContext,
                sessionStore: sessionStore,
                request: request
            )
        }

        // session_id from body OR X-Session-Id header. Cache hit if SessionStore
        // has an entry for (session_id, modelId) — we then prefill ONLY the
        // latest user message instead of the full history.
        let sessionId = payload.sessionId ?? request.headers[.init("X-Session-Id")!]
        let cacheHit: URL? = await {
            guard let sessionId, let sessionStore else { return nil }
            return await sessionStore.hit(
                sessionId: sessionId,
                modelId: modelId,
                cacheScope: sessionCacheScope
            )
        }()
        let promptForGeneration: String = (cacheHit != nil) ? lastUserMessageOnly(payload.messages) : userPrompt
        let effectiveInstructions: String? = (cacheHit != nil) ? nil : instructions

        // MTP fast path. If this main model has a paired speculative draft
        // and the request carries no vision input (images can't be expressed
        // through the iterator's token-only interface), route through the
        // MTP iterator. Tool specs ARE allowed (since A2): the chat template
        // renders them into the prompt and the iterator processes them as
        // regular tokens — only vision (and a session prompt-cache hit,
        // see below) falls through to ChatSession, so no caller is
        // regressed by enabling a draft.
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
                cachedTokens: 0,
                toolSpecs: toolSpecs
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
                sessionCacheScope: sessionCacheScope,
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
            let recovered = Self.recoverToolCalls(from: completion)
            if !recovered.toolCalls.isEmpty {
                completion = recovered.content
                collectedToolCalls = recovered.toolCalls
            }
        }
        completion = ThinkRepair.repairComplete(completion)

        if let sessionId, let sessionStore {
            await SessionCachePersistence.save(
                session: session,
                sessionId: sessionId,
                modelId: modelId,
                cacheScope: sessionCacheScope,
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

        let finishReason = MTPStopPolicy.finishReason(
            hasToolCalls: !collectedToolCalls.isEmpty,
            stoppedEarly: stoppedEarly,
            stopReason: info?.stopReason
        )

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

    // MARK: - Tool-history response (non-stream + dispatch to streaming)

    private func structuredMessagesResponse(
        payload: ChatCompletionRequest,
        container: ModelContainer,
        modelId: String,
        params: GenerateParameters,
        stopSequences: [String],
        imageBatch: VisionImageBatch,
        toolSpecs: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
        sessionStore: SessionStore?,
        request: Request
    ) async throws -> Response {
        if !imageBatch.images.isEmpty {
            return jsonError(
                .badRequest,
                code: "unsupported_request_error",
                message: "tool-history requests with image content are not supported yet"
            )
        }

        let instructions = Self.instructionsWithReasoningGuard(
            base: payload.system ?? extractSystem(from: payload.messages),
            modelId: modelId,
            effort: payload.reasoningEffort
        )

        // A1 — KV-bridge: reuse SessionStore to prefill only the delta
        let sessionId = payload.sessionId ?? request.headers[.init("X-Session-Id")!]
        let cacheHit: URL? = await {
            guard let sessionId, let sessionStore else { return nil }
            return await sessionStore.hit(
                sessionId: sessionId,
                modelId: modelId,
                cacheScope: Self.sessionCacheScope(additionalContext)
            )
        }()

        let session: ChatSession
        var cachedTokens = 0
        var prompt: String = ""

        if let cacheHit {
            do {
                let (loaded, _) = try loadPromptCache(url: cacheHit)
                cachedTokens = loaded.first?.offset ?? 0
                session = ChatSession(
                    container, instructions: nil, cache: loaded,
                    generateParameters: params,
                    additionalContext: additionalContext, tools: toolSpecs
                )
                // Cache hit: only send the new message (delta), not the full transcript
                prompt = payload.messages.last?.content?.asPlainText ?? ""
            } catch {
                FileHandle.standardError.write(Data("[telemak.kv] loadPromptCache failed: \(error)\n".utf8))
                let chatMessages = toChatMessages(payload.messages, system: payload.system)
                let lastMsg = chatMessages.last
                session = ChatSession(
                    container, instructions: nil,
                    history: Array(chatMessages.dropLast()),
                    generateParameters: params,
                    additionalContext: additionalContext, tools: toolSpecs
                )
                prompt = lastMsg?.content ?? ""
            }
        } else {
            let chatMessages = toChatMessages(payload.messages, system: payload.system)
            let lastMsg = chatMessages.last
            session = ChatSession(
                container, instructions: nil,
                history: Array(chatMessages.dropLast()),
                generateParameters: params,
                additionalContext: additionalContext, tools: toolSpecs
            )
            prompt = lastMsg?.content ?? ""
        }

        let lastRole: Chat.Message.Role = (payload.messages.last?.role == "tool") ? .tool : .user

        if payload.stream == true {
            return streamingStructuredMessagesResponse(
                container: container,
                session: session,
                prompt: prompt,
                cachedTokens: cachedTokens,
                role: lastRole,
                params: params,
                modelId: modelId,
                stopSequences: stopSequences,
                toolSpecs: toolSpecs,
                additionalContext: additionalContext,
                sessionId: sessionId,
                sessionStore: sessionStore,
                sessionCacheScope: Self.sessionCacheScope(additionalContext)
            )
        }

        let activityId = await activity.begin(model: modelId, phase: .prefill)
        let genStart = Date()
        var completion = ""
        var stopChecker = StopChecker(stops: stopSequences)
        var info: GenerateCompletionInfo?
        var collectedToolCalls: [ChatToolCall] = []

        do {
            await activity.setPhase(activityId, .decode)
            try await runWithOptionalWiredLimit {
                for try await gen in session.streamDetails(to: prompt, role: lastRole, images: [], videos: []) {
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
            let recovered = Self.recoverToolCalls(from: completion)
            if !recovered.toolCalls.isEmpty {
                completion = recovered.content
                collectedToolCalls = recovered.toolCalls
            }
        }

        if !collectedToolCalls.isEmpty {
            completion = cleanContentBeforeToolCall(completion)
        }
        completion = ThinkRepair.repairComplete(completion)

        // A1 — Save KV cache for next turn
        if let sessionId, let sessionStore {
            await SessionCachePersistence.save(
                session: session,
                sessionId: sessionId,
                modelId: modelId,
                cacheScope: Self.sessionCacheScope(additionalContext),
                sessionStore: sessionStore
            )
        }

        let promptTokens = info?.promptTokenCount ?? max(1, prompt.count / 4)
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
        let finishReason = MTPStopPolicy.finishReason(
            hasToolCalls: !collectedToolCalls.isEmpty,
            stoppedEarly: stopChecker.hit,
            stopReason: info?.stopReason
        )
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

    // MARK: - Helpers

    private static func sessionCacheScope(_ context: [String: any Sendable]?) -> String {
        guard let context, !context.isEmpty else { return "" }
        return context.keys.sorted().map { key in
            let value = context[key].map { "\($0)" } ?? ""
            return "\(key)=\(value)"
        }.joined(separator: ";")
    }

    /// Run `body` under a wired-memory active ticket when a coordinator is
    /// configured. Pass-through otherwise — keeps the non-prod / unit-test
    /// paths uncluttered. Body is NOT @Sendable so it can capture
    /// non-Sendable state (ChatSession, mutable locals from the request
    /// handler).
    ///
    /// `internal` (not `private`) because `ChatCompletionsStreaming.swift`
    /// (extension on this handler) calls it from the streaming path.
    func runWithOptionalWiredLimit<R>(_ body: () async throws -> R) async throws -> R {
        guard let wiredMemory else { return try await body() }
        let ticket = await wiredMemory.makeInferenceTicket(
            workspaceBytes: WiredMemoryCoordinator.defaultWorkspaceBytes
        )
        return try await WiredMemoryTicket.withWiredLimit(ticket, body)
    }

    /// `internal` (not `private`) because `ChatCompletionsStreaming.swift`
    /// (extension on this handler) calls it from the streaming-with-tool-
    /// history path.
    func makeRawGenerationStream(
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
    ///
    /// `internal` (not `private`) because `ChatCompletionsStreaming.swift`
    /// (extension on this handler) calls it from the streaming tool-call
    /// delta path.
    func toolCallToChat(_ call: ToolCall) -> ChatToolCall {
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

    // MARK: - Tool-call recovery (A3)

    /// Best-effort extraction of tool calls emitted as text by models whose
    /// chat template does not surface native `ToolCall` events.
    ///
    /// Supported envelopes:
    ///   - `<tool_call>{json}</tool_call>` (Qwen-style JSON body)
    ///   - `<minimax:tool_call><invoke …>…</minimax:tool_call>` (XML body)
    ///   - `<function_calls><invoke …>…</function_calls>` (XML body)
    ///
    /// A truncated block (missing closing tag, e.g. max_tokens cut) is still
    /// attempted. JSON bodies go through a repair pass (control characters,
    /// unterminated strings, unbalanced braces) before being dropped.
    /// Unrecoverable blocks are logged and skipped — never thrown.
    struct ToolCallRecovery: Sendable {
        /// Text with every recognized tool-call block removed. Equals the
        /// original text when no call could be recovered.
        var content: String
        /// Parsed calls, in order of appearance. Empty means "no tool
        /// calls" — callers must not treat it as an error.
        var toolCalls: [ChatToolCall]
        /// Blocks that looked like tool calls but survived no repair.
        var droppedBlocks: Int
    }

    private struct ToolCallBlock {
        enum Kind { case json, xmlInvoke }
        let kind: Kind
        let body: String
        let range: Range<String.Index>
    }

    /// Recover tool calls from generated text. Internal (not private) so the
    /// MTP path in ChatCompletionsMTP.swift applies the same recovery.
    static func recoverToolCalls(from text: String) -> ToolCallRecovery {
        let blocks = extractToolCallBlocks(from: text)
        guard !blocks.isEmpty else {
            return ToolCallRecovery(content: text, toolCalls: [], droppedBlocks: 0)
        }

        var calls: [ChatToolCall] = []
        var dropped = 0
        for block in blocks {
            if let call = parseToolCallBlock(block) {
                calls.append(call)
            } else {
                dropped += 1
                let preview = String(block.body.prefix(160))
                    .replacingOccurrences(of: "\n", with: " ")
                FileHandle.standardError.write(Data(
                    "[telemak.tools] dropped unrecoverable tool-call block: \(preview)\n".utf8
                ))
            }
        }

        // Nothing parseable: leave the text untouched rather than mangling
        // what might be plain prose.
        guard !calls.isEmpty else {
            return ToolCallRecovery(content: text, toolCalls: [], droppedBlocks: dropped)
        }

        var content = text
        for block in blocks.reversed() {
            content.removeSubrange(block.range)
        }
        return ToolCallRecovery(
            content: content.trimmingCharacters(in: .whitespacesAndNewlines),
            toolCalls: calls,
            droppedBlocks: dropped
        )
    }

    private static func extractToolCallBlocks(from text: String) -> [ToolCallBlock] {
        let envelopes: [(open: String, close: String, kind: ToolCallBlock.Kind)] = [
            ("<minimax:tool_call>", "</minimax:tool_call>", .xmlInvoke),
            ("<function_calls>", "</function_calls>", .xmlInvoke),
            ("<tool_call>", "</tool_call>", .json),
        ]
        var blocks: [ToolCallBlock] = []
        var cursor = text.startIndex
        while cursor < text.endIndex {
            var nearest: (range: Range<String.Index>, close: String, kind: ToolCallBlock.Kind)?
            for envelope in envelopes {
                guard let r = text.range(of: envelope.open, range: cursor ..< text.endIndex) else { continue }
                if nearest == nil || r.lowerBound < nearest!.range.lowerBound {
                    nearest = (r, envelope.close, envelope.kind)
                }
            }
            guard let found = nearest else { break }
            let bodyStart = found.range.upperBound
            let endRange = text.range(of: found.close, range: bodyStart ..< text.endIndex)
            let bodyEnd = endRange?.lowerBound ?? text.endIndex
            let blockEnd = endRange?.upperBound ?? text.endIndex
            blocks.append(ToolCallBlock(
                kind: found.kind,
                body: String(text[bodyStart ..< bodyEnd]),
                range: found.range.lowerBound ..< blockEnd
            ))
            cursor = blockEnd
        }
        return blocks
    }

    private static func parseToolCallBlock(_ block: ToolCallBlock) -> ChatToolCall? {
        switch block.kind {
        case .json:
            return parseJSONToolCall(block.body)
        case .xmlInvoke:
            return parseXMLInvokeToolCall(block.body)
        }
    }

    private static func makeRecoveredCall(name: String, arguments: String) -> ChatToolCall {
        ChatToolCall(
            id: "call_\(UUID().uuidString.lowercased().prefix(24))",
            type: "function",
            function: ChatToolCallFunction(name: name, arguments: arguments)
        )
    }

    // MARK: JSON tool-call bodies

    private static func parseJSONToolCall(_ body: String) -> ChatToolCall? {
        guard let object = parseObjectWithRepair(body),
              let name = object["name"] as? String, !name.isEmpty
        else { return nil }
        return makeRecoveredCall(name: name, arguments: normalizedArguments(object["arguments"]))
    }

    /// Parse a JSON object, first verbatim then after a repair pass.
    private static func parseObjectWithRepair(_ body: String) -> [String: Any]? {
        let trimmed = body.trimmingCharacters(in: .whitespacesAndNewlines)
        if let object = tryParseJSONObject(trimmed) { return object }
        return tryParseJSONObject(Self.repairJSON(trimmed))
    }

    private static func tryParseJSONObject(_ text: String) -> [String: Any]? {
        guard let data = text.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data),
              let dict = object as? [String: Any]
        else { return nil }
        return dict
    }

    /// OpenAI wire format wants `arguments` as a JSON string. Model output
    /// gives either an object (serialize it) or a string (already wire
    /// shape — parse + reserialize to normalize when possible).
    private static func normalizedArguments(_ value: Any?) -> String {
        switch value {
        case let dict as [String: Any]:
            return serializeJSONObject(dict) ?? "{}"
        case let string as String:
            if let object = parseObjectWithRepair(string),
               let serialized = serializeJSONObject(object)
            {
                return serialized
            }
            return string.isEmpty ? "{}" : string
        case .none:
            return "{}"
        default:
            if let value, let serialized = serializeJSONObject(["value": value]) {
                return serialized
            }
            return "{}"
        }
    }

    private static func serializeJSONObject(_ object: [String: Any]) -> String? {
        guard JSONSerialization.isValidJSONObject(object),
              let data = try? JSONSerialization.data(withJSONObject: object),
              let string = String(data: data, encoding: .utf8)
        else { return nil }
        return string
    }

    /// Repair common JSON damage seen in tool-call payloads:
    ///   - raw control characters inside strings (real newlines in values)
    ///   - truncated output (unterminated string, unbalanced braces/brackets)
    ///   - trailing commas before a closer
    /// Best-effort only — callers still validate by parsing afterwards.
    static func repairJSON(_ input: String) -> String {
        func dropTrailingComma(_ s: inout String) {
            while let last = s.last, last.isWhitespace { s.removeLast() }
            if s.last == "," { s.removeLast() }
        }

        var out = ""
        out.reserveCapacity(input.count + 8)
        var inString = false
        var escaped = false
        var stack: [Character] = []

        for ch in input {
            if inString {
                if escaped {
                    out.append(ch)
                    escaped = false
                } else if ch == "\\" {
                    out.append(ch)
                    escaped = true
                } else if ch == "\"" {
                    out.append(ch)
                    inString = false
                } else if ch == "\n" {
                    out.append("\\n")
                } else if ch == "\r" {
                    out.append("\\r")
                } else if ch == "\t" {
                    out.append("\\t")
                } else if let ascii = ch.asciiValue, ascii < 0x20 {
                    // drop other raw control characters inside strings
                } else {
                    out.append(ch)
                }
            } else {
                if ch == "\"" {
                    inString = true
                    out.append(ch)
                } else if ch == "{" || ch == "[" {
                    stack.append(ch)
                    out.append(ch)
                } else if ch == "}" || ch == "]" {
                    dropTrailingComma(&out)
                    let expected: Character = (ch == "}") ? "{" : "["
                    if stack.last == expected { stack.removeLast() }
                    out.append(ch)
                } else if let ascii = ch.asciiValue, ascii < 0x20, !ch.isWhitespace {
                    // drop stray control characters outside strings
                } else {
                    out.append(ch)
                }
            }
        }

        if escaped, out.last == "\\" {
            out.removeLast()
        }
        if inString {
            out.append("\"")
        }
        dropTrailingComma(&out)
        while let open = stack.popLast() {
            out.append(open == "{" ? "}" : "]")
        }
        return out
    }

    // MARK: XML invoke tool-call bodies (MiniMax / function_calls)

    private static func parseXMLInvokeToolCall(_ body: String) -> ChatToolCall? {
        guard let name = firstRegexGroup(#"<invoke\s+name="([^"]+)">"#, in: body) else {
            return nil
        }

        var args: [String: String] = [:]
        let pattern = #"<parameter\s+name="([^"]+)">([\s\S]*?)</parameter>"#
        if let regex = try? NSRegularExpression(pattern: pattern) {
            let nsRange = NSRange(body.startIndex ..< body.endIndex, in: body)
            for match in regex.matches(in: body, range: nsRange) {
                guard match.numberOfRanges == 3,
                      let keyRange = Range(match.range(at: 1), in: body),
                      let valueRange = Range(match.range(at: 2), in: body)
                else { continue }
                let key = String(body[keyRange])
                let value = String(body[valueRange]).trimmingCharacters(in: .whitespacesAndNewlines)
                args[key] = value
            }
        }

        // Truncated tail: an open <parameter name="…"> whose closing tag
        // never arrived (max_tokens cut) — salvage the partial value.
        let openPattern = #"<parameter\s+name="([^"]+)">"#
        if let openRegex = try? NSRegularExpression(pattern: openPattern) {
            let nsRange = NSRange(body.startIndex ..< body.endIndex, in: body)
            let opens = openRegex.matches(in: body, range: nsRange)
            let closeCount = body.components(separatedBy: "</parameter>").count - 1
            if opens.count > closeCount,
               let last = opens.last,
               let keyRange = Range(last.range(at: 1), in: body),
               let wholeRange = Range(last.range, in: body)
            {
                let value = String(body[wholeRange.upperBound...])
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                args[String(body[keyRange])] = value
            }
        }

        let argsString = serializeJSONObject(args) ?? "{}"
        return makeRecoveredCall(name: name, arguments: argsString)
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

    private static func firstRegexGroup(_ pattern: String, in text: String) -> String? {
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

    /// Convert OpenAI wire messages to mlx-swift-lm `Chat.Message` for
    /// `ChatSession` history rehydration (A1 KV-bridge, cache-miss path).
    /// `Chat.Message` has no tool-call fields, so assistant tool calls are
    /// serialized into `content` as Qwen-style tool_call blocks (JSON
    /// body, arguments as an object) — matching what the chat template
    /// renders for native tool calls. Tool results keep role .tool so the
    /// template wraps them as tool_response. `tool_call_id`/`name` have no
    /// channel through `Chat.Message`; ids are regenerated client-side on
    /// the next turn.
    private func toChatMessages(_ messages: [ChatMessage], system: String?) -> [Chat.Message] {
        var result: [Chat.Message] = []
        if let system, !messages.contains(where: { $0.role == "system" }) {
            result.append(.system(system))
        }
        for message in messages {
            let text = message.content?.asPlainText ?? ""
            switch message.role {
            case "system":
                result.append(.system(text))
            case "assistant":
                result.append(.assistant(text + Self.serializeToolCallsForHistory(message.toolCalls)))
            case "tool":
                result.append(.tool(text))
            default:
                result.append(.user(text))
            }
        }
        return result
    }

    private static func serializeToolCallsForHistory(_ toolCalls: [ChatToolCall]?) -> String {
        guard let toolCalls, !toolCalls.isEmpty else { return "" }
        return toolCalls.map { call in
            var arguments = call.function.arguments
            if let data = arguments.data(using: .utf8),
               let parsed = try? JSONSerialization.jsonObject(with: data),
               let pretty = try? JSONSerialization.data(withJSONObject: parsed),
               let prettyString = String(data: pretty, encoding: .utf8)
            {
                arguments = prettyString
            }
            return "<tool_call>\n{\"name\": \"\(call.function.name)\", \"arguments\": \(arguments)}\n</tool_call>"
        }.joined(separator: "\n")
    }

    private static func instructionsWithReasoningGuard(
        base: String?,
        modelId: String,
        effort: String?
    ) -> String? {
        let model = modelId.lowercased()
        guard model.contains("step-3.7") || model.contains("step3p7") else {
            return base
        }
        guard let effort = effort?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() else {
            return base
        }

        let guardrail: String? = switch effort {
        case "minimal":
            "Keep reasoning extremely brief. Close </think> within 80 tokens, then write the requested answer."
        case "low":
            "Keep reasoning brief. Close </think> within 200 tokens, then write the requested answer."
        default:
            nil
        }
        guard let guardrail else { return base }
        guard let base, !base.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return guardrail
        }
        return "\(guardrail)\n\n\(base)"
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
