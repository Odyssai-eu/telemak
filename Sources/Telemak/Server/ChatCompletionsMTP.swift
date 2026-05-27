import Foundation
import Hummingbird
import MLX
import MLXLLM
import MLXLMCommon
import MLXRandom

// MARK: - MTP fast path for /v1/chat/completions
//
// When the requested model has a paired MTP draft (loaded via /admin/load
// with `draft_model:`), the chat completion route can run through the
// speculative iterator instead of `MLXLMCommon.ChatSession`. This file
// is the integration glue.
//
// The iterator path skips a few features the regular path supports — tool
// calls, vision images, the on-disk prompt cache — because the draft
// can't propose tool calls or vision tokens, and the cache snapshot/replay
// would interfere with the fork's `targetVerify`/`rollbackSpeculativeCache`
// semantics. Whenever a caller would need those features, the regular
// `ChatSession` path is used as a fall-through (no regression).
//
// Companion's user-facing speedup metric comes from this file: with a
// draft paired, `/v1/chat/completions` produces fewer real-target forward
// passes per emitted token, so Companion's `Speed: NN tok/s` row reflects
// the MTP win.

extension ChatCompletionsHandler {

    /// Decides whether the request can be served by the MTP iterator.
    /// Returns the loaded draft when yes, nil when the regular ChatSession
    /// path must be used.
    func mtpDraftIfEligible(
        for modelId: String,
        toolSpecs: [[String: any Sendable]]?,
        imageBatch: VisionImageBatch
    ) async -> ModelRegistry.LoadedDraft? {
        // Tools and images can't be expressed through the speculative
        // iterator's token-only interface — fall through to ChatSession
        // when either is present.
        if let toolSpecs, !toolSpecs.isEmpty { return nil }
        if !imageBatch.images.isEmpty { return nil }

        guard let draftId = await registry.draftId(for: modelId),
              let draft = await registry.getDraft(draftId)
        else { return nil }
        return draft
    }

    /// Drive the speculative iterator for one chat request. Both streaming
    /// and non-streaming responses go through here.
    ///
    /// The prompt is prepared via the model processor (same path as
    /// `/admin/mtp/smoke`) so the chat template applied here matches what
    /// the regular `ChatSession` path produces — necessary for tokenizer
    /// alignment with the draft.
    func runMTPChat(
        container: ModelContainer,
        draftEntry: ModelRegistry.LoadedDraft,
        modelId: String,
        userPrompt: String,
        instructions: String?,
        params: GenerateParameters,
        stopSequences: [String],
        stream: Bool,
        seed: UInt64?,
        cachedTokens: Int
    ) async throws -> Response {
        if let seed { MLXRandom.seed(seed) }

        // Render system + user into a single chat string. The processor
        // applies the chat template on this; multi-turn requests already
        // arrived merged through `renderUserPrompt`.
        let chatText: String
        if let sys = instructions, !sys.isEmpty {
            chatText = "[system]\n\(sys)\n\n[user]\n\(userPrompt)"
        } else {
            chatText = userPrompt
        }

        if stream {
            return mtpStreamingResponse(
                container: container,
                draftEntry: draftEntry,
                modelId: modelId,
                chatText: chatText,
                params: params,
                stopSequences: stopSequences,
                cachedTokens: cachedTokens
            )
        }

        return try await mtpNonStreamingResponse(
            container: container,
            draftEntry: draftEntry,
            modelId: modelId,
            chatText: chatText,
            params: params,
            stopSequences: stopSequences,
            cachedTokens: cachedTokens
        )
    }

    // MARK: - Non-streaming

    private func mtpNonStreamingResponse(
        container: ModelContainer,
        draftEntry: ModelRegistry.LoadedDraft,
        modelId: String,
        chatText: String,
        params: GenerateParameters,
        stopSequences: [String],
        cachedTokens: Int
    ) async throws -> Response {
        let genStart = Date()
        let result: MTPRunResult = try await container.perform { ctx in
            let promptTokens = try await prepareMTPPrompt(ctx: ctx, chatText: chatText)
            guard !promptTokens.isEmpty else {
                throw HTTPError(.badRequest, message: "empty prompt after tokenization")
            }
            return try runMTPIteratorCollecting(
                ctx: ctx,
                draftEntry: draftEntry,
                promptTokens: promptTokens,
                params: params,
                stopSequences: stopSequences
            )
        }
        let elapsed = Date().timeIntervalSince(genStart)
        await stats.recordRequest(tokens: result.tokensGenerated, elapsedSeconds: elapsed)

        let response = ChatCompletionResponse(
            id: "chatcmpl-\(UUID().uuidString.lowercased())",
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: modelId,
            choices: [
                .init(
                    index: 0,
                    message: ChatMessage(role: "assistant", content: result.text, toolCalls: nil),
                    finishReason: "stop"
                )
            ],
            usage: ChatCompletionResponse.Usage(
                promptTokens: result.promptTokens,
                completionTokens: result.tokensGenerated,
                totalTokens: result.promptTokens + result.tokensGenerated,
                promptTokensDetails: cachedTokens > 0 ? .init(cachedTokens: cachedTokens) : nil
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
    //
    // Pre-computes the full token stream inside `container.perform` (the
    // only place we can safely access the model + tokenizer), buffers each
    // visible piece in an array, then streams the buffered pieces out
    // through the response writer. This avoids the Sendable-capture issue
    // of holding the writer inside the model-actor closure.
    //
    // Trade-off : the user perceives streaming but the underlying tokens
    // arrive in one batch after the generation finishes. Acceptable for
    // an initial wire — the next iteration can wrap the iterator into an
    // AsyncStream that yields per-token in real time. The MTP speedup
    // (acceptance + tok/s) still applies because the generation work
    // happens at speculative speed; only the SSE emission timing differs.

    private func mtpStreamingResponse(
        container: ModelContainer,
        draftEntry: ModelRegistry.LoadedDraft,
        modelId: String,
        chatText: String,
        params: GenerateParameters,
        stopSequences: [String],
        cachedTokens: Int
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

            // Role chunk — same shape as the regular streaming path so
            // Companion can render the assistant role pill consistently.
            let role = ChatCompletionChunk(
                id: id, object: "chat.completion.chunk",
                created: created, model: modelId,
                choices: [.init(index: 0, delta: .init(role: "assistant", content: nil), finishReason: nil)]
            )
            try await send(role)

            let genStart = Date()
            var promptTokenCount = 0
            var completionTokenCount = 0
            var visiblePieces: [String] = []

            do {
                let result: MTPStreamingResult = try await container.perform { ctx in
                    let promptTokens = try await prepareMTPPrompt(ctx: ctx, chatText: chatText)
                    guard !promptTokens.isEmpty else {
                        throw HTTPError(.badRequest, message: "empty prompt after tokenization")
                    }
                    let collected = try runMTPIteratorCollectingPieces(
                        ctx: ctx,
                        draftEntry: draftEntry,
                        promptTokens: promptTokens,
                        params: params,
                        stopSequences: stopSequences
                    )
                    return MTPStreamingResult(
                        promptTokens: promptTokens.count,
                        tokensGenerated: collected.tokensGenerated,
                        pieces: collected.pieces
                    )
                }
                promptTokenCount = result.promptTokens
                completionTokenCount = result.tokensGenerated
                visiblePieces = result.pieces

                // Emit each piece as its own SSE chunk so the client sees
                // a stream-shaped delta sequence even though the work is
                // already done. UI behaviour is identical to a real-time
                // stream of the same payload size.
                for piece in visiblePieces where !piece.isEmpty {
                    let chunk = ChatCompletionChunk(
                        id: id, object: "chat.completion.chunk",
                        created: created, model: modelId,
                        choices: [.init(index: 0, delta: .init(role: nil, content: piece), finishReason: nil)]
                    )
                    try await send(chunk)
                }

                let stop = ChatCompletionChunk(
                    id: id, object: "chat.completion.chunk",
                    created: created, model: modelId,
                    choices: [.init(index: 0, delta: .init(role: nil, content: nil), finishReason: "stop")]
                )
                try await send(stop)

                // Usage chunk between finish and [DONE].
                var usageBlock: [String: Any] = [
                    "prompt_tokens": promptTokenCount,
                    "completion_tokens": completionTokenCount,
                    "total_tokens": promptTokenCount + completionTokenCount,
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
                let payload = #"{"error":{"message":"mtp generation failed: \#(error)","type":"generation_failed"}}"#
                try? await writer.write(ByteBuffer(string: "data: \(payload)\n\n"))
            }

            let elapsed = Date().timeIntervalSince(genStart)
            await stats.recordRequest(tokens: completionTokenCount, elapsedSeconds: elapsed)

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

    // MARK: - Shared helpers

    /// Tokenize through the model's processor — identical to what
    /// `/admin/mtp/smoke` does — so the chat template applied lines up
    /// with what the draft was trained against.
    private func prepareMTPPrompt(
        ctx: ModelContext,
        chatText: String
    ) async throws -> [Int] {
        let input = try await ctx.processor.prepare(input: UserInput(prompt: chatText))
        let tokenArray = input.text.tokens.reshaped(-1).asType(.int32)
        eval(tokenArray)
        return tokenArray.asArray(Int32.self).map(Int.init)
    }

    /// Non-streaming run: collect all tokens, decode at end, apply stop
    /// sequences, return aggregated stats + text.
    private func runMTPIteratorCollecting(
        ctx: ModelContext,
        draftEntry: ModelRegistry.LoadedDraft,
        promptTokens: [Int],
        params: GenerateParameters,
        stopSequences: [String]
    ) throws -> MTPRunResult {
        let collected = try runMTPIteratorCollectingPieces(
            ctx: ctx,
            draftEntry: draftEntry,
            promptTokens: promptTokens,
            params: params,
            stopSequences: stopSequences
        )
        return MTPRunResult(
            text: collected.pieces.joined(),
            promptTokens: promptTokens.count,
            tokensGenerated: collected.tokensGenerated,
            acceptance: collected.acceptance,
            accepted: collected.accepted,
            proposed: collected.proposed
        )
    }

    /// Runs the iterator to exhaustion and returns each visible piece
    /// (filtered by `StopChecker`) plus aggregated stats. Used by both
    /// the streaming and non-streaming paths so the logic stays in one
    /// place.
    private func runMTPIteratorCollectingPieces(
        ctx: ModelContext,
        draftEntry: ModelRegistry.LoadedDraft,
        promptTokens: [Int],
        params: GenerateParameters,
        stopSequences: [String]
    ) throws -> MTPPiecesResult {
        var generated: [Int] = []
        var stopChecker = StopChecker(stops: stopSequences)
        var pieces: [String] = []
        var lastDecoded = ""
        let maxTokens = params.maxTokens ?? 512

        let acceptance: Double
        let accepted: Int
        let proposed: Int

        switch draftEntry.model {
        case .qwen35(let draftModel):
            guard let qwen = ctx.model as? any Qwen35HiddenStateProvider else {
                throw HTTPError(.badRequest,
                                message: "main model is not Qwen3.5/3.6 (\(type(of: ctx.model))); MTP unsupported")
            }
            let iterator = MTPSpeculativeIterator(
                main: qwen, draft: draftModel,
                promptTokens: promptTokens, maxTokens: maxTokens,
                blockSize: nil, parameters: params
            )
            while let tok = iterator.next() {
                generated.append(tok)
                let decoded = ctx.tokenizer.decode(tokenIds: generated)
                let delta = String(decoded.dropFirst(lastDecoded.count))
                if !delta.isEmpty {
                    let visible = stopChecker.feed(delta)
                    if !visible.isEmpty { pieces.append(visible) }
                    lastDecoded = decoded
                }
                if stopChecker.hit { break }
            }
            acceptance = iterator.acceptanceRate
            accepted = iterator.totalAccepted
            proposed = iterator.totalProposed

        case .gemma4Assistant(let draftModel):
            guard let gemma = ctx.model as? Gemma4Model else {
                throw HTTPError(.badRequest,
                                message: "main model is not Gemma4 (\(type(of: ctx.model))); Gemma4Assistant MTP unsupported")
            }
            let iterator = Gemma4AssistantSpeculativeIterator(
                main: gemma, draft: draftModel,
                promptTokens: promptTokens, maxTokens: maxTokens,
                blockSize: 6, parameters: params
            )
            while let tok = try iterator.next() {
                generated.append(tok)
                let decoded = ctx.tokenizer.decode(tokenIds: generated)
                let delta = String(decoded.dropFirst(lastDecoded.count))
                if !delta.isEmpty {
                    let visible = stopChecker.feed(delta)
                    if !visible.isEmpty { pieces.append(visible) }
                    lastDecoded = decoded
                }
                if stopChecker.hit { break }
            }
            acceptance = iterator.acceptanceRate
            accepted = iterator.totalAccepted
            proposed = iterator.totalProposed
        }

        if !stopChecker.hit {
            let tail = stopChecker.flushRemaining()
            if !tail.isEmpty { pieces.append(tail) }
        }

        return MTPPiecesResult(
            pieces: pieces,
            tokensGenerated: generated.count,
            acceptance: acceptance,
            accepted: accepted,
            proposed: proposed
        )
    }
}

// MARK: - Result types

private struct MTPRunResult {
    let text: String
    let promptTokens: Int
    let tokensGenerated: Int
    let acceptance: Double
    let accepted: Int
    let proposed: Int
}

private struct MTPPiecesResult {
    let pieces: [String]
    let tokensGenerated: Int
    let acceptance: Double
    let accepted: Int
    let proposed: Int
}

private struct MTPStreamingResult: Sendable {
    let promptTokens: Int
    let tokensGenerated: Int
    let pieces: [String]
}
