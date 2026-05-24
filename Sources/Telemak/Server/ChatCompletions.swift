import Foundation
import Hummingbird
import MLXLMCommon

/// POST /v1/chat/completions
struct ChatCompletionsHandler: Sendable {
    let registry: ModelRegistry
    let stats: StatsTracker

    func add(to router: Router<BasicRequestContext>) {
        router.post("/v1/chat/completions") { request, context async throws -> Response in
            try await self.handle(request, context: context)
        }
    }

    func handle(_ request: Request, context: BasicRequestContext) async throws -> Response {
        let body = try await request.body.collect(upTo: 1 << 20)   // 1 MB cap on request body
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
        do {
            container = try await registry.ensureLoaded(modelId)
        } catch {
            return jsonError(.serviceUnavailable, code: "model_load_failed",
                              message: "could not load model '\(modelId)': \(error)")
        }

        var params = GenerateParameters()
        if let maxTokens = payload.maxTokens { params.maxTokens = maxTokens }
        if let temperature = payload.temperature { params.temperature = temperature }
        if let topP = payload.topP { params.topP = topP }

        let instructions = payload.system ?? extractSystem(from: payload.messages)
        let userPrompt = renderUserPrompt(from: payload.messages)
        if userPrompt.isEmpty {
            return jsonError(.badRequest, code: "invalid_request_error",
                              message: "no user message to generate from")
        }

        let session = ChatSession(
            container,
            instructions: instructions,
            generateParameters: params
        )

        if payload.stream == true {
            return streamingResponse(
                container: container,
                instructions: instructions,
                params: params,
                userPrompt: userPrompt,
                modelId: modelId,
                stats: stats
            )
        }

        let genStart = Date()
        let completion: String
        do {
            completion = try await session.respond(to: userPrompt)
        } catch {
            return jsonError(.internalServerError, code: "generation_failed",
                              message: "model generation failed: \(error)")
        }
        let genElapsed = Date().timeIntervalSince(genStart)

        let promptChars = userPrompt.count + (instructions?.count ?? 0)
        let promptTokens = max(1, promptChars / 4)
        let completionTokens = max(1, completion.count / 4)
        await stats.recordRequest(tokens: completionTokens, elapsedSeconds: genElapsed)

        let response = ChatCompletionResponse(
            id: "chatcmpl-\(UUID().uuidString.lowercased())",
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: modelId,
            choices: [
                .init(
                    index: 0,
                    message: ChatMessage(role: "assistant", content: completion),
                    finishReason: "stop"
                )
            ],
            usage: .init(
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                totalTokens: promptTokens + completionTokens
            )
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
        stats: StatsTracker
    ) -> Response {
        let id = "chatcmpl-\(UUID().uuidString.lowercased())"
        let created = Int(Date().timeIntervalSince1970)

        let body = ResponseBody(contentLength: nil) { writer in
            let encoder = JSONEncoder()
            let session = ChatSession(
                container,
                instructions: instructions,
                generateParameters: params
            )

            func send(_ chunk: ChatCompletionChunk) async throws {
                let data = try encoder.encode(chunk)
                var buffer = ByteBuffer()
                buffer.writeString("data: ")
                buffer.writeBytes(data)
                buffer.writeString("\n\n")
                try await writer.write(buffer)
            }

            let genStart = Date()
            var completionChars = 0

            do {
                let role = ChatCompletionChunk(
                    id: id, object: "chat.completion.chunk",
                    created: created, model: modelId,
                    choices: [.init(index: 0, delta: .init(role: "assistant", content: nil), finishReason: nil)]
                )
                try await send(role)

                for try await piece in session.streamResponse(to: userPrompt) {
                    completionChars += piece.count
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

                try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
            } catch {
                let payload = #"{"error":{"message":"streaming aborted: \#(error)","type":"generation_failed"}}"#
                try? await writer.write(ByteBuffer(string: "data: \(payload)\n\n"))
            }
            let elapsed = Date().timeIntervalSince(genStart)
            let estTokens = max(1, completionChars / 4)
            await stats.recordRequest(tokens: estTokens, elapsedSeconds: elapsed)
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

    private func extractSystem(from messages: [ChatMessage]) -> String? {
        let systemParts = messages.filter { $0.role == "system" }.map(\.content)
        return systemParts.isEmpty ? nil : systemParts.joined(separator: "\n\n")
    }

    /// MVP user-prompt rendering: concatenate non-system messages, prefixed by
    /// role for any non-user line. Multi-turn fidelity is a V1 concern — we'll
    /// switch to `ChatSession` history-init once we keep state across requests.
    private func renderUserPrompt(from messages: [ChatMessage]) -> String {
        let nonSystem = messages.filter { $0.role != "system" }
        if nonSystem.count == 1, nonSystem[0].role == "user" {
            return nonSystem[0].content
        }
        return nonSystem.map { msg in
            msg.role == "user" ? msg.content : "[\(msg.role)] \(msg.content)"
        }.joined(separator: "\n\n")
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
