import Foundation
import Hummingbird
import MLX

/// OpenAI-compatible `POST /v1/embeddings`.
struct EmbeddingsHandler: Sendable {
    let registry: ModelRegistry

    func add(to router: Router<BasicRequestContext>) {
        router.post("/v1/embeddings") { request, _ async throws -> Response in
            try await self.handle(request)
        }
    }

    private func handle(_ request: Request) async throws -> Response {
        let buf = try await request.body.collect(upTo: 1 << 20)
        let body: EmbeddingsRequest
        do {
            body = try JSONDecoder().decode(EmbeddingsRequest.self, from: Data(buffer: buf))
        } catch {
            return jsonError(
                .badRequest,
                code: "invalid_request_error",
                message: "expected {\"model\":\"<id>\",\"input\":\"text\"|[\"text\"],\"encoding_format\":\"float\"?}: \(error)"
            )
        }

        let format = body.encodingFormat ?? .float
        let modelId = ModelLoader.canonicalIdentifier(body.model)
        guard let loaded = await registry.getEmbedder(modelId) else {
            return jsonError(
                .notFound,
                code: "model_not_loaded",
                message: "embedder model '\(modelId)' is not loaded"
            )
        }

        let inputs = body.input.values
        do {
            let (vectors, tokenCount) = try await embed(inputs: inputs, using: loaded)
            let data = vectors.enumerated().map { index, vector in
                EmbeddingsResponse.Item(
                    object: "embedding",
                    index: index,
                    embedding: EmbeddingValue(vector, format: format)
                )
            }
            let payload = EmbeddingsResponse(
                object: "list",
                data: data,
                model: modelId,
                usage: .init(promptTokens: tokenCount, totalTokens: tokenCount)
            )
            let encoded = try JSONEncoder().encode(payload)
            return jsonResponse(.ok, data: encoded)
        } catch {
            return jsonError(.serviceUnavailable, code: "embedding_failed", message: "\(error)")
        }
    }

    private func embed(inputs: [String], using loaded: ModelRegistry.LoadedEmbedder) async throws -> ([[Float]], Int) {
        if inputs.isEmpty {
            return ([], 0)
        }
        return try await loaded.container.perform { context in
            let tokenized = inputs.map { text in
                context.tokenizer.encode(text: text, addSpecialTokens: true)
            }
            let maxPositions = context.model.maxPositionEmbeddings
            let clipped = tokenized.map { ids in
                guard let maxPositions else { return ids }
                return Array(ids.prefix(maxPositions))
            }
            let promptTokens = clipped.reduce(0) { $0 + $1.count }
            let padId = context.tokenizer.eosTokenId ?? context.tokenizer.unknownTokenId ?? 0
            let maxLength = max(1, clipped.map(\.count).max() ?? 1)
            let padded = stacked(
                clipped.map { ids in
                    let row = ids.isEmpty ? [padId] : ids
                    return MLXArray(row + Array(repeating: padId, count: maxLength - row.count))
                }
            )
            let mask = (padded .!= padId)
            let tokenTypes = MLXArray.zeros(like: padded)
            let result = context.pooling(
                context.model(
                    padded,
                    positionIds: nil,
                    tokenTypeIds: tokenTypes,
                    attentionMask: mask
                ),
                mask: mask,
                normalize: true,
                applyLayerNorm: true
            )
            result.eval()
            return (result.map { $0.asArray(Float.self) }, promptTokens)
        }
    }

    private func jsonResponse(_ status: HTTPResponse.Status, data: Data) -> Response {
        Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }

    private func jsonError(_ status: HTTPResponse.Status, code: String, message: String) -> Response {
        let payload: [String: [String: String]] = [
            "error": ["type": code, "code": code, "message": message]
        ]
        let data = (try? JSONEncoder().encode(payload)) ?? Data("{}".utf8)
        return jsonResponse(status, data: data)
    }
}

private struct EmbeddingsRequest: Decodable {
    let model: String
    let input: EmbeddingsInput
    let encodingFormat: EncodingFormat?

    enum CodingKeys: String, CodingKey {
        case model
        case input
        case encodingFormat = "encoding_format"
    }
}

private enum EmbeddingsInput: Decodable {
    case single(String)
    case many([String])

    var values: [String] {
        switch self {
        case .single(let value): return [value]
        case .many(let values): return values
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(String.self) {
            self = .single(value)
            return
        }
        if let values = try? container.decode([String].self) {
            self = .many(values)
            return
        }
        throw DecodingError.typeMismatch(
            EmbeddingsInput.self,
            .init(codingPath: container.codingPath, debugDescription: "input must be string or [string]")
        )
    }
}

private enum EncodingFormat: String, Decodable {
    case float
    case base64
}

private enum EmbeddingValue: Encodable {
    case floats([Float])
    case base64(String)

    init(_ floats: [Float], format: EncodingFormat) {
        switch format {
        case .float:
            self = .floats(floats)
        case .base64:
            self = .base64(Self.base64(floats))
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .floats(let values):
            try container.encode(values)
        case .base64(let value):
            try container.encode(value)
        }
    }

    private static func base64(_ floats: [Float]) -> String {
        var data = Data(capacity: floats.count * MemoryLayout<Float>.size)
        for value in floats {
            var bitPattern = value.bitPattern.littleEndian
            withUnsafeBytes(of: &bitPattern) { bytes in
                data.append(contentsOf: bytes)
            }
        }
        return data.base64EncodedString()
    }
}

private struct EmbeddingsResponse: Encodable {
    let object: String
    let data: [Item]
    let model: String
    let usage: Usage

    struct Item: Encodable {
        let object: String
        let index: Int
        let embedding: EmbeddingValue
    }

    struct Usage: Encodable {
        let promptTokens: Int
        let totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case totalTokens = "total_tokens"
        }
    }
}
