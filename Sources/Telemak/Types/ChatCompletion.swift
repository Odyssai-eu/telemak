import Foundation

/// OpenAI-compatible chat-completion request.
///
/// Only the fields Telemak actually consumes are decoded; unknown fields are
/// silently ignored. Match Odysseus' `scripts/api.py` `ChatCompletionRequest`.
struct ChatCompletionRequest: Codable, Sendable {
    var model: String?
    var messages: [ChatMessage]
    var maxTokens: Int?
    var temperature: Float?
    var topP: Float?
    var stream: Bool?
    var system: String?

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case stream
        case system
    }
}

struct ChatMessage: Codable, Sendable {
    var role: String
    var content: String
}

/// OpenAI-compatible non-streamed response.
struct ChatCompletionResponse: Codable, Sendable {
    var id: String
    var object: String
    var created: Int
    var model: String
    var choices: [Choice]
    var usage: Usage

    struct Choice: Codable, Sendable {
        var index: Int
        var message: ChatMessage
        var finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index
            case message
            case finishReason = "finish_reason"
        }
    }

    struct Usage: Codable, Sendable {
        var promptTokens: Int
        var completionTokens: Int
        var totalTokens: Int

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }
}

/// OpenAI-compatible streaming chunk.
///
/// Optional `usage` is populated only on the final chunk per the OpenAI
/// `stream_options.include_usage: true` convention. Telemak always emits it
/// because the cost of one extra JSON object at end-of-stream is negligible
/// and clients that don't expect it just ignore the field.
struct ChatCompletionChunk: Codable, Sendable {
    var id: String
    var object: String        // "chat.completion.chunk"
    var created: Int
    var model: String
    var choices: [Choice]
    var usage: ChatCompletionResponse.Usage?

    struct Choice: Codable, Sendable {
        var index: Int
        var delta: Delta
        var finishReason: String?

        enum CodingKeys: String, CodingKey {
            case index
            case delta
            case finishReason = "finish_reason"
        }
    }

    struct Delta: Codable, Sendable {
        var role: String?
        var content: String?
    }
}
