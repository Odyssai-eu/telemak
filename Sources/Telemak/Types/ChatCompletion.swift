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
    var sessionId: String?

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case stream
        case system
        case sessionId = "session_id"
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
        var promptTokensDetails: PromptTokensDetails?

        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
            case promptTokensDetails = "prompt_tokens_details"
        }
    }

    struct PromptTokensDetails: Codable, Sendable {
        var cachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }
    }
}

/// OpenAI-compatible streaming chunk.
struct ChatCompletionChunk: Codable, Sendable {
    var id: String
    var object: String        // "chat.completion.chunk"
    var created: Int
    var model: String
    var choices: [Choice]

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
