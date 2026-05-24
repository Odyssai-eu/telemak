import Foundation

/// Anthropic `/v1/messages` request. Minimal but functional shape.
///
/// `content` blocks can be a plain string or an array of `{type, text|...}`
/// — we accept both, normalize to the array form internally.
struct AnthropicMessagesRequest: Codable, Sendable {
    var model: String?
    var system: AnthropicSystem?
    var messages: [AnthropicMessage]
    var maxTokens: Int
    var temperature: Float?
    var topP: Float?
    var topK: Int?
    var stream: Bool?
    var sessionId: String?

    enum CodingKeys: String, CodingKey {
        case model
        case system
        case messages
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case stream
        case sessionId = "session_id"
    }
}

/// Anthropic accepts `system: "..."` OR `system: [{type:"text", text:"..."}]`.
enum AnthropicSystem: Codable, Sendable {
    case string(String)
    case blocks([AnthropicContentBlock])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
            return
        }
        if let blocks = try? container.decode([AnthropicContentBlock].self) {
            self = .blocks(blocks)
            return
        }
        throw DecodingError.typeMismatch(
            AnthropicSystem.self,
            .init(codingPath: container.codingPath,
                  debugDescription: "system must be string or [content block]")
        )
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .blocks(let b): try container.encode(b)
        }
    }

    var asString: String {
        switch self {
        case .string(let s): return s
        case .blocks(let blocks):
            return blocks.compactMap { $0.text }.joined(separator: "\n\n")
        }
    }
}

struct AnthropicMessage: Codable, Sendable {
    var role: String
    var content: AnthropicMessageContent
}

enum AnthropicMessageContent: Codable, Sendable {
    case string(String)
    case blocks([AnthropicContentBlock])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
            return
        }
        if let blocks = try? container.decode([AnthropicContentBlock].self) {
            self = .blocks(blocks)
            return
        }
        throw DecodingError.typeMismatch(
            AnthropicMessageContent.self,
            .init(codingPath: container.codingPath,
                  debugDescription: "content must be string or [content block]")
        )
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .blocks(let b): try container.encode(b)
        }
    }

    var asPlainText: String {
        switch self {
        case .string(let s): return s
        case .blocks(let blocks): return blocks.compactMap { $0.text }.joined(separator: "\n\n")
        }
    }
}

struct AnthropicContentBlock: Codable, Sendable {
    var type: String     // "text" | "image" | "tool_use" | "tool_result"
    var text: String?
    // V1: ignore everything except text. Vision + tool blocks land later.
}

// MARK: - Response

struct AnthropicMessageResponse: Codable, Sendable {
    var id: String
    var type: String           // "message"
    var role: String           // "assistant"
    var model: String
    var content: [AnthropicContentBlock]
    var stopReason: String?    // "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"
    var stopSequence: String?
    var usage: Usage

    struct Usage: Codable, Sendable {
        var inputTokens: Int
        var outputTokens: Int

        enum CodingKeys: String, CodingKey {
            case inputTokens = "input_tokens"
            case outputTokens = "output_tokens"
        }
    }

    enum CodingKeys: String, CodingKey {
        case id
        case type
        case role
        case model
        case content
        case stopReason = "stop_reason"
        case stopSequence = "stop_sequence"
        case usage
    }
}
