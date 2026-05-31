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
    var topK: Int?
    var minP: Float?
    var repetitionPenalty: Float?
    var stop: StopSequence?
    var seed: UInt64?
    var stream: Bool?
    var system: String?
    var sessionId: String?
    var enableThinking: Bool?
    var reasoningEffort: String?
    /// OpenAI tool array. We accept any JSON, pass through to mlx-swift-lm.
    var tools: [JSONValue]?
    var toolChoice: JSONValue?

    // KV cache quantization knobs (Block 3 / C perf follow-up).
    // `kvBits` of nil leaves the cache at full precision (default).
    var kvBits: Int?
    var kvGroupSize: Int?
    var quantizedKvStart: Int?

    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case maxTokens = "max_tokens"
        case temperature
        case topP = "top_p"
        case topK = "top_k"
        case minP = "min_p"
        case repetitionPenalty = "repetition_penalty"
        case stop
        case seed
        case stream
        case system
        case sessionId = "session_id"
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
        case tools
        case toolChoice = "tool_choice"
        case kvBits = "kv_bits"
        case kvGroupSize = "kv_group_size"
        case quantizedKvStart = "quantized_kv_start"
    }
}

/// Tagged JSON value — used to pass through opaque OpenAI fields like `tools`
/// without losing the structure. mlx-swift-lm's `ToolSpec` is just
/// `[String: any Sendable]` so we round-trip via this.
enum JSONValue: Codable, Sendable {
    case null
    case bool(Bool)
    case int(Int)
    case double(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null; return }
        if let b = try? container.decode(Bool.self) { self = .bool(b); return }
        if let i = try? container.decode(Int.self) { self = .int(i); return }
        if let d = try? container.decode(Double.self) { self = .double(d); return }
        if let s = try? container.decode(String.self) { self = .string(s); return }
        if let a = try? container.decode([JSONValue].self) { self = .array(a); return }
        if let o = try? container.decode([String: JSONValue].self) { self = .object(o); return }
        throw DecodingError.dataCorruptedError(
            in: container, debugDescription: "unknown JSON value"
        )
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let b): try container.encode(b)
        case .int(let i): try container.encode(i)
        case .double(let d): try container.encode(d)
        case .string(let s): try container.encode(s)
        case .array(let a): try container.encode(a)
        case .object(let o): try container.encode(o)
        }
    }

    /// Unwrap into an `any Sendable` for mlx-swift-lm's ToolSpec/dispatch
    /// dictionaries (which are `[String: any Sendable]`).
    func toSendable() -> any Sendable {
        switch self {
        case .null: return NSNull()
        case .bool(let b): return b
        case .int(let i): return i
        case .double(let d): return d
        case .string(let s): return s
        case .array(let a): return a.map { $0.toSendable() }
        case .object(let o): return o.mapValues { $0.toSendable() }
        }
    }

    /// Unwrap JSON for HuggingFace chat templates.
    ///
    /// The swift-transformers Jinja bridge cannot convert `NSNull`/optional
    /// values reliably. Tool schemas may contain JSON nulls in optional fields;
    /// dropping them matches how Python template renderers treat absent
    /// optional schema keys and avoids model-specific 500s.
    func toTemplateSendable() -> (any Sendable)? {
        switch self {
        case .null:
            return nil
        case .bool(let b):
            return b
        case .int(let i):
            return i
        case .double(let d):
            return d
        case .string(let s):
            return s
        case .array(let values):
            return values.compactMap { $0.toTemplateSendable() }
        case .object(let object):
            var result: [String: any Sendable] = [:]
            for (key, value) in object {
                if let sendable = value.toTemplateSendable() {
                    result[key] = sendable
                }
            }
            return result
        }
    }
}

/// OpenAI accepts `stop` as either a single string or array of strings.
/// Decode both, normalize to `[String]`.
enum StopSequence: Codable, Sendable {
    case single(String)
    case multiple([String])

    var asArray: [String] {
        switch self {
        case .single(let s): return [s]
        case .multiple(let arr): return arr
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .single(s)
        } else if let arr = try? container.decode([String].self) {
            self = .multiple(arr)
        } else {
            throw DecodingError.typeMismatch(
                StopSequence.self,
                .init(codingPath: container.codingPath,
                      debugDescription: "stop must be string or [string]")
            )
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .single(let s): try container.encode(s)
        case .multiple(let arr): try container.encode(arr)
        }
    }
}

struct ChatMessage: Codable, Sendable {
    var role: String
    var content: ChatMessageContent?
    var toolCalls: [ChatToolCall]?
    var toolCallId: String?
    var name: String?

    enum CodingKeys: String, CodingKey {
        case role
        case content
        case toolCalls = "tool_calls"
        case toolCallId = "tool_call_id"
        case name
    }

    init(
        role: String,
        content: String?,
        toolCalls: [ChatToolCall]? = nil,
        toolCallId: String? = nil,
        name: String? = nil
    ) {
        self.role = role
        self.content = content.map { .string($0) }
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
        self.name = name
    }
}

enum ChatMessageContent: Codable, Sendable {
    case string(String)
    case blocks([ChatContentBlock])

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) {
            self = .string(s)
            return
        }
        if let blocks = try? container.decode([ChatContentBlock].self) {
            self = .blocks(blocks)
            return
        }
        throw DecodingError.typeMismatch(
            ChatMessageContent.self,
            .init(codingPath: container.codingPath,
                  debugDescription: "content must be string or [content block]")
        )
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s):
            try container.encode(s)
        case .blocks(let blocks):
            try container.encode(blocks)
        }
    }

    var asPlainText: String {
        switch self {
        case .string(let s):
            return s
        case .blocks(let blocks):
            return blocks.compactMap(\.text).joined(separator: "\n\n")
        }
    }
}

struct ChatContentBlock: Codable, Sendable {
    var type: String
    var text: String?
    var imageURL: ChatImageURL?

    enum CodingKeys: String, CodingKey {
        case type
        case text
        case imageURL = "image_url"
    }
}

struct ChatImageURL: Codable, Sendable {
    var url: String
    var detail: String?
}

struct ChatToolCall: Codable, Sendable {
    var id: String
    var type: String
    var function: ChatToolCallFunction

    struct Encoder {
        // placeholder to keep the struct namespace
    }
}

struct ChatToolCallFunction: Codable, Sendable {
    var name: String
    /// OpenAI emits arguments as a JSON-encoded string, not a structured object.
    var arguments: String
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
        var toolCalls: [ChatToolCall]?

        enum CodingKeys: String, CodingKey {
            case role
            case content
            case toolCalls = "tool_calls"
        }
    }
}
