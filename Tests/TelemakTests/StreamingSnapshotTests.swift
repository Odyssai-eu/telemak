import Foundation
import Testing
import Telemak
@testable import Telemak

// MARK: - SSE frame parser (shared between OpenAI and Anthropic tests)

struct SSEEvent: Equatable {
    let event: String?
    let data: String
}

func parseSSE(_ stream: String) -> [SSEEvent] {
    var events: [SSEEvent] = []
    for chunk in stream.components(separatedBy: "\n\n") where !chunk.isEmpty {
        var event: String?
        var dataLines: [String] = []
        for line in chunk.components(separatedBy: "\n") {
            if line.hasPrefix("event: ") {
                event = String(line.dropFirst("event: ".count))
            } else if line.hasPrefix("data: ") {
                dataLines.append(String(line.dropFirst("data: ".count)))
            }
        }
        if !dataLines.isEmpty {
            events.append(SSEEvent(event: event, data: dataLines.joined(separator: "\n")))
        }
    }
    return events
}

// MARK: - OpenAI streaming wire format

@Test func openAIChunkRoleEncoding() throws {
    let chunk = ChatCompletionChunk(
        id: "chatcmpl-abc",
        object: "chat.completion.chunk",
        created: 1_700_000_000,
        model: "test-model",
        choices: [
            .init(
                index: 0,
                delta: .init(role: "assistant", content: nil, toolCalls: nil),
                finishReason: nil
            )
        ]
    )
    let data = try JSONEncoder().encode(chunk)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    #expect(json["id"] as? String == "chatcmpl-abc")
    #expect(json["object"] as? String == "chat.completion.chunk")
    let choices = json["choices"] as! [[String: Any]]
    #expect(choices.count == 1)
    let delta = choices[0]["delta"] as! [String: Any]
    #expect(delta["role"] as? String == "assistant")
    // `content` is optional — nil → absent from JSON
    #expect(delta["content"] == nil)
    #expect(choices[0]["finish_reason"] == nil)
}

@Test func openAIChunkContentDeltaEncoding() throws {
    let chunk = ChatCompletionChunk(
        id: "chatcmpl-abc",
        object: "chat.completion.chunk",
        created: 1_700_000_000,
        model: "test-model",
        choices: [
            .init(
                index: 0,
                delta: .init(role: nil, content: "hello world", toolCalls: nil),
                finishReason: nil
            )
        ]
    )
    let data = try JSONEncoder().encode(chunk)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    let delta = ((json["choices"] as! [[String: Any]])[0])["delta"] as! [String: Any]
    #expect(delta["content"] as? String == "hello world")
    #expect(delta["role"] == nil)
}

@Test func openAIChunkFinishReasonEncoding() throws {
    let chunk = ChatCompletionChunk(
        id: "chatcmpl-abc",
        object: "chat.completion.chunk",
        created: 1_700_000_000,
        model: "test-model",
        choices: [
            .init(
                index: 0,
                delta: .init(role: nil, content: nil, toolCalls: nil),
                finishReason: "stop"
            )
        ]
    )
    let data = try JSONEncoder().encode(chunk)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    let choice = (json["choices"] as! [[String: Any]])[0]
    #expect(choice["finish_reason"] as? String == "stop")
}

@Test func openAIUsageChunkWithCachedTokens() throws {
    // Mirrors the dict the streaming path emits in
    // ChatCompletions.swift `streamingResponse` (lines 614–629): usage
    // chunk between finish_reason and [DONE], with `prompt_tokens_details`
    // present only when cachedTokens > 0.
    let usageDict: [String: Any] = [
        "id": "chatcmpl-abc",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "test-model",
        "choices": [],
        "usage": [
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": ["cached_tokens": 75],
        ] as [String: Any],
    ]
    let data = try JSONSerialization.data(withJSONObject: usageDict)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    let usage = json["usage"] as! [String: Any]
    #expect(usage["prompt_tokens"] as? Int == 100)
    #expect(usage["completion_tokens"] as? Int == 50)
    #expect(usage["total_tokens"] as? Int == 150)
    let details = usage["prompt_tokens_details"] as! [String: Any]
    #expect(details["cached_tokens"] as? Int == 75)
}

@Test func openAIUsageChunkOmitsDetailsWhenNoCache() throws {
    // Mirrors the `if cachedTokens > 0` branch in ChatCompletions.swift —
    // when no session-cache hit, the field is absent (not null).
    let usageDict: [String: Any] = [
        "id": "chatcmpl-abc",
        "object": "chat.completion.chunk",
        "created": 1_700_000_000,
        "model": "test-model",
        "choices": [],
        "usage": [
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        ] as [String: Any],
    ]
    let data = try JSONSerialization.data(withJSONObject: usageDict)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    let usage = json["usage"] as! [String: Any]
    #expect(usage["prompt_tokens"] as? Int == 100)
    #expect(usage["prompt_tokens_details"] == nil)
}

@Test func openAIStreamOrderRoleDeltaFinishUsageDone() throws {
    // Builds a complete OpenAI stream in the order ChatCompletions.swift
    // emits it, then verifies each frame parses back to the right chunk.
    let id = "chatcmpl-abc"
    let roleChunk = ChatCompletionChunk(
        id: id, object: "chat.completion.chunk", created: 1, model: "m",
        choices: [.init(index: 0, delta: .init(role: "assistant", content: nil, toolCalls: nil), finishReason: nil)]
    )
    let deltaChunk = ChatCompletionChunk(
        id: id, object: "chat.completion.chunk", created: 1, model: "m",
        choices: [.init(index: 0, delta: .init(role: nil, content: "hi", toolCalls: nil), finishReason: nil)]
    )
    let finishChunk = ChatCompletionChunk(
        id: id, object: "chat.completion.chunk", created: 1, model: "m",
        choices: [.init(index: 0, delta: .init(role: nil, content: nil, toolCalls: nil), finishReason: "stop")]
    )
    let usageDict: [String: Any] = [
        "id": id, "object": "chat.completion.chunk", "created": 1, "model": "m", "choices": [],
        "usage": [
            "prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6,
            "prompt_tokens_details": ["cached_tokens": 4],
        ] as [String: Any],
    ]
    let usageData = try JSONSerialization.data(withJSONObject: usageDict)
    let usageJSON = String(data: usageData, encoding: .utf8)!

    let stream = [
        "data: \(String(data: try JSONEncoder().encode(roleChunk), encoding: .utf8)!)\n\n",
        "data: \(String(data: try JSONEncoder().encode(deltaChunk), encoding: .utf8)!)\n\n",
        "data: \(String(data: try JSONEncoder().encode(finishChunk), encoding: .utf8)!)\n\n",
        "data: \(usageJSON)\n\n",
        "data: [DONE]\n\n",
    ].joined()

    let events = parseSSE(stream)
    #expect(events.count == 5)
    #expect(events.map(\.event) == [nil, nil, nil, nil, nil])  // OpenAI has no `event:` prefix
    #expect(events[4].data == "[DONE]")

    let e0 = try JSONSerialization.jsonObject(with: Data(events[0].data.utf8)) as! [String: Any]
    #expect((((e0["choices"] as! [[String: Any]])[0])["delta"] as! [String: Any])["role"] as? String == "assistant")

    let e1 = try JSONSerialization.jsonObject(with: Data(events[1].data.utf8)) as! [String: Any]
    #expect((((e1["choices"] as! [[String: Any]])[0])["delta"] as! [String: Any])["content"] as? String == "hi")

    let e2 = try JSONSerialization.jsonObject(with: Data(events[2].data.utf8)) as! [String: Any]
    #expect(((e2["choices"] as! [[String: Any]])[0])["finish_reason"] as? String == "stop")

    let e3 = try JSONSerialization.jsonObject(with: Data(events[3].data.utf8)) as! [String: Any]
    let usage3 = e3["usage"] as! [String: Any]
    #expect(usage3["prompt_tokens"] as? Int == 5)
    #expect(((usage3["prompt_tokens_details"] as! [String: Any]))["cached_tokens"] as? Int == 4)
}

@Test func openAIErrorEventShape() throws {
    // Mirrors the error chunk emitted in `streamingResponse`'s catch
    // (ChatCompletions.swift line 641). It's a `data:` line carrying a
    // JSON error object, NOT a separate `event:` frame.
    let errorPayload = #"{"error":{"message":"streaming aborted: foo","type":"generation_failed"}}"#
    let stream = "data: \(errorPayload)\n\n"

    let events = parseSSE(stream)
    #expect(events.count == 1)
    #expect(events[0].event == nil)
    let json = try JSONSerialization.jsonObject(with: Data(events[0].data.utf8)) as! [String: Any]
    let error = json["error"] as! [String: Any]
    #expect(error["type"] as? String == "generation_failed")
    #expect((error["message"] as? String)?.hasPrefix("streaming aborted:") == true)
}

// MARK: - Anthropic streaming wire format

@Test func anthropicUsageWithCacheReadInputTokens() throws {
    // AnthropicMessageResponse.Usage (non-streaming) — uses the typed
    // Codable struct. Maps to Anthropic spec `usage.cache_read_input_tokens`.
    let usage = AnthropicMessageResponse.Usage(
        inputTokens: 100,
        outputTokens: 50,
        cacheReadInputTokens: 75
    )
    let data = try JSONEncoder().encode(usage)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    #expect(json["input_tokens"] as? Int == 100)
    #expect(json["output_tokens"] as? Int == 50)
    #expect(json["cache_read_input_tokens"] as? Int == 75)
}

@Test func anthropicUsageWithoutCacheReadInputTokensOmitsField() throws {
    // Swift's default JSONEncoder omits nil optionals from the payload,
    // so `cache_read_input_tokens` is absent (not `null`) when there's no
    // cache hit. Matches the streaming path's `if cachedTokens > 0` guard
    // — both shapes stay consistent. Clients see "field not present" the
    // same as "field is null".
    let usage = AnthropicMessageResponse.Usage(
        inputTokens: 100,
        outputTokens: 50,
        cacheReadInputTokens: nil
    )
    let data = try JSONEncoder().encode(usage)
    let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    #expect(json["cache_read_input_tokens"] == nil)
    #expect(json["input_tokens"] as? Int == 100)
    #expect(json["output_tokens"] as? Int == 50)
}

@Test func anthropicStreamOrderMessageStartToMessageStop() throws {
    // Builds a complete Anthropic stream in the order AnthropicMessages.swift
    // emits it, then verifies the event sequence. Streaming payload is
    // built via [String: Any] in the closure, so we mirror that here to
    // lock the wire shape before the planned #58 refactor.
    let stream = """
    event: message_start
    data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}

    event: content_block_start
    data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

    event: content_block_delta
    data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}

    event: content_block_stop
    data: {"type":"content_block_stop","index":0}

    event: message_delta
    data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":5,"output_tokens":1}}

    event: message_stop
    data: {"type":"message_stop"}

    """

    let events = parseSSE(stream)
    #expect(events.count == 6)
    #expect(events.map(\.event) == [
        "message_start", "content_block_start", "content_block_delta",
        "content_block_stop", "message_delta", "message_stop",
    ])
}

@Test func anthropicMessageDeltaUsageIncludesCacheReadInputTokens() throws {
    // Streaming path emits cache_read_input_tokens inside the message_delta
    // event's usage block, conditional on cachedTokens > 0
    // (AnthropicMessages.swift `message_delta` payload).
    let usageDict: [String: Any] = [
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 75,
    ]
    let usageData = try JSONSerialization.data(withJSONObject: usageDict)
    let usageJSON = String(data: usageData, encoding: .utf8)!
    let stream = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":\(usageJSON)}\n\n"

    let events = parseSSE(stream)
    #expect(events.count == 1)
    #expect(events[0].event == "message_delta")
    let payload = try JSONSerialization.jsonObject(with: Data(events[0].data.utf8)) as! [String: Any]
    let usage = payload["usage"] as! [String: Any]
    #expect(usage["input_tokens"] as? Int == 100)
    #expect(usage["output_tokens"] as? Int == 50)
    #expect(usage["cache_read_input_tokens"] as? Int == 75)
}

@Test func anthropicMessageDeltaUsageOmitsCacheReadInputTokensWhenZero() throws {
    // Streaming path: when cachedTokens == 0, the field is absent
    // (the `if cachedTokens > 0` branch in AnthropicMessages.swift).
    let usageDict: [String: Any] = [
        "input_tokens": 100,
        "output_tokens": 50,
    ]
    let usageData = try JSONSerialization.data(withJSONObject: usageDict)
    let usageJSON = String(data: usageData, encoding: .utf8)!
    let stream = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":\(usageJSON)}\n\n"

    let events = parseSSE(stream)
    #expect(events.count == 1)
    let payload = try JSONSerialization.jsonObject(with: Data(events[0].data.utf8)) as! [String: Any]
    let usage = payload["usage"] as! [String: Any]
    #expect(usage["cache_read_input_tokens"] == nil)
}
