import Foundation
import Hummingbird

/// Shared Server-Sent-Events writer for the three streaming handlers
/// (`/v1/chat/completions`, `/admin/mtp/smoke` via ChatCompletionsMTP,
/// `/v1/messages`). Frame formats we support:
///
/// - **OpenAI chunk** — `data: <json>\n\n`  (no `event:` line)
/// - **Anthropic event** — `event: <name>\ndata: <json>\n\n`
/// - **OpenAI `[DONE]`** — `data: [DONE]\n\n`
/// - **Stream error** — `data: {"error":{...}}\n\n` (no `event:`)
///
/// This is the same byte-level wire format the handlers emitted before
/// the refactor (issue #58) — the SSE snapshot tests in
/// `Tests/TelemakTests/StreamingSnapshotTests.swift` lock the shape
/// before and after. Don't add anything that changes a byte.
///
/// Created inside a `ResponseBody { ... }` callback so it borrows the
/// Hummingbird `ResponseBodyWriter` for the duration of the response
/// stream. The writer is `Sendable`-confined to a single response; do
/// not share across concurrent responses.
///
/// Class (not struct) because `ResponseBodyWriter.write` is `mutating`
/// — wrapping in a class lets callers keep a `let sse = ...` binding
/// while the underlying writer's buffer is mutated in place. Each
/// response creates its own instance; the class is single-use.
final class SSEWriter {
    /// `var` (not `let`) because `ResponseBodyWriter.write` is `mutating`
    /// — the existential box gets reassigned as the underlying writer
    /// consumes ByteBuffers. With a class-backed writer (the common
    /// Hummingbird case) the box copy preserves the same instance, so
    /// mutations propagate back to the caller.
    private var writer: any ResponseBodyWriter
    private let encoder: JSONEncoder

    init(writer: any ResponseBodyWriter, encoder: JSONEncoder = JSONEncoder()) {
        self.writer = writer
        self.encoder = encoder
    }

    // MARK: - OpenAI / default

    /// `data: <chunk-json>\n\n`. Used for every `ChatCompletionChunk`
    /// (role, content delta, tool call, finish_reason) and for the
    /// post-finish `usage` chunk.
    func write<T: Encodable>(data chunk: T) async throws {
        let bytes = try encoder.encode(chunk)
        try await writeRawDataLine(bytes)
    }

    /// `data: <dict-json>\n\n`. Used for the OpenAI `usage` chunk
    /// (hand-built `[String: Any]` because the field is conditional on
    /// `cached_tokens > 0` — see the chat-completions spec for the
    /// `prompt_tokens_details` shape).
    func write(data dict: [String: Any]) async throws {
        let bytes = (try? JSONSerialization.data(withJSONObject: dict)) ?? Data("{}".utf8)
        try await writeRawDataLine(bytes)
    }

    /// `data: [DONE]\n\n`. OpenAI terminator. Sent AFTER the `usage`
    /// chunk so clients that only read usage still see the end marker.
    func writeDone() async throws {
        try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
    }

    /// `data: {"error":{...}}\n\n`. Best-effort — failures during the
    /// error path itself are swallowed (we're already on the error
    /// branch; no point throwing further).
    func writeError(message: String, type: String = "generation_failed") async throws {
        let payload: [String: Any] = [
            "error": [
                "message": message,
                "type": type,
            ]
        ]
        let bytes = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        var buf = ByteBuffer()
        buf.writeString("data: ")
        buf.writeBytes(bytes)
        buf.writeString("\n\n")
        try? await writer.write(buf)
    }

    // MARK: - Anthropic

    /// `event: <name>\ndata: <json>\n\n`. Anthropic events carry the
    /// `event:` line so the client can route by event type instead of
    /// by parsing the payload. Used for `message_start`,
    /// `content_block_start/delta/stop`, `message_delta`, `message_stop`.
    func write(event name: String, data dict: [String: Any]) async throws {
        let bytes = (try? JSONSerialization.data(withJSONObject: dict)) ?? Data("{}".utf8)
        var buf = ByteBuffer()
        buf.writeString("event: ")
        buf.writeString(name)
        buf.writeString("\n")
        buf.writeString("data: ")
        buf.writeBytes(bytes)
        buf.writeString("\n\n")
        try await writer.write(buf)
    }

    /// `event: error\ndata: <json>\n\n`. Anthropic-shaped error event
    /// (the wire is event-prefixed, unlike the OpenAI stream error
    /// which is just a `data:` line).
    func writeEventError(message: String, type: String = "api_error") async throws {
        let payload: [String: Any] = [
            "type": "error",
            "error": ["type": type, "message": message],
        ]
        try await write(event: "error", data: payload)
    }

    // MARK: - shared

    /// Internal: `data: <bytes>\n\n`. The two `data:` overloads above
    /// both delegate here so the framing is defined in exactly one
    /// place.
    private func writeRawDataLine(_ bytes: Data) async throws {
        var buf = ByteBuffer()
        buf.writeString("data: ")
        buf.writeBytes(bytes)
        buf.writeString("\n\n")
        try await writer.write(buf)
    }
}
