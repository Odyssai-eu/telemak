import Foundation
import MLXLMCommon

/// Shared best-effort persist of a `ChatSession`'s KV cache to a fresh
/// file in the operator's session store. Used by both
/// `/v1/chat/completions` and `/v1/messages` (issue #58 — before this
/// helper the two handlers had a copy-pasted version with subtly
/// different error handling: ChatCompletions logged to stderr and
/// distinguished `ChatSessionError.noCacheAvailable` as a no-op;
/// AnthropicMessages silently swallowed everything, which hid
/// disk-full / permission failures from the operator).
///
/// The behavior here matches the former ChatCompletions version:
///   - `noCacheAvailable` (empty generation, tool-only response) is
///     a no-op, not an error — the URL is cleaned up and we move on.
///   - any other failure is logged to stderr with the session id and
///     underlying error, so the operator can spot disk issues. The
///     response is NEVER failed because of a cache-save miss.
///
/// Callers pass the `cacheScope` they used for the matching
/// `SessionStore.hit(...)` call, so the rehydrated cache on the next
/// turn only matches when the prompt-template context (enable_thinking,
/// reasoning_effort) is unchanged.
public enum SessionCachePersistence {

    public static func save(
        session: ChatSession,
        sessionId: String,
        modelId: String,
        cacheScope: String,
        sessionStore: SessionStore
    ) async {
        let url = await sessionStore.nextCacheURL(for: sessionId)
        do {
            try await session.saveCache(to: url)
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
            await sessionStore.update(
                sessionId: sessionId,
                modelId: modelId,
                cacheScope: cacheScope,
                cacheURL: url,
                byteSize: size
            )
        } catch ChatSessionError.noCacheAvailable {
            // No cache to save (empty generation, tool-only response, etc.).
            // Not an error — just skip persistence.
            try? FileManager.default.removeItem(at: url)
        } catch {
            // Genuinely unexpected: log so operators can see if disk is full,
            // permissions broke, etc. Don't fail the response.
            FileHandle.standardError.write(Data("[telemak.kv] saveCache failed for session \(sessionId): \(error)\n".utf8))
            try? FileManager.default.removeItem(at: url)
        }
    }
}
