import Foundation
import MLXLMCommon

/// Per-session KV cache storage on disk.
///
/// mlx-swift-lm doesn't expose a public way to extract the live `[KVCache]`
/// from a `ChatSession`, but `session.saveCache(to:)` and the free function
/// `loadPromptCache(url:)` are public. So we persist each session's cache
/// to a file under `~/.telemak/sessions/` after every turn, and rehydrate
/// it on the next turn keyed by `session_id`.
///
/// LRU eviction: when entry count exceeds `maxSessions`, the oldest by
/// `lastUsed` is dropped (cache file deleted).
public actor SessionStore {

    public struct Entry: Sendable {
        public let sessionId: String
        public let modelId: String
        public let cacheURL: URL
        public var lastUsed: Date
        public var byteSize: Int64
    }

    private var entries: [String: Entry] = [:]
    private let maxSessions: Int
    private let directory: URL

    public init(
        directory: URL? = nil,
        maxSessions: Int = SessionStore.defaultMaxSessions()
    ) {
        let dir = directory ?? FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent(".telemak")
            .appendingPathComponent("sessions")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        self.directory = dir
        self.maxSessions = maxSessions
    }

    public static func defaultMaxSessions() -> Int {
        if let env = ProcessInfo.processInfo.environment["TELEMAK_MAX_SESSIONS"],
           let n = Int(env), n > 0 {
            return n
        }
        return 32
    }

    public var sessionCount: Int { entries.count }

    // MARK: - Snapshots for admin

    public struct Snapshot: Sendable, Encodable {
        public let id: String
        public let model: String
        public let lastUsedAgoSeconds: Double
        public let kvSizeMB: Double

        enum CodingKeys: String, CodingKey {
            case id
            case model
            case lastUsedAgoSeconds = "last_used_s"
            case kvSizeMB = "kv_size_mb"
        }
    }

    public func snapshots() -> [Snapshot] {
        let now = Date()
        return entries.values
            .map {
                Snapshot(
                    id: $0.sessionId,
                    model: $0.modelId,
                    lastUsedAgoSeconds: now.timeIntervalSince($0.lastUsed),
                    kvSizeMB: Double($0.byteSize) / 1_048_576.0
                )
            }
            .sorted { $0.lastUsedAgoSeconds < $1.lastUsedAgoSeconds }
    }

    // MARK: - Hit / miss

    /// Return the cache URL for `(sessionId, modelId)` if present. If a
    /// session is found but the model differs (operator loaded a new model
    /// for this id), the old cache is dropped and `nil` is returned —
    /// caller should run full prefill.
    public func hit(sessionId: String, modelId: String) async -> URL? {
        guard let entry = entries[sessionId] else { return nil }
        if entry.modelId != modelId {
            removeFile(at: entry.cacheURL)
            entries.removeValue(forKey: sessionId)
            return nil
        }
        return entry.cacheURL
    }

    /// Update the cache file URL after a successful generation. If new entry
    /// pushes us over `maxSessions`, evicts the LRU entry.
    public func update(
        sessionId: String,
        modelId: String,
        cacheURL: URL,
        byteSize: Int64
    ) async {
        let now = Date()
        // If we're replacing an existing entry with a different file path,
        // remove the old file.
        if let existing = entries[sessionId], existing.cacheURL != cacheURL {
            removeFile(at: existing.cacheURL)
        }
        entries[sessionId] = Entry(
            sessionId: sessionId,
            modelId: modelId,
            cacheURL: cacheURL,
            lastUsed: now,
            byteSize: byteSize
        )
        evictIfNeeded()
    }

    @discardableResult
    public func clear(sessionId: String) async -> Bool {
        guard let entry = entries.removeValue(forKey: sessionId) else { return false }
        removeFile(at: entry.cacheURL)
        return true
    }

    public func clearAll() async -> [String] {
        let ids = Array(entries.keys)
        for (_, entry) in entries {
            removeFile(at: entry.cacheURL)
        }
        entries.removeAll()
        return ids
    }

    /// Drop every session that targets `modelId` — used when the operator
    /// unloads a model. The cached KV state is bound to specific weights so
    /// it would mis-fire if reused against a different model with the same
    /// session_id.
    @discardableResult
    public func invalidateModel(_ modelId: String) async -> [String] {
        let toRemove = entries.filter { $0.value.modelId == modelId }.map(\.key)
        for sid in toRemove {
            if let entry = entries.removeValue(forKey: sid) {
                removeFile(at: entry.cacheURL)
            }
        }
        return toRemove
    }

    // MARK: - File helpers

    /// Generate a fresh URL for writing a new cache snapshot. Caller is
    /// expected to write to this path then call `update` with the resulting
    /// byte size.
    public func nextCacheURL(for sessionId: String) -> URL {
        let suffix = UUID().uuidString.lowercased()
        let safeId = sanitize(sessionId)
        return directory.appendingPathComponent("\(safeId)-\(suffix).safetensors")
    }

    private func sanitize(_ raw: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(.init(charactersIn: "-_"))
        return String(raw.unicodeScalars.filter { allowed.contains($0) }.prefix(40))
    }

    private func removeFile(at url: URL) {
        try? FileManager.default.removeItem(at: url)
    }

    private func evictIfNeeded() {
        guard entries.count > maxSessions else { return }
        let toEvict = entries.count - maxSessions
        let lru = entries.values.sorted { $0.lastUsed < $1.lastUsed }.prefix(toEvict)
        for entry in lru {
            removeFile(at: entry.cacheURL)
            entries.removeValue(forKey: entry.sessionId)
        }
    }
}
