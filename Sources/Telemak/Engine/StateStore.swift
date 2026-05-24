import Foundation

/// Persistent on-disk record of which models the operator wants loaded.
///
/// Persisted to `~/.telemak/state.json`. The serve command reads it at boot
/// and replays the loads (best-effort: failures are logged, not fatal).
///
/// Phase 5 stays single-model (the registry only holds one container) but the
/// schema is already an array so Block 1 can drop a multi-model registry on
/// top without changing the on-disk format.
public struct PersistedState: Codable, Sendable {
    public var loadedModels: [String]
    public var lastUpdated: String

    enum CodingKeys: String, CodingKey {
        case loadedModels = "loaded_models"
        case lastUpdated = "last_updated"
    }

    public init(loadedModels: [String], lastUpdated: Date = Date()) {
        self.loadedModels = loadedModels
        self.lastUpdated = Self.iso8601(lastUpdated)
    }

    private static func iso8601(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }
}

public actor StateStore {
    public static let defaultPath: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".telemak").appendingPathComponent("state.json")
    }()

    private let path: URL

    public init(path: URL = StateStore.defaultPath) {
        self.path = path
    }

    public func read() -> PersistedState? {
        guard FileManager.default.fileExists(atPath: path.path) else { return nil }
        guard let data = try? Data(contentsOf: path) else { return nil }
        return try? JSONDecoder().decode(PersistedState.self, from: data)
    }

    public func write(_ state: PersistedState) throws {
        let dir = path.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(state)
        try data.write(to: path, options: .atomic)
    }

    public func update(loadedModels: [String]) throws {
        try write(PersistedState(loadedModels: loadedModels))
    }
}
