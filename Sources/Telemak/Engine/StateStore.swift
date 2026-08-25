import Foundation

/// Persistent on-disk record of which models the operator wants loaded.
///
/// Persisted to `~/.telemak/state.json`. The serve command reads it at boot
/// and replays the loads (best-effort: failures are logged, not fatal).
///
/// `attemptingReplay` is a write-ahead (WAL) marker for the boot replay:
/// written BEFORE a replay load, cleared after it returns. A residual
/// marker at boot means the previous boot DIED during that load — MLX
/// GPU-timeout aborts the process from a Metal completion handler
/// (`std::terminate` → SIGABRT on a framework thread), which Swift cannot
/// catch. The serve command's circuit breaker consumes the marker and
/// drops that model from the replay list so the next boot doesn't
/// crash-loop on it (issue #75).
///
/// Phase 5 stays single-model (the registry only holds one container) but the
/// schema is already an array so Block 1 can drop a multi-model registry on
/// top without changing the on-disk format.
public struct PersistedState: Codable, Sendable {
    public var loadedModels: [String]
    public var attemptingReplay: String?
    public var lastUpdated: String

    enum CodingKeys: String, CodingKey {
        case loadedModels = "loaded_models"
        case attemptingReplay = "attempting_replay"
        case lastUpdated = "last_updated"
    }

    public init(loadedModels: [String], attemptingReplay: String? = nil, lastUpdated: Date = Date()) {
        self.loadedModels = loadedModels
        self.attemptingReplay = attemptingReplay
        self.lastUpdated = Self.iso8601(lastUpdated)
    }

    private static func iso8601(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }

    /// Copy with a new loaded set and a fresh timestamp, **preserving the
    /// write-ahead marker** (see `StateStore.update(loadedModels:)` for why
    /// the marker must survive loaded-set writes).
    public func withLoadedModels(_ ids: [String]) -> PersistedState {
        PersistedState(loadedModels: ids, attemptingReplay: attemptingReplay)
    }

    /// Circuit breaker, pure half (unit-testable without Metal): consume a
    /// residual write-ahead marker — the model whose load killed the
    /// previous boot — by dropping it from the replay list. Returns the
    /// healed state and the dropped model id (nil when there was no
    /// marker to consume).
    public func consumingStuckReplay() -> (state: PersistedState, stuckModel: String?) {
        guard let stuck = attemptingReplay else { return (self, nil) }
        var healed = self
        healed.attemptingReplay = nil
        healed.loadedModels.removeAll { $0 == stuck }
        return (healed, stuck)
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

    /// Replace the loaded-model set, **preserving any write-ahead replay
    /// marker** (actor-serialized read-modify-write).
    ///
    /// Why RMW instead of a blind write: `ModelRegistry.persistState()` calls
    /// this after every successful load — including the loads performed by
    /// the boot replay. A blind write would reset `attemptingReplay` to nil
    /// mid-replay and the WAL's correctness would silently depend on the
    /// exact ordering of registry persists vs replay markers. Preserving the
    /// field makes the marker sticky until an explicit
    /// mark/clear/consumeStuckReplay, so the WAL stays coherent regardless
    /// of call order. Every mutation below is a read-modify-write fully
    /// contained in one actor method, so no interleaving can tear one.
    public func update(loadedModels: [String]) throws {
        let base = read() ?? PersistedState(loadedModels: [])
        try write(base.withLoadedModels(loadedModels))
    }

    /// Write-ahead: record the model the replay is about to load. Must hit
    /// disk BEFORE `ModelRegistry.load` runs — if that load kills the
    /// process (uncatchable Metal abort), this marker is what the next
    /// boot's `consumeStuckReplay()` breaker consumes.
    public func markAttemptingReplay(_ id: String) throws {
        var state = read() ?? PersistedState(loadedModels: [])
        state.attemptingReplay = id
        try write(state)
    }

    /// Clear the write-ahead marker. Called when the load returned without
    /// killing the process — success or a recoverable Swift error.
    public func clearAttemptingReplay() throws {
        var state = read() ?? PersistedState(loadedModels: [])
        guard state.attemptingReplay != nil else { return }
        state.attemptingReplay = nil
        try write(state)
    }

    /// Circuit breaker at boot: if the previous boot died mid-replay
    /// (residual marker), drop that model from the replay list and clear
    /// the marker so this boot doesn't crash-loop on the same load.
    /// Returns the dropped model id, if any.
    public func consumeStuckReplay() throws -> String? {
        guard let current = read() else { return nil }
        let (healed, stuck) = current.consumingStuckReplay()
        guard let stuck else { return nil }
        try write(healed)
        return stuck
    }
}
