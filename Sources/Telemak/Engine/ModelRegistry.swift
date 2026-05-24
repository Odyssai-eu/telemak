import Foundation
import MLXLMCommon

/// Holds N loaded models, allowing concurrent inference on different models.
///
/// Concurrency model (V1):
///   - Each loaded model is wrapped in its own `ModelContainer` actor (from
///     mlx-swift-lm), which serializes requests on that specific model.
///   - Different models run truly in parallel — Task groups on different
///     containers don't block each other.
///   - The registry actor itself only serializes the metadata-mutating
///     operations (`add`, `remove`); reads (`get`, `list`) are fast.
///
/// Memory policy (V1):
///   - No auto-eviction. If `add` would exceed the ceiling, the caller is
///     expected to surface a 400 with a memory breakdown.
///   - The operator unloads explicitly via `/admin/unload`.
public actor ModelRegistry {

    public struct Loaded: Sendable {
        public let id: String
        public let container: ModelContainer
        public let loadedAt: Date
        public let ramEstimateBytes: Int64
    }

    private var entries: [String: Loaded] = [:]
    private let stateStore: StateStore?
    private let sessionStore: SessionStore?

    public init(stateStore: StateStore? = nil, sessionStore: SessionStore? = nil) {
        self.stateStore = stateStore
        self.sessionStore = sessionStore
    }

    // MARK: - Reads

    public var loadedModelIds: [String] { entries.keys.sorted() }
    public var loadedModels: [Loaded] { entries.values.sorted { $0.id < $1.id } }

    public func get(_ id: String) -> Loaded? { entries[id] }

    public func contains(_ id: String) -> Bool { entries[id] != nil }

    public func usedRamBytes() -> Int64 {
        entries.values.reduce(0) { $0 + $1.ramEstimateBytes }
    }

    public func perModelRamBytes() -> [String: Int64] {
        var out: [String: Int64] = [:]
        for (id, entry) in entries { out[id] = entry.ramEstimateBytes }
        return out
    }

    // MARK: - Writes

    public enum LoadError: Error, Sendable {
        /// Model load would push us over the RAM ceiling. Contains the
        /// breakdown so the HTTP layer can return a structured 400.
        case insufficientMemory(neededBytes: Int64, availableBytes: Int64, ceilingBytes: Int64, currentlyLoaded: [String])
        case loadFailed(underlying: String)
    }

    /// Load `id` if not already loaded. Returns the container either way
    /// (no-op if hit). Throws `LoadError.insufficientMemory` when the
    /// requested model would push us over the ceiling — the operator must
    /// `unload` first.
    @discardableResult
    public func load(_ id: String) async throws -> ModelContainer {
        if let existing = entries[id] {
            return existing.container
        }

        let neededBytes = RamBudget.estimate(modelId: id) ?? 0
        let usedBytes = usedRamBytes()
        let ceiling = RamBudget.ceilingBytes()

        // If we couldn't estimate (model not in any known dir), let the
        // loader try anyway — it'll fail with a clearer error. We only
        // refuse when we DO have an estimate that exceeds the ceiling.
        if neededBytes > 0, ceiling > 0, usedBytes + neededBytes > ceiling {
            throw LoadError.insufficientMemory(
                neededBytes: neededBytes,
                availableBytes: max(0, ceiling - usedBytes),
                ceilingBytes: ceiling,
                currentlyLoaded: loadedModelIds
            )
        }

        let container: ModelContainer
        do {
            container = try await ModelLoader.load(identifier: id)
        } catch {
            throw LoadError.loadFailed(underlying: "\(error)")
        }

        let loaded = Loaded(
            id: id, container: container,
            loadedAt: Date(),
            ramEstimateBytes: neededBytes
        )
        entries[id] = loaded
        await persistState()
        return container
    }

    /// Unload one model by id. Returns true if the model was loaded.
    /// Also drops every session whose cache is bound to this model.
    @discardableResult
    public func unload(_ id: String) async -> Bool {
        guard entries.removeValue(forKey: id) != nil else { return false }
        await sessionStore?.invalidateModel(id)
        await persistState()
        return true
    }

    /// Unload everything (admin convenience). Clears all sessions.
    public func unloadAll() async -> [String] {
        let ids = Array(entries.keys)
        entries.removeAll()
        _ = await sessionStore?.clearAll()
        await persistState()
        return ids
    }

    // MARK: - State persistence

    private func persistState() async {
        guard let stateStore else { return }
        try? await stateStore.update(loadedModels: loadedModelIds)
    }
}
