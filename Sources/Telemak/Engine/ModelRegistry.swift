import Foundation
import MLXEmbedders
import MLXLMCommon
import TelemakMTP

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
        public let isVision: Bool
        public let mtpCompatibility: MTPCompatibility
    }

    /// MTP-draft companions to main models. Drafts have very different
    /// lifecycle from main models : no tokenizer, no chat session, can't
    /// be served standalone, only invoked from inside the speculative
    /// loop when verifying a paired target. We keep them in a separate
    /// store so they don't pollute `/v1/models` or chat dispatch.
    public struct LoadedDraft: Sendable {
        public let id: String
        public let mainId: String
        public let model: MTPModelLoader.LoadedDraftModel
        public let loadedAt: Date
        public let ramEstimateBytes: Int64
        public let mtpCompatibility: MTPCompatibility
    }

    public struct LoadedEmbedder: Sendable {
        public let id: String
        public let container: EmbedderModelContainer
        public let loadedAt: Date
        public let ramEstimateBytes: Int64
    }

    private var entries: [String: Loaded] = [:]
    private var draftEntries: [String: LoadedDraft] = [:]
    private var embedderEntries: [String: LoadedEmbedder] = [:]
    /// `mainId → draftId`. Each main model can have ≤ 1 paired draft.
    private var pairing: [String: String] = [:]
    private let stateStore: StateStore?
    private let sessionStore: SessionStore?
    private let wiredMemory: WiredMemoryCoordinator?
    private var pendingLoadBytes: Int64 = 0

    public init(
        stateStore: StateStore? = nil,
        sessionStore: SessionStore? = nil,
        wiredMemory: WiredMemoryCoordinator? = nil
    ) {
        self.stateStore = stateStore
        self.sessionStore = sessionStore
        self.wiredMemory = wiredMemory
    }

    // MARK: - Reads

    public var loadedModelIds: [String] { entries.keys.sorted() }
    public var loadedModels: [Loaded] { entries.values.sorted { $0.id < $1.id } }
    public var loadedEmbedders: [LoadedEmbedder] { embedderEntries.values.sorted { $0.id < $1.id } }

    public func get(_ id: String) -> Loaded? { entries[id] }
    public func getEmbedder(_ id: String) -> LoadedEmbedder? { embedderEntries[id] }

    public func contains(_ id: String) -> Bool { entries[id] != nil || embedderEntries[id] != nil }

    public func usedRamBytes() -> Int64 {
        let mainBytes = entries.values.reduce(0) { $0 + $1.ramEstimateBytes }
        let draftBytes = draftEntries.values.reduce(0) { $0 + $1.ramEstimateBytes }
        let embedderBytes = embedderEntries.values.reduce(0) { $0 + $1.ramEstimateBytes }
        return mainBytes + draftBytes + embedderBytes
    }

    public func perModelRamBytes() -> [String: Int64] {
        var out: [String: Int64] = [:]
        for (id, entry) in entries { out[id] = entry.ramEstimateBytes }
        for (id, entry) in draftEntries { out[id] = entry.ramEstimateBytes }
        for (id, entry) in embedderEntries { out[id] = entry.ramEstimateBytes }
        return out
    }

    /// Returns the draft model id paired with `mainId`, if any.
    public func draftId(for mainId: String) -> String? { pairing[mainId] }

    /// Returns the loaded draft entry for a given draft id (not main id).
    public func getDraft(_ draftId: String) -> LoadedDraft? { draftEntries[draftId] }

    /// All currently loaded draft models. Excluded from `/v1/models`.
    public var loadedDrafts: [LoadedDraft] {
        draftEntries.values.sorted { $0.id < $1.id }
    }

    /// Pairs currently active.
    public var activePairs: [(main: String, draft: String)] {
        pairing.sorted { $0.key < $1.key }.map { ($0.key, $0.value) }
    }

    public struct ActivePairInfo: Sendable {
        public let main: String
        public let draft: String
        public let compatibility: MTPCompatibility
    }

    public var activePairInfo: [ActivePairInfo] {
        pairing.sorted { $0.key < $1.key }.map { mainId, draftId in
            ActivePairInfo(
                main: mainId,
                draft: draftId,
                compatibility: draftEntries[draftId]?.mtpCompatibility
                    ?? MTPCompatibility.unavailable(modelId: draftId)
            )
        }
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
    public func load(_ id: String, forceLLM: Bool = false) async throws -> ModelContainer {
        let id = ModelLoader.canonicalIdentifier(id)
        Self.dbg("entry id=\(id) forceLLM=\(forceLLM)")
        if let existing = entries[id] {
            if forceLLM && existing.isVision {
                throw LoadError.loadFailed(
                    underlying: "model '\(id)' is already loaded as VLM; unload it before pairing an MTP draft"
                )
            }
            Self.dbg("hit id=\(id)")
            return existing.container
        }

        Self.dbg("estimate start id=\(id)")
        let estimateStart = Date()
        let neededBytes = RamBudget.estimate(modelId: id) ?? 0
        Self.dbg("estimate done id=\(id) bytes=\(neededBytes) took=\(Date().timeIntervalSince(estimateStart))s")
        var reservedBudget = try reservePendingLoadBudget(neededBytes: neededBytes)
        defer {
            if reservedBudget {
                releasePendingLoadBudget(neededBytes)
            }
        }

        Self.dbg("before await ModelLoader.load id=\(id)")
        let loadStart = Date()
        let container: ModelContainer
        do {
            container = try await ModelLoader.load(identifier: id, forceLLM: forceLLM)
        } catch {
            Self.dbg("ModelLoader.load THREW id=\(id) error=\(error)")
            throw LoadError.loadFailed(underlying: "\(error)")
        }
        Self.dbg("after await ModelLoader.load id=\(id) took=\(Date().timeIntervalSince(loadStart))s")

        let loaded = Loaded(
            id: id, container: container,
            loadedAt: Date(),
            ramEstimateBytes: neededBytes,
            isVision: forceLLM ? false : ModelLoader.isVisionModel(identifier: id),
            mtpCompatibility: Self.mtpCompatibility(for: id, isDraft: false)
        )
        entries[id] = loaded
        if reservedBudget {
            releasePendingLoadBudget(neededBytes)
            reservedBudget = false
        }
        Self.dbg("entries updated id=\(id)")
        if neededBytes > 0 {
            await wiredMemory?.reserveModel(id, weightBytes: Int(neededBytes))
        }
        await persistState()
        Self.dbg("persistState done id=\(id) total_wall=\(Date().timeIntervalSince(estimateStart))s")
        return container
    }

    @discardableResult
    public func loadEmbedder(_ id: String) async throws -> EmbedderModelContainer {
        let id = ModelLoader.canonicalIdentifier(id)
        if let existing = embedderEntries[id] {
            return existing.container
        }
        let neededBytes = RamBudget.estimate(modelId: id) ?? 0
        var reservedBudget = try reservePendingLoadBudget(neededBytes: neededBytes)
        defer {
            if reservedBudget {
                releasePendingLoadBudget(neededBytes)
            }
        }

        let container: EmbedderModelContainer
        do {
            container = try await EmbedderLoader.load(identifier: id)
        } catch {
            throw LoadError.loadFailed(underlying: "\(error)")
        }
        embedderEntries[id] = LoadedEmbedder(
            id: id,
            container: container,
            loadedAt: Date(),
            ramEstimateBytes: neededBytes
        )
        if reservedBudget {
            releasePendingLoadBudget(neededBytes)
            reservedBudget = false
        }
        if neededBytes > 0 {
            await wiredMemory?.reserveModel(id, weightBytes: Int(neededBytes))
        }
        await persistState()
        return container
    }

    /// Load an MTP draft `draftId` paired with `mainId`. The main model
    /// must already be loaded (caller's responsibility — typically the
    /// /admin/load handler loads main first, then draft).
    ///
    /// The draft can't be served standalone — it lives only in the
    /// `draftEntries` map and is invoked by the speculative loop when
    /// chat completions target `mainId`.
    @discardableResult
    public func loadDraft(
        _ draftId: String,
        pairedWith mainId: String,
        allowUnverified: Bool = false
    ) async throws -> MTPModelLoader.LoadedDraftModel {
        let draftId = ModelLoader.canonicalIdentifier(draftId)
        let mainId = ModelLoader.canonicalIdentifier(mainId)
        if let existing = draftEntries[draftId] {
            // Reuse if already loaded. Update the pairing in case the
            // same draft got attached to a different main (rare).
            guard existing.mtpCompatibility.canRun(allowUnverified: allowUnverified || Self.allowUnverifiedMTPFromEnvironment()) else {
                throw LoadError.loadFailed(underlying: existing.mtpCompatibility.rejectedMessage(modelId: draftId))
            }
            pairing[mainId] = draftId
            await persistState()
            return existing.model
        }
        guard entries[mainId] != nil else {
            throw LoadError.loadFailed(
                underlying: "main model '\(mainId)' not loaded — pair draft after main"
            )
        }
        let isEmbeddedDraft = draftId == mainId
        let compatibility = Self.mtpCompatibility(for: draftId, isDraft: !isEmbeddedDraft)
        guard compatibility.canRun(allowUnverified: allowUnverified || Self.allowUnverifiedMTPFromEnvironment()) else {
            throw LoadError.loadFailed(underlying: compatibility.rejectedMessage(modelId: draftId))
        }
        let neededBytes = RamBudget.estimate(modelId: draftId) ?? 0
        var reservedBudget = try reservePendingLoadBudget(neededBytes: neededBytes)
        defer {
            if reservedBudget {
                releasePendingLoadBudget(neededBytes)
            }
        }
        let model: MTPModelLoader.LoadedDraftModel
        do {
            model = try MTPModelLoader.load(identifier: draftId, embedded: isEmbeddedDraft)
        } catch {
            throw LoadError.loadFailed(underlying: "\(error)")
        }
        draftEntries[draftId] = LoadedDraft(
            id: draftId, mainId: mainId, model: model,
            loadedAt: Date(), ramEstimateBytes: neededBytes,
            mtpCompatibility: compatibility
        )
        if reservedBudget {
            releasePendingLoadBudget(neededBytes)
            reservedBudget = false
        }
        pairing[mainId] = draftId
        if neededBytes > 0 {
            await wiredMemory?.reserveModel(draftId, weightBytes: Int(neededBytes))
        }
        await persistState()
        return model
    }

    /// Stderr diagnostic, gated on `TELEMAK_LOAD_DEBUG` env var. Off by
    /// default so production logs don't fill up. Set the var on the
    /// LaunchAgent plist to capture load timing breakdowns:
    ///
    ///   <key>TELEMAK_LOAD_DEBUG</key><string>1</string>
    ///
    /// Output format: `[telemak.mrl] <step> id=<model> [took=<seconds>s]`.
    private static let loadDebugEnabled: Bool = {
        !(ProcessInfo.processInfo.environment["TELEMAK_LOAD_DEBUG"] ?? "").isEmpty
    }()

    private static func dbg(_ msg: String) {
        guard loadDebugEnabled else { return }
        FileHandle.standardError.write(Data("[telemak.mrl] \(msg)\n".utf8))
    }

    private func reservePendingLoadBudget(neededBytes: Int64) throws -> Bool {
        guard neededBytes > 0 else { return false }
        let ceiling = RamBudget.ceilingBytes()
        guard ceiling > 0 else { return false }

        let committedBytes = usedRamBytes()
        let availableBytes = max(0, ceiling - committedBytes - pendingLoadBytes)
        if neededBytes > availableBytes {
            throw LoadError.insufficientMemory(
                neededBytes: neededBytes,
                availableBytes: availableBytes,
                ceilingBytes: ceiling,
                currentlyLoaded: allLoadedIds()
            )
        }

        pendingLoadBytes += neededBytes
        return true
    }

    private func releasePendingLoadBudget(_ bytes: Int64) {
        guard bytes > 0 else { return }
        pendingLoadBytes = max(0, pendingLoadBytes - bytes)
    }

    private func allLoadedIds() -> [String] {
        loadedModelIds + embedderEntries.keys.sorted() + draftEntries.keys.sorted()
    }

    private static func mtpCompatibility(for id: String, isDraft: Bool) -> MTPCompatibility {
        guard let directory = ModelLoader.resolvedModelDirectory(for: id) else {
            return MTPCompatibility.unavailable(modelId: id)
        }
        return MTPCompatibility.inspect(modelId: id, directory: directory, isDraft: isDraft)
    }

    private static func allowUnverifiedMTPFromEnvironment() -> Bool {
        let value = (ProcessInfo.processInfo.environment["TELEMAK_MTP_ALLOW_UNVERIFIED"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        return ["1", "true", "yes", "on"].contains(value)
    }

    /// Unload one model by id. Returns true if the model was loaded.
    /// Drops every session bound to it, the paired draft (if any), and
    /// any pairing entry. The id can refer to either a main model
    /// (drops main + paired draft) or a draft directly (drops draft +
    /// clears the pairing back-reference).
    @discardableResult
    public func unload(_ id: String) async -> Bool {
        let id = ModelLoader.canonicalIdentifier(id)
        if entries.removeValue(forKey: id) != nil {
            await wiredMemory?.endReservation(id)
            await sessionStore?.invalidateModel(id)
            // If this main had a paired draft, drop it too.
            if let draftId = pairing.removeValue(forKey: id) {
                draftEntries.removeValue(forKey: draftId)
                await wiredMemory?.endReservation(draftId)
            }
            await persistState()
            return true
        }
        if let draft = draftEntries.removeValue(forKey: id) {
            await wiredMemory?.endReservation(id)
            // Clear the pairing back-reference so the main no longer
            // claims to have a draft attached.
            pairing.removeValue(forKey: draft.mainId)
            await persistState()
            return true
        }
        if embedderEntries.removeValue(forKey: id) != nil {
            await wiredMemory?.endReservation(id)
            await persistState()
            return true
        }
        return false
    }

    /// Unload everything (admin convenience). Clears all sessions, all
    /// drafts, all pairings.
    public func unloadAll() async -> [String] {
        let ids = Array(entries.keys) + Array(draftEntries.keys) + Array(embedderEntries.keys)
        entries.removeAll()
        draftEntries.removeAll()
        embedderEntries.removeAll()
        pairing.removeAll()
        for id in ids {
            await wiredMemory?.endReservation(id)
        }
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
