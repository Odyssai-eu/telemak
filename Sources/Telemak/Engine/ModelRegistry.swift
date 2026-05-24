import Foundation
import MLXLMCommon

/// Holds the currently-loaded model, serializes load/unload/inference.
///
/// MVP policy (V0 + Phase 5): at most one model in memory. Requesting a
/// different model unloads the current one first. Concurrent inference
/// requests on the same model are serialized inside the actor —
/// single-flight by design. Block 1 widens this to a multi-model registry.
///
/// State is mirrored to a `StateStore` (when provided) so the serve command
/// can replay the last-loaded set at boot.
public actor ModelRegistry {

    public struct Loaded: Sendable {
        public let id: String
        public let container: ModelContainer
        public let loadedAt: Date
    }

    private var current: Loaded?
    private let stateStore: StateStore?

    public init(stateStore: StateStore? = nil) {
        self.stateStore = stateStore
    }

    public var loadedModelId: String? { current?.id }

    public var loadedModel: Loaded? { current }

    /// Return the container for `id`, loading it if necessary. If a different
    /// model is currently loaded, it is unloaded first.
    public func ensureLoaded(_ id: String) async throws -> ModelContainer {
        if let current, current.id == id {
            return current.container
        }
        if current != nil {
            current = nil   // drop the reference; ARC frees the model weights
        }
        let container = try await ModelLoader.load(identifier: id)
        let loaded = Loaded(id: id, container: container, loadedAt: Date())
        current = loaded
        await persistState()
        return container
    }

    /// Explicitly unload the current model (no-op if none loaded).
    @discardableResult
    public func unload() async -> String? {
        let id = current?.id
        current = nil
        await persistState()
        return id
    }

    /// Return the id of the loaded model only if it matches `id`. Used for
    /// "is this model loaded right now?" checks without mutating state.
    public func isLoaded(_ id: String) -> Bool {
        current?.id == id
    }

    private func persistState() async {
        guard let stateStore else { return }
        let ids = current.map { [$0.id] } ?? []
        try? await stateStore.update(loadedModels: ids)
    }
}
