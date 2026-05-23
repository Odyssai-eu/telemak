import Foundation
import MLXLMCommon

/// Holds the currently-loaded model, serializes load/unload/inference.
///
/// MVP policy: at most one model in memory. Requesting a different model
/// unloads the current one first. Concurrent inference requests on the same
/// model are serialized inside the actor — single-flight by design.
public actor ModelRegistry {

    public struct Loaded: Sendable {
        public let id: String
        public let container: ModelContainer
        public let loadedAt: Date
    }

    private var current: Loaded?

    public init() {}

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
        return container
    }

    /// Explicitly unload the current model (no-op if none loaded).
    @discardableResult
    public func unload() -> String? {
        let id = current?.id
        current = nil
        return id
    }

    /// Return the id of the loaded model only if it matches `id`. Used for
    /// "is this model loaded right now?" checks without mutating state.
    public func isLoaded(_ id: String) -> Bool {
        current?.id == id
    }
}
