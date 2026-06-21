import Foundation
import TelemakVersion
import HuggingFace
import MLXEmbedders
import MLXHuggingFace
import MLXLMCommon
import Tokenizers

/// Loader for sentence-embedding checkpoints backed by `MLXEmbedders`.
///
/// Resolution mirrors `ModelLoader`: absolute local path first,
/// `TELEMAK_MODELS_DIR` second, Hugging Face cache/download last.
public enum EmbedderLoader {

    public enum LoadError: Error, CustomStringConvertible {
        case directoryNotFound(id: String)
        case unsupportedModelType(id: String, modelType: String)

        public var description: String {
            switch self {
            case .directoryNotFound(let id):
                return "embedder '\(id)' not found locally"
            case .unsupportedModelType(let id, let modelType):
                return "model '\(id)' has model_type '\(modelType)', not a supported embedder"
            }
        }
    }

    private static let embedderOnlyTypes: Set<String> = [
        "bert", "roberta", "xlm-roberta", "distilbert", "nomic_bert", "gemma3_text",
    ]

    private static let ambiguousEmbedderTypes: Set<String> = [
        "qwen3", "gemma3", "gemma3n",
    ]

    public static func isEmbedder(identifier: String) -> Bool {
        if let dir = resolveLocalDirectory(identifier),
           let modelType = modelType(in: dir) {
            return isEmbedder(identifier: identifier, modelType: modelType)
        }
        return looksLikeEmbeddingId(identifier)
    }

    public static func load(
        identifier: String,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> EmbedderModelContainer {
        if let dir = resolveLocalDirectory(identifier) {
            let modelType = modelType(in: dir) ?? ""
            guard isEmbedder(identifier: identifier, modelType: modelType) else {
                throw LoadError.unsupportedModelType(id: identifier, modelType: modelType)
            }
            let prepared = (try? ModelLoader.prepareConfigForMLX(originalDir: dir, id: identifier)) ?? dir
            return try await EmbedderModelFactory.shared.loadContainer(
                from: prepared,
                using: #huggingFaceTokenizerLoader()
            )
        }

        guard looksLikeEmbeddingId(identifier) else {
            throw LoadError.directoryNotFound(id: identifier)
        }
        return try await EmbedderModelFactory.shared.loadContainer(
            from: #hubDownloader(),
            using: #huggingFaceTokenizerLoader(),
            configuration: ModelConfiguration(id: identifier),
            progressHandler: progressHandler
        )
    }

    private static func resolveLocalDirectory(_ identifier: String) -> URL? {
        if identifier.hasPrefix("/") {
            return ModelLoader.resolveDirectory(at: identifier)
        }
        if let modelsDir = ModelsConfig.shared.effectiveDir(),
           let url = ModelLoader.resolveModelsDir(modelsDir, id: identifier) {
            return url
        }
        let hfPath = ProcessInfo.processInfo.environment["HF_HOME"]
            .map { "\($0)/hub" }
            ?? "\(FileManager.default.homeDirectoryForCurrentUser.path)/.cache/huggingface/hub"
        return resolveHFCache(hfPath, id: identifier)
    }

    private static func resolveHFCache(_ root: String, id: String) -> URL? {
        let cacheId = "models--" + id.replacingOccurrences(of: "/", with: "--")
        let modelBase = (root as NSString).appendingPathComponent(cacheId)
        let snapshotsBase = (modelBase as NSString).appendingPathComponent("snapshots")
        let fm = FileManager.default
        guard let snapshots = try? fm.contentsOfDirectory(atPath: snapshotsBase) else {
            return nil
        }
        let candidates = snapshots
            .filter { !$0.hasPrefix(".") }
            .map { (snapshotsBase as NSString).appendingPathComponent($0) }
            .sorted()
        if let chosen = candidates.last(where: { ModelLoader.hasConfigJSON(at: $0) }) {
            return URL(fileURLWithPath: chosen)
        }
        return nil
    }

    private static func modelType(in directory: URL) -> String? {
        let configPath = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configPath),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let modelType = root["model_type"] as? String
        else {
            return nil
        }
        return modelType
    }

    private static func isEmbedder(identifier: String, modelType: String) -> Bool {
        if embedderOnlyTypes.contains(modelType) { return true }
        if ambiguousEmbedderTypes.contains(modelType) {
            return looksLikeEmbeddingId(identifier)
        }
        return false
    }

    private static func looksLikeEmbeddingId(_ identifier: String) -> Bool {
        let lower = identifier.lowercased()
        return lower.contains("embedding")
            || lower.contains("embed")
            || lower.contains("bge")
            || lower.contains("minilm")
            || lower.contains("gte")
            || lower.contains("e5-")
            || lower.contains("e5_")
            || lower.contains("snowflake")
            || lower.contains("mxbai")
            || lower.contains("nomic")
    }
}
