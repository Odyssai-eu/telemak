import Foundation
import MLXLMCommon
import MLXLLM
import MLXVLM   // Registers the VLM trampoline so qwen3_5 (Qwen3.6 VLM) loads.
import MLXHuggingFace
import HuggingFace
import Tokenizers

/// Resolve a model identifier into a loaded ``ModelContainer``.
///
/// Resolution order:
///   1. If `identifier` is an absolute path to a directory containing
///      `config.json`, load it directly.
///   2. If env var `TELEMAK_MODELS_DIR` is set, try the Odysseus models-dir
///      layout under that root: `<root>/<id>/snapshots/<hash>/`, falling back
///      to `<root>/<id>/` if no snapshot subdir.
///   3. Otherwise download (or hit the HF cache) via HubClient.
///
/// The Odysseus layout is `<org>/<name>/snapshots/<hash>/<files>` — NOT the
/// standard HF cache `models--<org>--<name>/snapshots/...`. We don't try to
/// reuse HubClient for it; we go straight to local-directory loading.
public enum ModelLoader {

    public static func load(
        identifier: String,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        if identifier.hasPrefix("/") {
            if let url = resolveDirectory(at: identifier) {
                return try await loadModelContainer(
                    from: url,
                    using: #huggingFaceTokenizerLoader()
                )
            }
        }

        if let modelsDir = ProcessInfo.processInfo.environment["TELEMAK_MODELS_DIR"],
           let url = resolveModelsDir(modelsDir, id: identifier) {
            return try await loadModelContainer(
                from: url,
                using: #huggingFaceTokenizerLoader()
            )
        }

        let configuration = ModelConfiguration(id: identifier)
        return try await #huggingFaceLoadModelContainer(
            configuration: configuration,
            progressHandler: progressHandler
        )
    }

    /// Resolve `<root>/<id>/snapshots/<hash>/` (or `<root>/<id>/` if no
    /// snapshot dir) into a directory URL that has `config.json`.
    private static func resolveModelsDir(_ root: String, id: String) -> URL? {
        let base = (root as NSString).appendingPathComponent(id)
        let snapshotsBase = (base as NSString).appendingPathComponent("snapshots")
        let fm = FileManager.default

        if let snapshots = try? fm.contentsOfDirectory(atPath: snapshotsBase) {
            let candidates = snapshots
                .filter { !$0.hasPrefix(".") }
                .map { (snapshotsBase as NSString).appendingPathComponent($0) }
                .sorted()   // deterministic — prefer the lexicographically last (newest hex hash often is)
            if let chosen = candidates.last(where: { hasConfigJSON(at: $0) }) {
                return URL(fileURLWithPath: chosen)
            }
        }
        if hasConfigJSON(at: base) {
            return URL(fileURLWithPath: base)
        }
        return nil
    }

    private static func resolveDirectory(at path: String) -> URL? {
        guard hasConfigJSON(at: path) else { return nil }
        return URL(fileURLWithPath: path)
    }

    private static func hasConfigJSON(at directory: String) -> Bool {
        var isDir: ObjCBool = false
        let configPath = (directory as NSString).appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: configPath, isDirectory: &isDir) && !isDir.boolValue
    }
}
