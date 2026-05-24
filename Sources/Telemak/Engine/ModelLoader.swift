import Foundation
import MLXLMCommon
import MLXLLM
import MLXVLM
import MLXHuggingFace
import HuggingFace
import Tokenizers

/// Force-link `MLXVLM`'s ObjC trampoline class.
///
/// `import MLXVLM` alone isn't enough — Swift's linker dead-strips classes
/// nothing references. `ModelFactoryRegistry` finds the trampoline via
/// `NSClassFromString("MLXVLM.TrampolineModelFactory")` at runtime; if the
/// class metadata isn't in the binary, lookup returns nil and the registry
/// falls back to LLM-only loading. Touching the class with a static
/// reference pins it in.
@MainActor
private let _mlxvlmTrampolineLoad: VLMModelFactory.Type = VLMModelFactory.self

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
                let prepared = (try? prepareConfigForMLX(originalDir: url, id: identifier)) ?? url
                return try await loadModelContainer(
                    from: prepared,
                    using: #huggingFaceTokenizerLoader()
                )
            }
        }

        if let modelsDir = ProcessInfo.processInfo.environment["TELEMAK_MODELS_DIR"],
           let url = resolveModelsDir(modelsDir, id: identifier) {
            let prepared = (try? prepareConfigForMLX(originalDir: url, id: identifier)) ?? url
            return try await loadModelContainer(
                from: prepared,
                using: #huggingFaceTokenizerLoader()
            )
        }

        let configuration = ModelConfiguration(id: identifier)
        return try await #huggingFaceLoadModelContainer(
            configuration: configuration,
            progressHandler: progressHandler
        )
    }

    /// Some inferencerlabs Qwen3.6 releases ship `quantization_config` at
    /// the top level instead of the `quantization` key mlx-swift-lm reads
    /// via `BaseConfiguration`. Without the metadata mlx-swift loads the
    /// `lm_head` Linear at unpacked shape and fails with
    /// `mismatchedSize([248320, 5120] vs [248320, 1280])`.
    ///
    /// Patch the config on the fly: re-write the JSON with a top-level
    /// `quantization` mirror, and stage the model in a symlinked dir under
    /// `~/.telemak/staged-models/<id>/`. All safetensors stay in place via
    /// symlinks; only `config.json` is materialised fresh.
    ///
    /// Idempotent — re-uses an existing staged dir if its config.json is up
    /// to date.
    static func prepareConfigForMLX(originalDir: URL, id: String) throws -> URL {
        let configURL = originalDir.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              var root = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) as? [String: Any]
        else {
            return originalDir
        }

        // Already has `quantization`? Nothing to fix.
        guard root["quantization"] == nil, let quantConfig = root["quantization_config"] else {
            return originalDir
        }

        // Some inferencerlabs configs omit `bits` from `quantization_config`
        // (e.g. Qwen3.6-27B's `{"group_size": 32}` only). Default to 8 — the
        // packing factor in the on-disk safetensors confirms 8-bit, and all
        // observed inferencerlabs releases are 8-bit so far. The model
        // name's "-9bit" suffix is a marketing term, not the actual bit
        // width of the packed weights.
        //
        // Larger Qwen3.6 releases (35B-A3B MoE) embed per-layer overrides
        // alongside the top-level config — those entries are also missing
        // `bits`. Walk every dictionary-valued child and patch them all.
        var enriched = quantConfig as? [String: Any] ?? [:]
        if enriched["bits"] == nil {
            enriched["bits"] = 8
        }
        for (key, value) in enriched {
            if var nested = value as? [String: Any], nested["bits"] == nil,
               nested["group_size"] != nil
            {
                nested["bits"] = 8
                enriched[key] = nested
            }
        }
        root["quantization"] = enriched

        let stagedRoot = FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent(".telemak/staged-models")
        let safeID = id.replacingOccurrences(of: "/", with: "--")
        let stagedDir = stagedRoot.appendingPathComponent(safeID)
        try FileManager.default.createDirectory(at: stagedDir, withIntermediateDirectories: true)

        // Symlink every file in originalDir → stagedDir, except config.json
        // which we write from the patched JSON.
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: originalDir.path)) ?? []
        for name in entries where !name.hasPrefix(".") && name != "config.json" {
            let src = originalDir.appendingPathComponent(name)
            let dst = stagedDir.appendingPathComponent(name)
            try? FileManager.default.removeItem(at: dst)
            try FileManager.default.createSymbolicLink(at: dst, withDestinationURL: src)
        }

        let patchedConfig = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
        let stagedConfigURL = stagedDir.appendingPathComponent("config.json")
        try patchedConfig.write(to: stagedConfigURL, options: .atomic)

        return stagedDir
    }

    /// Resolve `<root>/<id>/snapshots/<hash>/` (or `<root>/<id>/` if no
    /// snapshot dir) into a directory URL that has `config.json`.
    static func resolveModelsDir(_ root: String, id: String) -> URL? {
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

    static func resolveDirectory(at path: String) -> URL? {
        guard hasConfigJSON(at: path) else { return nil }
        return URL(fileURLWithPath: path)
    }

    static func hasConfigJSON(at directory: String) -> Bool {
        var isDir: ObjCBool = false
        let configPath = (directory as NSString).appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: configPath, isDirectory: &isDir) && !isDir.boolValue
    }
}
