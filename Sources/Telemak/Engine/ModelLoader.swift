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

// Thrown when an absolute model path escapes the configured models dir.
// ModelRegistry.load() catches any error from ModelLoader.load() and
// wraps it as LoadError.loadFailed(underlying:) — this struct ensures the
// wrapped message is the same as a normal not-found, removing the oracle.
private struct PathOutsideModelsDir: Error, CustomStringConvertible {
    let path: String
    init(_ path: String) { self.path = path }
    var description: String { "model not found: '\(path)'" }
}

public enum ModelLoader {
    /// Convert an absolute model path under `TELEMAK_MODELS_DIR` back to
    /// its stable repository id (`org/name`). The registry, staged config
    /// directory, session cache and wired-memory reservations are keyed by
    /// model id; accepting both `/Volumes/.../org/name` and `org/name` for
    /// the same files creates duplicate identities for one model.
    public static func canonicalIdentifier(_ identifier: String) -> String {
        let trimmed = identifier.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("/"),
              let modelsDir = ProcessInfo.processInfo.environment["TELEMAK_MODELS_DIR"]
        else {
            return trimmed
        }

        let modelPath = URL(fileURLWithPath: trimmed).standardized.path
        let rootPath = URL(fileURLWithPath: modelsDir).standardized.path
        guard isPath(modelPath, sameOrDescendantOf: rootPath) else {
            return trimmed
        }

        let relative = String(modelPath.dropFirst(rootPath.count))
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let parts = relative.split(separator: "/").map(String.init)
        guard parts.count >= 2 else { return trimmed }

        let candidate = "\(parts[0])/\(parts[1])"
        guard let resolved = resolveModelsDir(modelsDir, id: candidate) else {
            return trimmed
        }

        let candidateRoot = URL(fileURLWithPath: (modelsDir as NSString).appendingPathComponent(candidate))
            .standardized.path
        let resolvedPath = resolved.standardized.path
        if pathsMatch(modelPath, resolvedPath) || isPath(modelPath, sameOrDescendantOf: candidateRoot) {
            return candidate
        }
        return trimmed
    }

    public static func load(
        identifier: String,
        forceLLM: Bool = false,
        progressHandler: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> ModelContainer {
        if identifier.hasPrefix("/") {
            // Defense-in-depth: when TELEMAK_MODELS_DIR is set, reject absolute
            // paths outside it. Returns the same error as a failed load so the
            // caller can't distinguish "denied" from "missing config.json".
            if let modelsDir = ProcessInfo.processInfo.environment["TELEMAK_MODELS_DIR"] {
                let rootPath = URL(fileURLWithPath: modelsDir).standardized.path
                let modelPath = URL(fileURLWithPath: identifier).standardized.path
                guard isPath(modelPath, sameOrDescendantOf: rootPath) else {
                    throw PathOutsideModelsDir(identifier)
                }
            }
            if let url = resolveDirectory(at: identifier) {
                let prepared = (try? prepareConfigForMLX(originalDir: url, id: identifier)) ?? url
                return try await dispatchedLoad(prepared: prepared, forceLLM: forceLLM)
            }
        }

        if let modelsDir = ProcessInfo.processInfo.environment["TELEMAK_MODELS_DIR"],
           let url = resolveModelsDir(modelsDir, id: identifier) {
            let prepared = (try? prepareConfigForMLX(originalDir: url, id: identifier)) ?? url
            return try await dispatchedLoad(prepared: prepared, forceLLM: forceLLM)
        }

        let configuration = ModelConfiguration(id: identifier)
        if forceLLM {
            return try await LLMModelFactory.shared.loadContainer(
                from: #hubDownloader(),
                using: #huggingFaceTokenizerLoader(),
                configuration: configuration,
                progressHandler: progressHandler
            )
        }
        return try await #huggingFaceLoadModelContainer(
            configuration: configuration,
            progressHandler: progressHandler
        )
    }

    /// Inspect the staged config and pick the right factory.
    ///
    /// Qwen3.5 / Qwen3.6 ship as VLM checkpoints even when there's no
    /// vision payload (`vision_config: {}`, no vision_tower weights).
    /// The default `loadModelContainer` factory dispatch picks
    /// `MLXVLM.Qwen35` because of the `architectures` hint, but that
    /// class blocks our V2 MTP path (different hidden-state extraction
    /// ABI). Force LLM dispatch for text-only Qwen3.5/3.6 ; everything
    /// else falls through to the generic factory.
    private static func dispatchedLoad(prepared: URL, forceLLM: Bool) async throws -> ModelContainer {
        if forceLLM {
            return try await LLMModelFactory.shared.loadContainer(
                from: prepared,
                using: #huggingFaceTokenizerLoader()
            )
        }
        let configURL = prepared.appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: configURL),
           let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        {
            let modelType = (root["model_type"] as? String) ?? ""
            if modelType == "step3p7" || modelType == "step3p5" {
                return try await LLMModelFactory.shared.loadContainer(
                    from: prepared,
                    using: #huggingFaceTokenizerLoader()
                )
            }
            let visionConfig = root["vision_config"] as? [String: Any]
            let isVisionEmpty = visionConfig == nil || (visionConfig?.isEmpty ?? true)
            let isQwen35 = modelType == "qwen3_5" || modelType == "qwen3_5_moe"
                || modelType == "qwen3_5_text"
            if isQwen35 && isVisionEmpty {
                return try await LLMModelFactory.shared.loadContainer(
                    from: prepared,
                    using: #huggingFaceTokenizerLoader()
                )
            }
            if isVisionModel(root: root) {
                return try await VLMModelFactory.shared.loadContainer(
                    from: prepared,
                    using: #huggingFaceTokenizerLoader()
                )
            }
        }
        return try await loadModelContainer(
            from: prepared,
            using: #huggingFaceTokenizerLoader()
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

        // Detect text-only Qwen3.5/3.6 mislabeled as VLM via empty
        // `vision_config: {}`. The VLMModelFactory wins the factory
        // dispatch and routes to MLXVLM's Qwen35 class even though
        // there are no vision tower weights — fine for text generation
        // but blocks MTP (different forward shape, no clean hidden
        // state extraction). Strip the empty vision_config to force
        // LLM dispatch.
        var droppedEmptyVision = false
        if let vc = root["vision_config"] as? [String: Any], vc.isEmpty {
            root["vision_config"] = nil
            droppedEmptyVision = true
        }
        let tokenizerPatch = tokenizerConfigPatch(originalDir: originalDir, root: root)
        let aliasedMiniMaxM2 = (root["model_type"] as? String) == "minimax_m2"
        if aliasedMiniMaxM2 {
            root["model_type"] = "minimax"
        }
        let normalizedMistral3EOS = (root["model_type"] as? String) == "mistral3"
            && root["eos_token_id"] is Int
        if normalizedMistral3EOS, let eos = root["eos_token_id"] as? Int {
            root["eos_token_id"] = [eos]
        }

        // Already has `quantization`? Nothing to fix (modulo the
        // vision_config strip above).
        guard root["quantization"] == nil, let quantConfig = root["quantization_config"] else {
            if !droppedEmptyVision && !aliasedMiniMaxM2 && !normalizedMistral3EOS && tokenizerPatch == nil {
                return originalDir
            }
            // Still need to stage the dir to rewrite config.json with
            // vision_config removed ; fall through to the staging
            // block below by injecting an empty quantization carrier.
            root["__telemak_force_stage__"] = true
            return try writeStaged(
                originalDir: originalDir,
                root: root,
                id: id,
                patchedFiles: tokenizerPatch.map { ["tokenizer_config.json": $0] } ?? [:]
            )
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
        if enriched["mode"] == nil {
            enriched["mode"] = "affine"
        }
        for (key, value) in enriched {
            if var nested = value as? [String: Any], nested["bits"] == nil,
               nested["group_size"] != nil
            {
                nested["bits"] = 8
                if nested["mode"] == nil {
                    nested["mode"] = "affine"
                }
                enriched[key] = nested
            }
        }
        root["quantization"] = enriched
        root["quantization_config"] = enriched

        return try writeStaged(
            originalDir: originalDir,
            root: root,
            id: id,
            patchedFiles: tokenizerPatch.map { ["tokenizer_config.json": $0] } ?? [:]
        )
    }

    private static func writeStaged(
        originalDir: URL,
        root: [String: Any],
        id: String,
        patchedFiles: [String: [String: Any]] = [:]
    ) throws -> URL {
        var root = root
        root["__telemak_force_stage__"] = nil  // never persist this marker
        let stagedRoot = FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent(".telemak/staged-models")
        let safeID = id.replacingOccurrences(of: "/", with: "--")
        let stagedDir = stagedRoot.appendingPathComponent(safeID)
        try FileManager.default.createDirectory(at: stagedDir, withIntermediateDirectories: true)

        // Symlink every file in originalDir → stagedDir, except config.json
        // and explicitly patched JSON files.
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: originalDir.path)) ?? []
        for name in entries where !name.hasPrefix(".") && name != "config.json" && patchedFiles[name] == nil {
            let src = originalDir.appendingPathComponent(name)
            let dst = stagedDir.appendingPathComponent(name)
            try? FileManager.default.removeItem(at: dst)
            try FileManager.default.createSymbolicLink(at: dst, withDestinationURL: src)
        }

        let patchedConfig = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
        let stagedConfigURL = stagedDir.appendingPathComponent("config.json")
        try patchedConfig.write(to: stagedConfigURL, options: .atomic)

        for (name, json) in patchedFiles {
            let data = try JSONSerialization.data(withJSONObject: json, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: stagedDir.appendingPathComponent(name), options: .atomic)
        }

        return stagedDir
    }

    private static func tokenizerConfigPatch(originalDir: URL, root: [String: Any]) -> [String: Any]? {
        let modelType = (root["model_type"] as? String ?? "").lowercased()
        let textModelType = ((root["text_config"] as? [String: Any])?["model_type"] as? String ?? "").lowercased()
        let isQwen35Family = [
            "qwen3_5", "qwen3_5_text", "qwen3_next",
        ].contains(modelType) || [
            "qwen3_5_text", "qwen3_next",
        ].contains(textModelType)
        guard isQwen35Family else { return nil }

        let tokenizerURL = originalDir.appendingPathComponent("tokenizer_config.json")
        var tokenizerRoot: [String: Any] = [:]
        if let data = try? Data(contentsOf: tokenizerURL),
           let decoded = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) as? [String: Any]
        {
            tokenizerRoot = decoded
        }
        let tokenizerClass = tokenizerRoot["tokenizer_class"] as? String
        if tokenizerClass == "TokenizersBackend" {
            return nil
        }
        tokenizerRoot["tokenizer_class"] = "TokenizersBackend"
        return tokenizerRoot
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

    public static func isVisionModel(identifier: String) -> Bool {
        guard let directory = resolvedModelDirectory(for: identifier) else {
            return identifier.lowercased().contains("vlm")
                || identifier.lowercased().contains("-vl-")
                || identifier.lowercased().contains("_vl_")
        }
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let root = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) as? [String: Any]
        else { return false }

        let modelType = (root["model_type"] as? String ?? "").lowercased()
        if modelType == "step3p7" || modelType == "step3p5" {
            // Telemak currently supports Step-3.7 as a text-only LLM even
            // though the upstream checkpoint is a VLM wrapper.
            return false
        }
        if let visionConfig = root["vision_config"] as? [String: Any], !visionConfig.isEmpty {
            return true
        }
        let knownVLMTypes: Set<String> = [
            "paligemma", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "idefics3",
            "gemma4", "smolvlm", "fastvlm", "llava_qwen2", "pixtral",
            "mistral3", "lfm2_vl", "lfm2-vl", "glm_ocr",
        ]
        if knownVLMTypes.contains(modelType) {
            return true
        }

        let architectures = (root["architectures"] as? [String] ?? []).map { $0.lowercased() }
        let haystack = ([modelType] + architectures).joined(separator: " ")
        if haystack.contains("vlm") || haystack.contains("vision") || haystack.contains("_vl") || haystack.contains("vl_") {
            return true
        }
        return false
    }

    private static func isVisionModel(root: [String: Any]) -> Bool {
        let modelType = (root["model_type"] as? String ?? "").lowercased()
        if modelType == "step3p7" || modelType == "step3p5" {
            // Text-only Step-3.7 support is routed through MLXLLM.
            return false
        }
        if let visionConfig = root["vision_config"] as? [String: Any], !visionConfig.isEmpty {
            return true
        }
        let knownVLMTypes: Set<String> = [
            "paligemma", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "idefics3",
            "gemma4", "smolvlm", "fastvlm", "llava_qwen2", "pixtral",
            "mistral3", "lfm2_vl", "lfm2-vl", "glm_ocr",
        ]
        if knownVLMTypes.contains(modelType) {
            return true
        }

        let architectures = (root["architectures"] as? [String] ?? []).map { $0.lowercased() }
        let haystack = ([modelType] + architectures).joined(separator: " ")
        return haystack.contains("vlm")
            || haystack.contains("vision")
            || haystack.contains("_vl")
            || haystack.contains("vl_")
    }

    public static func modelType(identifier: String) -> String? {
        guard let directory = resolvedModelDirectory(for: identifier) else { return nil }
        let configURL = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: configURL),
              let root = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) as? [String: Any]
        else { return nil }
        return root["model_type"] as? String
    }

    static func hasConfigJSON(at directory: String) -> Bool {
        var isDir: ObjCBool = false
        let configPath = (directory as NSString).appendingPathComponent("config.json")
        return FileManager.default.fileExists(atPath: configPath, isDirectory: &isDir) && !isDir.boolValue
    }

    static func isPath(_ path: String, sameOrDescendantOf root: String) -> Bool {
        let normalizedPath = comparablePath(path)
        let normalizedRoot = comparablePath(root)
        return normalizedPath == normalizedRoot || normalizedPath.hasPrefix(normalizedRoot + "/")
    }

    private static func pathsMatch(_ lhs: String, _ rhs: String) -> Bool {
        comparablePath(lhs) == comparablePath(rhs)
    }

    private static func comparablePath(_ path: String) -> String {
        URL(fileURLWithPath: path)
            .standardized
            .path
            .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            .lowercased()
    }

    static func resolvedModelDirectory(for identifier: String) -> URL? {
        if identifier.hasPrefix("/") {
            return resolveDirectory(at: identifier)
        }
        if let modelsDir = ProcessInfo.processInfo.environment["TELEMAK_MODELS_DIR"],
           let url = resolveModelsDir(modelsDir, id: identifier) {
            return url
        }
        guard let hfCache = AvailableModels.defaultHFCache() else { return nil }
        let cacheDir = "models--" + identifier.replacingOccurrences(of: "/", with: "--")
        let cachePath = (hfCache as NSString).appendingPathComponent(cacheDir)
        let snapshotsPath = (cachePath as NSString).appendingPathComponent("snapshots")
        let fm = FileManager.default
        guard let snapshots = try? fm.contentsOfDirectory(atPath: snapshotsPath) else { return nil }
        let candidates = snapshots
            .filter { !$0.hasPrefix(".") }
            .map { (snapshotsPath as NSString).appendingPathComponent($0) }
            .sorted()
        guard let chosen = candidates.last(where: { hasConfigJSON(at: $0) }) else { return nil }
        return URL(fileURLWithPath: chosen)
    }
}
