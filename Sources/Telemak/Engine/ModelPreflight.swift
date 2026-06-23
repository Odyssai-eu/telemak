import Foundation
import TelemakVersion

/// Structured outcome of a `/admin/load` preflight check. Runs BEFORE the
/// heavy MLX load and gives actionable, typed errors (missing dir, missing
/// config, parse failure, incomplete shards, unsupported model type) instead
/// of the generic `configurationDecodingError` or `model not found` that
/// would otherwise surface late from the loader.
///
/// Resolution mirrors `ModelLoader`:
///   - absolute path → `resolveDirectory(at:)`;
///   - `org/name`     → `ModelsConfig.effectiveDir()` (the master/slave
///                      `config.json` or legacy env) + HF cache;
///   - **no network fetch** — if the identifier is not local, we hand off
///     to the load path with `.remote` and let the HF download happen there.
///
/// Lives in `Engine/` (not `TelemakVersion`) because the load path owns the
/// resolution logic and the existing `resolvedModelDirectory` / `isPath`
/// helpers — and we deliberately want to depend on the same `ModelLoader`
/// surface the loader uses, so a new layout or repo convention in
/// `ModelLoader` flows here for free.
public enum ModelPreflight {

    /// What the preflight decided. Either we found a local dir to validate
    /// (and did, successfully), or we couldn't find one locally and the
    /// load path will fetch from the hub.
    public enum Outcome: Sendable, Equatable {
        /// Local dir resolved and looks loadable — continue to the heavy load.
        case local(id: String, dir: URL, modelType: String?, weightFiles: [String])
        /// Identifier is not present on disk; the load path will fetch from
        /// the Hugging Face hub. No preflight possible.
        case remote(id: String)
    }

    /// Typed, structured preflight failures. Each case carries enough
    /// context to render an actionable error to the operator — the path
    /// the loader tried, the model type it saw, the shards it expected
    /// vs. found.
    public enum Error: Swift.Error, Sendable, Equatable, CustomStringConvertible {
        /// The local dir the identifier resolved to (absolute path or
        /// `<models_dir>/<org>/<name>/snapshots/<hash>`) does not exist.
        case modelDirMissing(id: String, path: String)
        /// The local dir exists but has no `config.json` — the snapshot
        /// is incomplete or the path points at the wrong level.
        case configMissing(id: String, dir: String)
        /// `config.json` is present but its JSON body failed to parse.
        case configParseFailed(id: String, dir: String, reason: String)
        /// `model.safetensors.index.json` references shard files that
        /// are not all on disk — likely a partial / aborted download.
        /// 503 (transient): retrying after the download resumes should pass.
        case shardsIncomplete(id: String, dir: String, missing: [String])
        /// The local dir has weights, but no `model_type` and no
        /// `architectures` in the config — mlx-swift-lm has no factory
        /// to dispatch on. Or the weight files are not in a format we
        /// recognize (e.g. raw `.bin` PyTorch dumps without an index).
        case unsupportedModelType(id: String, dir: String, modelType: String?)

        public var description: String {
            switch self {
            case .modelDirMissing(let id, let path):
                return "model '\(id)' directory not found: \(path)"
            case .configMissing(let id, let dir):
                return "model '\(id)' directory has no config.json: \(dir)"
            case .configParseFailed(let id, let dir, let reason):
                return "model '\(id)' config.json failed to parse in \(dir): \(reason)"
            case .shardsIncomplete(let id, let dir, let missing):
                let listed = missing.prefix(5).joined(separator: ", ")
                let more = missing.count > 5 ? " (+\(missing.count - 5) more)" : ""
                return "model '\(id)' in \(dir) is missing safetensors shards: \(listed)\(more)"
            case .unsupportedModelType(let id, let dir, let modelType):
                let mt = modelType ?? "<missing>"
                return "model '\(id)' in \(dir) has no supported model_type ('\(mt)')"
            }
        }
    }

    // MARK: - public API

    /// Run the preflight for `identifier`. Returns `.local` (caller proceeds
    /// to load), `.remote` (caller proceeds to fetch from hub), or throws
    /// `Error` for a structured failure the HTTP layer can surface directly.
    ///
    /// `modelsDir` is the effective models dir to resolve `org/name`
    /// identifiers under. Defaults to `ModelsConfig.shared.effectiveDir()`
    /// — tests pass an explicit value to avoid mutating the global.
    public static func check(
        identifier: String,
        modelsDir: String? = ModelsConfig.shared.effectiveDir()
    ) throws -> Outcome {
        let trimmed = identifier.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw Error.modelDirMissing(id: identifier, path: "<empty>")
        }

        // Absolute paths get stricter treatment than hub ids: a missing
        // local dir is an *error* (the operator explicitly pointed at a
        // path that doesn't exist), whereas a missing hub id is just
        // "we'll fetch from the hub when the load runs".
        if trimmed.hasPrefix("/") {
            return try checkAbsolutePath(id: trimmed)
        }

        if let resolved = resolveLocalDirectory(identifier: trimmed, modelsDir: modelsDir) {
            return try validateResolved(id: trimmed, dir: resolved)
        }
        return .remote(id: trimmed)
    }

    /// Absolute-path branch: existence + config.json must be present, no
    /// resolution from telemak-models-dir / HF cache.
    private static func checkAbsolutePath(id: String) throws -> Outcome {
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: id, isDirectory: &isDir)
        if !exists || !isDir.boolValue {
            throw Error.modelDirMissing(id: id, path: id)
        }
        if !ModelLoader.hasConfigJSON(at: id) {
            throw Error.configMissing(id: id, dir: id)
        }
        return try validateResolved(id: id, dir: URL(fileURLWithPath: id))
    }

    // MARK: - resolution

    /// Mirror of `ModelLoader.resolvedModelDirectory` / `EmbedderLoader.resolveLocalDirectory`
    /// with an injectable `modelsDir` — lets tests run without mutating the
    /// global `ModelsConfig.shared`.
    private static func resolveLocalDirectory(identifier: String, modelsDir: String?) -> URL? {
        if identifier.hasPrefix("/") {
            // Absolute path — accept if it has a config.json directly under
            // it. Do not follow symlinks into the models dir; the caller
            // already knows where they want to load from.
            return ModelLoader.resolveDirectory(at: identifier)
        }
        if let modelsDir,
           let url = ModelLoader.resolveModelsDir(modelsDir, id: identifier) {
            return url
        }
        // HF cache fallback — same layout the loader uses for org/name lookups
        // when the operator has an old-style HF download in `~/.cache/...`.
        guard let hfCache = AvailableModels.defaultHFCache() else { return nil }
        let cacheId = "models--" + identifier.replacingOccurrences(of: "/", with: "--")
        let modelBase = (hfCache as NSString).appendingPathComponent(cacheId)
        let snapshotsBase = (modelBase as NSString).appendingPathComponent("snapshots")
        let fm = FileManager.default
        guard let snapshots = try? fm.contentsOfDirectory(atPath: snapshotsBase) else { return nil }
        let candidates = snapshots
            .filter { !$0.hasPrefix(".") }
            .map { (snapshotsBase as NSString).appendingPathComponent($0) }
            .sorted()
        if let chosen = candidates.last(where: { ModelLoader.hasConfigJSON(at: $0) }) {
            return URL(fileURLWithPath: chosen)
        }
        return nil
    }

    // MARK: - validation

    private static func validateResolved(id: String, dir: URL) throws -> Outcome {
        let dirPath = dir.path

        // (1) dir exists + is a directory
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: dirPath, isDirectory: &isDir)
        if !exists {
            throw Error.modelDirMissing(id: id, path: dirPath)
        }
        if !isDir.boolValue {
            // A file where the dir should be — give the operator an actionable
            // error rather than the generic "not found" from the loader.
            throw Error.modelDirMissing(id: id, path: dirPath)
        }

        // (2) config.json present
        let configURL = dir.appendingPathComponent("config.json")
        guard FileManager.default.fileExists(atPath: configURL.path) else {
            throw Error.configMissing(id: id, dir: dirPath)
        }

        // (3) config.json parses
        let root: [String: Any]
        do {
            let data = try Data(contentsOf: configURL)
            guard let parsed = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) as? [String: Any] else {
                throw Error.configParseFailed(id: id, dir: dirPath,
                                              reason: "config.json is not a JSON object")
            }
            root = parsed
        } catch let prefError as Error {
            throw prefError
        } catch {
            throw Error.configParseFailed(id: id, dir: dirPath, reason: "\(error)")
        }

        let modelType = (root["model_type"] as? String)?.lowercased()
        let architectures = (root["architectures"] as? [String]) ?? []

        // (4) shard completeness — only when an index is present.
        let weightFiles = listWeightFiles(in: dir)
        if let indexURL = findShardIndex(in: dir) {
            let missing = missingShards(for: indexURL, dir: dir)
            if !missing.isEmpty {
                throw Error.shardsIncomplete(id: id, dir: dirPath, missing: missing)
            }
        }

        // (5) model_type / architectures plausibility — we can't enumerate
        // every supported type, but we can flag "no factory could dispatch
        // on this". Empty model_type + empty architectures is the strongest
        // signal: mlx-swift-lm factory dispatch relies on one of them.
        if (modelType == nil || modelType?.isEmpty == true), architectures.isEmpty {
            // Some checkpoints (esp. embedders, GGUF-with-sidecar) ship
            // without model_type but with a tokenizer — surface a soft
            // skip here: trust the loader to reject if it really is unknown.
            // The preflight only fails when there is *nothing* for any
            // factory to key on AND the weights are not safetensors/gguf.
            if !weightFiles.contains(where: { $0.hasSuffix(".safetensors") || $0.hasSuffix(".gguf") }) {
                throw Error.unsupportedModelType(id: id, dir: dirPath, modelType: modelType)
            }
        }

        return .local(id: id, dir: dir, modelType: modelType, weightFiles: weightFiles)
    }

    // MARK: - shard helpers

    /// Returns the names of all `.safetensors` / `.gguf` files in `dir`
    /// (top level only — mlx-swift-lm and HF cache convention).
    private static func listWeightFiles(in dir: URL) -> [String] {
        let fm = FileManager.default
        let entries = (try? fm.contentsOfDirectory(atPath: dir.path)) ?? []
        return entries.filter { name in
            (name.hasSuffix(".safetensors") || name.hasSuffix(".gguf"))
                && !name.hasSuffix(".index.json")
        }
    }

    /// Find `model.safetensors.index.json` (or the alternate name
    /// `model.safetensors.index.json` — same thing) under `dir`. HF also
    /// allows a per-shard alternative format, but the index file is the
    /// common case we care about for completeness.
    private static func findShardIndex(in dir: URL) -> URL? {
        let fm = FileManager.default
        let candidates = ["model.safetensors.index.json"]
        for name in candidates {
            let url = dir.appendingPathComponent(name)
            if fm.fileExists(atPath: url.path) { return url }
        }
        return nil
    }

    /// Parse the HF weight_map from the shard index and return the unique
    /// shard filenames that are NOT on disk.
    private static func missingShards(for indexURL: URL, dir: URL) -> [String] {
        guard let data = try? Data(contentsOf: indexURL),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = root["weight_map"] as? [String: String]
        else {
            return []   // unparseable index — let the loader deal with it.
        }
        let fm = FileManager.default
        var missing: [String] = []
        var seen: Set<String> = []
        for shard in weightMap.values where seen.insert(shard).inserted {
            let path = dir.appendingPathComponent(shard).path
            if !fm.fileExists(atPath: path) {
                missing.append(shard)
            }
        }
        return missing.sorted()
    }
}
