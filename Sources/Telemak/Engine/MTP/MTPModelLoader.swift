import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Parallel loader for `qwen3_5_mtp` draft models.
///
/// mlx-swift-lm's `LLMTypeRegistry.shared` only registers types whose
/// model class conforms to `LanguageModel`. Our `Qwen35MTPDraftModel`
/// is plain `Module + BaseLanguageModel` — it has no `lm_head` /
/// `embed_tokens` of its own (those are borrowed from the target at
/// bind time), so it can't fulfil the `LanguageModel` contract. We
/// short-circuit the factory dispatch and instantiate + load the
/// draft directly from a local directory.
///
/// Resolution mirrors `ModelLoader.load` :
/// 1. Absolute path → load straight from that dir.
/// 2. `TELEMAK_MODELS_DIR` env var → `<root>/<id>/snapshots/<hash>/`.
/// 3. *(no HF download path)* — the draft is always paired with a
///    target that's already resolved, and HF cache layouts are too
///    variable for the staged-config path. Surface a clear error if
///    we can't find it locally.
public enum MTPModelLoader {

    public enum LoadError: Error, CustomStringConvertible {
        case directoryNotFound(id: String)
        case configMissing(dir: String)
        case configDecodeFailed(dir: String, underlying: String)
        case wrongModelType(dir: String, gotType: String)
        case noSafetensors(dir: String)
        case weightLoadFailed(underlying: String)

        public var description: String {
            switch self {
            case .directoryNotFound(let id):
                return
                    "MTP draft '\(id)' not found locally. Set TELEMAK_MODELS_DIR or pass an absolute path."
            case .configMissing(let dir):
                return "config.json missing in \(dir)"
            case .configDecodeFailed(let dir, let why):
                return "config.json decode failed in \(dir): \(why)"
            case .wrongModelType(let dir, let got):
                return
                    "expected model_type 'qwen3_5_mtp' or embedded Qwen3.5 MTP weights in \(dir)/config.json, got '\(got)'"
            case .noSafetensors(let dir):
                return "no *.safetensors files found in \(dir)"
            case .weightLoadFailed(let why):
                return "weight loading failed: \(why)"
            }
        }
    }

    /// Resolve `identifier` to a directory, stage its config for MLX
    /// (handles the inferencerlabs `quantization_config` quirk the
    /// same way `ModelLoader` does), then instantiate + load weights.
    public static func load(identifier: String) throws -> Qwen35MTPDraftModel {
        let dir = try resolveDir(identifier)
        let staged = (try? ModelLoader.prepareConfigForMLX(originalDir: dir, id: identifier)) ?? dir
        let configURL = staged.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: configURL) else {
            throw LoadError.configMissing(dir: staged.path)
        }
        let decoder = JSONDecoder()
        let mtpConfig: Qwen35MTPConfiguration
        do {
            mtpConfig = try decoder.decode(Qwen35MTPConfiguration.self, from: configData)
        } catch {
            throw LoadError.configDecodeFailed(dir: staged.path, underlying: "\(error)")
        }
        if mtpConfig.modelType != "qwen3_5_mtp", !hasEmbeddedHead(in: dir) {
            throw LoadError.wrongModelType(dir: staged.path, gotType: mtpConfig.modelType)
        }
        // BaseConfiguration handles the quantization shape +
        // per-layer overrides. `prepareConfigForMLX` ensures the
        // top-level `quantization` key exists with sane defaults.
        let baseConfig: BaseConfiguration
        do {
            baseConfig = try decoder.decode(BaseConfiguration.self, from: configData)
        } catch {
            throw LoadError.configDecodeFailed(dir: staged.path, underlying: "\(error)")
        }
        // Sanity : at least one safetensors file.
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: staged.path)) ?? []
        guard entries.contains(where: { $0.hasSuffix(".safetensors") }) else {
            throw LoadError.noSafetensors(dir: staged.path)
        }
        let model = Qwen35MTPDraftModel(mtpConfig)
        do {
            try loadWeights(
                modelDirectory: staged,
                model: model,
                perLayerQuantization: baseConfig.perLayerQuantization
            )
        } catch {
            throw LoadError.weightLoadFailed(underlying: "\(error)")
        }
        return model
    }

    public static func embeddedDraftId(for mainId: String) -> String {
        "\(mainId)#embedded-mtp"
    }

    public static func hasEmbeddedHead(identifier: String) -> Bool {
        guard let dir = try? resolveDir(identifier) else { return false }
        return hasEmbeddedHead(in: dir)
    }

    private static func hasEmbeddedHead(in dir: URL) -> Bool {
        let indexURL = dir.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL),
              let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = root["weight_map"] as? [String: Any]
        else {
            return false
        }
        return weightMap.keys.contains {
            $0.hasPrefix("language_model.mtp.") || $0.hasPrefix("mtp.")
        }
    }

    private static func resolveDir(_ identifier: String) throws -> URL {
        if identifier.hasPrefix("/") {
            if let url = ModelLoader.resolveDirectory(at: identifier) {
                return url
            }
            throw LoadError.directoryNotFound(id: identifier)
        }
        if let modelsDir = ProcessInfo.processInfo.environment["TELEMAK_MODELS_DIR"],
           let url = ModelLoader.resolveModelsDir(modelsDir, id: identifier)
        {
            return url
        }
        throw LoadError.directoryNotFound(id: identifier)
    }
}
