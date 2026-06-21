import Foundation
import TelemakVersion
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/// Parallel loader for sidecar MTP draft models.
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

    public enum LoadedDraftModel: @unchecked Sendable {
        case qwen35(Qwen35MTPDraftModel)
        case gemma4Assistant(Gemma4AssistantModel)
    }

    public enum LoadError: Error, CustomStringConvertible {
        case directoryNotFound(id: String)
        case configMissing(dir: String)
        case configDecodeFailed(dir: String, underlying: String)
        case wrongModelType(dir: String, gotType: String)
        case embeddedMTPMissing(dir: String)
        case embeddedMTPStageFailed(underlying: String)
        case noSafetensors(dir: String)
        case weightLoadFailed(underlying: String)

        public var description: String {
            switch self {
            case .directoryNotFound(let id):
                return
                    "MTP draft '\(id)' not found locally. Set your models directory (OdyssAI-X or the menubar) or pass an absolute path."
            case .configMissing(let dir):
                return "config.json missing in \(dir)"
            case .configDecodeFailed(let dir, let why):
                return "config.json decode failed in \(dir): \(why)"
            case .wrongModelType(let dir, let got):
                return "unsupported MTP draft model_type '\(got)' in \(dir)/config.json"
            case .embeddedMTPMissing(let dir):
                return "embedded MTP contract found, but no mtp.safetensors artifact exists in \(dir)"
            case .embeddedMTPStageFailed(let why):
                return "could not stage embedded MTP weights: \(why)"
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
    public static func load(identifier: String, embedded: Bool = false) throws -> LoadedDraftModel {
        let dir = try resolveDir(identifier)
        let staged = (try? ModelLoader.prepareConfigForMLX(originalDir: dir, id: identifier)) ?? dir
        let configURL = staged.appendingPathComponent("config.json")
        guard let configData = try? Data(contentsOf: configURL) else {
            throw LoadError.configMissing(dir: staged.path)
        }
        guard let root = try? JSONSerialization.jsonObject(with: configData) as? [String: Any] else {
            throw LoadError.configDecodeFailed(dir: staged.path, underlying: "invalid JSON")
        }
        let modelType = (root["model_type"] as? String) ?? ""
        if embedded {
            return .qwen35(try loadEmbeddedQwen35MTP(staged: staged, configData: configData, root: root))
        }
        switch modelType {
        case "qwen3_5_mtp":
            return .qwen35(try loadQwen35MTP(staged: staged, configData: configData))
        case "gemma4_assistant":
            return .gemma4Assistant(try loadGemma4Assistant(staged: staged, configData: configData))
        default:
            throw LoadError.wrongModelType(dir: staged.path, gotType: modelType)
        }
    }

    /// Load a native dense MTP head embedded in the main model directory.
    ///
    /// MTPLX-style artifacts keep the trunk shards and `mtp.safetensors`
    /// together. `loadWeights` loads every safetensors file under a
    /// directory, so loading the draft directly from the model root would
    /// feed trunk weights into `Qwen35MTPDraftModel` and fail verification.
    /// Stage only `config.json` + the declared MTP safetensors through
    /// symlinks, then reuse the normal draft loader.
    private static func loadEmbeddedQwen35MTP(
        staged: URL,
        configData: Data,
        root: [String: Any]
    ) throws -> Qwen35MTPDraftModel {
        let mtpFile = embeddedMTPFile(root: root) ?? "mtp.safetensors"
        let mtpURL = staged.appendingPathComponent(mtpFile)
        guard FileManager.default.fileExists(atPath: mtpURL.path) else {
            throw LoadError.embeddedMTPMissing(dir: staged.path)
        }

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("telemak-embedded-mtp-\(UUID().uuidString)")
        do {
            try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
            let draftConfigData = try embeddedDraftConfigData(root: root, fallback: configData)
            try draftConfigData.write(to: tmp.appendingPathComponent("config.json"), options: .atomic)
            try FileManager.default.createSymbolicLink(
                at: tmp.appendingPathComponent((mtpFile as NSString).lastPathComponent),
                withDestinationURL: mtpURL
            )
            defer { try? FileManager.default.removeItem(at: tmp) }
            return try loadQwen35MTP(staged: tmp, configData: draftConfigData)
        } catch let error as LoadError {
            throw error
        } catch {
            throw LoadError.embeddedMTPStageFailed(underlying: "\(error)")
        }
    }

    private static func embeddedMTPFile(root: [String: Any]) -> String? {
        guard let extras = root["mlx_lm_extra_tensors"] as? [String: Any],
              let file = extras["mtp_file"] as? String,
              !file.isEmpty
        else { return nil }
        return file
    }

    private static func embeddedDraftConfigData(root: [String: Any], fallback: Data) throws -> Data {
        var draftRoot = root
        if let mtpQuantization = root["mtplx_mtp_quantization"] as? [String: Any] {
            let quantization = compactQuantization(mtpQuantization)
            draftRoot["quantization"] = quantization
            draftRoot["quantization_config"] = quantization
            return try JSONSerialization.data(
                withJSONObject: draftRoot,
                options: [.prettyPrinted, .sortedKeys]
            )
        }
        return fallback
    }

    private static func compactQuantization(_ raw: [String: Any]) -> [String: Any] {
        var out: [String: Any] = [:]
        for key in ["bits", "group_size", "mode"] {
            if let value = raw[key] {
                out[key] = value
            }
        }
        if out["mode"] == nil {
            out["mode"] = "affine"
        }
        return out
    }

    private static func loadQwen35MTP(staged: URL, configData: Data) throws -> Qwen35MTPDraftModel {
        let decoder = JSONDecoder()
        let mtpConfig: Qwen35MTPConfiguration
        do {
            mtpConfig = try decoder.decode(Qwen35MTPConfiguration.self, from: configData)
        } catch {
            throw LoadError.configDecodeFailed(dir: staged.path, underlying: "\(error)")
        }
        let baseConfig: BaseConfiguration
        do {
            baseConfig = try decoder.decode(BaseConfiguration.self, from: configData)
        } catch {
            throw LoadError.configDecodeFailed(dir: staged.path, underlying: "\(error)")
        }
        try ensureSafetensors(in: staged)
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

    private static func loadGemma4Assistant(staged: URL, configData: Data) throws -> Gemma4AssistantModel {
        let decoder = JSONDecoder.json5()
        let config: Gemma4AssistantConfiguration
        do {
            config = try decoder.decode(Gemma4AssistantConfiguration.self, from: configData)
        } catch {
            throw LoadError.configDecodeFailed(dir: staged.path, underlying: "\(error)")
        }
        let baseConfig: BaseConfiguration
        do {
            baseConfig = try decoder.decode(BaseConfiguration.self, from: configData)
        } catch {
            throw LoadError.configDecodeFailed(dir: staged.path, underlying: "\(error)")
        }
        try ensureSafetensors(in: staged)
        let model = Gemma4AssistantModel(config)
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

    private static func ensureSafetensors(in staged: URL) throws {
        let entries = (try? FileManager.default.contentsOfDirectory(atPath: staged.path)) ?? []
        guard entries.contains(where: { $0.hasSuffix(".safetensors") }) else {
            throw LoadError.noSafetensors(dir: staged.path)
        }
    }

    private static func resolveDir(_ identifier: String) throws -> URL {
        if identifier.hasPrefix("/") {
            if let url = ModelLoader.resolveDirectory(at: identifier) {
                return url
            }
            throw LoadError.directoryNotFound(id: identifier)
        }
        if let modelsDir = ModelsConfig.shared.effectiveDir(),
           let url = ModelLoader.resolveModelsDir(modelsDir, id: identifier)
        {
            return url
        }
        throw LoadError.directoryNotFound(id: identifier)
    }
}
