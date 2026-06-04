import Foundation

public enum MTPCompatibilityStatus: String, Codable, Sendable {
    case noMTP = "no_mtp"
    case sidecarOnly = "sidecar_only"
    case incompatibleArchitecture = "incompatible_architecture"
    case unverifiedEmbeddedMTP = "unverified_embedded_mtp"
    case verifiedEmbeddedMTP = "verified_embedded_mtp"
}

public struct MTPCompatibility: Codable, Equatable, Sendable {
    public let status: MTPCompatibilityStatus
    public let reason: String
    public let source: String
    public let contractPath: String?
    public let overrideRequired: Bool
    public let autoEnabled: Bool

    enum CodingKeys: String, CodingKey {
        case status
        case reason
        case source
        case contractPath = "contract_path"
        case overrideRequired = "override_required"
        case autoEnabled = "auto_enabled"
    }

    public init(
        status: MTPCompatibilityStatus,
        reason: String,
        source: String,
        contractPath: String? = nil,
        overrideRequired: Bool,
        autoEnabled: Bool
    ) {
        self.status = status
        self.reason = reason
        self.source = source
        self.contractPath = contractPath
        self.overrideRequired = overrideRequired
        self.autoEnabled = autoEnabled
    }

    public var canRunWithoutOverride: Bool {
        status == .verifiedEmbeddedMTP
    }

    public func canRun(allowUnverified: Bool) -> Bool {
        canRunWithoutOverride || (overrideRequired && allowUnverified)
    }

    public func rejectedMessage(modelId: String) -> String {
        "\(modelId): \(reason). Pass allow_unverified_mtp=true on /admin/load or set TELEMAK_MTP_ALLOW_UNVERIFIED=1 to override."
    }

    public static func unavailable(modelId: String) -> MTPCompatibility {
        MTPCompatibility(
            status: .noMTP,
            reason: "no local config/index metadata found for MTP inspection",
            source: modelId,
            overrideRequired: false,
            autoEnabled: false
        )
    }

    public static func inspect(modelId: String, directory: URL, isDraft: Bool = false) -> MTPCompatibility {
        let config = readJSON(directory.appendingPathComponent("config.json"))
        let runtimeContractURL = firstExisting([
            directory.appendingPathComponent("telemak_mtp.json"),
            directory.appendingPathComponent("mtplx_runtime.json"),
        ])
        let modelType = string(config["model_type"]).lowercased()
        let textConfig = (config["text_config"] as? [String: Any]) ?? config
        let textModelType = string(textConfig["model_type"]).lowercased()
        let mtpLayerCount = int(textConfig["mtp_num_hidden_layers"])
            ?? int(textConfig["num_nextn_predict_layers"])
            ?? int(config["num_nextn_predict_layers"])
            ?? 0
        let keys = weightKeys(in: directory)
        let hasMTPWeights = keys.contains { isMTPWeightKey($0) }
            || fileExists(directory.appendingPathComponent("mtp.safetensors"))
            || fileExists(directory.appendingPathComponent("mtp/weights.safetensors"))
            || fileExists(directory.appendingPathComponent("model-mtp.safetensors"))
        let hasMTPMarkers = mtpLayerCount > 0 || hasMTPWeights

        if isDraft || modelType == "qwen3_5_mtp" || modelType == "gemma4_assistant" {
            return MTPCompatibility(
                status: .sidecarOnly,
                reason: "standalone MTP sidecar draft; Telemak requires explicit admin override before pairing unverified external MTP artifacts",
                source: directory.path,
                contractPath: runtimeContractURL?.path,
                overrideRequired: true,
                autoEnabled: false
            )
        }

        guard hasMTPMarkers else {
            return MTPCompatibility(
                status: .noMTP,
                reason: "no embedded MTP markers found in config or safetensors index",
                source: directory.path,
                overrideRequired: false,
                autoEnabled: false
            )
        }

        if isMoEMTPHead(modelType: modelType, textModelType: textModelType, textConfig: textConfig, keys: keys) {
            return MTPCompatibility(
                status: .incompatibleArchitecture,
                reason: "embedded MTP head is MoE; dense Qwen3.6-27B MTP is supported first, MoE MTP stays blocked on #70",
                source: directory.path,
                contractPath: runtimeContractURL?.path,
                overrideRequired: false,
                autoEnabled: false
            )
        }

        guard isSupportedDenseEmbeddedArchitecture(modelType: modelType, textModelType: textModelType) else {
            return MTPCompatibility(
                status: .incompatibleArchitecture,
                reason: "MTP markers exist, but model_type '\(modelType.isEmpty ? "unknown" : modelType)' is not supported by Telemak's native MTP path",
                source: directory.path,
                contractPath: runtimeContractURL?.path,
                overrideRequired: false,
                autoEnabled: false
            )
        }

        if let runtimeContractURL {
            let archId = runtimeArchId(runtimeContractURL)
            if let archId, archId != "qwen3-next-mtp" {
                return MTPCompatibility(
                    status: .incompatibleArchitecture,
                    reason: "MTP runtime contract arch_id '\(archId)' is not supported by Telemak's dense Qwen path",
                    source: directory.path,
                    contractPath: runtimeContractURL.path,
                    overrideRequired: false,
                    autoEnabled: false
                )
            }
            return MTPCompatibility(
                status: .verifiedEmbeddedMTP,
                reason: "embedded MTP markers and runtime contract found",
                source: directory.path,
                contractPath: runtimeContractURL.path,
                overrideRequired: false,
                autoEnabled: true
            )
        }

        return MTPCompatibility(
            status: .unverifiedEmbeddedMTP,
            reason: "embedded MTP markers found, but no telemak_mtp.json or mtplx_runtime.json runtime contract is present",
            source: directory.path,
            overrideRequired: true,
            autoEnabled: false
        )
    }

    private static func isSupportedDenseEmbeddedArchitecture(
        modelType: String,
        textModelType: String
    ) -> Bool {
        [
            "qwen3_5",
            "qwen3_5_text",
            "qwen3_next",
        ].contains(modelType) || [
            "qwen3_5_text",
            "qwen3_next",
        ].contains(textModelType)
    }

    private static func isMoEMTPHead(
        modelType: String,
        textModelType: String,
        textConfig: [String: Any],
        keys: [String]
    ) -> Bool {
        if modelType.contains("moe") || textModelType.contains("moe") { return true }
        let numExperts = int(textConfig["num_experts"]) ?? 0
        if numExperts > 0 { return true }
        return keys.contains { key in
            key.contains(".mlp.experts.") || key.contains(".mlp.shared_expert")
        }
    }

    private static func runtimeArchId(_ url: URL) -> String? {
        let root = readJSON(url)
        return string(root["arch_id"]).lowercased().isEmpty ? nil : string(root["arch_id"]).lowercased()
    }

    private static func isMTPWeightKey(_ key: String) -> Bool {
        key.hasPrefix("mtp.") || key.hasPrefix("language_model.mtp.")
    }

    private static func weightKeys(in directory: URL) -> [String] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: directory.path) else { return [] }
        let indexFiles = entries.filter { $0.hasSuffix(".safetensors.index.json") || $0 == "model.safetensors.index.json" }
        var out: [String] = []
        for file in indexFiles {
            let url = directory.appendingPathComponent(file)
            let root = readJSON(url)
            if let map = root["weight_map"] as? [String: Any] {
                out.append(contentsOf: map.keys)
            }
        }
        return out
    }

    private static func readJSON(_ url: URL) -> [String: Any] {
        guard let data = try? Data(contentsOf: url),
              let root = try? JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]) as? [String: Any]
        else { return [:] }
        return root
    }

    private static func firstExisting(_ urls: [URL]) -> URL? {
        urls.first(where: fileExists)
    }

    private static func fileExists(_ url: URL) -> Bool {
        FileManager.default.fileExists(atPath: url.path)
    }

    private static func string(_ value: Any?) -> String {
        value as? String ?? ""
    }

    private static func int(_ value: Any?) -> Int? {
        if let intValue = value as? Int { return intValue }
        if let number = value as? NSNumber { return number.intValue }
        if let stringValue = value as? String { return Int(stringValue) }
        return nil
    }
}
