import Foundation
import MLXLMCommon

/// Subset of `qwen3_5*` text-config fields needed by the MTP draft
/// model. mlx-swift-lm exposes `Qwen35TextConfiguration` as a public
/// struct but its stored properties are module-internal, so we can't
/// read them from outside MLXLLM. We re-decode the same JSON with our
/// own struct that surfaces only what the MTP wrapper actually uses.
public struct Qwen35TextFields: Decodable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let attentionBias: Bool
    public let attnOutputGate: Bool
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let partialRotaryFactor: Float
    public let ropeScaling: [String: StringOrNumber]?
    public let fullAttentionInterval: Int
    public let intermediateSize: Int
    public let numExperts: Int
    public let numExpertsPerTok: Int
    public let normTopkProb: Bool
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case attentionBias = "attention_bias"
        case attnOutputGate = "attn_output_gate"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case fullAttentionInterval = "full_attention_interval"
        case intermediateSize = "intermediate_size"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case normTopkProb = "norm_topk_prob"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? ""
        self.hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        self.rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        self.attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        self.kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        let head = try c.decodeIfPresent(Int.self, forKey: .headDim)
        let attHeads = self.attentionHeads
        let hidden = self.hiddenSize
        self.headDim = head ?? (attHeads > 0 ? hidden / attHeads : 0)
        self.attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.attnOutputGate = try c.decodeIfPresent(Bool.self, forKey: .attnOutputGate) ?? true
        self.maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131072
        // RoPE params : prefer the nested `rope_parameters` dict (Qwen3.5
        // shape) ; fall back to top-level keys (older configs).
        let nestedRope = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        if let nestedRope {
            self.ropeTheta = nestedRope["rope_theta"]?.asFloat() ?? 10000000
            self.partialRotaryFactor = nestedRope["partial_rotary_factor"]?.asFloat() ?? 0.25
            var fixed = nestedRope
            if fixed["type"] == nil, let t = fixed["rope_type"] {
                fixed["type"] = t
            }
            self.ropeScaling = fixed
        } else {
            self.ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000000
            self.partialRotaryFactor =
                try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.25
            self.ropeScaling = try c.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeScaling)
        }
        self.fullAttentionInterval = try c.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
        self.intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 0
        self.numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        self.numExpertsPerTok = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
        self.normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        self.moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        self.sharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        self.tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

/// Config for `qwen3_5_mtp` — the Qwen3.5 / Qwen3.6 MTP draft model.
///
/// Port of `mlx_vlm/speculative/drafters/qwen3_5_mtp/config.py` from
/// `Blaizzy/mlx-vlm`. The HF `config.json` shape is :
///
/// ```jsonc
/// {
///   "model_type": "qwen3_5_mtp",
///   "block_size": 3,
///   "text_config": {
///     "model_type": "qwen3_5_moe_text",
///     "hidden_size": 2048,
///     "mtp_num_hidden_layers": 1,
///     "mtp_use_dedicated_embeddings": false,
///     "tie_word_embeddings": false,
///     ...
///   }
/// }
/// ```
public struct Qwen35MTPConfiguration: Decodable, Sendable {
    public let modelType: String
    public let textConfig: Qwen35TextFields
    public let mtpNumHiddenLayers: Int
    public let mtpUseDedicatedEmbeddings: Bool
    public let blockSize: Int
    public let tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case blockSize = "block_size"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    private struct MTPExtras: Decodable {
        let mtpNumHiddenLayers: Int?
        let mtpUseDedicatedEmbeddings: Bool?
        let tieWordEmbeddings: Bool?
        enum CodingKeys: String, CodingKey {
            case mtpNumHiddenLayers = "mtp_num_hidden_layers"
            case mtpUseDedicatedEmbeddings = "mtp_use_dedicated_embeddings"
            case tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_5_mtp"
        self.textConfig = try c.decode(Qwen35TextFields.self, forKey: .textConfig)
        let extras = try c.decodeIfPresent(MTPExtras.self, forKey: .textConfig)
            ?? MTPExtras(
                mtpNumHiddenLayers: nil, mtpUseDedicatedEmbeddings: nil, tieWordEmbeddings: nil)
        let mtpLayers = extras.mtpNumHiddenLayers ?? 1
        self.mtpNumHiddenLayers = mtpLayers
        self.mtpUseDedicatedEmbeddings = extras.mtpUseDedicatedEmbeddings ?? false
        // block_size default = mtp_num_hidden_layers + 2 (matches Blaizzy from_dict).
        self.blockSize = try c.decodeIfPresent(Int.self, forKey: .blockSize) ?? (mtpLayers + 2)
        let topLevelTie = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        self.tieWordEmbeddings = extras.tieWordEmbeddings ?? topLevelTie
    }
}
