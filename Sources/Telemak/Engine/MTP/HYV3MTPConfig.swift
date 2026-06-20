import Foundation
import MLXLMCommon

/// Config for the Hy3 MoE-MTP draft head — decodes the sidecar `config.json`
/// (`model_type: "hy_v3_mtp"`) produced by `scripts/hy3_mtp_extract.py`.
public struct HYV3MTPConfiguration: Decodable, Sendable {
    public var fields: HYV3MTPFields
    public var mtpNumHiddenLayers: Int
    public var blockSize: Int
    public var vocabSize: Int
    public var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case numExperts = "num_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case numSharedExperts = "num_shared_experts"
        case routerScalingFactor = "router_scaling_factor"
        case routeNorm = "route_norm"
        case rmsNormEps = "rms_norm_eps"
        case ropeParameters = "rope_parameters"
        case maxPositionEmbeddings = "max_position_embeddings"
        case numMtpLayers = "num_mtp_layers"
        case vocabSize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
    }
    private enum RopeKeys: String, CodingKey { case ropeTheta = "rope_theta" }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let hidden = try c.decode(Int.self, forKey: .hiddenSize)

        var theta: Float = 11_158_840.0
        if let rp = try? c.nestedContainer(keyedBy: RopeKeys.self, forKey: .ropeParameters) {
            theta = (try? rp.decode(Float.self, forKey: .ropeTheta)) ?? theta
        }

        self.fields = HYV3MTPFields(
            hiddenSize: hidden,
            moeIntermediateSize: try c.decode(Int.self, forKey: .moeIntermediateSize),
            numAttentionHeads: try c.decode(Int.self, forKey: .numAttentionHeads),
            numKeyValueHeads: try c.decode(Int.self, forKey: .numKeyValueHeads),
            headDim: try c.decode(Int.self, forKey: .headDim),
            numExperts: try c.decode(Int.self, forKey: .numExperts),
            numExpertsPerTok: try c.decode(Int.self, forKey: .numExpertsPerTok),
            numSharedExperts: try c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 1,
            routerScalingFactor: try c.decodeIfPresent(Float.self, forKey: .routerScalingFactor) ?? 1.0,
            routeNorm: try c.decodeIfPresent(Bool.self, forKey: .routeNorm) ?? true,
            rmsNormEps: try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5,
            ropeTheta: theta,
            ropeScaling: nil,
            maxPositionEmbeddings: try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
        )
        self.mtpNumHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numMtpLayers) ?? 1
        self.vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        self.tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        // D2 is the M3-class optimum (per #69): draft 2 candidates → verify block of 3.
        self.blockSize = 3
    }
}
