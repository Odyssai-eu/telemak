import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Qwen3.5 / Qwen3.6 MTP draft model — Swift port of
/// `Blaizzy/mlx-vlm`'s `Qwen3_5MTPDraftModel`.
///
/// A *draft* model in speculative decoding proposes a block of N candidate
/// next-tokens that the main (target) model then verifies in a single
/// forward pass. When the candidates match, you get N tokens per pass
/// instead of one → speedup. The MTP variant differs from classic
/// drafters in that the draft receives `(embedded_token, hidden_state)`
/// straight from the target's last-layer hidden state, not just a token
/// id — tight coupling, better acceptance rate.
///
/// ## Architecture
///
/// One `Qwen35DecoderLayer` (reused from mlx-swift-lm's main model
/// implementation), preceded by two RMSNorms (one for the token
/// embedding, one for the hidden state) and a Linear that concatenates
/// + projects `[2 * hidden_size] → [hidden_size]`. After the layer, a
/// final RMSNorm. Weights total ~906 MB for Qwen3.6-35B-A3B-MTP at 9-bit.
///
/// ## Status — V2 Unit 1 complete (architecture port)
///
/// Wrapper class + config wiring + weight remap + decoder layer all
/// in place. The decoder layer (`MTPDecoderLayer` in
/// `Qwen35MTPLayer.swift`) is vendored from mlx-swift-lm's
/// `Qwen35DecoderLayer`, trimmed to just the full-attention + MoE
/// path that MTP actually uses (MTP forces
/// `full_attention_interval=1` ; no linear-attention branch).
///
/// References :
/// - `Blaizzy/mlx-vlm/mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py`
/// - `docs/V2-MTP-DRAFT-PORT.md`
public final class Qwen35MTPDraftModel: Module, BaseLanguageModel, @unchecked Sendable {

    public let config: Qwen35MTPConfiguration
    public let blockSize: Int

    /// Pre-projection norm applied to the embedded next-token.
    @ModuleInfo(key: "pre_fc_norm_embedding") public var preFcNormEmbedding: RMSNorm
    /// Pre-projection norm applied to the main model's hidden state.
    @ModuleInfo(key: "pre_fc_norm_hidden") public var preFcNormHidden: RMSNorm
    /// Concatenation projection : `[2 * hidden] → [hidden]`, no bias.
    @ModuleInfo(key: "fc") public var fc: Linear
    /// Final norm before LM head.
    @ModuleInfo(key: "norm") public var norm: RMSNorm
    /// MTP transformer layers (exactly `mtp_num_hidden_layers`, almost
    /// always 1 in published Qwen3.5/3.6 MTP repos).
    public let layers: [MTPDecoderLayer]

    public init(_ config: Qwen35MTPConfiguration) {
        self.config = config
        self.blockSize = config.blockSize
        let textConfig = config.textConfig
        let hidden = textConfig.hiddenSize
        let eps = textConfig.rmsNormEps
        let mtpLayers = max(1, config.mtpNumHiddenLayers)
        self._fc.wrappedValue = Linear(2 * hidden, hidden, bias: false)
        self._preFcNormEmbedding.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self._preFcNormHidden.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self._norm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self.layers = (0 ..< mtpLayers).map { _ in MTPDecoderLayer(textConfig) }
        super.init()
    }

    /// Forward pass : `(embed(token), hidden) → next-token hidden`.
    /// Called by the speculative loop in Unit 2.
    public func callAsFunction(
        tokenEmbedding: MLXArray,
        hidden: MLXArray,
        cache: [KVCache?]?,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        let normedEmbed = preFcNormEmbedding(tokenEmbedding)
        let normedHidden = preFcNormHidden(hidden)
        var h = concatenated([normedEmbed, normedHidden], axis: -1)
        h = fc(h)
        let caches = cache ?? Array(repeating: nil, count: layers.count)
        for (layer, layerCache) in zip(layers, caches) {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = layerCache.map {
                createAttentionMask(h: h, cache: $0)
            } ?? (h.shape[1] > 1 ? .causal : .none)
            h = layer(h, mask: mask, cache: layerCache, positionIds: positionIds)
        }
        return norm(h)
    }

    /// Weight remap — port of Python `sanitize()`. Two steps :
    ///
    /// 1. Strip leading `mtp.` prefix if the safetensors archive
    ///    namespaces the MTP weights under it (matches Blaizzy split.py
    ///    output).
    /// 2. Split a fused `mlp.experts.gate_up_proj` tensor into separate
    ///    `mlp.switch_mlp.gate_proj` and `mlp.switch_mlp.up_proj`
    ///    weights (mirrors the same step in mlx-swift-lm's
    ///    `Qwen35MoEModel.sanitize`).
    ///
    /// The "+1 to norm weights" step in the Python (`value + 1.0`) is
    /// Qwen3.5/3.6's pre-fused-residual norm convention — the
    /// vendored `Qwen35DecoderLayer` already bakes this in. We keep
    /// the explicit add here so the weight tensors load identically
    /// regardless of which layer impl we end up using.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var working = weights

        // 1a. Strip `mtp.` prefix (if Blaizzy split.py emitted it).
        let stripped: [String: MLXArray] = working.reduce(into: [:]) { acc, kv in
            let key = kv.key.hasPrefix("mtp.") ? String(kv.key.dropFirst("mtp.".count)) : kv.key
            acc[key] = kv.value
        }
        working = stripped

        // 1b. Split fused `experts.gate_up_proj` → `switch_mlp.gate_proj`
        // + `switch_mlp.up_proj` on every layer that has one (typically
        // layers.0 only on a single-layer MTP).
        let gateUpKeys = working.keys.filter { $0.hasSuffix(".experts.gate_up_proj") }
        for fullKey in gateUpKeys {
            let prefix = String(fullKey.dropLast(".experts.gate_up_proj".count))
            guard let gateUp = working[fullKey] else { continue }
            working[fullKey] = nil
            let mid = gateUp.dim(-2) / 2
            working["\(prefix).switch_mlp.gate_proj.weight"] =
                gateUp[.ellipsis, 0 ..< mid, 0...]
            working["\(prefix).switch_mlp.up_proj.weight"] =
                gateUp[.ellipsis, mid..., 0...]
            let downKey = "\(prefix).experts.down_proj"
            if let down = working[downKey] {
                working[downKey] = nil
                working["\(prefix).switch_mlp.down_proj.weight"] = down
            }
        }

        // 2. RMSNorm weight +1 fold for Qwen3.5/3.6 norms.
        let normSuffixes: [String] = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
            "norm.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
        ]
        for (key, value) in working {
            var v = value
            if normSuffixes.contains(where: { key.hasSuffix($0) }), value.ndim == 1 {
                if value.dtype == .float16 || value.dtype == .float32 || value.dtype == .bfloat16 {
                    v = value + 1.0
                }
            }
            out[key] = v
        }
        return out
    }
}

