import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Decoder layer for the Hy3 (HunYuan-3) MoE-MTP draft head.
///
/// Vendored from the fork's `HYV3.swift` (`HYV3Attention` / `HYV3MoE` /
/// `HYV3Router`), which are `internal` (invisible across the module boundary) —
/// same vendoring rationale as `Qwen35MTPLayer.swift`. Two architectural
/// differences from the Qwen3.5 MTP layer:
///   - attention is plain GQA + qk_norm, NO output gate (Hy3 has no `attn_output_gate`);
///   - the MoE router is **sigmoid + expert-correction bias + route-norm + router_scaling**
///     (DeepSeek-V3 style), with a plain `shared_mlp` and **no** `shared_expert_gate`.
///
/// Reference: vLLM `hy_v3_mtp.py` / the fork `HYV3.swift` (the trunk this mirrors).

/// Config fields the Hy3 MTP head + layer need (from the sidecar `config.json`).
struct HYV3MTPFields {
    var hiddenSize: Int
    var moeIntermediateSize: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var headDim: Int
    var numExperts: Int
    var numExpertsPerTok: Int
    var numSharedExperts: Int
    var routerScalingFactor: Float
    var routeNorm: Bool
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int
}

// MARK: - Routing (DeepSeek-V3: sigmoid + bias correction) — identical to HYV3.swift

private func hyv3MTPExpertSelect(
    gates: MLXArray, expertBias: MLXArray, topK: Int,
    routedScalingFactor: Float, normTopkProb: Bool
) -> (MLXArray, MLXArray) {
    let scores = sigmoid(gates.asType(.float32))
    let originalScores = scores
    let selectionScores = scores + expertBias.asType(.float32)
    let inds = argPartition(-selectionScores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    var weights = takeAlong(originalScores, inds, axis: -1)
    if topK > 1, normTopkProb {
        weights = weights / weights.sum(axis: -1, keepDims: true)
    }
    weights = weights * routedScalingFactor
    return (inds, weights)
}

// MARK: - Attention (GQA + qk_norm, no output gate)

final class HYV3MTPAttention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ a: HYV3MTPFields) {
        self.nHeads = a.numAttentionHeads
        self.nKVHeads = a.numKeyValueHeads
        self.headDim = a.headDim
        self.scale = pow(Float(a.headDim), -0.5)
        _qProj.wrappedValue = Linear(a.hiddenSize, nHeads * headDim, bias: false)
        _kProj.wrappedValue = Linear(a.hiddenSize, nKVHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(a.hiddenSize, nKVHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(nHeads * headDim, a.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: a.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: a.rmsNormEps)
        self.rope = initializeRope(
            dims: headDim, base: a.ropeTheta, traditional: false,
            scalingConfig: a.ropeScaling, maxPositionEmbeddings: a.maxPositionEmbeddings)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(B, L, nHeads, -1)
        var k = kProj(x).reshaped(B, L, nKVHeads, -1)
        var v = vProj(x).reshaped(B, L, nKVHeads, -1)
        q = qNorm(q).transposed(0, 2, 1, 3)
        k = kNorm(k).transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)
        q = applyRotaryPosition(rope, to: q, cache: cache)
        k = applyRotaryPosition(rope, to: k, cache: cache)
        let o = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return oProj(o)
    }
}

// MARK: - MLP (dense / shared expert)

final class HYV3MTPMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Router + MoE block (sigmoid + bias + shared_mlp, no shared_expert_gate)

final class HYV3MTPRouter: Module {
    let topK: Int
    let normTopkProb: Bool
    let routedScalingFactor: Float

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray

    init(_ a: HYV3MTPFields) {
        self.topK = a.numExpertsPerTok
        self.normTopkProb = a.routeNorm
        self.routedScalingFactor = a.routerScalingFactor
        _gate.wrappedValue = Linear(a.hiddenSize, a.numExperts, bias: false)
        _expertBias.wrappedValue = MLXArray.zeros([a.numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        hyv3MTPExpertSelect(
            gates: gate(x), expertBias: expertBias, topK: topK,
            routedScalingFactor: routedScalingFactor, normTopkProb: normTopkProb)
    }
}

final class HYV3MTPMoE: Module, UnaryLayer {
    @ModuleInfo(key: "router") var router: HYV3MTPRouter
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_mlp") var sharedMLP: HYV3MTPMLP

    init(_ a: HYV3MTPFields) {
        _router.wrappedValue = HYV3MTPRouter(a)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: a.hiddenSize, hiddenDims: a.moeIntermediateSize, numExperts: a.numExperts)
        _sharedMLP.wrappedValue = HYV3MTPMLP(
            hiddenSize: a.hiddenSize, intermediateSize: a.moeIntermediateSize * a.numSharedExperts)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, weights) = router(x)
        var y = switchMLP(x, inds)
        y = (y * weights[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        return y + sharedMLP(x)
    }
}

// MARK: - Decoder Layer

public final class HYV3MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: HYV3MTPAttention
    @ModuleInfo(key: "mlp") var mlp: HYV3MTPMoE
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ a: HYV3MTPFields) {
        _selfAttn.wrappedValue = HYV3MTPAttention(a)
        _mlp.wrappedValue = HYV3MTPMoE(a)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: a.hiddenSize, eps: a.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: a.hiddenSize, eps: a.rmsNormEps)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        _ = positionIds  // RoPE pulls the offset from cache
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}
