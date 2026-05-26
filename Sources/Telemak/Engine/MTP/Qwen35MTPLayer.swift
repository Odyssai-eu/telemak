import Foundation
import MLX
import MLXLMCommon
import MLXNN

/// Decoder layer for the Qwen3.5 / Qwen3.6 MTP draft model.
///
/// Vendored (and trimmed) from mlx-swift-lm's `Qwen35DecoderLayer`
/// (Libraries/MLXLLM/Models/Qwen35.swift) because the upstream class
/// is `final class … : Module` with no `public` modifier — invisible
/// across the module boundary. We could fork mlx-swift-lm, but
/// vendoring is :
///
/// - Lower-friction (no new repo to maintain or upstream PR
///   negotiation with `ml-explore`).
/// - Sufficient — MTP only needs the *full-attention* + MoE branch
///   of the layer. The original supports both linear-attention
///   (`Qwen35GatedDeltaNet`) and full-attention paths, multiplexed by
///   `layer_idx % full_attention_interval`. The MTP draft model
///   forces `full_attention_interval=1` (Python `replace(..., full_attention_interval=1)`
///   in `Qwen3_5MTPDraftModel.__init__`), so every MTP layer is full
///   attention. We can drop the linear-attention path entirely and
///   the file shrinks from 700+ LOC to ~250.
///
/// The public mlx-swift-lm helpers (`initializeRope`,
/// `applyRotaryPosition`, `attentionWithCacheUpdate`, `SwitchGLU`)
/// do the heavy lifting. We only need to wire the layer shape and
/// hand off to them.
///
/// Reference repo for the architecture :
/// - `inferencerlabs/Qwen3.6-35B-A3B-MTP-MLX-9bit`
/// - `Blaizzy/mlx-vlm/mlx_vlm/models/qwen3_5_moe/language.py`
///   (`Qwen3_5MoeDecoderLayer` — the Python original we mirror).

// MARK: - Helpers (internal, vendored from Qwen3Next.swift)

func sigmoidMultiply(_ x: MLXArray, _ gate: MLXArray) -> MLXArray {
    x * sigmoid(gate)
}

/// Two-Linear gated MLP used by both the dense path (when there are
/// no experts) and the MoE block's shared_expert. Vendored from
/// `Qwen3NextMLP` in `Libraries/MLXLLM/Models/Qwen3Next.swift`.
final class MTPGatedMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Attention

/// Full-attention with q/k norm + RoPE + optional output gate.
/// Mirrors `Qwen35Attention` in mlx-swift-lm, adapted to read from
/// our `Qwen35TextFields`. `attn_output_gate: true` (Qwen3.5/3.6
/// default) doubles the q_proj output dim — first half is the query,
/// second half is the gate fed through sigmoid at the end.
final class MTPAttention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let scale: Float
    let useOutputGate: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: Qwen35TextFields) {
        let headDim = args.headDim > 0 ? args.headDim : (args.hiddenSize / args.attentionHeads)
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(headDim), -0.5)
        self.useOutputGate = args.attnOutputGate

        let qMultiplier = args.attnOutputGate ? 2 : 1
        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * headDim * qMultiplier, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * headDim, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let ropeDims = Int(Float(headDim) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        let qProjOutput = qProj(x)
        var queries: MLXArray
        var gate: MLXArray? = nil
        if useOutputGate {
            let qSplit = qProjOutput.reshaped(B, L, attentionHeads, -1).split(parts: 2, axis: -1)
            queries = qSplit[0]
            gate = qSplit[1].reshaped(B, L, -1)
        } else {
            queries = qProjOutput.reshaped(B, L, attentionHeads, -1)
        }

        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        values = values.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, cache: cache)
        keys = applyRotaryPosition(rope, to: keys, cache: cache)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        if let gate {
            return oProj(sigmoidMultiply(output, gate))
        }
        return oProj(output)
    }
}

// MARK: - SparseMoeBlock

/// Top-k softmax-gated MoE + shared_expert + shared_expert_gate.
/// Mirrors `Qwen35SparseMoeBlock` in mlx-swift-lm, adapted to
/// `Qwen35TextFields`.
final class MTPSparseMoeBlock: Module, UnaryLayer {
    let normTopkProb: Bool
    let numExperts: Int
    let topK: Int

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    @ModuleInfo(key: "shared_expert") var sharedExpert: MTPGatedMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen35TextFields) {
        self.normTopkProb = args.normTopkProb
        self.numExperts = args.numExperts
        self.topK = args.numExpertsPerTok

        _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize,
            hiddenDims: args.moeIntermediateSize,
            numExperts: args.numExperts
        )

        _sharedExpert.wrappedValue = MTPGatedMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize
        )
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var gates = gate(x)
        gates = MLX.softmax(gates, axis: -1, precise: true)

        let k = topK
        let kth = gates.dim(-1) - k
        let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, (kth)...]
        var scores = MLX.takeAlong(gates, inds, axis: -1)
        if normTopkProb {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }

        let y = switchMLP(x, inds)
        let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

        var sharedY = sharedExpert(x)
        sharedY = sigmoid(sharedExpertGate(x)) * sharedY

        return combined + sharedY
    }
}

// MARK: - Decoder Layer (full-attention + MoE only)

/// Full-attention + MoE-MLP transformer block. MTP layers always run
/// full attention (the Python `replace(..., full_attention_interval=1)`
/// forces the layer-index modulo to always pick the self-attention
/// branch), so we trim out the linear-attention path entirely vs the
/// upstream `Qwen35DecoderLayer`.
public final class MTPDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: MTPAttention
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Module

    public init(_ args: Qwen35TextFields) {
        _selfAttn.wrappedValue = MTPAttention(args)
        if args.numExperts > 0 {
            _mlp.wrappedValue = MTPSparseMoeBlock(args)
        } else {
            _mlp.wrappedValue = MTPGatedMLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize
            )
        }
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue =
            RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        // `positionIds` is unused — RoPE pulls the offset from `cache`.
        // The Blaizzy Python takes explicit positions for the batched
        // verify path ; we'll add that in Unit 2 when the speculative
        // loop needs it.
        _ = positionIds
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
    }
}
