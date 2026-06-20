import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import TelemakMTP

/// Hy3 (HunYuan-3) MoE-MTP draft model — the embedded `model.layers.80` head,
/// extracted into a sidecar (`scripts/hy3_mtp_extract.py`) and paired with the
/// 9-bit trunk for native speculative decoding.
///
/// Structurally a calque of `Qwen35MTPDraftModel` (the draft-loop helpers are
/// architecture-agnostic — they only call `target.embed` / `target.applyLMHead`),
/// with three Hy3-specific divergences resolved from vLLM `hy_v3_mtp.py`:
///   - checkpoint keys are `enorm` / `hnorm` / `eh_proj` / `final_layernorm`
///     (not Qwen's `pre_fc_norm_*` / `fc` / `norm`);
///   - the decoder layer is the sigmoid/expert-bias MoE `HYV3MTPDecoderLayer`;
///   - **sanitize is a pass-through** — the sidecar is already bf16, experts
///     pre-stacked into `switch_mlp`, and Hy3 uses plain RMSNorm (NO Qwen `+1`).
///
/// Concat order is embedding-FIRST (`eh_proj(cat[enorm(embed), hnorm(hidden)])`),
/// and the head consumes the trunk's PRE-norm hidden (see `HYV3MTP.swift` ABI).
public final class HYV3MTPDraftModel: Module, BaseLanguageModel, @unchecked Sendable {

    public let config: HYV3MTPConfiguration
    public let blockSize: Int

    @ModuleInfo(key: "enorm") public var enorm: RMSNorm           // pre-proj norm on the next-token embedding
    @ModuleInfo(key: "hnorm") public var hnorm: RMSNorm           // pre-proj norm on the trunk hidden
    @ModuleInfo(key: "eh_proj") public var ehProj: Linear         // [2*hidden] -> [hidden], no bias
    @ModuleInfo(key: "final_layernorm") public var finalLayernorm: RMSNorm
    public let layers: [HYV3MTPDecoderLayer]

    public private(set) weak var boundTarget: (any HYV3HiddenStateProvider)?

    private var roundAppended: Int = 0
    private var seedSample: MTPSampledToken?
    private var seedHidden: MLXArray?

    public init(_ config: HYV3MTPConfiguration) {
        self.config = config
        self.blockSize = config.blockSize
        let f = config.fields
        let hidden = f.hiddenSize
        let eps = f.rmsNormEps
        let mtpLayers = max(1, config.mtpNumHiddenLayers)
        self._ehProj.wrappedValue = Linear(2 * hidden, hidden, bias: false)
        self._enorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self._hnorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self._finalLayernorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        self.layers = (0 ..< mtpLayers).map { _ in HYV3MTPDecoderLayer(f) }
        super.init()
    }

    /// Forward: `(embed(token), pre-norm trunk hidden) → next-token hidden`.
    /// Embedding FIRST in the concat (eh_proj column order), final_layernorm at the end.
    public func callAsFunction(
        tokenEmbedding: MLXArray,
        hidden: MLXArray,
        cache: [KVCache?]?,
        positionIds: MLXArray? = nil
    ) -> MLXArray {
        let normedEmbed = enorm(tokenEmbedding)
        let normedHidden = hnorm(hidden)
        var h = concatenated([normedEmbed, normedHidden], axis: -1)
        h = ehProj(h)
        let caches = cache ?? Array(repeating: nil, count: layers.count)
        for (layer, layerCache) in zip(layers, caches) {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode = layerCache.map {
                createAttentionMask(h: h, cache: $0)
            } ?? (h.shape[1] > 1 ? .causal : .none)
            h = layer(h, mask: mask, cache: layerCache, positionIds: positionIds)
        }
        return finalLayernorm(h)
    }

    /// Pass-through — the sidecar already matches the module tree (bf16,
    /// experts pre-stacked into switch_mlp, plain RMSNorm with no `+1` offset).
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    // MARK: - Speculative-loop helpers (architecture-agnostic; calque of Qwen35MTPDraftModel)

    public func bind(_ target: any HYV3HiddenStateProvider) {
        self.boundTarget = target
        self.seedSample = nil
        self.seedHidden = nil
        self.roundAppended = 0
    }

    public func makeCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }

    private func forwardToken(
        _ tokenId: MLXArray, previousHidden: MLXArray, cache: [KVCache]
    ) -> MLXArray {
        guard let target = boundTarget else {
            fatalError("HYV3MTPDraftModel.forwardToken called before bind(_:).")
        }
        let tokEmbed = target.embed(tokenId)
        return callAsFunction(
            tokenEmbedding: tokEmbed, hidden: previousHidden, cache: cache.map { $0 as KVCache? })
    }

    func prefillFromTargetHidden(
        inputIds: MLXArray, hidden: MLXArray, bonusToken: Int,
        cache: [KVCache], sampler: MTPTokenSampler
    ) {
        guard let target = boundTarget else {
            fatalError("HYV3MTPDraftModel.prefillFromTargetHidden called before bind(_:).")
        }
        let promptLen = inputIds.shape[1]
        guard promptLen > 0 else { return }
        let shifted: MLXArray
        if promptLen > 1 {
            let bonusArr = MLXArray([Int32(bonusToken)], [1, 1])
            shifted = concatenated([inputIds[0..., 1 ..< promptLen], bonusArr], axis: 1)
        } else {
            shifted = MLXArray([Int32(bonusToken)], [1, 1])
        }
        let shiftedLen = shifted.shape[1]
        let useHidden = hidden[0..., 0 ..< shiftedLen, 0...]
        let tokEmbed = target.embed(shifted)
        let h = callAsFunction(
            tokenEmbedding: tokEmbed, hidden: useHidden, cache: cache.map { $0 as KVCache? })
        let lastH = h[0..., (shiftedLen - 1) ..< shiftedLen, 0...]
        let logits = target.applyLMHead(lastH)
        self.seedSample = sampler.sample(logits: logits)
        self.seedHidden = lastH
    }

    struct DraftBlock {
        let tokens: MLXArray
        let samples: [MTPSampledToken]
    }

    func draftBlock(
        lastBonus: Int, hidden: MLXArray, cache: [KVCache],
        blockSize: Int, sampler: MTPTokenSampler
    ) -> DraftBlock {
        guard boundTarget != nil else {
            fatalError("HYV3MTPDraftModel.draftBlock called before bind(_:).")
        }
        var tok = MLXArray([Int32(lastBonus)], [1, 1])
        var hPrev = hidden
        var samples: [MTPSampledToken] = []
        self.roundAppended = 0

        if let seed = self.seedSample, let sh = self.seedHidden {
            tok = MLXArray([Int32(seed.token)], [1, 1])
            hPrev = sh
            samples.append(seed)
            self.seedSample = nil
            self.seedHidden = nil
        }

        while samples.count < blockSize - 1 {
            hPrev = forwardToken(tok, previousHidden: hPrev, cache: cache)
            self.roundAppended += 1
            let logits = boundTarget!.applyLMHead(hPrev)
            let sample = sampler.sample(logits: logits)
            tok = MLXArray([Int32(sample.token)], [1, 1])
            samples.append(sample)
        }
        let tokenIds = samples.map { Int32($0.token) }
        return DraftBlock(tokens: MLXArray(tokenIds, [1, tokenIds.count]), samples: samples)
    }

    func acceptVerifiedTokens(
        verifyHidden: MLXArray, draftTokens: MLXArray, accepted: Int,
        newTokens: [Int], cache: [KVCache], sampler: MTPTokenSampler
    ) {
        guard let target = boundTarget else {
            fatalError("HYV3MTPDraftModel.acceptVerifiedTokens called before bind(_:).")
        }
        let acceptedCount = min(accepted, roundAppended)
        let trim = roundAppended - acceptedCount
        if trim > 0 {
            for c in cache { _ = c.trim(trim) }
        }

        var tokenChunks: [MLXArray] = []
        var hiddenChunks: [MLXArray] = []
        for draftIdx in acceptedCount ..< accepted {
            tokenChunks.append(draftTokens[0..., draftIdx ..< (draftIdx + 1)])
            hiddenChunks.append(verifyHidden[0..., draftIdx ..< (draftIdx + 1), 0...])
        }
        if let bonus = newTokens.last {
            tokenChunks.append(MLXArray([Int32(bonus)], [1, 1]))
            hiddenChunks.append(verifyHidden[0..., accepted ..< (accepted + 1), 0...])
        }
        if !tokenChunks.isEmpty {
            let tokens = concatenated(tokenChunks, axis: 1)
            let hiddens = concatenated(hiddenChunks, axis: 1)
            let tokEmbed = target.embed(tokens)
            let h = callAsFunction(
                tokenEmbedding: tokEmbed, hidden: hiddens, cache: cache.map { $0 as KVCache? })
            let lastH = h[0..., (h.shape[1] - 1) ..< h.shape[1], 0...]
            let logits = target.applyLMHead(lastH)
            self.seedSample = sampler.sample(logits: logits)
            self.seedHidden = lastH
        }
        self.roundAppended = 0
    }
}
