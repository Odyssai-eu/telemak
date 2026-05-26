import Foundation
import MLX
import MLXLLM
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

    /// Target (main) model the draft borrows embed_tokens + lm_head
    /// from. Set by `bind(_:)`. The draft holds the reference for the
    /// lifetime of one chat completion. The protocol lets us accept
    /// the LLM-side `Qwen35Model`.
    public private(set) weak var boundTarget: (any Qwen35HiddenStateProvider)?

    /// Per-round draft state — number of draft positions appended to
    /// the draft's KV cache during the *current* round, so we know how
    /// many to trim on rejection.
    private var roundAppended: Int = 0

    /// Seed token + hidden from the last `prefill_from_target_hidden`
    /// or `accept_verified_tokens` call. Consumed at the start of the
    /// next `draftBlock` to recover the position lost to the rejected
    /// candidates of the previous round.
    private var seedToken: MLXArray?
    private var seedHidden: MLXArray?

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

    // MARK: - Speculative-loop helpers

    /// Bind the draft to its target. The draft borrows the target's
    /// embed_tokens (for next-token lookup) and lm_head (to score
    /// proposals). Call at the start of every chat completion ; rebind
    /// is safe — old reference is dropped.
    public func bind(_ target: any Qwen35HiddenStateProvider) {
        self.boundTarget = target
        self.seedToken = nil
        self.seedHidden = nil
        self.roundAppended = 0
    }

    /// Build a fresh KV cache for the draft's layers (one per layer).
    /// MTP layers are always full-attention so `KVCacheSimple` suffices
    /// — no MambaCache plumbing needed.
    public func makeCache() -> [KVCache] {
        layers.map { _ in KVCacheSimple() }
    }

    /// Forward one token through the draft : embed the token id, run
    /// the MTP forward with the previous hidden as the "hidden" input,
    /// return the new hidden state. Used by `draftBlock` to chain
    /// `block_size - 1` candidate predictions.
    private func forwardToken(
        _ tokenId: MLXArray,
        previousHidden: MLXArray,
        cache: [KVCache]
    ) -> MLXArray {
        guard let target = boundTarget else {
            fatalError("Qwen35MTPDraftModel.forwardToken called before bind(_:).")
        }
        let tokEmbed = target.embed(tokenId)
        return callAsFunction(
            tokenEmbedding: tokEmbed,
            hidden: previousHidden,
            cache: cache.map { $0 as KVCache? }
        )
    }

    /// Seed the draft from the target's prefill : feed shifted
    /// `(input_ids[1:] + first_bonus)` plus the target's hidden states
    /// at those positions. This primes the draft's KV cache so the
    /// first `draftBlock` call has the correct K/V to attend over.
    public func prefillFromTargetHidden(
        inputIds: MLXArray,
        hidden: MLXArray,
        bonusToken: Int,
        cache: [KVCache]
    ) {
        guard let target = boundTarget else {
            fatalError("Qwen35MTPDraftModel.prefillFromTargetHidden called before bind(_:).")
        }
        let promptLen = inputIds.shape[1]
        guard promptLen > 0 else { return }
        let shifted: MLXArray
        if promptLen > 1 {
            let bonusArr = MLXArray([Int32(bonusToken)], [1, 1])
            shifted = concatenated([inputIds[0..., 1 ..< promptLen], bonusArr], axis: 1)
        } else {
            // Single-token prompt edge case — just the bonus.
            shifted = MLXArray([Int32(bonusToken)], [1, 1])
        }
        let shiftedLen = shifted.shape[1]
        let useHidden = hidden[0..., 0 ..< shiftedLen, 0...]
        let tokEmbed = target.embed(shifted)
        let h = callAsFunction(
            tokenEmbedding: tokEmbed,
            hidden: useHidden,
            cache: cache.map { $0 as KVCache? }
        )
        // Seed for the upcoming draftBlock — use the *last* hidden
        // produced by the prefill and the bonus token as the seed.
        let lastH = h[0..., (shiftedLen - 1) ..< shiftedLen, 0...]
        let logits = target.applyLMHead(lastH)
        let argmax = MLX.argMax(logits, axis: -1)
        self.seedToken = argmax
        self.seedHidden = lastH
    }

    /// Propose `blockSize - 1` candidate tokens autoregressively. The
    /// first candidate either consumes the seed (from prefill /
    /// previous accept) or comes from a fresh `(lastBonus, hidden)`
    /// forward through the draft.
    public func draftBlock(
        lastBonus: Int,
        hidden: MLXArray,
        cache: [KVCache],
        blockSize: Int
    ) -> MLXArray {
        guard let target = boundTarget else {
            fatalError("Qwen35MTPDraftModel.draftBlock called before bind(_:).")
        }
        var tok = MLXArray([Int32(lastBonus)], [1, 1])
        var hPrev = hidden
        var tokens: [MLXArray] = []
        self.roundAppended = 0

        if let st = self.seedToken, let sh = self.seedHidden {
            tok = st
            hPrev = sh
            tokens.append(tok)
            self.seedToken = nil
            self.seedHidden = nil
        }

        while tokens.count < blockSize - 1 {
            hPrev = forwardToken(tok, previousHidden: hPrev, cache: cache)
            self.roundAppended += 1
            let logits = target.applyLMHead(hPrev)
            tok = MLX.argMax(logits, axis: -1)
            tokens.append(tok)
        }
        return concatenated(tokens, axis: 1)
    }

    /// Update the draft's KV cache after the target has verified the
    /// proposed block. Trim the cache by the rejected count (positions
    /// past the first reject), then re-forward the (newly-accepted +
    /// bonus) tokens to bring the cache back into sync with what the
    /// target accepted.
    public func acceptVerifiedTokens(
        verifyHidden: MLXArray,
        draftTokens: MLXArray,
        accepted: Int,
        newTokens: [Int],
        cache: [KVCache]
    ) {
        guard let target = boundTarget else {
            fatalError("Qwen35MTPDraftModel.acceptVerifiedTokens called before bind(_:).")
        }
        let acceptedCount = min(accepted, roundAppended)
        let trim = roundAppended - acceptedCount
        if trim > 0 {
            for c in cache { _ = c.trim(trim) }
        }

        // Build the tail of tokens we still need to push into the
        // draft's cache : the candidates from `acceptedCount` up to
        // `accepted`, plus the new bonus (last entry of newTokens).
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
                tokenEmbedding: tokEmbed,
                hidden: hiddens,
                cache: cache.map { $0 as KVCache? }
            )
            // Seed for next round : use the last position's hidden +
            // its argmax as the seed token.
            let lastH = h[0..., (h.shape[1] - 1) ..< h.shape[1], 0...]
            let logits = target.applyLMHead(lastH)
            self.seedToken = MLX.argMax(logits, axis: -1)
            self.seedHidden = lastH
        }
        self.roundAppended = 0
    }
}
