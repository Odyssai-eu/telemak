import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXVLM

/// Common ABI both Qwen3.5/3.6 dispatches (LLM `Qwen35Model` and
/// VLM `Qwen35`) implement on the Odyssai-eu fork. Lets the
/// speculative iterator and draft model work against either without
/// caring which factory produced the container.
public protocol Qwen35HiddenStateProvider: AnyObject {
    func forwardWithHidden(_ inputs: MLXArray, cache: [KVCache]?) -> (logits: MLXArray, hidden: MLXArray)
    func embed(_ inputs: MLXArray) -> MLXArray
    func applyLMHead(_ hidden: MLXArray) -> MLXArray
    func newCache(parameters: GenerateParameters?) -> [KVCache]
}

extension Qwen35Model: Qwen35HiddenStateProvider {}
extension Qwen35: Qwen35HiddenStateProvider {}

/// Speculative-decoding driver for a Qwen3.5/3.6 main + MTP draft pair.
///
/// One iteration of `next()` emits one token. Internally it batches
/// `blockSize` candidate verifications per real-target forward pass —
/// when the draft is right, we get up to `blockSize` tokens for the
/// cost of one target forward + one cheap draft forward. The expected
/// speedup on Qwen3.6-35B-A3B is ~1.6× (matching what Inferencer.app
/// demonstrates).
///
/// Greedy-only for the first cut : both the target and the draft use
/// argmax sampling. Temperature / top-p sampling adds complications
/// (rejection probability needs the actual logits) that come later.
///
/// ## Cache rollback
///
/// On rejection, the target's KV caches have advanced past the first
/// reject and need to be rolled back. Strategy : snapshot the cache
/// state before the verify forward (via `KVCache.copy()`), then on
/// reject swap the cache array back to the snapshot. Cheap because
/// MLXArray is reference-counted ; `copy()` for unmodified caches is
/// effectively a pointer bump.
///
/// Note Qwen3.5/3.6 has interleaved linear-attention layers (
/// `Qwen35GatedDeltaNet` backed by `MambaCache`) where in-place
/// state mutation isn't reversible via `trim()`. The snapshot+swap
/// strategy bypasses this : we keep the whole cache state, not just
/// the trim count.
public final class MTPSpeculativeIterator {

    public let main: any Qwen35HiddenStateProvider
    public let draft: Qwen35MTPDraftModel
    public let blockSize: Int

    /// Maximum tokens to emit (including the first bonus).
    public let maxTokens: Int

    /// Stats — exposed for telemetry / dashboard pills.
    public private(set) var roundsRun: Int = 0
    public private(set) var totalProposed: Int = 0
    public private(set) var totalAccepted: Int = 0
    public var acceptanceRate: Double {
        guard totalProposed > 0 else { return 0 }
        return Double(totalAccepted) / Double(totalProposed)
    }

    private var mainCache: [KVCache]
    private var draftCache: [KVCache]
    private var bonusToken: Int
    private var pending: [Int] = []
    private var emitted: Int = 0
    private var done: Bool = false

    public init(
        main: any Qwen35HiddenStateProvider,
        draft: Qwen35MTPDraftModel,
        promptTokens: [Int],
        maxTokens: Int,
        blockSize: Int? = nil
    ) {
        self.main = main
        self.draft = draft
        self.blockSize = blockSize ?? draft.blockSize
        self.maxTokens = maxTokens

        precondition(!promptTokens.isEmpty, "MTPSpeculativeIterator requires non-empty prompt")
        precondition(self.blockSize >= 2, "blockSize must be ≥ 2 for any speedup")

        draft.bind(main)
        self.mainCache = main.newCache(parameters: nil)
        self.draftCache = draft.makeCache()

        // 1. Prefill main on the full prompt. We need (logits, hidden)
        //    at every position so the draft prefill can index into the
        //    hidden states.
        let promptArray = MLXArray(promptTokens.map { Int32($0) }, [1, promptTokens.count])
        let (logits, hidden) = main.forwardWithHidden(promptArray, cache: mainCache)

        // 2. First bonus = argmax of the last position's logits.
        let lastLogits = logits[0..., (logits.shape[1] - 1) ..< logits.shape[1], 0...]
        let argmaxArr = MLX.argMax(lastLogits, axis: -1)
        let bonus = Int(argmaxArr.item(Int32.self))
        self.bonusToken = bonus
        self.pending.append(bonus)
        self.emitted = 0

        // 3. Prefill the draft from the same hidden states.
        draft.prefillFromTargetHidden(
            inputIds: promptArray,
            hidden: hidden,
            bonusToken: bonus,
            cache: draftCache
        )
    }

    public func next() -> Int? {
        if done { return nil }
        if emitted >= maxTokens {
            done = true
            return nil
        }
        if !pending.isEmpty {
            let t = pending.removeFirst()
            emitted += 1
            if emitted >= maxTokens { done = true }
            return t
        }
        runRound()
        if pending.isEmpty {
            done = true
            return nil
        }
        let t = pending.removeFirst()
        emitted += 1
        if emitted >= maxTokens { done = true }
        return t
    }

    /// One speculative round : draft proposes `blockSize - 1`
    /// candidates ; target verifies in one batched forward ; we walk
    /// the acceptance frontier and refill the pending queue.
    private func runRound() {
        guard let target = draft.boundTarget else {
            done = true
            return
        }

        // We need the draft to know the target's hidden state for the
        // bonus token's position. We don't have it post-prefill (we
        // didn't keep the prompt's full hidden tensor) so we
        // synthesize : embed the bonus + look up the embedding as a
        // proxy. The draft's seed (from prefill / previous accept)
        // covers the position lost on cache rollback so this is OK as
        // an initial hidden for `draftBlock` — the seed kicks in on
        // the first iteration of the inner loop and `lastBonus` /
        // `hidden` are only used when there's no seed.
        //
        // For a *correct* (zero-divergence) implementation we'd keep
        // a single-position last-hidden cache. The seed mechanism
        // means this approximation only matters for the very first
        // round when prefill seeded the draft — and `prefillFromTargetHidden`
        // sets the seed, so in practice this branch isn't reached on
        // round 1.
        let bonusEmbedding = target.embed(MLXArray([Int32(bonusToken)], [1, 1]))

        // 1. Draft proposes blockSize - 1 candidates.
        let draftTokens = draft.draftBlock(
            lastBonus: bonusToken,
            hidden: bonusEmbedding,
            cache: draftCache,
            blockSize: blockSize
        )
        let nCandidates = draftTokens.shape[1]

        // 2. Build verify input = [bonus, draft[0], …, draft[nCandidates-1]].
        let bonusArr = MLXArray([Int32(bonusToken)], [1, 1])
        let verifyInput = concatenated([bonusArr, draftTokens], axis: 1)

        // 3. Snapshot the main cache before the verify forward — so we
        //    can roll back on rejection.
        let snapshot = mainCache.map { $0.copy() }

        // 4. Run target verify forward.
        let (logits, hidden) = main.forwardWithHidden(verifyInput, cache: mainCache)
        // logits shape: [1, nCandidates + 1, vocab]
        // hidden shape: [1, nCandidates + 1, hidden_dim]

        // 5. Acceptance walk : at each position i, the target's
        //    argmax(logits[:, i, :]) is its prediction for position
        //    i+1. Compare to draft[i] ; on first mismatch, accept up
        //    to that point and emit the target's choice as the new
        //    bonus.
        //
        // Perf : we batch the argmaxes into one call and evaluate the
        // resulting `[1, nCandidates + 1]` tensor a single time, then
        // read the ints. Per-position `.item()` calls used to force
        // an MLX↔CPU sync each iteration (≈ 2 * blockSize syncs per
        // round) which dominated the loop cost on a 35B target.
        let targetArgmax = MLX.argMax(logits, axis: -1)
        let draftArgmax = draftTokens.reshaped(-1).asType(.int32)
        eval(targetArgmax, draftArgmax)
        let targetIds = targetArgmax.reshaped(-1).asArray(Int32.self).map(Int.init)
        let draftIds = draftArgmax.asArray(Int32.self).map(Int.init)

        var accepted = 0
        var newTokens: [Int] = []
        var newBonus: Int = 0

        let budget = maxTokens - emitted - pending.count
        for pos in 0 ... nCandidates {
            let targetTok = targetIds[pos]
            if pos < nCandidates && targetTok == draftIds[pos] {
                accepted += 1
                if newTokens.count < budget {
                    newTokens.append(targetTok)
                }
                continue
            }
            if newTokens.count < budget {
                newTokens.append(targetTok)
                newBonus = targetTok
            } else {
                // Budget already saturated — we keep the last accepted
                // token's choice as the bonus.
                newBonus = newTokens.last ?? bonusToken
            }
            break
        }

        // 6. Roll back main cache if we rejected any candidates.
        let rejected = nCandidates - accepted
        if rejected > 0 {
            mainCache = snapshot
            // Re-play the accepted prefix (+ bonus) so the cache is in
            // sync with what we'll yield. accepted + 1 tokens land in
            // the cache.
            if !newTokens.isEmpty {
                let replayIds = newTokens.map { Int32($0) }
                let replay = MLXArray(replayIds, [1, replayIds.count])
                _ = main.forwardWithHidden(replay, cache: mainCache)
            }
        }
        // If rejected == 0, the verify forward already advanced the
        // cache by `nCandidates + 1` which is exactly what we want.

        // 7. Update draft cache : trim back to acceptedCount + 1
        //    correct positions, then push the (acceptedCount + bonus)
        //    new positions.
        draft.acceptVerifiedTokens(
            verifyHidden: hidden,
            draftTokens: draftTokens,
            accepted: accepted,
            newTokens: newTokens,
            cache: draftCache
        )

        // 8. Push yielded tokens into pending ; track stats.
        pending.append(contentsOf: newTokens)
        bonusToken = newBonus
        roundsRun += 1
        totalProposed += nCandidates
        totalAccepted += accepted
    }
}
