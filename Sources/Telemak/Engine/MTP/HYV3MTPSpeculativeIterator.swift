import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import TelemakMTP

/// Hidden-state ABI the Hy3 LLM dispatch implements (in the fork, `HYV3MTP.swift`).
/// Hy3 is full-attention (no SSM), so — unlike `Qwen35HiddenStateProvider` — there
/// is no rollback capture buffer: `forwardWithHidden` covers prefill AND verify,
/// and cache rollback on rejection is a plain KV trim done by the iterator.
public protocol HYV3HiddenStateProvider: AnyObject {
    func forwardWithHidden(_ inputs: MLXArray, cache: [KVCache]?) -> (logits: MLXArray, hidden: MLXArray)
    func embed(_ inputs: MLXArray) -> MLXArray
    func applyLMHead(_ hidden: MLXArray) -> MLXArray
    func newMTPCache() -> [KVCache]
}

extension HYV3Model: HYV3HiddenStateProvider {}

/// Speculative-decoding driver for a Hy3 main + MoE-MTP draft pair.
///
/// Calque of `MTPSpeculativeIterator` (the Qwen3.5 driver), simplified: Hy3 has
/// no linear-attention/SSM layers, so the per-step capture buffer + `targetVerify`
/// are replaced by `forwardWithHidden` + a KV-cache `trim` on rejection.
public final class HYV3MTPSpeculativeIterator {

    public let main: any HYV3HiddenStateProvider
    public let draft: HYV3MTPDraftModel
    public let blockSize: Int
    public let maxTokens: Int

    public private(set) var roundsRun: Int = 0
    public private(set) var totalProposed: Int = 0
    public private(set) var totalAccepted: Int = 0
    public private(set) var mainPrefillSeconds: Double = 0
    public private(set) var draftSeconds: Double = 0
    public private(set) var verifySeconds: Double = 0
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
    private let sampler: MTPTokenSampler

    public init(
        main: any HYV3HiddenStateProvider,
        draft: HYV3MTPDraftModel,
        promptTokens: [Int],
        maxTokens: Int,
        blockSize: Int? = nil,
        parameters: GenerateParameters? = nil
    ) {
        self.main = main
        self.draft = draft
        self.blockSize = blockSize ?? draft.blockSize
        self.maxTokens = maxTokens
        self.sampler = MTPTokenSampler(parameters: parameters)

        precondition(!promptTokens.isEmpty, "HYV3MTPSpeculativeIterator requires non-empty prompt")
        precondition(self.blockSize >= 2, "blockSize must be ≥ 2 for any speedup")

        draft.bind(main)
        self.mainCache = main.newMTPCache()
        self.draftCache = draft.makeCache()

        let promptArray = MLXArray(promptTokens.map { Int32($0) }, [1, promptTokens.count])
        let mainPrefillStart = Date()
        let (logits, hidden) = main.forwardWithHidden(promptArray, cache: mainCache)
        self.mainPrefillSeconds = Date().timeIntervalSince(mainPrefillStart)

        let lastLogits = logits[0..., (logits.shape[1] - 1) ..< logits.shape[1], 0...]
        let bonus = sampler.sample(logits: lastLogits).token
        self.bonusToken = bonus
        self.pending.append(bonus)
        self.emitted = 0

        draft.prefillFromTargetHidden(
            inputIds: promptArray, hidden: hidden, bonusToken: bonus,
            cache: draftCache, sampler: sampler)
    }

    public func next() -> Int? {
        if done { return nil }
        if emitted >= maxTokens { done = true; return nil }
        if !pending.isEmpty {
            let t = pending.removeFirst()
            emitted += 1
            if emitted >= maxTokens { done = true }
            return t
        }
        runRound()
        if pending.isEmpty { done = true; return nil }
        let t = pending.removeFirst()
        emitted += 1
        if emitted >= maxTokens { done = true }
        return t
    }

    private func runRound() {
        guard let target = draft.boundTarget else { done = true; return }

        let bonusEmbedding = target.embed(MLXArray([Int32(bonusToken)], [1, 1]))

        let draftStart = Date()
        let draftBlock = draft.draftBlock(
            lastBonus: bonusToken, hidden: bonusEmbedding,
            cache: draftCache, blockSize: blockSize, sampler: sampler)
        draftSeconds += Date().timeIntervalSince(draftStart)
        let draftTokens = draftBlock.tokens
        let nCandidates = draftTokens.shape[1]

        let bonusArr = MLXArray([Int32(bonusToken)], [1, 1])
        let verifyInput = concatenated([bonusArr, draftTokens], axis: 1)

        // Hy3: one forward covers verify; no SSM capture buffer.
        let verifyStart = Date()
        let (logits, hidden) = main.forwardWithHidden(verifyInput, cache: mainCache)
        verifySeconds += Date().timeIntervalSince(verifyStart)

        let acceptance = sampler.isGreedy
            ? greedyAcceptance(logits: logits, draftTokens: draftTokens, nCandidates: nCandidates)
            : samplingAcceptance(logits: logits, draftSamples: draftBlock.samples, nCandidates: nCandidates)
        let accepted = acceptance.accepted
        let newTokens = acceptance.newTokens
        let newBonus = acceptance.newBonus

        // Rollback = plain KV trim: the verify advanced the cache by (nCandidates+1);
        // keep (accepted+1) (bonus + accepted draft), trim the rejected suffix.
        let rejected = nCandidates - accepted
        if rejected > 0 {
            for c in mainCache { _ = c.trim(rejected) }
        }

        draft.acceptVerifiedTokens(
            verifyHidden: hidden, draftTokens: draftTokens, accepted: accepted,
            newTokens: newTokens, cache: draftCache, sampler: sampler)

        pending.append(contentsOf: newTokens)
        bonusToken = newBonus
        roundsRun += 1
        totalProposed += nCandidates
        totalAccepted += accepted
    }

    private struct AcceptanceResult {
        let accepted: Int
        let newTokens: [Int]
        let newBonus: Int
    }

    private func greedyAcceptance(
        logits: MLXArray, draftTokens: MLXArray, nCandidates: Int
    ) -> AcceptanceResult {
        let targetArgmax = MLX.argMax(logits, axis: -1)
        let draftArgmax = draftTokens.reshaped(-1).asType(.int32)
        eval(targetArgmax, draftArgmax)
        let targetIds = targetArgmax.reshaped(-1).asArray(Int32.self).map(Int.init)
        let draftIds = draftArgmax.asArray(Int32.self).map(Int.init)

        var accepted = 0
        var newTokens: [Int] = []
        var newBonus = 0
        let budget = maxTokens - emitted - pending.count
        for pos in 0 ... nCandidates {
            let targetTok = targetIds[pos]
            if pos < nCandidates && targetTok == draftIds[pos] {
                accepted += 1
                if newTokens.count < budget { newTokens.append(targetTok) }
                continue
            }
            if newTokens.count < budget {
                newTokens.append(targetTok)
                newBonus = targetTok
            } else {
                newBonus = newTokens.last ?? bonusToken
            }
            break
        }
        return AcceptanceResult(accepted: accepted, newTokens: newTokens, newBonus: newBonus)
    }

    private func samplingAcceptance(
        logits: MLXArray, draftSamples: [MTPSampledToken], nCandidates: Int
    ) -> AcceptanceResult {
        var accepted = 0
        var newTokens: [Int] = []
        var newBonus = bonusToken
        let budget = maxTokens - emitted - pending.count

        for pos in 0 ..< nCandidates {
            let targetLogits = logits[0..., pos ..< (pos + 1), 0...]
            let targetDistribution = sampler.distribution(from: targetLogits)
            let draftSample = draftSamples[pos]
            let decision = MTPSampling.verify(
                target: targetDistribution, draft: draftSample.distribution, draftToken: draftSample.token)
            if decision.accepted {
                accepted += 1
                if newTokens.count < budget {
                    newTokens.append(draftSample.token)
                    newBonus = draftSample.token
                }
                continue
            }
            if newTokens.count < budget {
                newTokens.append(decision.token)
                newBonus = decision.token
            }
            return AcceptanceResult(accepted: accepted, newTokens: newTokens, newBonus: newBonus)
        }

        let bonusLogits = logits[0..., nCandidates ..< (nCandidates + 1), 0...]
        let bonusSample = sampler.sample(logits: bonusLogits)
        if newTokens.count < budget {
            newTokens.append(bonusSample.token)
            newBonus = bonusSample.token
        }
        return AcceptanceResult(accepted: accepted, newTokens: newTokens, newBonus: newBonus)
    }
}
