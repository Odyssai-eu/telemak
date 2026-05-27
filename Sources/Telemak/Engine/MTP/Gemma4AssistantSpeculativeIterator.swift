import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import TelemakMTP

/// Speculative-decoding driver for a Gemma4 target + Gemma4Assistant sidecar pair.
///
/// Gemma's assistant drafter is not a standalone LanguageModel: it consumes
/// target-token embeddings plus the previous target hidden state, and attends
/// over selected target KV caches. This iterator keeps that native contract
/// instead of routing through MLXLMCommon's generic SpeculativeTokenIterator.
public final class Gemma4AssistantSpeculativeIterator {

    public let main: Gemma4Model
    public let draft: Gemma4AssistantModel
    public let blockSize: Int
    public let maxTokens: Int

    public private(set) var roundsRun: Int = 0
    public private(set) var totalProposed: Int = 0
    public private(set) var totalAccepted: Int = 0
    public var acceptanceRate: Double {
        guard totalProposed > 0 else { return 0 }
        return Double(totalAccepted) / Double(totalProposed)
    }

    private var mainCache: [KVCache]
    private var bonusToken: Int
    private var feedbackHidden: MLXArray
    private var pending: [Int] = []
    private var emitted: Int = 0
    private var done: Bool = false
    private let sampler: MTPTokenSampler

    public init(
        main: Gemma4Model,
        draft: Gemma4AssistantModel,
        promptTokens: [Int],
        maxTokens: Int,
        blockSize: Int = 6,
        parameters: GenerateParameters? = nil
    ) {
        precondition(!promptTokens.isEmpty, "Gemma4AssistantSpeculativeIterator requires non-empty prompt")
        precondition(blockSize >= 2, "blockSize must be >= 2 for Gemma4 MTP")

        self.main = main
        self.draft = draft
        self.blockSize = blockSize
        self.maxTokens = maxTokens
        self.sampler = MTPTokenSampler(parameters: parameters)
        self.mainCache = main.newCache(parameters: nil)

        let promptArray = MLXArray(promptTokens.map { Int32($0) }, [1, promptTokens.count])
        let (logits, hidden) = main.forwardWithHidden(promptArray, cache: mainCache)
        let last = logits.shape[1] - 1
        let lastLogits = logits[0..., last ..< (last + 1), 0...]
        let lastHidden = hidden[0..., last ..< (last + 1), 0...]
        let bonus = sampler.sample(logits: lastLogits).token

        self.bonusToken = bonus
        self.feedbackHidden = lastHidden
        self.pending.append(bonus)
    }

    public func next() throws -> Int? {
        if done { return nil }
        if emitted >= maxTokens {
            done = true
            return nil
        }
        if !pending.isEmpty {
            let token = pending.removeFirst()
            emitted += 1
            if emitted >= maxTokens { done = true }
            return token
        }

        try runRound()
        if pending.isEmpty {
            done = true
            return nil
        }
        let token = pending.removeFirst()
        emitted += 1
        if emitted >= maxTokens { done = true }
        return token
    }

    private func runRound() throws {
        let budget = maxTokens - emitted - pending.count
        guard budget > 0 else {
            done = true
            return
        }

        let planned = min(blockSize - 1, max(0, budget - 1))
        var inputToken = bonusToken
        var hidden = feedbackHidden
        var draftTokenArrays: [MLXArray] = []
        var draftSamples: [MTPSampledToken] = []

        if planned > 0 {
            let sharedKV = try draft.sharedKeyValues(
                from: mainCache,
                targetLayerTypes: main.layerTypes,
                targetNumKVSharedLayers: main.numKVSharedLayers
            )

            for _ in 0 ..< planned {
                let tokenInput = MLXArray([Int32(inputToken)], [1, 1])
                let output = draft.draft(
                    inputsEmbeds: main.inputEmbeddings(tokenInput),
                    feedbackHiddenStates: hidden,
                    sharedKeyValues: sharedKV
                )
                let logits = output.logits[0..., (output.logits.shape[1] - 1) ..< output.logits.shape[1], 0...]
                let sample = sampler.sample(logits: logits)
                let tokenArray = MLXArray([Int32(sample.token)], [1, 1])
                draftSamples.append(sample)
                draftTokenArrays.append(tokenArray)
                inputToken = sample.token
                hidden = output.backboneHiddenStates[0..., (output.backboneHiddenStates.shape[1] - 1) ..< output.backboneHiddenStates.shape[1], 0...]
            }
        }

        let bonusArray = MLXArray([Int32(bonusToken)], [1, 1])
        if draftTokenArrays.isEmpty {
            let (logits, verifyHidden) = main.forwardWithHidden(bonusArray, cache: mainCache)
            let sample = sampler.sample(logits: logits[0..., 0 ..< 1, 0...])
            feedbackHidden = verifyHidden[0..., 0 ..< 1, 0...]
            pending.append(sample.token)
            bonusToken = sample.token
            roundsRun += 1
            return
        }

        let draftTokens = concatenated(draftTokenArrays, axis: 1)
        let verifyInput = concatenated([bonusArray, draftTokens], axis: 1)
        let (logits, verifyHidden) = main.forwardWithHidden(verifyInput, cache: mainCache)
        let acceptance = sampler.isGreedy
            ? greedyAcceptance(logits: logits, draftTokens: draftTokens, nCandidates: draftSamples.count)
            : samplingAcceptance(logits: logits, draftSamples: draftSamples, nCandidates: draftSamples.count)

        let rejected = draftSamples.count - acceptance.accepted
        if rejected > 0 {
            trimPromptCache(mainCache, numTokens: rejected)
        }

        let keptHiddenIndex = acceptance.accepted
        feedbackHidden = verifyHidden[0..., keptHiddenIndex ..< (keptHiddenIndex + 1), 0...]
        pending.append(contentsOf: acceptance.newTokens)
        bonusToken = acceptance.newBonus
        roundsRun += 1
        totalProposed += draftSamples.count
        totalAccepted += acceptance.accepted
    }

    private struct AcceptanceResult {
        let accepted: Int
        let newTokens: [Int]
        let newBonus: Int
    }

    private func greedyAcceptance(
        logits: MLXArray,
        draftTokens: MLXArray,
        nCandidates: Int
    ) -> AcceptanceResult {
        let targetArgmax = MLX.argMax(logits, axis: -1)
        let draftArgmax = draftTokens.reshaped(-1).asType(.int32)
        eval(targetArgmax, draftArgmax)

        let targetIds = targetArgmax.reshaped(-1).asArray(Int32.self).map(Int.init)
        let draftIds = draftArgmax.asArray(Int32.self).map(Int.init)
        let budget = maxTokens - emitted - pending.count

        var accepted = 0
        var newTokens: [Int] = []
        var newBonus = bonusToken

        for pos in 0 ... nCandidates {
            let targetToken = targetIds[pos]
            if pos < nCandidates && targetToken == draftIds[pos] {
                accepted += 1
                if newTokens.count < budget {
                    newTokens.append(targetToken)
                    newBonus = targetToken
                }
                continue
            }
            if newTokens.count < budget {
                newTokens.append(targetToken)
            }
            newBonus = targetToken
            break
        }

        return AcceptanceResult(accepted: accepted, newTokens: newTokens, newBonus: newBonus)
    }

    private func samplingAcceptance(
        logits: MLXArray,
        draftSamples: [MTPSampledToken],
        nCandidates: Int
    ) -> AcceptanceResult {
        let budget = maxTokens - emitted - pending.count
        var accepted = 0
        var newTokens: [Int] = []
        var newBonus = bonusToken

        for pos in 0 ..< nCandidates {
            let targetLogits = logits[0..., pos ..< (pos + 1), 0...]
            let targetDistribution = sampler.distribution(from: targetLogits)
            let draftSample = draftSamples[pos]
            let decision = MTPSampling.verify(
                target: targetDistribution,
                draft: draftSample.distribution,
                draftToken: draftSample.token
            )
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
            }
            newBonus = decision.token
            return AcceptanceResult(accepted: accepted, newTokens: newTokens, newBonus: newBonus)
        }

        let bonusLogits = logits[0..., nCandidates ..< (nCandidates + 1), 0...]
        let bonusSample = sampler.sample(logits: bonusLogits)
        if newTokens.count < budget {
            newTokens.append(bonusSample.token)
        }
        newBonus = bonusSample.token
        return AcceptanceResult(accepted: accepted, newTokens: newTokens, newBonus: newBonus)
    }
}
