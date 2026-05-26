import Foundation
import MLX
import MLXLMCommon

public struct MTPSamplingConfig {
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float

    public init(parameters: GenerateParameters?) {
        self.temperature = parameters?.temperature ?? 0
        self.topP = parameters?.topP ?? 1
        self.topK = parameters?.topK ?? 0
        self.minP = parameters?.minP ?? 0
    }

    public init(temperature: Float, topP: Float = 1, topK: Int = 0, minP: Float = 0) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
    }

    public var isGreedy: Bool { temperature <= 0 }
}

public struct MTPDistribution: Equatable {
    public var probabilities: [Double]

    public init(_ probabilities: [Double]) {
        let clipped = probabilities.map { $0.isFinite && $0 > 0 ? $0 : 0 }
        let total = clipped.reduce(0, +)
        if total > 0 {
            self.probabilities = clipped.map { $0 / total }
        } else if let first = clipped.indices.first {
            var fallback = Array(repeating: 0.0, count: clipped.count)
            fallback[first] = 1
            self.probabilities = fallback
        } else {
            self.probabilities = []
        }
    }

    public func probability(_ token: Int) -> Double {
        guard probabilities.indices.contains(token) else { return 0 }
        return probabilities[token]
    }

    public func sample(random: () -> Double = { Double.random(in: 0 ..< 1) }) -> Int {
        guard !probabilities.isEmpty else { return 0 }
        let draw = min(max(random(), 0), 0.9999999999999999)
        var cumulative = 0.0
        for (idx, p) in probabilities.enumerated() {
            cumulative += p
            if draw < cumulative {
                return idx
            }
        }
        return probabilities.count - 1
    }

    public func residual(after draft: MTPDistribution) -> MTPDistribution {
        let count = max(probabilities.count, draft.probabilities.count)
        var residual = Array(repeating: 0.0, count: count)
        for idx in 0 ..< count {
            let p = idx < probabilities.count ? probabilities[idx] : 0
            let q = idx < draft.probabilities.count ? draft.probabilities[idx] : 0
            residual[idx] = max(p - q, 0)
        }
        let total = residual.reduce(0, +)
        return total > 0 ? MTPDistribution(residual) : self
    }

    public static func oneHot(token: Int, vocabularySize: Int) -> MTPDistribution {
        var probabilities = Array(repeating: 0.0, count: max(vocabularySize, token + 1))
        probabilities[token] = 1
        return MTPDistribution(probabilities)
    }
}

public struct MTPSamplingDecision: Equatable {
    public let accepted: Bool
    public let token: Int
    public let acceptanceProbability: Double

    public init(accepted: Bool, token: Int, acceptanceProbability: Double) {
        self.accepted = accepted
        self.token = token
        self.acceptanceProbability = acceptanceProbability
    }
}

public struct MTPSampledToken {
    public let token: Int
    public let distribution: MTPDistribution

    public init(token: Int, distribution: MTPDistribution) {
        self.token = token
        self.distribution = distribution
    }
}

public enum MTPSampling {
    public static func distribution(from logits: [Double], config: MTPSamplingConfig) -> MTPDistribution {
        guard !logits.isEmpty else { return MTPDistribution([]) }
        if config.isGreedy {
            return .oneHot(token: argmax(logits), vocabularySize: logits.count)
        }

        let usesFilters = (config.topP > 0 && config.topP < 1) || config.topK > 0 || config.minP > 0
        if !usesFilters {
            return MTPDistribution(softmax(logits.map { $0 / Double(config.temperature) }))
        }

        var logProbs = logSoftmax(logits)
        if config.topP > 0 && config.topP < 1 {
            applyTopP(&logProbs, topP: Double(config.topP))
        }
        if config.minP > 0 {
            applyMinP(&logProbs, minP: Double(config.minP))
        }
        if config.topK > 0 {
            applyTopK(&logProbs, topK: config.topK)
        }
        return MTPDistribution(softmax(logProbs.map { $0 / Double(config.temperature) }))
    }

    public static func verify(
        target: MTPDistribution,
        draft: MTPDistribution,
        draftToken: Int,
        random: () -> Double = { Double.random(in: 0 ..< 1) }
    ) -> MTPSamplingDecision {
        let p = target.probability(draftToken)
        let q = draft.probability(draftToken)
        let acceptanceProbability: Double
        if q <= 0 {
            acceptanceProbability = p > 0 ? 1 : 0
        } else {
            acceptanceProbability = min(1, p / q)
        }
        if random() <= acceptanceProbability {
            return MTPSamplingDecision(
                accepted: true,
                token: draftToken,
                acceptanceProbability: acceptanceProbability
            )
        }
        return MTPSamplingDecision(
            accepted: false,
            token: target.residual(after: draft).sample(random: random),
            acceptanceProbability: acceptanceProbability
        )
    }

    public static func outputMarginal(target: MTPDistribution, draft: MTPDistribution) -> MTPDistribution {
        let count = max(target.probabilities.count, draft.probabilities.count)
        var out = Array(repeating: 0.0, count: count)
        for token in 0 ..< count {
            let q = draft.probability(token)
            guard q > 0 else { continue }
            let p = target.probability(token)
            let accept = q <= 0 ? (p > 0 ? 1.0 : 0.0) : min(1.0, p / q)
            out[token] += q * accept
            if accept < 1 {
                let residual = target.residual(after: draft)
                for idx in 0 ..< count {
                    out[idx] += q * (1 - accept) * residual.probability(idx)
                }
            }
        }
        return MTPDistribution(out)
    }

    private static func argmax(_ values: [Double]) -> Int {
        var bestIndex = 0
        var bestValue = values[0]
        for idx in values.indices.dropFirst() where values[idx] > bestValue {
            bestIndex = idx
            bestValue = values[idx]
        }
        return bestIndex
    }

    private static func logSoftmax(_ values: [Double]) -> [Double] {
        let maxValue = values.max() ?? 0
        let shifted = values.map { exp($0 - maxValue) }
        let logSum = log(shifted.reduce(0, +)) + maxValue
        return values.map { $0 - logSum }
    }

    private static func softmax(_ values: [Double]) -> [Double] {
        let finite = values.filter { $0.isFinite }
        guard let maxValue = finite.max() else {
            return Array(repeating: 0.0, count: values.count)
        }
        let exps = values.map { $0.isFinite ? exp($0 - maxValue) : 0 }
        let total = exps.reduce(0, +)
        guard total > 0 else { return exps }
        return exps.map { $0 / total }
    }

    private static func applyTopP(_ logProbs: inout [Double], topP: Double) {
        let threshold = 1 - topP
        let indices = logProbs.indices.sorted { logProbs[$0] < logProbs[$1] }
        var cumulative = 0.0
        var keep = Array(repeating: false, count: logProbs.count)
        for idx in indices {
            cumulative += exp(logProbs[idx])
            if cumulative > threshold {
                keep[idx] = true
            }
        }
        for idx in logProbs.indices where !keep[idx] {
            logProbs[idx] = -Double.infinity
        }
    }

    private static func applyMinP(_ logProbs: inout [Double], minP: Double) {
        guard let maxLogProb = logProbs.filter({ $0.isFinite }).max() else { return }
        let threshold = maxLogProb + log(minP)
        for idx in logProbs.indices where logProbs[idx] < threshold {
            logProbs[idx] = -Double.infinity
        }
    }

    private static func applyTopK(_ logProbs: inout [Double], topK: Int) {
        guard topK > 0 && topK < logProbs.count else { return }
        let keep = Set(logProbs.indices.sorted { logProbs[$0] > logProbs[$1] }.prefix(topK))
        for idx in logProbs.indices where !keep.contains(idx) {
            logProbs[idx] = -Double.infinity
        }
    }
}

public final class MTPTokenSampler {
    public let config: MTPSamplingConfig

    public init(parameters: GenerateParameters?) {
        self.config = MTPSamplingConfig(parameters: parameters)
    }

    public var isGreedy: Bool { config.isGreedy }

    public func sample(logits: MLXArray) -> MTPSampledToken {
        let distribution = distribution(from: logits)
        let token = distribution.sample()
        return MTPSampledToken(token: token, distribution: distribution)
    }

    public func distribution(from logits: MLXArray) -> MTPDistribution {
        let flat = logits.asType(.float32).reshaped(-1)
        eval(flat)
        return MTPSampling.distribution(
            from: flat.asArray(Float.self).map(Double.init),
            config: config
        )
    }
}
