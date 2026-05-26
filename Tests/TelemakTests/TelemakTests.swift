import Testing
import TelemakMTP

@Test func greedyDistributionIsOneHotArgmax() {
    let distribution = MTPSampling.distribution(
        from: [0.1, 3.0, 2.0],
        config: MTPSamplingConfig(temperature: 0)
    )

    #expect(distribution.probabilities == [0.0, 1.0, 0.0])
    #expect(distribution.sample(random: { 0.99 }) == 1)
}

@Test func probabilityRatioRejectsIntoResidualDistribution() {
    let target = MTPDistribution([0.1, 0.7, 0.2])
    let draft = MTPDistribution([0.5, 0.2, 0.3])
    var draws = [0.9, 0.0]

    let decision = MTPSampling.verify(
        target: target,
        draft: draft,
        draftToken: 0,
        random: { draws.removeFirst() }
    )

    #expect(decision.accepted == false)
    #expect(abs(decision.acceptanceProbability - 0.2) < 0.000001)
    #expect(decision.token == 1)
}

@Test func speculativeOutputMarginalMatchesTargetDistribution() {
    let target = MTPDistribution([0.2, 0.5, 0.3])
    let draft = MTPDistribution([0.4, 0.4, 0.2])
    let marginal = MTPSampling.outputMarginal(target: target, draft: draft)

    for idx in target.probabilities.indices {
        #expect(abs(marginal.probability(idx) - target.probability(idx)) < 0.000001)
    }
}

@Test func topKFilterLimitsSamplingSupport() {
    let distribution = MTPSampling.distribution(
        from: [0.0, 1.0, 2.0, 3.0],
        config: MTPSamplingConfig(temperature: 0.6, topK: 2)
    )

    #expect(distribution.probability(0) == 0)
    #expect(distribution.probability(1) == 0)
    #expect(distribution.probability(2) > 0)
    #expect(distribution.probability(3) > 0)
}
