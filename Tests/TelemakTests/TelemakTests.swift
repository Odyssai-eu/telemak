import Foundation
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

@Test func mtpCompatibilityRequiresContractForEmbeddedMarkers() throws {
    let dir = try temporaryModelDir(config: [
        "model_type": "qwen3_5",
        "text_config": ["mtp_num_hidden_layers": 1],
    ])
    defer { try? FileManager.default.removeItem(at: dir) }

    let compatibility = MTPCompatibility.inspect(modelId: "test/qwen", directory: dir)

    #expect(compatibility.status == .unverifiedEmbeddedMTP)
    #expect(compatibility.overrideRequired == true)
    #expect(compatibility.autoEnabled == false)
}

@Test func mtpCompatibilityAcceptsVerifiedEmbeddedContract() throws {
    let dir = try temporaryModelDir(
        config: [
            "model_type": "qwen3_5",
            "text_config": ["mtp_num_hidden_layers": 1],
        ],
        runtimeContract: ["runtime": "telemak", "schema": 1]
    )
    defer { try? FileManager.default.removeItem(at: dir) }

    let compatibility = MTPCompatibility.inspect(modelId: "test/qwen", directory: dir)

    #expect(compatibility.status == .verifiedEmbeddedMTP)
    #expect(compatibility.overrideRequired == false)
    #expect(compatibility.autoEnabled == true)
}

@Test func mtpCompatibilityClassifiesSidecarDraftsAsOverrideOnly() throws {
    let dir = try temporaryModelDir(config: ["model_type": "qwen3_5_mtp"])
    defer { try? FileManager.default.removeItem(at: dir) }

    let compatibility = MTPCompatibility.inspect(modelId: "test/draft", directory: dir, isDraft: true)

    #expect(compatibility.status == .sidecarOnly)
    #expect(compatibility.overrideRequired == true)
    #expect(compatibility.canRun(allowUnverified: false) == false)
    #expect(compatibility.canRun(allowUnverified: true) == true)
}

@Test func mtpCompatibilityReportsNoMTPWithoutMarkers() throws {
    let dir = try temporaryModelDir(config: ["model_type": "llama"])
    defer { try? FileManager.default.removeItem(at: dir) }

    let compatibility = MTPCompatibility.inspect(modelId: "test/llama", directory: dir)

    #expect(compatibility.status == .noMTP)
    #expect(compatibility.overrideRequired == false)
}

private func temporaryModelDir(
    config: [String: Any],
    runtimeContract: [String: Any]? = nil
) throws -> URL {
    let dir = FileManager.default.temporaryDirectory
        .appendingPathComponent("telemak-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let configData = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
    try configData.write(to: dir.appendingPathComponent("config.json"))
    if let runtimeContract {
        let contractData = try JSONSerialization.data(withJSONObject: runtimeContract, options: [.prettyPrinted, .sortedKeys])
        try contractData.write(to: dir.appendingPathComponent("telemak_mtp.json"))
    }
    return dir
}
