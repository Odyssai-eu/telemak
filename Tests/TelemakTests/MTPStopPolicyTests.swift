import Foundation
import MLXLMCommon
import Testing
@testable import Telemak

// Tags split into concatenated literals like ToolCallRecoveryTests — keeps
// the special-token markers out of naive token-level tooling.
private let thinkOpen = "<think"
private let thinkClose = "</think"
private let imEnd = "<|im_end"
private let endOfText = "<|endoftext"

private func joined(_ pieces: [String]) -> String {
    pieces.joined()
}

// MARK: - MTPStopPolicy: stop-set construction (fork buildStopTokenIds parity)

@Test func stopPolicyUnionsAllStopSources() {
    let policy = MTPStopPolicy(
        eosTokenIds: [7, 8],
        tokenizerEosTokenId: 9,
        extraEOSTokens: ["<|im_end|>", "<|unknown-token|>"],
        convertTokenToId: { token -> Int? in
            switch token {
            case "<|im_end|>": return 10
            default: return nil
            }
        },
        unknownTokenId: 11
    )
    #expect(policy.isStopToken(7))
    #expect(policy.isStopToken(8))
    #expect(policy.isStopToken(9))
    #expect(policy.isStopToken(10)) // extraEOSTokens resolved through the tokenizer
    #expect(policy.isStopToken(11)) // unknown token id
    #expect(!policy.isStopToken(12))
    #expect(!policy.isStopToken(0))
}

@Test func stopPolicySkipsUnresolvableExtraTokens() {
    let policy = MTPStopPolicy(
        eosTokenIds: [7],
        tokenizerEosTokenId: nil,
        extraEOSTokens: ["<|not-in-vocab|>"],
        convertTokenToId: { _ in nil },
        unknownTokenId: nil
    )
    #expect(policy.stopTokenIds == [7])
}

// MARK: - MTPGenerationCollector: stop-token detection on assembled text

/// Fake iterator: drains a fixed token list. Mirrors what the speculative
/// iterators expose (`next() -> Int?`), stopping at nil like max_tokens
/// exhaustion.
private struct FakeIterator {
    let tokens: [Int]
    var index = 0
    mutating func next() -> Int? {
        guard index < tokens.count else { return nil }
        defer { index += 1 }
        return tokens[index]
    }
}

/// Identity-ish fake decode: token id n decodes to a unique character so
/// incremental decoding works like the real tokenizer.
private func fakeDecode(_ tokens: [Int]) -> String {
    String(tokens.map { Character(UnicodeScalar(65 + $0 % 26)!) })
}

@Test func collectorStopsOnStopTokenBeforeEmittingIt() {
    // Tokens 0, 1 are ordinary; 99 is in the stop set; 2 must NEVER be
    // consumed or decoded (the pre-fix behaviour decoded the EOS marker and
    // kept generating past it).
    var iterator = FakeIterator(tokens: [0, 1, 99, 2])
    let policy = MTPStopPolicy(eosTokenIds: [99], tokenizerEosTokenId: nil, extraEOSTokens: [], convertTokenToId: { _ in nil }, unknownTokenId: nil)
    let result = MTPGenerationCollector.collect(
        next: { iterator.next() },
        maxTokens: 10,
        stopPolicy: policy,
        stopSequences: [],
        decode: fakeDecode
    )
    #expect(joined(result.pieces) == "AB")
    #expect(result.tokensGenerated == 2)
    #expect(result.hitStop)
    #expect(!result.hitCeiling)
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, hitStop: result.hitStop, hitCeiling: result.hitCeiling) == "stop")
}

@Test func collectorReportsLengthWhenIteratorExhausts() {
    // Iterator ends on its own (max_tokens ceiling) with no stop seen.
    var iterator = FakeIterator(tokens: [0, 1, 2])
    let policy = MTPStopPolicy(eosTokenIds: [99], tokenizerEosTokenId: nil, extraEOSTokens: [], convertTokenToId: { _ in nil }, unknownTokenId: nil)
    let result = MTPGenerationCollector.collect(
        next: { iterator.next() },
        maxTokens: 3,
        stopPolicy: policy,
        stopSequences: [],
        decode: fakeDecode
    )
    #expect(joined(result.pieces) == "ABC")
    #expect(result.tokensGenerated == 3)
    #expect(!result.hitStop)
    #expect(result.hitCeiling)
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, hitStop: false, hitCeiling: true) == "length")
}

@Test func collectorStopsOnUserStopSequence() {
    // Decode spells "hello STOP world" one character per token; the user
    // stop string must truncate the visible text and flag hitStop.
    let text = "hello STOP world"
    let tokens = Array(0..<text.count)
    var iterator = FakeIterator(tokens: tokens)
    let policy = MTPStopPolicy(eosTokenIds: [99], tokenizerEosTokenId: nil, extraEOSTokens: [], convertTokenToId: { _ in nil }, unknownTokenId: nil)
    let result = MTPGenerationCollector.collect(
        next: { iterator.next() },
        maxTokens: 16,
        stopPolicy: policy,
        stopSequences: ["STOP"],
        decode: { ids in String(ids.map { Array(text)[$0] }) }
    )
    #expect(joined(result.pieces) == "hello ")
    #expect(result.hitStop)
    #expect(result.tokensGenerated == 10) // consumed through the token that completed "STOP"
}

@Test func collectorFlushesBufferedTailAfterStopToken() {
    // Short visible text with an active stop checker: the StopChecker
    // buffers the tail (potential stop-string prefix) while feeding, so the
    // stop-token break must flush it — otherwise text would be lost.
    var iterator = FakeIterator(tokens: [0, 1, 99])
    let policy = MTPStopPolicy(eosTokenIds: [99], tokenizerEosTokenId: nil, extraEOSTokens: [], convertTokenToId: { _ in nil }, unknownTokenId: nil)
    let result = MTPGenerationCollector.collect(
        next: { iterator.next() },
        maxTokens: 10,
        stopPolicy: policy,
        stopSequences: ["STOP"],
        decode: fakeDecode
    )
    #expect(joined(result.pieces) == "AB")
    #expect(result.tokensGenerated == 2)
    #expect(result.hitStop)
}

@Test func collectorDropsDecodedEOSMarkersFromQwenRepro() {
    // Regression shape of the .33 repro: the model answers, then emits the
    // EOS markers and hallucinated extra turns as PLAIN TOKENS because the
    // loop never checked for them. Here ids 90/91 stand for <|im_end|> /
    // <|endoftext|>; everything after them must vanish from the output.
    var iterator = FakeIterator(tokens: [7, 8, 90, 91, 3, 4])
    let policy = MTPStopPolicy(eosTokenIds: [90, 91], tokenizerEosTokenId: nil, extraEOSTokens: [], convertTokenToId: { _ in nil }, unknownTokenId: nil)
    let result = MTPGenerationCollector.collect(
        next: { iterator.next() },
        maxTokens: 10,
        stopPolicy: policy,
        stopSequences: [],
        decode: fakeDecode
    )
    #expect(joined(result.pieces) == "HI")
    #expect(result.tokensGenerated == 2)
    #expect(result.hitStop)
}

// MARK: - MINEUR 4: ceiling exhaustion with active stopSequences

@Test func collectorReportsLengthWhenCeilingHitWithActiveStopSequences() {
    // Gap the review flagged: exhaustion with a configured stop-string that
    // never occurs must still flush the buffered tail, keep hitStop false,
    // set hitCeiling, and map to "length" (not silently become a stop).
    var iterator = FakeIterator(tokens: [0, 1, 2])
    let policy = MTPStopPolicy(eosTokenIds: [99], tokenizerEosTokenId: nil, extraEOSTokens: [], convertTokenToId: { _ in nil }, unknownTokenId: nil)
    let result = MTPGenerationCollector.collect(
        next: { iterator.next() },
        maxTokens: 3,
        stopPolicy: policy,
        stopSequences: ["STOP"],
        decode: fakeDecode
    )
    #expect(joined(result.pieces) == "ABC") // tail flushed even though StopChecker was armed
    #expect(result.tokensGenerated == 3)
    #expect(!result.hitStop)
    #expect(result.hitCeiling)
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, hitStop: result.hitStop, hitCeiling: result.hitCeiling) == "length")
}

// MARK: - MINEUR 2: iterator death before the ceiling is NOT "length"

@Test func collectorReportsEarlyDeathWhenIteratorDiesBeforeCeiling() {
    // The iterator can return nil before max_tokens (e.g. MTPSpeculativeIterator
    // bails when draft.boundTarget is nil). That must not be reported as a
    // ceiling hit: both flags false, finish_reason falls to "stop" and the
    // call site logs a warning.
    var iterator = FakeIterator(tokens: [0, 1])
    let policy = MTPStopPolicy(eosTokenIds: [99], tokenizerEosTokenId: nil, extraEOSTokens: [], convertTokenToId: { _ in nil }, unknownTokenId: nil)
    let result = MTPGenerationCollector.collect(
        next: { iterator.next() },
        maxTokens: 10,
        stopPolicy: policy,
        stopSequences: [],
        decode: fakeDecode
    )
    #expect(joined(result.pieces) == "AB")
    #expect(result.tokensGenerated == 2)
    #expect(!result.hitStop)
    #expect(!result.hitCeiling)
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, hitStop: false, hitCeiling: false) == "stop")
}

// MARK: - finish_reason mapping (replaces the unconditional "stop")

@Test func finishReasonToolCallsWins() {
    #expect(MTPStopPolicy.finishReason(hasToolCalls: true, hitStop: true, hitCeiling: false) == "tool_calls")
    #expect(MTPStopPolicy.finishReason(hasToolCalls: true, hitStop: false, hitCeiling: true) == "tool_calls")
    #expect(MTPStopPolicy.finishReason(hasToolCalls: true, hitStop: false, hitCeiling: false) == "tool_calls")
}

@Test func finishReasonStopVersusLength() {
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, hitStop: true, hitCeiling: false) == "stop")
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, hitStop: true, hitCeiling: true) == "stop")
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, hitStop: false, hitCeiling: true) == "length")
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, hitStop: false, hitCeiling: false) == "stop")
}

// MARK: - finish_reason mapping, regular (ChatSession) paths

@Test func finishReasonRegularPathMapsStopReason() {
    // tool_calls wins over everything; a met stop-string is a legitimate
    // "stop"; fork stopReason .length maps to "length"; .stop/.cancelled/nil
    // all collapse to "stop" (no "cancelled" in the OpenAI contract).
    #expect(MTPStopPolicy.finishReason(hasToolCalls: true, stoppedEarly: true, stopReason: .stop) == "tool_calls")
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, stoppedEarly: true, stopReason: .length) == "stop")
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, stoppedEarly: false, stopReason: .length) == "length")
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, stoppedEarly: false, stopReason: .stop) == "stop")
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, stoppedEarly: false, stopReason: .cancelled) == "stop")
    #expect(MTPStopPolicy.finishReason(hasToolCalls: false, stoppedEarly: false, stopReason: nil) == "stop")
}

// MARK: - ThinkRepair on the MTP assembled text (volet 2)

@Test func thinkRepairPrependsOpenForOrphanClose() {
    // Qwen-style template pre-opens  in the prompt: the completion
    // carries only the closing tag. repairComplete rebuilds the pair.
    let text = "answer body" + thinkClose + ">rest"
    let repaired = ThinkRepair.repairComplete(text)
    #expect(repaired == thinkOpen + ">" + text)
}

@Test func thinkRepairLeavesWellFormedPairAlone() {
    let text = "\(thinkOpen)>\(thinkClose)>visible answer"
    #expect(ThinkRepair.repairComplete(text) == text)
}

@Test func thinkRepairLeavesTaglessTextAlone() {
    let text = "plain answer, no reasoning at all"
    #expect(ThinkRepair.repairComplete(text) == text)
}

@Test func thinkRepairStreamPrependsWhenNothingEmitted() {
    // Stream mode, orphan close in the FIRST piece: nothing went out yet,
    // the pair can be rebuilt in-band.
    var repair = ThinkRepair()
    let first = repair.feed("reasoning\(thinkClose)>answer")
    #expect(first == "\(thinkOpen)>reasoning\(thinkClose)>answer")
}

@Test func thinkRepairStreamStripsOrphanAfterEmission() {
    // Stream mode, orphan close arriving after visible content went out:
    // the pair can't be rebuilt — strip the orphan tag instead.
    var repair = ThinkRepair()
    _ = repair.feed("visible prefix ")
    let second = repair.feed("\(thinkClose)>answer")
    #expect(second == "answer")
}
