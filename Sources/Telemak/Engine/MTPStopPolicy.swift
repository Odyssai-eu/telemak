import Foundation
import MLXLMCommon

/// Stop-token policy for the MTP speculative path.
///
/// Mirrors the fork's private `buildStopTokenIds` + generation loops
/// (mlx-swift-lm `MLXLMCommon/Evaluate.swift`): the regular `ChatSession`
/// path drives BOTH `runSynchronousGenerationLoop` and its AsyncStream
/// variant (`streamDetails`, same stop semantics — verified identical),
/// and stops as soon as a generated token is an end-of-sequence token,
/// BEFORE the token is emitted. The MTP collecting loop originally had no
/// stop-token handling at all and ran to `max_tokens`, decoding EOS markers
/// (`<|im_end|>`, `<|endoftext|>`, …) into the visible text and
/// hallucinating extra turns past the model's answer.
///
/// The stop set unions every source the fork uses:
/// - `ModelConfiguration.eosTokenIds`
/// - `tokenizer.eosTokenId`
/// - ids of `ModelConfiguration.extraEOSTokens` (resolved via the tokenizer)
/// - `tokenizer.unknownTokenId` (the fork treats it as a stop too)
///
/// Kept as a duplication of the fork's 8-line builder (not a fork API change)
/// because the fork's version is `private`; this file is the Telemak-side
/// parity reference.
public struct MTPStopPolicy: Sendable {
    public let stopTokenIds: Set<Int>

    public init(
        eosTokenIds: Set<Int>,
        tokenizerEosTokenId: Int?,
        extraEOSTokens: Set<String>,
        convertTokenToId: (String) -> Int?,
        unknownTokenId: Int?
    ) {
        var ids = eosTokenIds
        if let tokenizerEosTokenId {
            ids.insert(tokenizerEosTokenId)
        }
        for token in extraEOSTokens {
            if let id = convertTokenToId(token) {
                ids.insert(id)
            }
        }
        if let unknownTokenId {
            ids.insert(unknownTokenId)
        }
        self.stopTokenIds = ids
    }

    /// True when `token` ends generation. The caller must stop WITHOUT
    /// emitting it — same contract as the fork's synchronous loop.
    public func isStopToken(_ token: Int) -> Bool {
        stopTokenIds.contains(token)
    }

    /// OpenAI `finish_reason` for the MTP paths, replacing the previous
    /// unconditional `"stop"` (which lied when generation merely hit the
    /// `max_tokens` ceiling).
    ///
    /// - tool calls recovered → `"tool_calls"` (wins over everything)
    /// - EOS/stop-token or stop-string reached → `"stop"`
    /// - `max_tokens` ceiling reached → `"length"`
    /// - iterator died early (before `max_tokens`, no stop) → `"stop"`:
    ///   the OpenAI contract has no `"cancelled"`, so `"stop"` is the
    ///   least-wrong value; the caller MUST log a warning for diagnosis.
    public static func finishReason(hasToolCalls: Bool, hitStop: Bool, hitCeiling: Bool) -> String {
        if hasToolCalls { return "tool_calls" }
        if hitStop { return "stop" }
        if hitCeiling { return "length" }
        return "stop"
    }

    /// OpenAI `finish_reason` for the REGULAR (ChatSession) paths. Maps the
    /// fork's `GenerateCompletionInfo.stopReason` (`.stop` / `.length` /
    /// `.cancelled`) which those paths already receive and previously ignored
    /// (unconditional `"stop"` lied on truncation).
    ///
    /// - tool calls recovered → `"tool_calls"` (wins over everything)
    /// - a user stop-string was reached → `"stop"` (legitimate stop: OpenAI
    ///   treats a met stop-sequence as a natural stop point)
    /// - fork stopReason `.length` → `"length"`
    /// - `.stop`, `.cancelled` (early stream abort) or nil → `"stop"`: the
    ///   OpenAI contract has no `"cancelled"`, so `"stop"` is the least-wrong
    ///   value for fork-side cancellation.
    public static func finishReason(
        hasToolCalls: Bool,
        stoppedEarly: Bool,
        stopReason: GenerateStopReason?
    ) -> String {
        if hasToolCalls { return "tool_calls" }
        if stoppedEarly { return "stop" }
        if stopReason == .length { return "length" }
        return "stop"
    }
}

/// Pure token-collection loop shared by the MTP streaming and non-streaming
/// responses (and by both iterator flavours, Qwen and Gemma4Assistant).
///
/// Extracted from the two previously duplicated while-loops in
/// `runMTPIteratorCollectingPieces` so the stop semantics live in exactly one
/// place and are unit-testable without Metal:
///
/// 1. A stop TOKEN (per `MTPStopPolicy`) breaks the loop before the token is
///    counted or decoded — EOS markers never reach the visible text.
/// 2. Stop STRINGS from the request ride through `StopChecker` exactly as
///    before: tokens producing the stop string are counted, the string itself
///    is withheld from the visible pieces.
/// 3. `hitStop` / `hitCeiling` report why generation ended. The iterator can
///    also die BEFORE `max_tokens` (e.g. `MTPSpeculativeIterator.runRound`
///    bails when `draft.boundTarget` is nil): then neither flag is set and
///    callers map it to the least-wrong `"stop"` plus a diagnostic warning
///    (`MTPStopPolicy.finishReason`).
public enum MTPGenerationCollector {
    public struct Result: Sendable {
        /// Visible text pieces (post stop-string filtering), in order.
        public let pieces: [String]
        /// Tokens actually emitted — stop tokens excluded.
        public let tokensGenerated: Int
        /// True when generation ended on a stop token or stop string.
        public let hitStop: Bool
        /// True when generation exhausted the `max_tokens` ceiling. False
        /// both on a stop AND on an early iterator death before the ceiling.
        public let hitCeiling: Bool
    }

    /// Drain `next` until a stop token, a stop string, or exhaustion
    /// (`maxTokens`). `decode` receives the full generated token list so
    /// callers can use an incremental tokenizer decode. `maxTokens` mirrors
    /// the iterator's own ceiling so an exhaustion-`nil` maps to `hitCeiling`
    /// instead of being mistaken for an early death (and vice versa).
    public static func collect(
        next: () throws -> Int?,
        maxTokens: Int,
        stopPolicy: MTPStopPolicy,
        stopSequences: [String],
        decode: ([Int]) -> String
    ) rethrows -> Result {
        var generated: [Int] = []
        var stopChecker = StopChecker(stops: stopSequences)
        var pieces: [String] = []
        var lastDecoded = ""
        var hitStopToken = false

        while let token = try next() {
            if stopPolicy.isStopToken(token) {
                hitStopToken = true
                break
            }
            generated.append(token)
            let decoded = decode(generated)
            let delta = String(decoded.dropFirst(lastDecoded.count))
            if !delta.isEmpty {
                let visible = stopChecker.feed(delta)
                if !visible.isEmpty { pieces.append(visible) }
                lastDecoded = decoded
            }
            if stopChecker.hit { break }
        }

        if !stopChecker.hit {
            let tail = stopChecker.flushRemaining()
            if !tail.isEmpty { pieces.append(tail) }
        }

        let hitStop = hitStopToken || stopChecker.hit
        return Result(
            pieces: pieces,
            tokensGenerated: generated.count,
            hitStop: hitStop,
            hitCeiling: !hitStop && generated.count >= maxTokens
        )
    }
}
