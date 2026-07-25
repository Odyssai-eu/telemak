import Foundation

/// Repairs reasoning-tag well-formedness for models whose chat template
/// PRE-OPENS the think block inside the prompt (poolside Laguna: the rendered
/// prompt ends with `<assistant><think>`; GLM-4.x has the same idiom).
///
/// Telemak's contract is "reasoning inline, downstream parses the
/// `<think>...</think>` pair". Qwen-style models emit the opening tag
/// themselves, so the pair is well-formed. Pre-opened templates leave the
/// opening tag in the PROMPT: the generated stream then carries an ORPHAN
/// `</think>` that downstream parsers can't match — the tag leaks into the
/// visible text (observed on Laguna-XS via /v1/chat/completions, 2026-07-25).
///
/// Repair rules (activation is content-driven, so models that emit their own
/// `<think>` are never touched):
/// - Non-stream: content contains `</think>` with no `<think>` before its
///   first occurrence → prepend `<think>` — downstream sees a well-formed pair.
/// - Stream: once text has been sent we can't retro-prepend. If nothing has
///   been emitted yet, prepend to the first chunk; otherwise strip the orphan
///   tag so at least no litter reaches clients (asymmetry documented).
/// - A model that never closes its think block (answers entirely inside it —
///   Laguna does this on short code prompts) is left untouched: no orphan
///   close, and prepending would flag the whole answer as reasoning.
///
/// Limitation: stream-side detection assumes `</think>` arrives within one
/// decoded piece. True for Laguna (single token, id 24-adjacent vocab entry
/// `</think>` id 19) and every currently served model; a model that splits
/// the closing tag across pieces would fall back to today's behaviour.
public struct ThinkRepair: Sendable {
    private var emittedAnything = false
    private var sawOpen = false
    private var done = false

    public init() {}

    /// Streaming: transform one about-to-be-sent piece.
    public mutating func feed(_ piece: String) -> String {
        if done { return piece }
        if piece.contains("<think>") {
            sawOpen = true
            done = true
            return piece
        }
        if let range = piece.range(of: "</think>"), !sawOpen {
            done = true
            if !emittedAnything {
                // Whole reasoning block arrived before the first flush —
                // rebuild the pair in-band.
                return "<think>" + piece
            }
            // Content already went out; the pair can't be rebuilt. Strip the
            // orphan tag so it doesn't litter the visible text.
            var repaired = piece
            repaired.removeSubrange(range)
            if repaired.isEmpty { repaired = "\n" }
            return repaired
        }
        if !piece.isEmpty { emittedAnything = true }
        return piece
    }

    /// Non-stream: repair a fully assembled completion.
    public static func repairComplete(_ text: String) -> String {
        guard let close = text.range(of: "</think>") else { return text }
        if let open = text.range(of: "<think>"),
            open.lowerBound < close.lowerBound
        {
            return text
        }
        return "<think>" + text
    }
}
