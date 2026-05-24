import Foundation

/// Streaming stop-sequence detector.
///
/// Token-by-token output may include any of the configured stop strings; once
/// a stop string is emitted, generation must end and the trailing bytes
/// (including the stop) must not appear in the response.
///
/// `feed(chunk)` returns the slice that's safe to emit immediately — bytes
/// that cannot possibly be the start of a stop sequence. The remainder is
/// buffered internally. When a stop sequence is detected, `hit` flips to
/// `true` and `safePrefixBeforeStop` returns the text up to (but not
/// including) the stop sequence start.
///
/// On end-of-generation without a stop hit, `flushRemaining()` returns the
/// buffered tail.
public struct StopChecker: Sendable {

    public let stops: [String]
    public private(set) var hit: Bool = false
    public private(set) var safePrefixBeforeStop: String = ""

    private var buffer: String = ""
    private let maxStopLength: Int

    public init(stops: [String]) {
        self.stops = stops
        self.maxStopLength = stops.map(\.count).max() ?? 0
    }

    /// Append `chunk` to the buffer, return the prefix that is safe to emit
    /// (bytes that can't be part of a stop sequence). If a stop string is
    /// detected, `hit` becomes true and subsequent calls return "".
    public mutating func feed(_ chunk: String) -> String {
        if hit { return "" }
        if stops.isEmpty {
            return chunk
        }
        buffer += chunk

        // Stop hit? Find the earliest position of any stop string.
        var earliest: String.Index?
        for stop in stops {
            if let r = buffer.range(of: stop), earliest == nil || r.lowerBound < earliest! {
                earliest = r.lowerBound
            }
        }
        if let earliest {
            let prefix = String(buffer[..<earliest])
            safePrefixBeforeStop += prefix
            hit = true
            buffer = ""
            return prefix
        }

        // No hit yet. Keep the trailing `maxStopLength - 1` bytes in the
        // buffer (they may extend into a stop on the next call), emit the
        // rest. Operates on Character count to avoid splitting a multi-byte
        // codepoint mid-stop.
        let keepCount = max(0, maxStopLength - 1)
        if buffer.count <= keepCount {
            return ""
        }
        let emitEnd = buffer.index(buffer.endIndex, offsetBy: -keepCount)
        let emit = String(buffer[..<emitEnd])
        buffer = String(buffer[emitEnd...])
        return emit
    }

    /// Return any bytes still buffered. Called when generation ends without
    /// a stop sequence having been hit.
    public mutating func flushRemaining() -> String {
        if hit { return "" }
        defer { buffer = "" }
        return buffer
    }
}
