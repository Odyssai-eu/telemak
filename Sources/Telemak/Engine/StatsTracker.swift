import Foundation

/// Tracks aggregate inference stats for `/health` (and later `/admin/metrics`).
///
/// Phase 5: tracks total requests served and a rolling window of recent
/// tok/s measurements. Steady-state tok/s for the last 32 requests is the
/// `avg_tok_s_recent` exposed in `/health`.
public actor StatsTracker {
    private var requestsServed: Int = 0
    private var recentTokensPerSecond: [Double] = []
    private let windowSize: Int

    public init(windowSize: Int = 32) {
        self.windowSize = windowSize
    }

    public func recordRequest(tokens: Int, elapsedSeconds: Double) {
        requestsServed += 1
        guard elapsedSeconds > 0, tokens > 0 else { return }
        let tokPerSec = Double(tokens) / elapsedSeconds
        recentTokensPerSecond.append(tokPerSec)
        if recentTokensPerSecond.count > windowSize {
            recentTokensPerSecond.removeFirst(recentTokensPerSecond.count - windowSize)
        }
    }

    public func snapshot() -> (requests: Int, avgTokPerSec: Double?) {
        let avg: Double?
        if recentTokensPerSecond.isEmpty {
            avg = nil
        } else {
            avg = recentTokensPerSecond.reduce(0, +) / Double(recentTokensPerSecond.count)
        }
        return (requestsServed, avg)
    }
}
