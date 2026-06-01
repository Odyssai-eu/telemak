import Foundation

public actor ActivityTracker {
    public enum Phase: String, Codable, Sendable {
        case loading
        case prefill
        case decode
        case streaming
        case idle
    }

    public struct Snapshot: Codable, Sendable {
        public let activeRequests: Int
        public let currentModel: String?
        public let currentRequestStartedAt: String?
        public let currentElapsedS: Double?
        public let currentGeneratedTokens: Int
        public let currentTokS: Double?
        public let currentPhase: Phase
        public let lastError: String?

        enum CodingKeys: String, CodingKey {
            case activeRequests = "active_requests"
            case currentModel = "current_model"
            case currentRequestStartedAt = "current_request_started_at"
            case currentElapsedS = "current_elapsed_s"
            case currentGeneratedTokens = "current_generated_tokens"
            case currentTokS = "current_tok_s"
            case currentPhase = "current_phase"
            case lastError = "last_error"
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(activeRequests, forKey: .activeRequests)
            try container.encode(currentModel, forKey: .currentModel)
            try container.encode(currentRequestStartedAt, forKey: .currentRequestStartedAt)
            try container.encode(currentElapsedS, forKey: .currentElapsedS)
            try container.encode(currentGeneratedTokens, forKey: .currentGeneratedTokens)
            try container.encode(currentTokS, forKey: .currentTokS)
            try container.encode(currentPhase, forKey: .currentPhase)
            try container.encode(lastError, forKey: .lastError)
        }
    }

    private struct RequestState {
        let id: String
        let model: String
        let startedAt: Date
        var generatedTokens: Int
        var phase: Phase
    }

    private var active: [String: RequestState] = [:]
    private var lastError: String?
    private let dateFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    public init() {}

    public func begin(model: String, phase: Phase = .prefill) -> String {
        let id = UUID().uuidString.lowercased()
        active[id] = RequestState(
            id: id,
            model: model,
            startedAt: Date(),
            generatedTokens: 0,
            phase: phase
        )
        return id
    }

    public func setPhase(_ id: String, _ phase: Phase) {
        guard var state = active[id] else { return }
        state.phase = phase
        active[id] = state
    }

    public func incrementGeneratedTokens(_ id: String, by count: Int = 1) {
        guard count > 0, var state = active[id] else { return }
        state.generatedTokens += count
        active[id] = state
    }

    public func setGeneratedTokens(_ id: String, _ tokens: Int) {
        guard var state = active[id] else { return }
        state.generatedTokens = max(0, tokens)
        active[id] = state
    }

    public func finish(_ id: String) {
        active[id] = nil
    }

    public func fail(_ id: String?, error: String) {
        lastError = error
        if let id {
            active[id] = nil
        }
    }

    public func snapshot() -> Snapshot {
        let current = active.values.sorted { lhs, rhs in
            lhs.startedAt > rhs.startedAt
        }.first
        let elapsed = current.map { max(0, Date().timeIntervalSince($0.startedAt)) } ?? 0
        let tokS: Double?
        if let current, current.generatedTokens > 0, elapsed > 0 {
            tokS = Double(current.generatedTokens) / elapsed
        } else {
            tokS = nil
        }

        return Snapshot(
            activeRequests: active.count,
            currentModel: current?.model,
            currentRequestStartedAt: current.map { dateFormatter.string(from: $0.startedAt) },
            currentElapsedS: current == nil ? nil : elapsed,
            currentGeneratedTokens: current?.generatedTokens ?? 0,
            currentTokS: tokS,
            currentPhase: current?.phase ?? .idle,
            lastError: lastError
        )
    }
}
