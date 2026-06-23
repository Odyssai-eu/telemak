import Foundation

public actor ActivityTracker {
    public enum Phase: String, Codable, Sendable {
        case loading
        case prefill
        case decode
        case streaming
        case idle
    }

    /// Tracks what kind of operation `begin/finish` is wrapping. Requests
    /// go into `recentRequests`; loads go into `recentLoads`. Both keep the
    /// same `begin → setPhase → finish` lifecycle.
    public enum Kind: Sendable {
        case request
        case load
    }

    public struct RecentRequest: Codable, Sendable {
        public let model: String
        public let startedAt: String
        public let elapsedS: Double
        public let generatedTokens: Int
        public let tokS: Double?
        public let phase: Phase
        public let error: String?

        enum CodingKeys: String, CodingKey {
            case model
            case startedAt = "started_at"
            case elapsedS = "elapsed_s"
            case generatedTokens = "generated_tokens"
            case tokS = "tok_s"
            case phase
            case error
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(model, forKey: .model)
            try container.encode(startedAt, forKey: .startedAt)
            try container.encode(elapsedS, forKey: .elapsedS)
            try container.encode(generatedTokens, forKey: .generatedTokens)
            try container.encode(tokS, forKey: .tokS)
            try container.encode(phase.rawValue, forKey: .phase)
            try container.encode(error, forKey: .error)
        }
    }

    public struct RecentLoad: Codable, Sendable {
        public let model: String
        public let startedAt: String
        public let elapsedS: Double
        public let success: Bool
        public let error: String?

        enum CodingKeys: String, CodingKey {
            case model
            case startedAt = "started_at"
            case elapsedS = "elapsed_s"
            case success
            case error
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(model, forKey: .model)
            try container.encode(startedAt, forKey: .startedAt)
            try container.encode(elapsedS, forKey: .elapsedS)
            try container.encode(success, forKey: .success)
            try container.encode(error, forKey: .error)
        }
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
        /// Most-recent-first, bounded by `historyLimit` (default 32).
        /// Each entry is one finished or failed inference request —
        /// chat completion, embedding, MTP smoke, etc.
        public let recentRequests: [RecentRequest]
        /// Most-recent-first, bounded by `historyLimit` (default 32).
        /// Each entry is one finished or failed `/admin/load` (or
        /// embedder / draft pair load).
        public let recentLoads: [RecentLoad]

        enum CodingKeys: String, CodingKey {
            case activeRequests = "active_requests"
            case currentModel = "current_model"
            case currentRequestStartedAt = "current_request_started_at"
            case currentElapsedS = "current_elapsed_s"
            case currentGeneratedTokens = "current_generated_tokens"
            case currentTokS = "current_tok_s"
            case currentPhase = "current_phase"
            case lastError = "last_error"
            case recentRequests = "recent_requests"
            case recentLoads = "recent_loads"
        }

        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(activeRequests, forKey: .activeRequests)
            try container.encode(currentModel, forKey: .currentModel)
            try container.encode(currentRequestStartedAt, forKey: .currentRequestStartedAt)
            try container.encode(currentElapsedS, forKey: .currentElapsedS)
            try container.encode(currentGeneratedTokens, forKey: .currentGeneratedTokens)
            try container.encode(currentTokS, forKey: .currentTokS)
            try container.encode(currentPhase.rawValue, forKey: .currentPhase)
            try container.encode(lastError, forKey: .lastError)
            try container.encode(recentRequests, forKey: .recentRequests)
            try container.encode(recentLoads, forKey: .recentLoads)
        }
    }

    private struct RequestState {
        let id: String
        let model: String
        let startedAt: Date
        var generatedTokens: Int
        var phase: Phase
        let kind: Kind
    }

    private var active: [String: RequestState] = [:]
    private var lastError: String?
    private var recentRequests: [RecentRequest] = []
    private var recentLoads: [RecentLoad] = []
    private let historyLimit: Int
    private let dateFormatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    public init(historyLimit: Int = 32) {
        self.historyLimit = historyLimit
    }

    /// Start a request-lifetime entry (chat completion, embedding, etc.).
    /// Lands in `recentRequests` on `finish` / `fail`.
    public func begin(model: String, phase: Phase = .prefill) -> String {
        begin(model: model, kind: .request, phase: phase)
    }

    /// Start a load-lifetime entry (chat model, embedder, MTP draft).
    /// Lands in `recentLoads` on `finish` / `fail`.
    public func beginLoad(model: String) -> String {
        begin(model: model, kind: .load, phase: .loading)
    }

    private func begin(model: String, kind: Kind, phase: Phase) -> String {
        let id = UUID().uuidString.lowercased()
        active[id] = RequestState(
            id: id,
            model: model,
            startedAt: Date(),
            generatedTokens: 0,
            phase: phase,
            kind: kind
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

    /// Mark `id` complete. Pushes into the right bounded history
    /// (`recentRequests` or `recentLoads`) based on the entry's `kind`.
    public func finish(_ id: String, error: String? = nil) {
        guard let state = active.removeValue(forKey: id) else { return }
        let elapsed = max(0, Date().timeIntervalSince(state.startedAt))
        let startedAt = dateFormatter.string(from: state.startedAt)

        switch state.kind {
        case .request:
            let tokS: Double? = (state.generatedTokens > 0 && elapsed > 0)
                ? Double(state.generatedTokens) / elapsed
                : nil
            appendRecentRequest(RecentRequest(
                model: state.model,
                startedAt: startedAt,
                elapsedS: elapsed,
                generatedTokens: state.generatedTokens,
                tokS: tokS,
                phase: state.phase,
                error: error
            ))
        case .load:
            appendRecentLoad(RecentLoad(
                model: state.model,
                startedAt: startedAt,
                elapsedS: elapsed,
                success: error == nil,
                error: error
            ))
        }
    }

    /// Mark `id` failed (or just record a top-level error if `id` is nil).
    /// Always updates `lastError`. If `id` is provided, also pushes the
    /// entry into its kind-appropriate history with the error attached.
    public func fail(_ id: String?, error: String) {
        lastError = error
        if let id {
            finish(id, error: error)
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

        // Append order is oldest-first; the dashboard wants newest-first.
        let requestsOut = Array(recentRequests.reversed())
        let loadsOut = Array(recentLoads.reversed())

        return Snapshot(
            activeRequests: active.count,
            currentModel: current?.model,
            currentRequestStartedAt: current.map { dateFormatter.string(from: $0.startedAt) },
            currentElapsedS: current == nil ? nil : elapsed,
            currentGeneratedTokens: current?.generatedTokens ?? 0,
            currentTokS: tokS,
            currentPhase: current?.phase ?? .idle,
            lastError: lastError,
            recentRequests: requestsOut,
            recentLoads: loadsOut
        )
    }

    private func appendRecentRequest(_ entry: RecentRequest) {
        recentRequests.append(entry)
        if recentRequests.count > historyLimit {
            recentRequests.removeFirst(recentRequests.count - historyLimit)
        }
    }

    private func appendRecentLoad(_ entry: RecentLoad) {
        recentLoads.append(entry)
        if recentLoads.count > historyLimit {
            recentLoads.removeFirst(recentLoads.count - historyLimit)
        }
    }
}
