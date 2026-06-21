import ArgumentParser
import Foundation
import Logging
import TelemakVersion
#if canImport(Darwin)
import Darwin
#endif

struct Serve: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "serve",
        abstract: "Run the HTTP server on the configured port."
    )

    @Option(name: .shortAndLong, help: "Port to bind to (default 8002).")
    var port: Int = 8002

    @Option(name: .long, help: "Host to bind to (default 0.0.0.0).")
    var host: String = "0.0.0.0"

    @Option(name: .long, help: "Override HF_HUB_CACHE for model downloads/lookups.")
    var hfHubCache: String?

    @Option(name: .long, help: "Log level (trace, debug, info, notice, warning, error, critical).")
    var logLevel: String = "info"

    @Flag(name: .long, help: "Skip replaying ~/.telemak/state.json at boot.")
    var noReplay: Bool = false

    func run() async throws {
        if let hfHubCache {
            setenv("HF_HUB_CACHE", hfHubCache, 1)
        }

        // Bootstrap structured JSON logging before any Logger is created
        // anywhere. Reads TELEMAK_LOG_LEVEL from the env; falls back to the
        // --log-level flag if set, or `info` otherwise.
        if !logLevel.isEmpty, ProcessInfo.processInfo.environment["TELEMAK_LOG_LEVEL"] == nil {
            setenv("TELEMAK_LOG_LEVEL", logLevel, 1)
        }
        LogConfig.bootstrap()
        let logger = Logger(label: "telemak")

        let startTime = Date()
        let stateStore = StateStore()
        let sessionStore = SessionStore()
        let wiredMemory = WiredMemoryCoordinator()
        let registry = ModelRegistry(
            stateStore: stateStore,
            sessionStore: sessionStore,
            wiredMemory: wiredMemory
        )
        let stats = StatsTracker()
        let activity = ActivityTracker()

        if !noReplay {
            await replayState(registry: registry, stateStore: stateStore, logger: logger)
        }

        let app = buildApplication(
            registry: registry,
            stats: stats,
            activity: activity,
            sessionStore: sessionStore,
            wiredMemory: wiredMemory,
            startTime: startTime,
            host: host,
            port: port,
            logger: logger
        )
        logger.info("telemak listening on http://\(host):\(port)")
        try await app.runService()
    }

    private func replayState(registry: ModelRegistry, stateStore: StateStore, logger: Logger) async {
        // A schema-incompatible (pre-upgrade) state.json fails to decode →
        // read() returns nil → we start empty. That is the intended one-time
        // reset; the master repopulates loads afterwards.
        guard let state = await stateStore.read(), !state.loadedModels.isEmpty else {
            logger.info("no persisted state — starting empty")
            return
        }
        // No models directory configured → nothing will resolve; don't even try
        // (the master will set the dir, then loads come back via the API).
        guard ModelsConfig.shared.effectiveDir() != nil else {
            logger.notice("no models directory configured — skipping replay of \(state.loadedModels.count) model(s); set it via OdyssAI-X or the menubar")
            return
        }
        logger.info("replaying persisted state: \(state.loadedModels)")
        for id in state.loadedModels {
            // Per-model: if the model isn't under the current effective dir
            // (e.g. the dir changed under us), the load below fails and is logged
            // — best-effort, non-fatal. No recordedDir bookkeeping needed.
            do {
                _ = try await registry.load(id)
                logger.info("replayed model: \(id)")
            } catch {
                logger.warning("failed to replay model \(id): \(error) — continuing")
            }
        }
    }
}
