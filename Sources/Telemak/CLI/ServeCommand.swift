import ArgumentParser
import Foundation
import Logging
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

    @Option(name: .long, help: "Host to bind to (default 127.0.0.1).")
    var host: String = "127.0.0.1"

    @Option(name: .long, help: "Override HF_HUB_CACHE for model downloads/lookups.")
    var hfHubCache: String?

    @Option(name: .long, help: "Log level (trace, debug, info, notice, warning, error, critical).")
    var logLevel: String = "info"

    func run() async throws {
        if let hfHubCache {
            setenv("HF_HUB_CACHE", hfHubCache, 1)
        }

        var logger = Logger(label: "telemak")
        if let level = Logger.Level(rawValue: logLevel) {
            logger.logLevel = level
        }

        let registry = ModelRegistry()
        let app = buildApplication(registry: registry, host: host, port: port, logger: logger)
        logger.info("telemak listening on http://\(host):\(port)")
        try await app.runService()
    }
}
