import Foundation
import Hummingbird
import Logging

/// Build a Hummingbird application configured for Telemak.
///
/// `registry` and `stats` are shared across handlers so requests can both
/// serialize on model state and feed the `/health` snapshot.
func buildApplication(
    registry: ModelRegistry,
    stats: StatsTracker,
    startTime: Date,
    host: String,
    port: Int,
    logger: Logger
) -> some ApplicationProtocol {
    let router = Router()

    router.add(middleware: LogRequestsMiddleware(.info))

    HealthHandler(registry: registry, stats: stats, startTime: startTime).add(to: router)
    ChatCompletionsHandler(registry: registry, stats: stats).add(to: router)
    ModelsHandler(registry: registry).add(to: router)

    let app = Application(
        router: router,
        configuration: .init(
            address: .hostname(host, port: port),
            serverName: "telemak"
        ),
        logger: logger
    )
    return app
}
