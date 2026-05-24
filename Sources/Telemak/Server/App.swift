import Foundation
import Hummingbird
import Logging

/// Build a Hummingbird application configured for Telemak.
func buildApplication(
    registry: ModelRegistry,
    stats: StatsTracker,
    sessionStore: SessionStore,
    startTime: Date,
    host: String,
    port: Int,
    logger: Logger
) -> some ApplicationProtocol {
    let router = Router()

    router.add(middleware: LogRequestsMiddleware(.info))

    HealthHandler(registry: registry, stats: stats, startTime: startTime).add(to: router)
    ChatCompletionsHandler(registry: registry, stats: stats, sessionStore: sessionStore).add(to: router)
    AnthropicMessagesHandler(registry: registry, stats: stats, sessionStore: sessionStore).add(to: router)
    ModelsHandler(registry: registry).add(to: router)
    SessionsHandler(sessionStore: sessionStore).add(to: router)
    CapabilitiesHandler(registry: registry).add(to: router)

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
