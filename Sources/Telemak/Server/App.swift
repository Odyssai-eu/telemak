import Foundation
import Hummingbird
import Logging

/// Build a Hummingbird application configured for Telemak.
func buildApplication(
    registry: ModelRegistry,
    stats: StatsTracker,
    activity: ActivityTracker,
    sessionStore: SessionStore,
    wiredMemory: WiredMemoryCoordinator,
    startTime: Date,
    host: String,
    port: Int,
    logger: Logger
) -> some ApplicationProtocol {
    let router = Router()

    router.add(middleware: CORSMiddleware<BasicRequestContext>())
    router.add(middleware: BearerAuthMiddleware<BasicRequestContext>())
    router.add(middleware: LogRequestsMiddleware(.info))

    HealthHandler(registry: registry, stats: stats, startTime: startTime).add(to: router)
    ActivityHandler(activity: activity).add(to: router)
    ChatCompletionsHandler(registry: registry, stats: stats, activity: activity, sessionStore: sessionStore, wiredMemory: wiredMemory).add(to: router)
    AnthropicMessagesHandler(registry: registry, stats: stats, activity: activity, sessionStore: sessionStore).add(to: router)
    EmbeddingsHandler(registry: registry, activity: activity).add(to: router)
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
