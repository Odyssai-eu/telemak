import Foundation
import Hummingbird
import Logging

/// Build a Hummingbird application configured for Telemak.
///
/// `registry` is shared across handlers so requests serialize on the same
/// loaded-model state.
func buildApplication(
    registry: ModelRegistry,
    host: String,
    port: Int,
    logger: Logger
) -> some ApplicationProtocol {
    let router = Router()

    router.add(middleware: LogRequestsMiddleware(.info))

    router.get("/health") { _, _ in
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(string: #"{"status":"ok"}"#))
        )
    }

    ChatCompletionsHandler(registry: registry).add(to: router)
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
