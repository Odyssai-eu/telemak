import Foundation
import Hummingbird

/// `GET /admin/sessions` + `POST /admin/sessions/clear`.
struct SessionsHandler: Sendable {
    let sessionStore: SessionStore

    func add(to router: Router<BasicRequestContext>) {
        router.get("/admin/sessions") { _, _ async throws -> Response in
            try await self.list()
        }
        router.post("/admin/sessions/clear") { request, _ async throws -> Response in
            try await self.clear(request)
        }
    }

    private func list() async throws -> Response {
        struct Payload: Encodable {
            let sessions: [SessionStore.Snapshot]
        }
        let snapshots = await sessionStore.snapshots()
        let data = try JSONEncoder().encode(Payload(sessions: snapshots))
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }

    private func clear(_ request: Request) async throws -> Response {
        let params = request.uri.queryParameters
        let queryAll = params["all"] == "true"
        let queryId: String? = params["session_id"].map(String.init)

        if queryAll {
            let cleared = await sessionStore.clearAll()
            let payload: [String: Any] = ["status": "cleared_all", "sessions": cleared]
            let data = try JSONSerialization.data(withJSONObject: payload)
            return jsonResponse(.ok, data: data)
        }

        guard let sessionId = queryId, !sessionId.isEmpty else {
            return jsonError(.badRequest, message: "specify ?session_id=<id> or ?all=true")
        }

        let wasPresent = await sessionStore.clear(sessionId: sessionId)
        let payload: [String: Any] = [
            "status": wasPresent ? "cleared" : "noop",
            "session_id": sessionId,
        ]
        let data = try JSONSerialization.data(withJSONObject: payload)
        return jsonResponse(.ok, data: data)
    }

    private func jsonResponse(_ status: HTTPResponse.Status, data: Data) -> Response {
        Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }

    private func jsonError(_ status: HTTPResponse.Status, message: String) -> Response {
        let payload: [String: [String: String]] = ["error": ["message": message]]
        let data = (try? JSONEncoder().encode(payload)) ?? Data("{}".utf8)
        return jsonResponse(status, data: data)
    }
}
