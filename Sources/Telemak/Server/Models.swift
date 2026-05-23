import Foundation
import Hummingbird

/// GET /v1/models  — OpenAI listing of currently-loaded models.
/// POST /admin/load  — load a model
/// POST /admin/unload  — unload current model
struct ModelsHandler: Sendable {
    let registry: ModelRegistry

    func add(to router: Router<BasicRequestContext>) {
        router.get("/v1/models") { _, _ async throws -> Response in
            try await self.list()
        }
        router.post("/admin/load") { request, _ async throws -> Response in
            try await self.load(request)
        }
        router.post("/admin/unload") { request, _ async throws -> Response in
            try await self.unload(request)
        }
    }

    private func list() async throws -> Response {
        struct ModelEntry: Encodable {
            let id: String
            let object: String
            let created: Int
            let owned_by: String
        }
        struct ModelList: Encodable {
            let object: String
            let data: [ModelEntry]
        }

        let entries: [ModelEntry]
        if let loaded = await registry.loadedModel {
            entries = [ModelEntry(
                id: loaded.id,
                object: "model",
                created: Int(loaded.loadedAt.timeIntervalSince1970),
                owned_by: "telemak"
            )]
        } else {
            entries = []
        }

        let payload = ModelList(object: "list", data: entries)
        let data = try JSONEncoder().encode(payload)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }

    private func load(_ request: Request) async throws -> Response {
        struct LoadBody: Decodable { let model: String }
        let buf = try await request.body.collect(upTo: 1 << 16)
        let body: LoadBody
        do {
            body = try JSONDecoder().decode(LoadBody.self, from: Data(buffer: buf))
        } catch {
            return jsonError(.badRequest, message: "expected {\"model\": \"<id>\"}: \(error)")
        }
        do {
            _ = try await registry.ensureLoaded(body.model)
        } catch {
            return jsonError(.serviceUnavailable, message: "load failed: \(error)")
        }
        let payload: [String: String] = ["status": "loaded", "model": body.model]
        let data = try JSONEncoder().encode(payload)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }

    private func unload(_ request: Request) async throws -> Response {
        let unloaded = await registry.unload()
        let payload: [String: String] = [
            "status": "unloaded",
            "model": unloaded ?? "",
        ]
        let data = try JSONEncoder().encode(payload)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }

    private func jsonError(_ status: HTTPResponse.Status, message: String) -> Response {
        let payload: [String: [String: String]] = [
            "error": ["message": message]
        ]
        let data = (try? JSONEncoder().encode(payload)) ?? Data("{}".utf8)
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }
}
