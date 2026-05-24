import Foundation
import Hummingbird

/// `GET /v1/models`, `GET /admin/models/available`, `GET /admin/memory`,
/// `POST /admin/load`, `POST /admin/unload`.
struct ModelsHandler: Sendable {
    let registry: ModelRegistry

    func add(to router: Router<BasicRequestContext>) {
        router.get("/v1/models") { _, _ async throws -> Response in
            try await self.list()
        }
        router.get("/admin/models/available") { _, _ async throws -> Response in
            self.available()
        }
        router.get("/admin/memory") { _, _ async throws -> Response in
            try await self.memory()
        }
        router.post("/admin/load") { request, _ async throws -> Response in
            try await self.load(request)
        }
        router.post("/admin/unload") { request, _ async throws -> Response in
            try await self.unload(request)
        }
    }

    // MARK: - GET /v1/models

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

        let loaded = await registry.loadedModels
        let entries = loaded.map {
            ModelEntry(
                id: $0.id,
                object: "model",
                created: Int($0.loadedAt.timeIntervalSince1970),
                owned_by: "telemak"
            )
        }
        let payload = ModelList(object: "list", data: entries)
        let data = try JSONEncoder().encode(payload)
        return jsonResponse(.ok, data: data)
    }

    // MARK: - GET /admin/models/available

    private func available() -> Response {
        struct Payload: Encodable {
            let models: [AvailableModels.Entry]
        }
        let entries = AvailableModels.scan()
        let payload = Payload(models: entries)
        let data = (try? JSONEncoder().encode(payload)) ?? Data(#"{"models":[]}"#.utf8)
        return jsonResponse(.ok, data: data)
    }

    // MARK: - GET /admin/memory

    private func memory() async throws -> Response {
        struct Payload: Encodable {
            let used_gb: Double
            let free_gb: Double
            let total_gb: Double
            let ceiling_gb: Double
            let per_model: [String: Double]
        }

        let used = await registry.usedRamBytes()
        let perModelBytes = await registry.perModelRamBytes()
        let ceiling = RamBudget.ceilingBytes()
        let total = RamBudget.totalRam()

        let payload = Payload(
            used_gb: bytesToGB(used),
            free_gb: bytesToGB(max(0, ceiling - used)),
            total_gb: bytesToGB(total),
            ceiling_gb: bytesToGB(ceiling),
            per_model: perModelBytes.mapValues { bytesToGB($0) }
        )
        let data = try JSONEncoder().encode(payload)
        return jsonResponse(.ok, data: data)
    }

    // MARK: - POST /admin/load

    private func load(_ request: Request) async throws -> Response {
        struct LoadBody: Decodable { let model: String }
        let buf = try await request.body.collect(upTo: 1 << 16)
        let body: LoadBody
        do {
            body = try JSONDecoder().decode(LoadBody.self, from: Data(buffer: buf))
        } catch {
            return jsonError(.badRequest, code: "invalid_request_error",
                              message: "expected {\"model\": \"<id>\"}: \(error)")
        }
        do {
            _ = try await registry.load(body.model)
        } catch ModelRegistry.LoadError.insufficientMemory(let needed, let available, let ceiling, let currentlyLoaded) {
            return insufficientMemoryError(
                neededBytes: needed,
                availableBytes: available,
                ceilingBytes: ceiling,
                currentlyLoaded: currentlyLoaded
            )
        } catch ModelRegistry.LoadError.loadFailed(let why) {
            return jsonError(.serviceUnavailable, code: "model_load_failed",
                              message: "could not load model '\(body.model)': \(why)")
        } catch {
            return jsonError(.serviceUnavailable, code: "model_load_failed",
                              message: "could not load model '\(body.model)': \(error)")
        }
        let payload: [String: String] = ["status": "loaded", "model": body.model]
        let data = try JSONEncoder().encode(payload)
        return jsonResponse(.ok, data: data)
    }

    // MARK: - POST /admin/unload

    private func unload(_ request: Request) async throws -> Response {
        struct UnloadBody: Decodable { let model: String? }
        let queryAll = request.uri.queryParameters["all"] == "true"

        let buf = try await request.body.collect(upTo: 1 << 16)
        let body: UnloadBody? = (try? JSONDecoder().decode(UnloadBody.self, from: Data(buffer: buf)))

        if queryAll {
            let ids = await registry.unloadAll()
            let payload: [String: Any] = ["status": "unloaded_all", "models": ids]
            let data = try JSONSerialization.data(withJSONObject: payload)
            return jsonResponse(.ok, data: data)
        }

        guard let modelId = body?.model, !modelId.isEmpty else {
            return jsonError(.badRequest, code: "invalid_request_error",
                              message: "specify model id or pass ?all=true")
        }

        let wasLoaded = await registry.unload(modelId)
        let payload: [String: Any] = [
            "status": wasLoaded ? "unloaded" : "noop",
            "model": modelId,
        ]
        let data = try JSONSerialization.data(withJSONObject: payload)
        return jsonResponse(.ok, data: data)
    }

    // MARK: - error helpers

    private func insufficientMemoryError(
        neededBytes: Int64,
        availableBytes: Int64,
        ceilingBytes: Int64,
        currentlyLoaded: [String]
    ) -> Response {
        let payload: [String: Any] = [
            "error": [
                "type": "insufficient_memory",
                "code": "insufficient_memory",
                "message": "loading this model would exceed the wired-memory ceiling. unload something first.",
                "needed_gb": bytesToGB(neededBytes),
                "available_gb": bytesToGB(availableBytes),
                "ceiling_gb": bytesToGB(ceilingBytes),
                "currently_loaded": currentlyLoaded,
            ]
        ]
        let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        return jsonResponse(.badRequest, data: data)
    }

    private func jsonResponse(_ status: HTTPResponse.Status, data: Data) -> Response {
        Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }

    private func jsonError(_ status: HTTPResponse.Status, code: String, message: String) -> Response {
        let payload: [String: [String: String]] = [
            "error": ["type": code, "code": code, "message": message]
        ]
        let data = (try? JSONEncoder().encode(payload)) ?? Data("{}".utf8)
        return jsonResponse(status, data: data)
    }

    private func bytesToGB(_ bytes: Int64) -> Double {
        Double(bytes) / 1_073_741_824.0
    }
}
