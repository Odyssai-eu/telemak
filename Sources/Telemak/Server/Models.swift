import Foundation
import Hummingbird
import MLXLLM
import MLXLMCommon
import TelemakMTP

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
        router.post("/admin/models/split-mtp") { request, _ async throws -> Response in
            try await self.splitMTP(request)
        }
        router.post("/admin/mtp/smoke") { request, _ async throws -> Response in
            try await self.mtpSmoke(request)
        }
    }

    // MARK: - GET /v1/models

    private func list() async throws -> Response {
        struct ModelEntry: Encodable {
            let id: String
            let object: String
            let created: Int
            let owned_by: String
            let xTelemak: ModelExt

            enum CodingKeys: String, CodingKey {
                case id
                case object
                case created
                case owned_by
                case xTelemak = "x_telemak"
            }
        }
        struct ModelExt: Encodable {
            let kind: String
            let mtp: MTPCompatibility
        }
        struct ModelList: Encodable {
            let object: String
            let data: [ModelEntry]
        }

        let loaded = await registry.loadedModels
        let embedders = await registry.loadedEmbedders
        let llmEntries = loaded.map {
            ModelEntry(
                id: $0.id,
                object: "model",
                created: Int($0.loadedAt.timeIntervalSince1970),
                owned_by: "telemak",
                xTelemak: .init(kind: $0.isVision ? "vlm" : "llm", mtp: $0.mtpCompatibility)
            )
        }
        let embedderEntries = embedders.map {
            ModelEntry(
                id: $0.id,
                object: "model",
                created: Int($0.loadedAt.timeIntervalSince1970),
                owned_by: "telemak",
                xTelemak: .init(
                    kind: "embedder",
                    mtp: MTPCompatibility(
                        status: .noMTP,
                        reason: "embedders do not support MTP decoding",
                        source: $0.id,
                        overrideRequired: false,
                        autoEnabled: false
                    )
                )
            )
        }
        let entries = (llmEntries + embedderEntries).sorted { $0.id < $1.id }
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
        struct LoadBody: Decodable {
            let model: String
            let draft_model: String?
            let allow_unverified_mtp: Bool?
        }
        let buf = try await request.body.collect(upTo: 1 << 16)
        let body: LoadBody
        do {
            body = try JSONDecoder().decode(LoadBody.self, from: Data(buffer: buf))
        } catch {
            return jsonError(.badRequest, code: "invalid_request_error",
                              message: "expected {\"model\": \"<id>\", \"draft_model\": \"<id?>\"}: \(error)")
        }
        let isEmbedder = EmbedderLoader.isEmbedder(identifier: body.model)
        if isEmbedder {
            if let draftId = body.draft_model, !draftId.isEmpty {
                return jsonError(.badRequest, code: "invalid_request_error",
                                  message: "draft_model is only valid for chat models")
            }
            do {
                _ = try await registry.loadEmbedder(body.model)
            } catch ModelRegistry.LoadError.insufficientMemory(let needed, let available, let ceiling, let currentlyLoaded) {
                return insufficientMemoryError(
                    neededBytes: needed,
                    availableBytes: available,
                    ceilingBytes: ceiling,
                    currentlyLoaded: currentlyLoaded
                )
            } catch ModelRegistry.LoadError.loadFailed(let why) {
                return jsonError(.serviceUnavailable, code: "model_load_failed",
                                  message: "could not load embedder '\(body.model)': \(why)")
            } catch {
                return jsonError(.serviceUnavailable, code: "model_load_failed",
                                  message: "could not load embedder '\(body.model)': \(error)")
            }
            let payload = ["status": "loaded", "model": body.model, "kind": "embedder"]
            let data = try JSONEncoder().encode(payload)
            return jsonResponse(.ok, data: data)
        } else {
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
        }
        // Optional draft pairing for MTP speculative decoding (V2).
        if let draftId = body.draft_model, !draftId.isEmpty {
            do {
                _ = try await registry.loadDraft(
                    draftId,
                    pairedWith: body.model,
                    allowUnverified: body.allow_unverified_mtp ?? false
                )
            } catch ModelRegistry.LoadError.insufficientMemory(let needed, let available, let ceiling, let currentlyLoaded) {
                return insufficientMemoryError(
                    neededBytes: needed,
                    availableBytes: available,
                    ceilingBytes: ceiling,
                    currentlyLoaded: currentlyLoaded
                )
            } catch ModelRegistry.LoadError.loadFailed(let why) {
                return jsonError(.serviceUnavailable, code: "draft_load_failed",
                                  message: "main loaded but draft '\(draftId)' failed: \(why)")
            } catch {
                return jsonError(.serviceUnavailable, code: "draft_load_failed",
                                  message: "main loaded but draft '\(draftId)' failed: \(error)")
            }
        }
        var payload: [String: String] = ["status": "loaded", "model": body.model]
        if let draftId = body.draft_model, !draftId.isEmpty {
            payload["draft_model"] = draftId
        }
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

    // MARK: - POST /admin/models/split-mtp

    /// Wraps Blaizzy's `mlx_vlm.speculative.drafters.qwen3_5_mtp.split`
    /// CLI as an admin endpoint. Lets a Telemak operator pop an MTP
    /// drafter out of any Qwen3.5 / Qwen3.6 source from the dashboard
    /// instead of hand-running Python on the host.
    ///
    /// Body : `{"source": "<hf-id or path>", "output": "<local path>"}`.
    /// Prereq on the host : `pip install mlx-vlm` — the endpoint
    /// surfaces a clear 503 if the Python module isn't importable.
    private func splitMTP(_ request: Request) async throws -> Response {
        struct SplitBody: Decodable {
            let source: String
            let output: String?
        }
        let buf = try await request.body.collect(upTo: 1 << 16)
        let body: SplitBody
        do {
            body = try JSONDecoder().decode(SplitBody.self, from: Data(buffer: buf))
        } catch {
            return jsonError(
                .badRequest, code: "invalid_request_error",
                message: "expected {\"source\": \"<id>\", \"output\": \"<path?>\"}: \(error)")
        }
        let pythonBin = ProcessInfo.processInfo.environment["TELEMAK_PYTHON"]
            ?? "/usr/bin/env"
        let module = "mlx_vlm.speculative.drafters.qwen3_5_mtp.split"
        var args: [String]
        if pythonBin.hasSuffix("env") {
            args = ["python3", "-m", module, body.source]
        } else {
            args = ["-m", module, body.source]
        }
        if let out = body.output, !out.isEmpty {
            args.append(contentsOf: ["--output", out])
        }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: pythonBin)
        process.arguments = args
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        do {
            try process.run()
        } catch {
            return jsonError(
                .serviceUnavailable, code: "split_failed",
                message:
                    "could not spawn '\(pythonBin) \(args.joined(separator: " "))': \(error). "
                    + "Install mlx-vlm on the host (`pip install mlx-vlm`) or point "
                    + "TELEMAK_PYTHON at a venv that has it.")
        }
        process.waitUntilExit()
        let stdoutData = outPipe.fileHandleForReading.readDataToEndOfFile()
        let stderrData = errPipe.fileHandleForReading.readDataToEndOfFile()
        let stdout = String(data: stdoutData, encoding: .utf8) ?? ""
        let stderr = String(data: stderrData, encoding: .utf8) ?? ""
        if process.terminationStatus != 0 {
            let payload: [String: Any] = [
                "error": [
                    "type": "split_failed",
                    "code": "split_failed",
                    "exit_code": Int(process.terminationStatus),
                    "stdout": stdout,
                    "stderr": stderr,
                    "command": ([pythonBin] + args).joined(separator: " "),
                ]
            ]
            let data =
                (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
            return jsonResponse(.serviceUnavailable, data: data)
        }
        let payload: [String: Any] = [
            "status": "ok",
            "source": body.source,
            "output": body.output ?? "(default)",
            "stdout": stdout,
            "stderr": stderr,
        ]
        let data =
            (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
        return jsonResponse(.ok, data: data)
    }

    // MARK: - POST /admin/mtp/smoke

    /// Drive the MTP speculative iterator directly, bypassing
    /// /v1/chat/completions. Returns generated text + acceptance stats
    /// + timing. Used to validate the spec-decoding math without going
    /// through the full chat pipeline.
    private func mtpSmoke(_ request: Request) async throws -> Response {
        struct SmokeBody: Decodable {
            let model: String
            let prompt: String
            let max_tokens: Int?
            let temperature: Float?
            let top_p: Float?
            let top_k: Int?
        }
        let buf = try await request.body.collect(upTo: 1 << 16)
        let body: SmokeBody
        do {
            body = try JSONDecoder().decode(SmokeBody.self, from: Data(buffer: buf))
        } catch {
            return jsonError(
                .badRequest, code: "invalid_request_error",
                message: "expected {\"model\":\"<id>\",\"prompt\":\"<text>\",\"max_tokens\":N}: \(error)")
        }
        let mainId = body.model
        guard let main = await registry.get(mainId) else {
            return jsonError(.notFound, code: "model_not_loaded",
                              message: "main model '\(mainId)' is not loaded")
        }
        guard !main.isVision else {
            return jsonError(.badRequest, code: "vision_mtp_unsupported",
                              message: "MTP smoke does not support vision models")
        }
        guard let draftId = await registry.draftId(for: mainId),
              let draftEntry = await registry.getDraft(draftId)
        else {
            return jsonError(.badRequest, code: "no_draft_paired",
                              message: "no MTP draft paired with main model '\(mainId)'")
        }
        let maxTok = body.max_tokens ?? 128
        let promptText = body.prompt
        let draftModel = draftEntry.model
        var generationParameters = GenerateParameters(maxTokens: maxTok, temperature: body.temperature ?? 0)
        if let topP = body.top_p { generationParameters.topP = topP }
        if let topK = body.top_k { generationParameters.topK = topK }
        let parameters = generationParameters

        return try await main.container.perform { ctx in
            // Tokenize the prompt with the main model's tokenizer.
            let promptTokens = ctx.tokenizer.encode(text: promptText)
            guard !promptTokens.isEmpty else {
                throw HTTPError(.badRequest, message: "empty prompt after tokenization")
            }
            // Cast the main model to our Qwen3.5/3.6 LLM protocol with
            // fork-added forwardWithHidden / targetVerify / rollback APIs.
            guard let qwen = ctx.model as? any Qwen35HiddenStateProvider else {
                throw HTTPError(.badRequest,
                                message: "main model is not Qwen3.5/3.6 (\(type(of: ctx.model))); MTP unsupported")
            }
            let start = Date()
            let iterator = MTPSpeculativeIterator(
                main: qwen,
                draft: draftModel,
                promptTokens: promptTokens,
                maxTokens: maxTok,
                parameters: parameters
            )
            var generated: [Int] = []
            while let tok = iterator.next() {
                generated.append(tok)
            }
            let elapsed = Date().timeIntervalSince(start)
            let text = ctx.tokenizer.decode(tokenIds: generated)
            let tps = elapsed > 0 ? Double(generated.count) / elapsed : 0
            let payload: [String: Any] = [
                "text": text,
                "tokens": generated.count,
                "rounds": iterator.roundsRun,
                "proposed": iterator.totalProposed,
                "accepted": iterator.totalAccepted,
                "acceptance_rate": iterator.acceptanceRate,
                "elapsed_s": elapsed,
                "tokens_per_sec": tps,
                "block_size": iterator.blockSize,
            ]
            let data = (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: ResponseBody(byteBuffer: ByteBuffer(data: data))
            )
        }
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
