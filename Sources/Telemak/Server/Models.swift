import Foundation
import TelemakVersion
import Hummingbird
import MLX
import MLXLLM
import MLXLMCommon
import TelemakMTP

/// `GET /v1/models`, `GET /admin/models/available`, `GET /admin/memory`,
/// `POST /admin/load`, `POST /admin/unload`.
struct ModelsHandler: Sendable {
    let registry: ModelRegistry
    let activity: ActivityTracker

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
        router.post("/admin/models-dir") { request, _ async throws -> Response in
            try await self.setModelsDir(request)
        }
        router.get("/admin/models-dir") { _, _ async throws -> Response in
            self.getModelsDir()
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
            // true when no models directory is configured (config.json + env both
            // absent) so clients show "set your models directory" instead of a
            // silent empty list.
            let models_dir_unset: Bool
        }
        let entries = AvailableModels.scan()
        let payload = Payload(models: entries,
                              models_dir_unset: ModelsConfig.shared.effectiveDir() == nil)
        let data = (try? JSONEncoder().encode(payload))
            ?? Data(#"{"models":[],"models_dir_unset":true}"#.utf8)
        return jsonResponse(.ok, data: data)
    }

    // MARK: - GET /admin/models-dir

    /// Reports the effective models directory + where it came from, so the
    /// menubar can show it (read-only when `managed`, editable when standalone).
    private func getModelsDir() -> Response {
        struct Payload: Encodable {
            let dir: String?
            let source: String   // "config" | "env" | "unset"
            let managed: Bool
        }
        let snap = ModelsConfig.shared.snapshot()
        let payload = Payload(dir: snap.dir, source: snap.source, managed: snap.managed)
        let data = (try? JSONEncoder().encode(payload))
            ?? Data(#"{"dir":null,"source":"unset","managed":false}"#.utf8)
        return jsonResponse(.ok, data: data)
    }

    // MARK: - POST /admin/models-dir

    /// Set the models directory (master = OdyssAI-X, or the standalone menubar).
    /// Validates the path; with `create:true` it `mkdir -p`s a missing dir, else
    /// returns 409 so the caller can offer to create. Persists to config.json +
    /// recomputes the cached resolver. Loaded models are NOT unloaded — only
    /// discovery + future loads use the new dir.
    private func setModelsDir(_ request: Request) async throws -> Response {
        struct Body: Decodable {
            let dir: String
            let create: Bool?
            let managed: Bool?
        }
        let buf = try await request.body.collect(upTo: 1 << 16)
        let body: Body
        do {
            body = try JSONDecoder().decode(Body.self, from: Data(buffer: buf))
        } catch {
            return jsonError(.badRequest, code: "invalid_request_error",
                              message: "expected {\"dir\": \"<path>\", \"create\": <bool?>, \"managed\": <bool?>}: \(error)")
        }
        let dir = body.dir.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !dir.isEmpty else {
            return jsonError(.badRequest, code: "invalid_request_error", message: "dir must not be empty")
        }
        var isDir: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: dir, isDirectory: &isDir)
        if !exists {
            if body.create == true {
                do {
                    try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
                } catch {
                    return jsonError(.badRequest, code: "create_failed",
                                      message: "could not create directory '\(dir)': \(error)")
                }
            } else {
                return jsonError(.conflict, code: "not_found",
                                  message: "directory '\(dir)' does not exist (pass create:true to create it)")
            }
        } else if !isDir.boolValue {
            return jsonError(.badRequest, code: "not_a_directory", message: "'\(dir)' is not a directory")
        }
        do {
            try ModelsConfig.shared.set(dir: dir, managed: body.managed ?? false)
        } catch {
            return jsonError(.internalServerError, code: "persist_failed",
                              message: "could not persist models directory: \(error)")
        }
        let snap = ModelsConfig.shared.snapshot()
        let payload: [String: String] = ["status": "ok", "dir": snap.dir ?? ""]
        let data = try JSONEncoder().encode(payload)
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
            let force_llm: Bool?
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
        let modelId = ModelLoader.canonicalIdentifier(body.model)
        let draftId = body.draft_model.map(ModelLoader.canonicalIdentifier)
        let activityId = await activity.begin(model: modelId, phase: .loading)
        let isEmbedder = EmbedderLoader.isEmbedder(identifier: modelId)
        if isEmbedder {
            if let draftId, !draftId.isEmpty {
                await activity.finish(activityId)
                return jsonError(.badRequest, code: "invalid_request_error",
                                  message: "draft_model is only valid for chat models")
            }
            do {
                _ = try await registry.loadEmbedder(modelId)
            } catch ModelRegistry.LoadError.insufficientMemory(let needed, let available, let ceiling, let currentlyLoaded) {
                await activity.fail(activityId, error: "insufficient memory loading embedder \(modelId)")
                return insufficientMemoryError(
                    neededBytes: needed,
                    availableBytes: available,
                    ceilingBytes: ceiling,
                    currentlyLoaded: currentlyLoaded
                )
            } catch ModelRegistry.LoadError.loadFailed(let why) {
                await activity.fail(activityId, error: "could not load embedder \(modelId): \(why)")
                return jsonError(.serviceUnavailable, code: "model_load_failed",
                                  message: "could not load embedder '\(modelId)': \(why)")
            } catch {
                await activity.fail(activityId, error: "could not load embedder \(modelId): \(error)")
                return jsonError(.serviceUnavailable, code: "model_load_failed",
                                  message: "could not load embedder '\(modelId)': \(error)")
            }
            await activity.finish(activityId)
            let payload = ["status": "loaded", "model": modelId, "kind": "embedder"]
            let data = try JSONEncoder().encode(payload)
            return jsonResponse(.ok, data: data)
        } else {
            do {
                let forceLLM = body.force_llm ?? draftId.map {
                    let draftType = ModelLoader.modelType(identifier: $0)
                    return draftType == "gemma4_assistant" || draftType == "qwen3_5_mtp"
                } ?? false
                _ = try await registry.load(modelId, forceLLM: forceLLM)
            } catch ModelRegistry.LoadError.insufficientMemory(let needed, let available, let ceiling, let currentlyLoaded) {
                await activity.fail(activityId, error: "insufficient memory loading model \(modelId)")
                return insufficientMemoryError(
                    neededBytes: needed,
                    availableBytes: available,
                    ceilingBytes: ceiling,
                    currentlyLoaded: currentlyLoaded
                )
            } catch ModelRegistry.LoadError.loadFailed(let why) {
                await activity.fail(activityId, error: "could not load model \(modelId): \(why)")
                return jsonError(.serviceUnavailable, code: "model_load_failed",
                                  message: "could not load model '\(modelId)': \(why)")
            } catch {
                await activity.fail(activityId, error: "could not load model \(modelId): \(error)")
                return jsonError(.serviceUnavailable, code: "model_load_failed",
                                  message: "could not load model '\(modelId)': \(error)")
            }
        }
        // Optional draft pairing for MTP speculative decoding (V2). A verified
        // embedded MTPLX artifact pairs the main model with itself; sidecars
        // remain explicit via `draft_model`.
        let autoDraftId: String?
        if let explicit = draftId, !explicit.isEmpty {
            autoDraftId = explicit
        } else if let loaded = await registry.get(modelId), loaded.mtpCompatibility.autoEnabled {
            autoDraftId = modelId
        } else {
            autoDraftId = nil
        }

        if let draftId = autoDraftId, !draftId.isEmpty {
            do {
                _ = try await registry.loadDraft(
                    draftId,
                    pairedWith: modelId,
                    allowUnverified: body.allow_unverified_mtp ?? false
                )
            } catch ModelRegistry.LoadError.insufficientMemory(let needed, let available, let ceiling, let currentlyLoaded) {
                await activity.fail(activityId, error: "insufficient memory loading draft \(draftId)")
                return insufficientMemoryError(
                    neededBytes: needed,
                    availableBytes: available,
                    ceilingBytes: ceiling,
                    currentlyLoaded: currentlyLoaded
                )
            } catch ModelRegistry.LoadError.loadFailed(let why) {
                await activity.fail(activityId, error: "main loaded but draft \(draftId) failed: \(why)")
                return jsonError(.serviceUnavailable, code: "draft_load_failed",
                                  message: "main loaded but draft '\(draftId)' failed: \(why)")
            } catch {
                await activity.fail(activityId, error: "main loaded but draft \(draftId) failed: \(error)")
                return jsonError(.serviceUnavailable, code: "draft_load_failed",
                                  message: "main loaded but draft '\(draftId)' failed: \(error)")
            }
        }
        await activity.finish(activityId)
        var payload: [String: String] = ["status": "loaded", "model": modelId]
        if let draftId = autoDraftId, !draftId.isEmpty {
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

        guard let rawModelId = body?.model, !rawModelId.isEmpty else {
            return jsonError(.badRequest, code: "invalid_request_error",
                              message: "specify model id or pass ?all=true")
        }
        let modelId = ModelLoader.canonicalIdentifier(rawModelId)

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
        // Validate source — must be a clean HuggingFace org/name ID.
        // Reject path separators, traversal sequences, and anything that
        // isn't a safe repo reference before it reaches the Python module.
        let sourcePattern = #"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"#
        guard body.source.range(of: sourcePattern, options: .regularExpression) != nil else {
            return jsonError(.badRequest, code: "invalid_request_error",
                             message: "source must be a HuggingFace model ID (org/name)")
        }

        // Validate output — must resolve under TELEMAK_MODELS_DIR or ~/.telemak.
        if let out = body.output, !out.isEmpty {
            let rootPaths: [String]
            if let modelsDir = ModelsConfig.shared.effectiveDir() {
                rootPaths = [URL(fileURLWithPath: modelsDir).standardized.path]
            } else {
                let home = FileManager.default.homeDirectoryForCurrentUser
                rootPaths = [home.appendingPathComponent(".telemak").standardized.path]
            }
            let outputPath = URL(fileURLWithPath: out).standardized.path
            let contained = rootPaths.contains { root in
                ModelLoader.isPath(outputPath, sameOrDescendantOf: root)
            }
            if !contained {
                return jsonError(.badRequest, code: "invalid_request_error",
                                 message: "output path must be under the models directory")
            }
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
            let draft_block_size: Int?
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
        guard let draftId = await registry.draftId(for: mainId),
              let draftEntry = await registry.getDraft(draftId)
        else {
            return jsonError(.badRequest, code: "no_draft_paired",
                              message: "no MTP draft paired with main model '\(mainId)'")
        }
        let maxTok = body.max_tokens ?? 128
        let blockSize = body.draft_block_size
        let promptText = body.prompt
        var generationParameters = GenerateParameters(maxTokens: maxTok, temperature: body.temperature ?? 0)
        if let topP = body.top_p { generationParameters.topP = topP }
        if let topK = body.top_k { generationParameters.topK = topK }
        let parameters = generationParameters

        return try await main.container.perform { ctx in
            // Prepare through the model processor so smoke uses the same chat
            // template path as /v1/chat/completions.
            let input = try await ctx.processor.prepare(input: UserInput(prompt: promptText))
            let tokenArray = input.text.tokens.reshaped(-1).asType(.int32)
            eval(tokenArray)
            let promptTokens = tokenArray.asArray(Int32.self).map(Int.init)
            guard !promptTokens.isEmpty else {
                throw HTTPError(.badRequest, message: "empty prompt after tokenization")
            }
            let start = Date()
            var generated: [Int] = []
            let rounds: Int
            let proposed: Int
            let accepted: Int
            let acceptanceRate: Double
            let effectiveBlockSize: Int
            let timing: [String: Double]

            switch draftEntry.model {
            case .qwen35(let draftModel):
                // Cast the main model to our Qwen3.5/3.6 LLM protocol with
                // fork-added forwardWithHidden / targetVerify / rollback APIs.
                guard let qwen = ctx.model as? any Qwen35HiddenStateProvider else {
                    throw HTTPError(.badRequest,
                                    message: "main model is not Qwen3.5/3.6 (\(type(of: ctx.model))); MTP unsupported")
                }
                let iterator = MTPSpeculativeIterator(
                    main: qwen,
                    draft: draftModel,
                    promptTokens: promptTokens,
                    maxTokens: maxTok,
                    blockSize: blockSize,
                    parameters: parameters
                )
                while let tok = iterator.next() {
                    generated.append(tok)
                }
                rounds = iterator.roundsRun
                proposed = iterator.totalProposed
                accepted = iterator.totalAccepted
                acceptanceRate = iterator.acceptanceRate
                effectiveBlockSize = iterator.blockSize
                timing = [
                    "main_prefill_s": iterator.mainPrefillSeconds,
                    "draft_prefill_s": iterator.draftPrefillSeconds,
                    "draft_s": iterator.draftSeconds,
                    "verify_s": iterator.verifySeconds,
                    "acceptance_s": iterator.acceptanceSeconds,
                    "rollback_s": iterator.rollbackSeconds,
                    "draft_update_s": iterator.draftUpdateSeconds,
                ]

            case .gemma4Assistant(let draftModel):
                guard let gemma = ctx.model as? Gemma4Model else {
                    throw HTTPError(.badRequest,
                                    message: "main model is not Gemma4 (\(type(of: ctx.model))); Gemma4Assistant MTP unsupported")
                }
                let iterator = Gemma4AssistantSpeculativeIterator(
                    main: gemma,
                    draft: draftModel,
                    promptTokens: promptTokens,
                    maxTokens: maxTok,
                    blockSize: blockSize ?? 6,
                    parameters: parameters
                )
                while let tok = try iterator.next() {
                    generated.append(tok)
                }
                rounds = iterator.roundsRun
                proposed = iterator.totalProposed
                accepted = iterator.totalAccepted
                acceptanceRate = iterator.acceptanceRate
                effectiveBlockSize = iterator.blockSize
                timing = [:]
            }

            let elapsed = Date().timeIntervalSince(start)
            let text = ctx.tokenizer.decode(tokenIds: generated)
            let tps = elapsed > 0 ? Double(generated.count) / elapsed : 0
            let payload: [String: Any] = [
                "text": text,
                "tokens": generated.count,
                "rounds": rounds,
                "proposed": proposed,
                "accepted": accepted,
                "acceptance_rate": acceptanceRate,
                "elapsed_s": elapsed,
                "tokens_per_sec": tps,
                "block_size": effectiveBlockSize,
                "timing": timing,
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
