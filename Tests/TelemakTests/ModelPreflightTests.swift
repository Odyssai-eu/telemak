import Foundation
import Testing
import Telemak
@testable import Telemak
@testable import TelemakVersion

/// Issue #64 — preflight validation for `/admin/load`. Builds real fixtures
/// on disk (config.json + safetensors shards + shard index) and asserts the
/// 5 typed errors and 2 happy paths.
@Suite struct ModelPreflightTests {

    // MARK: - fixtures

    private static let hfCacheEnvVar = "HF_HUB_CACHE"
    // Swift 6 strict concurrency: the saved env value is process-global by
    // definition (we mutate a libc env var). Mark `nonisolated(unsafe)` and
    // keep access confined to `isolateHFCache` / `restoreHFCache` — both
    // called from the suite's @Test funcs, which Swift Testing serializes
    // per suite by default for static funcs that mutate shared state.
    private nonisolated(unsafe) static var savedHFCache: String?

    /// Test-scoped temp root. Each test gets its own sub-dir, so concurrent
    /// test runs don't trample each other.
    private static func tmpRoot(testName: String = UUID().uuidString) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("preflight-\(testName)")
    }

    /// Build a model dir under `root/<id>/snapshots/<hash>/` with the given
    /// `configJSON` and a list of `shardFilenames` (empty bodies are fine —
    /// we only check existence, not safetensors format).
    private static func makeModelDir(
        in root: URL,
        id: String,
        configJSON: [String: Any],
        shardFilenames: [String] = [],
        indexWeightMap: [String: String]? = nil
    ) throws -> URL {
        let hash = "abc123"
        let snapshot = root
            .appendingPathComponent(id)
            .appendingPathComponent("snapshots")
            .appendingPathComponent(hash)
        try FileManager.default.createDirectory(at: snapshot, withIntermediateDirectories: true)

        let configData = try JSONSerialization.data(
            withJSONObject: configJSON, options: [.prettyPrinted, .sortedKeys])
        try configData.write(to: snapshot.appendingPathComponent("config.json"))

        for name in shardFilenames {
            try Data().write(to: snapshot.appendingPathComponent(name))
        }
        if let weightMap = indexWeightMap {
            let index: [String: Any] = [
                "metadata": ["total_size": 1_000_000_000],
                "weight_map": weightMap,
            ]
            let data = try JSONSerialization.data(
                withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: snapshot.appendingPathComponent("model.safetensors.index.json"))
        }
        return snapshot
    }

    private static func makeBareModelDir(in root: URL, id: String) throws -> URL {
        let bare = root.appendingPathComponent(id)
        try FileManager.default.createDirectory(at: bare, withIntermediateDirectories: true)
        return bare
    }

    private static func validQwenConfig() -> [String: Any] {
        [
            "model_type": "qwen3",
            "architectures": ["Qwen3ForCausalLM"],
            "hidden_size": 1024,
        ]
    }

    // MARK: - isolation

    /// Point the HF cache at an empty tmp dir so tests don't accidentally
    /// resolve a real model id from the operator's local HF cache. Restored
    /// on teardown.
    private static func isolateHFCache() {
        if savedHFCache == nil {
            savedHFCache = getenv(hfCacheEnvVar).flatMap { String(cString: $0) }
        }
        let empty = FileManager.default.temporaryDirectory
            .appendingPathComponent("preflight-hf-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        _ = setenv(hfCacheEnvVar, empty.path, 1)
    }

    private static func restoreHFCache() {
        if let saved = savedHFCache {
            _ = setenv(hfCacheEnvVar, saved, 1)
        } else {
            _ = unsetenv(hfCacheEnvVar)
        }
        savedHFCache = nil
    }

    // MARK: - happy paths

    @Test func localDirWithValidConfigReturnsLocalOutcome() throws {
        Self.isolateHFCache(); defer { Self.restoreHFCache() }
        let root = Self.tmpRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let snapshot = try Self.makeModelDir(
            in: root, id: "test-org/test-model",
            configJSON: Self.validQwenConfig(),
            shardFilenames: ["model-00001-of-00002.safetensors",
                             "model-00002-of-00002.safetensors"])
        let modelsDir = root.path
        let outcome = try ModelPreflight.check(
            identifier: "test-org/test-model", modelsDir: modelsDir)
        // URL.standardizedFileURL trims the trailing slash that
        // `appendingPathComponent` adds — both sides get the same shape.
        if case .local(_, let dir, let modelType, let weightFiles) = outcome {
            #expect(dir.standardizedFileURL == snapshot.standardizedFileURL)
            #expect(modelType == "qwen3")
            #expect(weightFiles.sorted() == [
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            ])
        } else {
            Issue.record("expected .local, got \(outcome)")
        }
    }

    @Test func absolutePathToValidDirReturnsLocalOutcome() throws {
        Self.isolateHFCache(); defer { Self.restoreHFCache() }
        let root = Self.tmpRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        let bare = try Self.makeBareModelDir(in: root, id: "abs")
        let configData = try JSONSerialization.data(
            withJSONObject: Self.validQwenConfig())
        try configData.write(to: bare.appendingPathComponent("config.json"))

        let outcome = try ModelPreflight.check(identifier: bare.path, modelsDir: nil)
        if case .local(_, _, let modelType, _) = outcome {
            #expect(modelType == "qwen3")
        } else {
            Issue.record("expected .local, got \(outcome)")
        }
    }

    @Test func hubIdNotLocalReturnsRemoteOutcome() throws {
        Self.isolateHFCache(); defer { Self.restoreHFCache() }
        let modelsDir = Self.tmpRoot().path
        defer { try? FileManager.default.removeItem(atPath: modelsDir) }

        // An id that won't be in the empty models dir or the empty HF cache.
        let outcome = try ModelPreflight.check(
            identifier: "zzztest-org/zzztest-model-98765-abcde",
            modelsDir: modelsDir)
        #expect(outcome == .remote(id: "zzztest-org/zzztest-model-98765-abcde"))
    }

    // MARK: - error paths

    @Test func absolutePathToMissingDirThrowsModelDirMissing() throws {
        Self.isolateHFCache(); defer { Self.restoreHFCache() }
        let missing = "/tmp/preflight-definitely-missing-\(UUID().uuidString)"
        #expect(throws: ModelPreflight.Error.self) {
            _ = try ModelPreflight.check(identifier: missing, modelsDir: nil)
        }
    }

    @Test func dirWithoutConfigJsonThrowsConfigMissing() throws {
        Self.isolateHFCache(); defer { Self.restoreHFCache() }
        let root = Self.tmpRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        // Absolute path to a dir that exists but has no config.json.
        let bare = try Self.makeBareModelDir(in: root, id: "no-cfg")

        do {
            _ = try ModelPreflight.check(identifier: bare.path, modelsDir: nil)
            Issue.record("expected configMissing")
        } catch let ModelPreflight.Error.configMissing(_, dir) {
            #expect(dir == bare.path)
        } catch {
            Issue.record("expected configMissing, got \(error)")
        }
    }

    @Test func malformedConfigJsonThrowsConfigParseFailed() throws {
        Self.isolateHFCache(); defer { Self.restoreHFCache() }
        let root = Self.tmpRoot()
        defer { try? FileManager.default.removeItem(at: root) }
        let snapshot = try Self.makeBareModelDir(in: root, id: "bad-cfg")
        try Data("not actually json".utf8).write(to: snapshot.appendingPathComponent("config.json"))

        #expect(throws: ModelPreflight.Error.self) {
            _ = try ModelPreflight.check(identifier: "bad-cfg", modelsDir: root.path)
        }
    }

    @Test func incompleteShardsThrowsShardsIncomplete() throws {
        Self.isolateHFCache(); defer { Self.restoreHFCache() }
        let root = Self.tmpRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        // Index references 3 shards, only 2 are on disk.
        let weightMap: [String: String] = [
            "model.embed_tokens.weight":   "model-00001-of-00003.safetensors",
            "model.layers.0.weight":       "model-00001-of-00003.safetensors",
            "model.layers.50.weight":      "model-00003-of-00003.safetensors",
        ]
        _ = try Self.makeModelDir(
            in: root, id: "sharded-org/sharded-model",
            configJSON: Self.validQwenConfig(),
            shardFilenames: ["model-00001-of-00003.safetensors",
                             "model-00002-of-00003.safetensors"],   // shard 2 missing
            indexWeightMap: weightMap)

        do {
            _ = try ModelPreflight.check(
                identifier: "sharded-org/sharded-model", modelsDir: root.path)
            Issue.record("expected shardsIncomplete")
        } catch let ModelPreflight.Error.shardsIncomplete(_, _, missing) {
            #expect(missing == ["model-00003-of-00003.safetensors"])
        } catch {
            Issue.record("expected shardsIncomplete, got \(error)")
        }
    }

    @Test func unsupportedModelTypeThrows() throws {
        Self.isolateHFCache(); defer { Self.restoreHFCache() }
        let root = Self.tmpRoot()
        defer { try? FileManager.default.removeItem(at: root) }

        // config has neither model_type nor architectures, AND the only
        // weights are a .bin (PyTorch dump — mlx-swift-lm can't load it).
        let config: [String: Any] = ["hidden_size": 1024]
        _ = try Self.makeModelDir(
            in: root, id: "weird-org/weird-model",
            configJSON: config,
            shardFilenames: ["pytorch_model.bin"])

        do {
            _ = try ModelPreflight.check(
                identifier: "weird-org/weird-model", modelsDir: root.path)
            Issue.record("expected unsupportedModelType")
        } catch let ModelPreflight.Error.unsupportedModelType(_, _, modelType) {
            #expect(modelType == nil)
        } catch {
            Issue.record("expected unsupportedModelType, got \(error)")
        }
    }

    @Test func emptyIdentifierThrowsModelDirMissing() {
        Self.isolateHFCache(); defer { Self.restoreHFCache() }
        #expect(throws: ModelPreflight.Error.self) {
            _ = try ModelPreflight.check(identifier: "   ", modelsDir: nil)
        }
    }
}
