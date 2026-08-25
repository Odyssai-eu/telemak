import Foundation
import Testing
@testable import Telemak

/// Issue #75 — circuit-breaker write-ahead replay.
///
/// MLX GPU-timeout aborts the process from a Metal completion handler, which
/// Swift cannot catch; `replayState`'s do/catch is dead code for that class
/// of error. The WAL (`attempting_replay` in state.json) plus the boot-time
/// breaker (`consumeStuckReplay`) guarantee the NEXT boot skips the model
/// that killed the previous one, instead of crash-looping.
///
/// The load itself needs Metal, so these tests exercise the persistence
/// state machine around it — which is all the breaker needs to be correct.
@Suite struct ReplayCircuitBreakerTests {

    private func tmpStateURL() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("telemak-state-\(UUID().uuidString).json")
    }

    // MARK: - Pure logic (PersistedState.consumingStuckReplay)

    @Test func stuckMarkerIsConsumedAndModelEvicted() {
        let state = PersistedState(loadedModels: ["minimax-m3-vl", "qwen-27b"], attemptingReplay: "minimax-m3-vl")
        let (healed, stuck) = state.consumingStuckReplay()

        #expect(stuck == "minimax-m3-vl")
        #expect(healed.loadedModels == ["qwen-27b"])
        #expect(healed.attemptingReplay == nil)
    }

    @Test func noMarkerMeansNoop() {
        let state = PersistedState(loadedModels: ["a", "b"])
        let (healed, stuck) = state.consumingStuckReplay()

        #expect(stuck == nil)
        #expect(healed.loadedModels == ["a", "b"])
        #expect(healed.attemptingReplay == nil)
    }

    @Test func markerWithoutMatchingModelIsStillCleared() {
        // Crash happened after a persistState already dropped the model from
        // loaded_models but before the marker was cleared. The breaker must
        // still clear the marker (otherwise every later boot logs a phantom
        // warning and, worse, a stale marker could evict a model re-added
        // by the operator).
        let state = PersistedState(loadedModels: ["a"], attemptingReplay: "b")
        let (healed, stuck) = state.consumingStuckReplay()

        #expect(stuck == "b")
        #expect(healed.loadedModels == ["a"])
        #expect(healed.attemptingReplay == nil)
    }

    @Test func legacyStateJsonDecodesWithNilMarker() throws {
        // state.json written by pre-#75 binaries has no attempting_replay
        // key — decoding must yield nil, not fail.
        let legacy = Data(
            #"{"loaded_models":["a","b"],"last_updated":"2026-08-24T10:00:00Z"}"#.utf8
        )
        let state = try JSONDecoder().decode(PersistedState.self, from: legacy)
        #expect(state.loadedModels == ["a", "b"])
        #expect(state.attemptingReplay == nil)
    }

    // MARK: - StateStore actor round-trips (real temp files)

    @Test func bootAfterCrashEvictsStuckModelFromDisk() async throws {
        let url = tmpStateURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let store = StateStore(path: url)

        // Simulate the disk exactly as the previous boot left it when it
        // died inside registry.load("bad"): the write-ahead marker was
        // persisted, the clear never ran.
        try await store.write(
            PersistedState(loadedModels: ["bad", "good"], attemptingReplay: "bad")
        )

        // Next boot: breaker fires before any load.
        let stuck = try await store.consumeStuckReplay()
        #expect(stuck == "bad")

        let after = await store.read()
        #expect(after?.loadedModels == ["good"])
        #expect(after?.attemptingReplay == nil)

        // Second boot on the healed state: nothing left to consume — the
        // loop is broken, not just deferred.
        let stuckAgain = try await store.consumeStuckReplay()
        #expect(stuckAgain == nil)
    }

    @Test func successSequenceLeavesNoResidualMarker() async throws {
        let url = tmpStateURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let store = StateStore(path: url)

        // Full happy-path replay of two models: mark → (load) → clear.
        try await store.markAttemptingReplay("a")
        try await store.clearAttemptingReplay()
        try await store.markAttemptingReplay("b")
        try await store.clearAttemptingReplay()

        let state = await store.read()
        #expect(state?.attemptingReplay == nil)
    }

    @Test func registryPersistDoesNotClobberWriteAheadMarker() async throws {
        // The coordination point with ModelRegistry.persistState(): after a
        // successful load the registry calls update(loadedModels:) — in the
        // MIDDLE of the mark/clear window. If that write reset the marker,
        // a crash in the NEXT model's load would evict nothing (marker nil)
        // and the crash-loop would survive. The marker must survive the
        // registry persist.
        let url = tmpStateURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let store = StateStore(path: url)

        try await store.write(PersistedState(loadedModels: ["a", "b"]))
        try await store.markAttemptingReplay("a")

        // registry.load("a") succeeded → persistState → update(loadedModels:)
        try await store.update(loadedModels: ["a"])

        let mid = await store.read()
        #expect(mid?.loadedModels == ["a"])
        #expect(mid?.attemptingReplay == "a", "registry persist must not reset the WAL marker")

        // And the marker is still consumable: crash before the clear → the
        // next boot still knows which model to evict.
        let stuck = try await store.consumeStuckReplay()
        #expect(stuck == "a")
        let after = await store.read()
        #expect(after?.loadedModels == [])
        #expect(after?.attemptingReplay == nil)
    }

    @Test func recoverableFailureAlsoClearsMarker() async throws {
        // A load that throws a catchable error (model missing, OOM refusal)
        // must NOT leave a marker behind: the process survived, so the next
        // boot must replay normally — no phantom eviction.
        let url = tmpStateURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let store = StateStore(path: url)

        try await store.write(PersistedState(loadedModels: ["a"]))
        try await store.markAttemptingReplay("a")
        try await store.clearAttemptingReplay()

        let state = await store.read()
        #expect(state?.loadedModels == ["a"])
        #expect(state?.attemptingReplay == nil)
    }

    @Test func updateOnMissingFileStartsFreshWithoutMarker() async throws {
        let url = tmpStateURL()
        defer { try? FileManager.default.removeItem(at: url) }
        let store = StateStore(path: url)

        // No prior state at all (first run): update must synthesise a fresh
        // state and consumeStuckReplay must be a no-op.
        try await store.update(loadedModels: ["a"])
        let state = await store.read()
        #expect(state?.loadedModels == ["a"])
        #expect(state?.attemptingReplay == nil)

        let stuck = try await store.consumeStuckReplay()
        #expect(stuck == nil)
    }
}
