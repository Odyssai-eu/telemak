import Foundation
import Testing
import TelemakVersion

/// Resolution order + persistence for the master/slave models-dir config.
@Suite struct ModelsConfigTests {
    private func tmp() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("mc-test-\(UUID().uuidString).json")
    }

    private func writeConfig(_ dir: String?, managed: Bool, to url: URL) throws {
        let cfg = ModelsDirConfig(modelsDir: dir, managed: managed)
        try JSONEncoder().encode(cfg).write(to: url)
    }

    @Test func configWinsOverEnv() throws {
        let path = tmp()
        defer { try? FileManager.default.removeItem(at: path) }
        try writeConfig("/config/dir", managed: true, to: path)
        let mc = ModelsConfig(path: path, env: "/env/dir")
        #expect(mc.effectiveDir() == "/config/dir")
        let snap = mc.snapshot()
        #expect(snap.source == "config")
        #expect(snap.managed == true)
    }

    @Test func envWhenNoConfig() {
        let mc = ModelsConfig(path: tmp(), env: "/env/dir")
        #expect(mc.effectiveDir() == "/env/dir")
        #expect(mc.snapshot().source == "env")
        #expect(mc.snapshot().managed == false)
    }

    @Test func unsetWhenNeither() {
        let mc = ModelsConfig(path: tmp(), env: nil)
        #expect(mc.effectiveDir() == nil)
        #expect(mc.snapshot().source == "unset")
    }

    @Test func emptyEnvIsTreatedAsUnset() {
        let mc = ModelsConfig(path: tmp(), env: "")
        #expect(mc.effectiveDir() == nil)
        #expect(mc.snapshot().source == "unset")
    }

    @Test func setPersistsCanonicalizesAndReReads() throws {
        let path = tmp()
        defer { try? FileManager.default.removeItem(at: path) }
        let mc = ModelsConfig(path: path, env: nil)
        try mc.set(dir: "/tmp/a/b/../c", managed: true)
        #expect(mc.effectiveDir() == "/tmp/a/c")        // `..` standardized away
        #expect(mc.snapshot().managed == true)
        // A fresh instance reads the persisted file — and config still beats env.
        let mc2 = ModelsConfig(path: path, env: "/env/ignored")
        #expect(mc2.effectiveDir() == "/tmp/a/c")
        #expect(mc2.snapshot().source == "config")
    }
}
