import Foundation

/// On-disk config the operator sets to tell Telemak which directory to
/// scan / serve / load models from. Set by OdyssAI-X (the master) on a paired
/// cluster, or by the standalone menubar. Persisted to `~/.telemak/config.json`
/// (the same data dir as `state.json` / `api-key.txt`).
public struct ModelsDirConfig: Codable, Sendable {
    public var modelsDir: String?
    public var managed: Bool
    public var lastUpdated: String?

    enum CodingKeys: String, CodingKey {
        case modelsDir = "models_dir"
        case managed
        case lastUpdated = "last_updated"
    }

    public init(modelsDir: String?, managed: Bool, lastUpdated: String? = nil) {
        self.modelsDir = modelsDir
        self.managed = managed
        self.lastUpdated = lastUpdated
    }
}

/// Resolves the effective models directory and persists operator changes.
///
/// Resolution order: `config.json.models_dir` (master) **>** env
/// `TELEMAK_MODELS_DIR` (legacy / standalone install) **>** `nil` (unset →
/// callers show "set your models directory" instead of a silent empty list).
///
/// The value is CACHED at init and recomputed on `set()` — never a per-call env
/// read. Lives in `TelemakVersion` so the server and the menubar (separate
/// executables that both depend on it) share one implementation. Thread-safe via
/// a lock around the cached snapshot.
public final class ModelsConfig: @unchecked Sendable {
    /// Process-wide instance used by every read site (the models dir is a
    /// process-level setting). Tests construct their own with `init(path:env:)`.
    public static let shared = ModelsConfig()

    public static let defaultPath: URL = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".telemak").appendingPathComponent("config.json")
    }()

    private let path: URL
    private let envDir: String?
    private let lock = NSLock()
    private var cached: ModelsDirConfig?

    public init(path: URL = ModelsConfig.defaultPath,
                env: String? = ProcessInfo.processInfo.environment["TELEMAK_MODELS_DIR"]) {
        self.path = path
        self.envDir = (env?.isEmpty == false) ? env : nil
        self.cached = ModelsConfig.readFile(path)
    }

    /// The directory to scan / load from, or `nil` when nothing is configured.
    public func effectiveDir() -> String? {
        lock.lock(); defer { lock.unlock() }
        if let d = cached?.modelsDir, !d.isEmpty { return d }
        return envDir
    }

    /// `(dir, source, managed)` for `GET /admin/models-dir` and the menubar.
    /// `managed` is only meaningful when `source == "config"`.
    public func snapshot() -> (dir: String?, source: String, managed: Bool) {
        lock.lock(); defer { lock.unlock() }
        if let d = cached?.modelsDir, !d.isEmpty {
            return (d, "config", cached?.managed ?? false)
        }
        if let e = envDir { return (e, "env", false) }
        return (nil, "unset", false)
    }

    /// Persist a new (canonicalized) directory + recompute the cache. The caller
    /// is responsible for validating existence / honoring a `create` flag first.
    public func set(dir: String, managed: Bool) throws {
        let canon = URL(fileURLWithPath: dir).standardized.path
        let cfg = ModelsDirConfig(modelsDir: canon, managed: managed,
                                  lastUpdated: ModelsConfig.iso8601(Date()))
        try ModelsConfig.writeFile(cfg, to: path)
        lock.lock(); cached = cfg; lock.unlock()
    }

    // MARK: - File IO

    private static func readFile(_ url: URL) -> ModelsDirConfig? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? JSONDecoder().decode(ModelsDirConfig.self, from: data)
    }

    private static func writeFile(_ cfg: ModelsDirConfig, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(cfg).write(to: url, options: .atomic)
    }

    private static func iso8601(_ date: Date) -> String {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime]
        return f.string(from: date)
    }
}
