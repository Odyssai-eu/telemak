import Foundation
import TelemakVersion

/// Filesystem scan for models locally available to Telemak — `TELEMAK_MODELS_DIR`
/// (Odysseus layout) and the HF cache (`~/.cache/huggingface/hub/`).
///
/// The output feeds `GET /admin/models/available` and the menu-bar app's
/// model picker. Both roots are scanned; entries are deduplicated by `id`
/// with `telemak-models-dir` source winning over `hf-cache` when both
/// contain the same id (operator likely curates the Odysseus dir).
public enum AvailableModels {

    public struct Entry: Codable, Sendable, Equatable {
        public let id: String
        public let source: String       // "telemak-models-dir" | "hf-cache"
        public let sizeBytes: Int64
        public let sizeGB: Double
        public let family: String?      // e.g. "qwen3_moe", "qwen2", "gemma2"; nil if config.json missing
        public let lastModified: String  // ISO 8601

        enum CodingKeys: String, CodingKey {
            case id
            case source
            case sizeBytes = "size_bytes"
            case sizeGB = "size_gb"
            case family
            case lastModified = "last_modified"
        }
    }

    public static func scan(
        telemakModelsDir: String? = ModelsConfig.shared.effectiveDir(),
        hfCacheDir: String? = AvailableModels.defaultHFCache()
    ) -> [Entry] {
        var byId: [String: Entry] = [:]

        if let hfCacheDir, let hfEntries = try? scanHFCache(hfCacheDir) {
            for entry in hfEntries { byId[entry.id] = entry }
        }
        if let telemakModelsDir, let teleEntries = try? scanTelemakDir(telemakModelsDir) {
            for entry in teleEntries { byId[entry.id] = entry }   // telemak-models-dir wins
        }

        return byId.values.sorted { $0.id < $1.id }
    }

    public static func defaultHFCache() -> String? {
        if let env = ProcessInfo.processInfo.environment["HF_HUB_CACHE"], !env.isEmpty {
            return env
        }
        let home = FileManager.default.homeDirectoryForCurrentUser.path
        return "\(home)/.cache/huggingface/hub"
    }

    /// Odysseus layout: `<root>/<org>/<name>/snapshots/<hash>/<files>`.
    private static func scanTelemakDir(_ root: String) throws -> [Entry] {
        let fm = FileManager.default
        guard let orgs = try? fm.contentsOfDirectory(atPath: root) else { return [] }

        var results: [Entry] = []
        for org in orgs where !org.hasPrefix(".") {
            let orgPath = (root as NSString).appendingPathComponent(org)
            var isDir: ObjCBool = false
            guard fm.fileExists(atPath: orgPath, isDirectory: &isDir), isDir.boolValue else { continue }

            guard let names = try? fm.contentsOfDirectory(atPath: orgPath) else { continue }
            for name in names where !name.hasPrefix(".") {
                let id = "\(org)/\(name)"
                let modelPath = (orgPath as NSString).appendingPathComponent(name)
                let snapshotsPath = (modelPath as NSString).appendingPathComponent("snapshots")

                let snapshotDir: String
                if fm.fileExists(atPath: snapshotsPath) {
                    let snaps = (try? fm.contentsOfDirectory(atPath: snapshotsPath)) ?? []
                    guard let chosen = snaps.filter({ !$0.hasPrefix(".") }).sorted().last else { continue }
                    snapshotDir = (snapshotsPath as NSString).appendingPathComponent(chosen)
                } else {
                    snapshotDir = modelPath
                }

                guard fileExists((snapshotDir as NSString).appendingPathComponent("config.json")) else { continue }

                let size = directorySize(at: snapshotDir)
                let mtime = lastModified(at: snapshotDir)
                results.append(Entry(
                    id: id,
                    source: "telemak-models-dir",
                    sizeBytes: size,
                    sizeGB: Double(size) / 1_073_741_824.0,
                    family: familyFor(id: id, snapshotDir: snapshotDir),
                    lastModified: iso8601(mtime)
                ))
            }
        }
        return results
    }

    /// HF cache layout: `<root>/models--<org>--<name>/snapshots/<hash>/<files>`.
    private static func scanHFCache(_ root: String) throws -> [Entry] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(atPath: root) else { return [] }

        var results: [Entry] = []
        for dirName in entries where dirName.hasPrefix("models--") {
            let raw = String(dirName.dropFirst("models--".count))
            let parts = raw.components(separatedBy: "--")
            guard parts.count >= 2 else { continue }
            // Org is parts[0], name is the rest joined by "-" (— wait: HF replaces "/" with "--" but the name itself may contain "-")
            // Convention: only the first "--" splits org/name. Rest is part of name with "--" → "-".
            let id = "\(parts[0])/\(parts.dropFirst().joined(separator: "--"))"

            let modelPath = (root as NSString).appendingPathComponent(dirName)
            let snapshotsPath = (modelPath as NSString).appendingPathComponent("snapshots")
            guard fm.fileExists(atPath: snapshotsPath) else { continue }
            let snaps = (try? fm.contentsOfDirectory(atPath: snapshotsPath)) ?? []
            guard let chosen = snaps.filter({ !$0.hasPrefix(".") }).sorted().last else { continue }
            let snapshotDir = (snapshotsPath as NSString).appendingPathComponent(chosen)

            guard fileExists((snapshotDir as NSString).appendingPathComponent("config.json")) else { continue }

            let size = directorySize(at: snapshotDir)
            let mtime = lastModified(at: snapshotDir)
            results.append(Entry(
                id: id,
                source: "hf-cache",
                sizeBytes: size,
                sizeGB: Double(size) / 1_073_741_824.0,
                family: familyFor(id: id, snapshotDir: snapshotDir),
                lastModified: iso8601(mtime)
            ))
        }
        return results
    }

    // MARK: - utilities

    /// Sniff `config.json` for HF `model_type` (canonical family id used by
    /// transformers + mlx-swift-lm). Falls back to a path heuristic when the
    /// file is missing or unreadable — handles bare GGUFs and odd repos that
    /// ship without a config.
    private static func familyFor(id: String, snapshotDir: String) -> String? {
        let configPath = (snapshotDir as NSString).appendingPathComponent("config.json")
        if let data = try? Data(contentsOf: URL(fileURLWithPath: configPath)),
           let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
           let modelType = json["model_type"] as? String,
           !modelType.isEmpty {
            return modelType
        }
        // Path heuristic: lowercased "name" segment, strip size/quant suffixes.
        let name = id.split(separator: "/").last.map(String.init) ?? id
        let lower = name.lowercased()
        for family in ["qwen3", "qwen2", "qwen", "llama", "gemma", "mistral", "phi", "deepseek"] {
            if lower.contains(family) { return family }
        }
        return nil
    }

    private static func fileExists(_ path: String) -> Bool {
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: path, isDirectory: &isDir) && !isDir.boolValue
    }

    /// Sum of all regular-file sizes under `dir`, following symlinks.
    private static func directorySize(at dir: String) -> Int64 {
        let fm = FileManager.default
        let url = URL(fileURLWithPath: dir)
        let resourceKeys: Set<URLResourceKey> = [.isRegularFileKey, .totalFileAllocatedSizeKey, .fileSizeKey]
        guard let enumerator = fm.enumerator(
            at: url,
            includingPropertiesForKeys: Array(resourceKeys),
            options: [],
            errorHandler: { _, _ in true }
        ) else { return 0 }

        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(forKeys: resourceKeys),
                  values.isRegularFile == true else { continue }
            if let size = values.totalFileAllocatedSize ?? values.fileSize {
                total += Int64(size)
            }
        }
        return total
    }

    private static func lastModified(at path: String) -> Date {
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        return (attrs?[.modificationDate] as? Date) ?? Date(timeIntervalSince1970: 0)
    }

    private static func iso8601(_ date: Date) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.string(from: date)
    }
}
