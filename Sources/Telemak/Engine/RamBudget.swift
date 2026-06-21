import Foundation
import TelemakVersion
#if canImport(Darwin)
import Darwin
#endif

/// RAM budget helpers — how big a model is on disk (≈ RAM use for MLX
/// quantized weights, which stay quantized in memory), and how much we
/// have to play with.
public enum RamBudget {

    /// Estimate RAM bytes needed to load `id`. Returns nil if the model isn't
    /// findable in `TELEMAK_MODELS_DIR` or the HF cache. Scoped to a single
    /// model directory — does NOT walk the entire models tree, which would
    /// be O(N×M) on hosts with several large models on disk.
    public static func estimate(modelId id: String) -> Int64? {
        if let dir = resolveModelDir(id: id) {
            return directorySize(at: dir)
        }
        return nil
    }

    /// Resolve `id` → snapshot directory URL. Mirrors ModelLoader's logic
    /// but doesn't iterate every other model on disk.
    private static func resolveModelDir(id: String) -> String? {
        let fm = FileManager.default

        // TELEMAK_MODELS_DIR — Odysseus layout `<root>/<id>` or
        // `<root>/<id>/snapshots/<hash>/`.
        if let root = ModelsConfig.shared.effectiveDir(), !root.isEmpty {
            let base = (root as NSString).appendingPathComponent(id)
            let snapshotsBase = (base as NSString).appendingPathComponent("snapshots")
            if let snaps = try? fm.contentsOfDirectory(atPath: snapshotsBase) {
                if let chosen = snaps.filter({ !$0.hasPrefix(".") }).sorted().last {
                    let candidate = (snapshotsBase as NSString).appendingPathComponent(chosen)
                    if fileExists((candidate as NSString).appendingPathComponent("config.json")) {
                        return candidate
                    }
                }
            }
            if fileExists((base as NSString).appendingPathComponent("config.json")) {
                return base
            }
        }

        // HF cache layout — `<root>/models--<org>--<name>/snapshots/<hash>/`.
        let hfRoot: String
        if let env = ProcessInfo.processInfo.environment["HF_HUB_CACHE"], !env.isEmpty {
            hfRoot = env
        } else {
            hfRoot = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub").path
        }
        let dirName = "models--" + id.replacingOccurrences(of: "/", with: "--")
        let cacheBase = (hfRoot as NSString).appendingPathComponent(dirName)
        let cacheSnapshots = (cacheBase as NSString).appendingPathComponent("snapshots")
        if let snaps = try? fm.contentsOfDirectory(atPath: cacheSnapshots) {
            if let chosen = snaps.filter({ !$0.hasPrefix(".") }).sorted().last {
                let candidate = (cacheSnapshots as NSString).appendingPathComponent(chosen)
                if fileExists((candidate as NSString).appendingPathComponent("config.json")) {
                    return candidate
                }
            }
        }
        return nil
    }

    /// Total system RAM in bytes.
    public static func totalRam() -> Int64 {
        var size: UInt64 = 0
        var sz = MemoryLayout<UInt64>.size
        let ret = sysctlbyname("hw.memsize", &size, &sz, nil, 0)
        return ret == 0 ? Int64(size) : 0
    }

    /// `iogpu.wired_limit_mb` sysctl in bytes, or 0 if unset.
    public static func wiredLimitBytes() -> Int64 {
        var value: Int = 0
        var sz = MemoryLayout<Int>.size
        let ret = sysctlbyname("iogpu.wired_limit_mb", &value, &sz, nil, 0)
        guard ret == 0, value > 0 else { return 0 }
        return Int64(value) * 1024 * 1024
    }

    /// Effective ceiling for model memory: prefer the wired-memory ceiling
    /// if set (the operator opted into a high limit), otherwise fall back to
    /// 75% of physical RAM as a soft cap.
    public static func ceilingBytes() -> Int64 {
        let wired = wiredLimitBytes()
        if wired > 0 { return wired }
        let physical = totalRam()
        return Int64(Double(physical) * 0.75)
    }

    // MARK: - utilities

    private static func fileExists(_ path: String) -> Bool {
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: path, isDirectory: &isDir) && !isDir.boolValue
    }

    private static func directorySize(at dir: String) -> Int64 {
        let url = URL(fileURLWithPath: dir)
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .totalFileAllocatedSizeKey, .fileSizeKey]
        guard let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles], errorHandler: { _, _ in true }
        ) else { return 0 }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            guard let values = try? fileURL.resourceValues(forKeys: keys),
                  values.isRegularFile == true else { continue }
            if let size = values.totalFileAllocatedSize ?? values.fileSize {
                total += Int64(size)
            }
        }
        return total
    }
}
