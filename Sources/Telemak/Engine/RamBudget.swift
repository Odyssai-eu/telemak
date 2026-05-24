import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// RAM budget helpers — how big a model is on disk (≈ RAM use for MLX
/// quantized weights, which stay quantized in memory), and how much we
/// have to play with.
public enum RamBudget {

    /// Estimate RAM bytes needed to load `id`. Returns nil if the model isn't
    /// findable in `TELEMAK_MODELS_DIR` or the HF cache (then load will fail
    /// anyway with a clearer error from ModelLoader).
    public static func estimate(modelId id: String) -> Int64? {
        // Reuse the AvailableModels scan and look up by id — it already
        // walks the right directories and sums file sizes.
        let entries = AvailableModels.scan()
        if let entry = entries.first(where: { $0.id == id }) {
            return Int64(entry.sizeGB * 1_073_741_824.0)
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
}
