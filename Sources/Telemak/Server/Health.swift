import Foundation
import Hummingbird
import MLX
import TelemakVersion

/// `GET /health` — enriched in Phase 5: uptime, loaded models, MLX memory,
/// request count, recent throughput.
struct HealthHandler: Sendable {
    let registry: ModelRegistry
    let stats: StatsTracker
    let startTime: Date

    func add(to router: Router<BasicRequestContext>) {
        router.get("/health") { _, _ async throws -> Response in
            try await self.handle()
        }
    }

    private func handle() async throws -> Response {
        let uptime = Date().timeIntervalSince(startTime)
        let loadedModelIds = await registry.loadedModelIds
        let (requestsServed, avgTokPerSec) = await stats.snapshot()

        let activeBytes = MLX.GPU.activeMemory
        let cacheBytes = MLX.GPU.cacheMemory
        let usedGB = Double(activeBytes + cacheBytes) / 1_073_741_824.0

        // wired memory free is harder: we'd need to query iogpu.wired_limit_mb
        // and subtract MLX usage. Phase 5 surface what MLX owns; full system
        // accounting can land alongside Block 5's metrics endpoint.
        let payload = HealthPayload(
            status: "ok",
            version: telemakVersion,
            uptimeS: uptime,
            modelsLoaded: loadedModelIds,
            wiredMemoryUsedGB: usedGB,
            wiredMemoryFreeGB: wiredMemoryFreeGB(usedGB: usedGB),
            wiredMemoryTotalGB: bytesToGB(RamBudget.totalRam()),
            wiredMemoryCeilingGB: bytesToGB(RamBudget.ceilingBytes()),
            requestsServed: requestsServed,
            avgTokSRecent: avgTokPerSec
        )

        let data = try JSONEncoder().encode(payload)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }

    /// Best-effort wired-memory free: read `iogpu.wired_limit_mb` via sysctl
    /// and subtract MLX usage. If sysctl unavailable, returns 0 (caller can
    /// detect by checking against system memory).
    private func wiredMemoryFreeGB(usedGB: Double) -> Double {
        let limitGB: Double
        if let wiredLimitMB = sysctlInt("iogpu.wired_limit_mb"), wiredLimitMB > 0 {
            limitGB = Double(wiredLimitMB) / 1024.0
        } else {
            limitGB = bytesToGB(RamBudget.ceilingBytes())
        }
        return max(0, limitGB - usedGB)
    }

    private func sysctlInt(_ name: String) -> Int? {
        var size: size_t = MemoryLayout<Int>.size
        var value: Int = 0
        let ret = sysctlbyname(name, &value, &size, nil, 0)
        return ret == 0 ? value : nil
    }

    private func bytesToGB(_ bytes: Int64) -> Double {
        Double(bytes) / 1_073_741_824.0
    }
}

private struct HealthPayload: Codable {
    let status: String
    let version: String
    let uptimeS: Double
    let modelsLoaded: [String]
    let wiredMemoryUsedGB: Double
    let wiredMemoryFreeGB: Double
    let wiredMemoryTotalGB: Double
    let wiredMemoryCeilingGB: Double
    let requestsServed: Int
    let avgTokSRecent: Double?

    enum CodingKeys: String, CodingKey {
        case status
        case version
        case uptimeS = "uptime_s"
        case modelsLoaded = "models_loaded"
        case wiredMemoryUsedGB = "wired_memory_used_gb"
        case wiredMemoryFreeGB = "wired_memory_free_gb"
        case wiredMemoryTotalGB = "wired_memory_total_gb"
        case wiredMemoryCeilingGB = "wired_memory_ceiling_gb"
        case requestsServed = "requests_served"
        case avgTokSRecent = "avg_tok_s_recent"
    }
}
