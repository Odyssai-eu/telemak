import Foundation
import MLX
import MLXLMCommon

/// Centralises wired-memory tickets so model weights stay GPU-resident and
/// inference passes claim transient workspace explicitly.
///
/// Pattern (per mlx-swift README + Apple's "Wired Memory" docs):
///   - One **reservation** ticket per loaded model, kept open for the
///     duration of the model's life in the registry. Tells macOS "these
///     bytes are needed; don't page them out".
///   - One **active** ticket per inference call, wrapping the
///     `streamDetails` / `respond` loop. Tells macOS "an additional
///     workspace + KV cache is in use right now".
///
/// `WiredSumPolicy` adds these together to derive the desired wired limit,
/// clamped to `iogpu.wired_limit_mb` (or 75 % of RAM as a soft cap).
///
/// On a Mac without an active sysctl wired-limit grant, mlx-swift falls back
/// to policy-only admission (no kernel calls). So this is safe to wire in
/// unconditionally — when the sysctl is set, it enforces residency; when
/// not, it's a no-op.
public actor WiredMemoryCoordinator {
    private let policy: MLXLMCommon.WiredSumPolicy = MLXLMCommon.WiredSumPolicy(cap: nil)
    private var reservations: [String: WiredMemoryTicket] = [:]   // model_id → ticket

    public init() {}

    /// Open a reservation ticket for `modelId` sized to `weightBytes`. Held
    /// until `endReservation` is called. Idempotent — replacing an existing
    /// ticket for the same id is safe.
    public func reserveModel(_ modelId: String, weightBytes: Int) async {
        if let existing = reservations.removeValue(forKey: modelId) {
            _ = await existing.end()
        }
        let ticket = WiredMemoryTicket(
            size: weightBytes,
            policy: policy,
            kind: .reservation
        )
        _ = await ticket.start()
        reservations[modelId] = ticket
    }

    /// Release the reservation ticket for `modelId`.
    public func endReservation(_ modelId: String) async {
        guard let ticket = reservations.removeValue(forKey: modelId) else { return }
        _ = await ticket.end()
    }

    /// Build an active ticket sized to the inference workspace. Callers
    /// invoke `WiredMemoryTicket.withWiredLimit(ticket, body)` themselves
    /// — keeps the closure's sendability up to the caller (the ChatSession
    /// body captures non-Sendable state, can't be `@Sendable`).
    public func makeInferenceTicket(workspaceBytes: Int) -> WiredMemoryTicket {
        WiredMemoryTicket(
            size: workspaceBytes,
            policy: policy,
            kind: .active
        )
    }

    /// Heuristic for inference workspace — KV cache for ~16k context plus
    /// MLX transient buffers. Used by the chat handler when the caller
    /// didn't request a specific size.
    public static let defaultWorkspaceBytes: Int = 4 * 1024 * 1024 * 1024   // 4 GB
}
