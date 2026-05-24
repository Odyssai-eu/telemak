import Foundation
import Hummingbird

/// `GET /.well-known/inference-engine.json` — capability contract that
/// Odysseus (and future clients) can auto-discover instead of hardcoding
/// what Telemak supports.
struct CapabilitiesHandler: Sendable {
    let registry: ModelRegistry

    /// Telemak engine semantic version — bump on contract changes.
    /// 0.3.0 = +speculative_decoding capability (V2 MTP support).
    static let engineVersion = "0.3.0"

    func add(to router: Router<BasicRequestContext>) {
        router.get("/.well-known/inference-engine.json") { _, _ async throws -> Response in
            try await self.handle()
        }
    }

    private func handle() async throws -> Response {
        let loaded = await registry.loadedModels

        struct ModelInfo: Encodable {
            let id: String
            let xTelemak: ModelExt

            enum CodingKeys: String, CodingKey {
                case id
                case xTelemak = "x_telemak"
            }
        }

        struct ModelExt: Encodable {
            let sizeGB: Double
            let loadedAt: String

            enum CodingKeys: String, CodingKey {
                case sizeGB = "size_gb"
                case loadedAt = "loaded_at"
            }
        }

        struct SpeculativePair: Encodable {
            let main: String
            let draft: String
        }

        struct SpeculativeCapability: Encodable {
            let supported: Bool
            let modes: [String]
            let activePairs: [SpeculativePair]

            enum CodingKeys: String, CodingKey {
                case supported
                case modes
                case activePairs = "active_pairs"
            }
        }

        struct Capabilities: Encodable {
            let stream: Bool
            let tools: Bool
            let vision: Bool
            let maxContext: Int
            let sessionCache: Bool
            let openaiCompat: String
            let anthropicCompat: String
            let speculativeDecoding: SpeculativeCapability

            enum CodingKeys: String, CodingKey {
                case stream
                case tools
                case vision
                case maxContext = "max_context"
                case sessionCache = "session_cache"
                case openaiCompat = "openai_compat"
                case anthropicCompat = "anthropic_compat"
                case speculativeDecoding = "speculative_decoding"
            }
        }

        struct Payload: Encodable {
            let engine: String
            let version: String
            let capabilities: Capabilities
            let models: [ModelInfo]
        }

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]

        let modelEntries: [ModelInfo] = loaded.map { entry in
            ModelInfo(
                id: entry.id,
                xTelemak: ModelExt(
                    sizeGB: Double(entry.ramEstimateBytes) / 1_073_741_824.0,
                    loadedAt: iso.string(from: entry.loadedAt)
                )
            )
        }

        // V1 surface — tools is `true` because Block 4 wires tool-call
        // routing through; Anthropic compat lands in Block 4 too. Both
        // flip to false in the build that ships before Block 4 if needed.
        // V2 surface — speculative decoding via paired MTP drafters
        // (Issue #24). The loop is still being wired ; `supported` flips
        // to `true` once Unit 2 lands, but the API surface
        // (`/admin/load` `draft_model`) accepts pairs today so the wire
        // contract stays stable.
        let pairs = await registry.activePairs.map { SpeculativePair(main: $0.main, draft: $0.draft) }
        let speculative = SpeculativeCapability(
            supported: !pairs.isEmpty,
            modes: ["mtp_adapter"],
            activePairs: pairs
        )
        let capabilities = Capabilities(
            stream: true,
            tools: true,
            vision: false,
            maxContext: 32_768,
            sessionCache: true,
            openaiCompat: "v1",
            anthropicCompat: "v1",
            speculativeDecoding: speculative
        )

        let payload = Payload(
            engine: "telemak",
            version: Self.engineVersion,
            capabilities: capabilities,
            models: modelEntries
        )

        let data = try JSONEncoder().encode(payload)
        return Response(
            status: .ok,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(data: data))
        )
    }
}
