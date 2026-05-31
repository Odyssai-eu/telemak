import Foundation
import Hummingbird
import Logging
import HTTPTypes

/// CORS middleware — `Access-Control-Allow-Origin` configurable via env
/// `TELEMAK_CORS_ORIGIN` (default `*`). Handles `OPTIONS` preflight with
/// the standard reply headers for the methods/headers Telemak speaks.
struct CORSMiddleware<Context: RequestContext>: RouterMiddleware {
    let origin: String

    init(origin: String? = nil) {
        let env = ProcessInfo.processInfo.environment["TELEMAK_CORS_ORIGIN"]
        self.origin = origin ?? env ?? "*"
    }

    func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws -> Response {
        if request.method == .options {
            return Response(
                status: .noContent,
                headers: corsHeaders()
            )
        }
        var response = try await next(request, context)
        for field in corsHeaders() {
            response.headers[field.name] = field.value
        }
        return response
    }

    private func corsHeaders() -> HTTPFields {
        var fields = HTTPFields()
        fields[.init("Access-Control-Allow-Origin")!] = origin
        fields[.init("Access-Control-Allow-Methods")!] = "GET, POST, OPTIONS"
        fields[.init("Access-Control-Allow-Headers")!] = "Content-Type, Authorization, X-Session-Id"
        fields[.init("Access-Control-Max-Age")!] = "86400"
        return fields
    }
}

/// Optional bearer-token auth for operator endpoints. Telemak is LAN-first:
/// inference endpoints stay reachable by Odysseus on the local network, while
/// `/admin/*` can be hardened by setting `TELEMAK_API_KEY`.
struct BearerAuthMiddleware<Context: RequestContext>: RouterMiddleware {
    let expectedKey: String?
    let protectedPathPrefixes: [String]

    init(expectedKey: String? = nil, protectedPathPrefixes: [String] = ["/admin/"]) {
        self.expectedKey = expectedKey ?? ProcessInfo.processInfo.environment["TELEMAK_API_KEY"]
        self.protectedPathPrefixes = protectedPathPrefixes
    }

    func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws -> Response {
        guard let expected = expectedKey, !expected.isEmpty else {
            return try await next(request, context)
        }

        let path = request.uri.path
        let isProtected = protectedPathPrefixes.contains { prefix in
            let normalized = prefix.hasSuffix("/") ? String(prefix.dropLast()) : prefix
            return path == normalized || path.hasPrefix(prefix)
        }
        if !isProtected {
            return try await next(request, context)
        }

        let header = request.headers[.authorization] ?? ""
        let prefix = "Bearer "
        if header.hasPrefix(prefix), Self.constantTimeEquals(String(header.dropFirst(prefix.count)), expected) {
            return try await next(request, context)
        }

        let payload = #"{"error":{"type":"unauthorized","code":"unauthorized","message":"missing or invalid bearer token"}}"#
        return Response(
            status: .unauthorized,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(string: payload))
        )
    }

    private static func constantTimeEquals(_ candidate: String, _ expected: String) -> Bool {
        let candidateBytes = Array(candidate.utf8)
        let expectedBytes = Array(expected.utf8)
        var diff = UInt8(candidateBytes.count == expectedBytes.count ? 0 : 1)

        for index in expectedBytes.indices {
            let candidateByte = index < candidateBytes.count ? candidateBytes[index] : 0
            diff |= candidateByte ^ expectedBytes[index]
        }

        return diff == 0
    }
}
