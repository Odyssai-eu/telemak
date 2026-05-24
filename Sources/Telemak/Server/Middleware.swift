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

/// Optional bearer-token auth. Enforced only when `TELEMAK_API_KEY` env var
/// is set. The healthcheck and `.well-known` endpoints are always open so
/// orchestrators (Odysseus health probes, capability discovery) can poll
/// without a key.
struct BearerAuthMiddleware<Context: RequestContext>: RouterMiddleware {
    let expectedKey: String?
    let openPathPrefixes: [String]

    init(expectedKey: String? = nil, openPathPrefixes: [String] = ["/health", "/.well-known/"]) {
        self.expectedKey = expectedKey ?? ProcessInfo.processInfo.environment["TELEMAK_API_KEY"]
        self.openPathPrefixes = openPathPrefixes
    }

    func handle(_ request: Request, context: Context, next: (Request, Context) async throws -> Response) async throws -> Response {
        guard let expected = expectedKey, !expected.isEmpty else {
            return try await next(request, context)
        }

        let path = request.uri.path
        if openPathPrefixes.contains(where: { path == $0 || path.hasPrefix($0) }) {
            return try await next(request, context)
        }

        let header = request.headers[.authorization] ?? ""
        let prefix = "Bearer "
        if header.hasPrefix(prefix), String(header.dropFirst(prefix.count)) == expected {
            return try await next(request, context)
        }

        let payload = #"{"error":{"type":"unauthorized","code":"unauthorized","message":"missing or invalid bearer token"}}"#
        return Response(
            status: .unauthorized,
            headers: [.contentType: "application/json"],
            body: ResponseBody(byteBuffer: ByteBuffer(string: payload))
        )
    }
}
