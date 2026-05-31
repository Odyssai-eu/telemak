import CoreImage
import Darwin
import Foundation
@preconcurrency import MLXLMCommon

struct VisionImageBatch: @unchecked Sendable {
    let images: [UserInput.Image]
}

enum VisionInputError: LocalizedError, Sendable {
    case tooManyImages(Int)
    case unsupportedImageSource(String)
    case invalidBase64
    case imageTooLarge(Int)
    case urlFetchFailed(String)
    case invalidImage
    case blockedPrivateImageURL(String)

    var errorDescription: String? {
        switch self {
        case .tooManyImages(let count):
            return "multiple images are not supported yet (received \(count))"
        case .unsupportedImageSource(let source):
            return "unsupported image source: \(source)"
        case .invalidBase64:
            return "invalid base64 image data"
        case .imageTooLarge(let bytes):
            return "image payload exceeds 25 MB decoded limit (\(bytes) bytes)"
        case .urlFetchFailed(let message):
            return "image URL fetch failed: \(message)"
        case .invalidImage:
            return "image payload could not be decoded"
        case .blockedPrivateImageURL(let message):
            return "blocked private image URL: \(message)"
        }
    }
}

enum VisionInputs {
    private static let maxDecodedBytes = 25 * 1024 * 1024

    static func collectOpenAIImages(from messages: [ChatMessage]) async throws -> VisionImageBatch {
        var refs: [ImageReference] = []
        for message in messages where message.role == "user" {
            guard case .blocks(let blocks)? = message.content else { continue }
            for block in blocks where block.type == "image_url" {
                guard let url = block.imageURL?.url else {
                    throw VisionInputError.unsupportedImageSource("missing image_url.url")
                }
                refs.append(.openAIURL(url))
            }
        }
        return try await load(refs)
    }

    static func collectAnthropicImages(from messages: [AnthropicMessage]) async throws -> VisionImageBatch {
        var refs: [ImageReference] = []
        for message in messages where message.role == "user" {
            guard case .blocks(let blocks) = message.content else { continue }
            for block in blocks where block.type == "image" {
                guard let source = block.source else {
                    throw VisionInputError.unsupportedImageSource("missing image source")
                }
                switch source.type {
                case "base64":
                    guard let data = source.data else {
                        throw VisionInputError.unsupportedImageSource("missing base64 data")
                    }
                    refs.append(.base64(data))
                case "url":
                    guard let url = source.url else {
                        throw VisionInputError.unsupportedImageSource("missing source.url")
                    }
                    refs.append(.remoteURL(url))
                default:
                    throw VisionInputError.unsupportedImageSource(source.type)
                }
            }
        }
        return try await load(refs)
    }

    private static func load(_ refs: [ImageReference]) async throws -> VisionImageBatch {
        guard refs.count <= 1 else { throw VisionInputError.tooManyImages(refs.count) }
        var images: [UserInput.Image] = []
        for ref in refs {
            let data = try await data(for: ref)
            images.append(try decode(data))
        }
        return VisionImageBatch(images: images)
    }

    private static func data(for ref: ImageReference) async throws -> Data {
        switch ref {
        case .base64(let raw):
            return try decodeBase64(raw)
        case .openAIURL(let raw):
            if raw.lowercased().hasPrefix("data:") {
                return try decodeDataURI(raw)
            }
            return try await fetch(urlString: raw)
        case .remoteURL(let raw):
            return try await fetch(urlString: raw)
        }
    }

    private static func decodeBase64(_ raw: String) throws -> Data {
        guard let data = Data(base64Encoded: raw, options: [.ignoreUnknownCharacters]) else {
            throw VisionInputError.invalidBase64
        }
        try checkDecodedSize(data)
        return data
    }

    private static func decodeDataURI(_ raw: String) throws -> Data {
        guard let comma = raw.firstIndex(of: ",") else {
            throw VisionInputError.unsupportedImageSource("malformed data URI")
        }
        let metadata = raw[..<comma].lowercased()
        guard metadata.hasPrefix("data:image/"), metadata.contains(";base64") else {
            throw VisionInputError.unsupportedImageSource(String(metadata))
        }
        return try decodeBase64(String(raw[raw.index(after: comma)...]))
    }

    private static func fetch(urlString: String) async throws -> Data {
        guard let url = URL(string: urlString),
              let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https"
        else {
            throw VisionInputError.unsupportedImageSource(urlString)
        }
        try validateRemoteImageURL(url)

        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 5
        config.timeoutIntervalForResource = 5
        let session = URLSession(
            configuration: config,
            delegate: ImageURLFetchDelegate(),
            delegateQueue: nil
        )
        defer { session.finishTasksAndInvalidate() }

        do {
            let (data, response) = try await session.data(from: url)
            if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                throw VisionInputError.urlFetchFailed("HTTP \(http.statusCode)")
            }
            if let mime = response.mimeType, !mime.lowercased().hasPrefix("image/") {
                throw VisionInputError.unsupportedImageSource("URL returned \(mime)")
            }
            try checkDecodedSize(data)
            return data
        } catch let error as VisionInputError {
            throw error
        } catch {
            throw VisionInputError.urlFetchFailed("\(error)")
        }
    }

    private static func decode(_ data: Data) throws -> UserInput.Image {
        guard let image = CIImage(data: data) else { throw VisionInputError.invalidImage }
        return .ciImage(resizeIfNeeded(image))
    }

    private static func resizeIfNeeded(_ image: CIImage) -> CIImage {
        let maxDim = maxImageDimension()
        let width = image.extent.width
        let height = image.extent.height
        guard width.isFinite, height.isFinite, width > 0, height > 0 else { return image }
        let currentMax = max(width, height)
        guard currentMax > maxDim else { return image }
        let scale = maxDim / currentMax
        return image.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
    }

    private static func checkDecodedSize(_ data: Data) throws {
        if data.count > maxDecodedBytes {
            throw VisionInputError.imageTooLarge(data.count)
        }
    }

    private static func maxImageDimension() -> CGFloat {
        let raw = ProcessInfo.processInfo.environment["TELEMAK_MAX_IMAGE_DIM"] ?? ""
        if let value = Double(raw), value > 0 {
            return CGFloat(value)
        }
        return 2048
    }

    fileprivate static func validateRemoteImageURL(_ url: URL) throws {
        guard let scheme = url.scheme?.lowercased(), scheme == "http" || scheme == "https" else {
            throw VisionInputError.unsupportedImageSource(url.absoluteString)
        }
        guard let host = url.host, !host.isEmpty else {
            throw VisionInputError.unsupportedImageSource(url.absoluteString)
        }
        guard !allowPrivateImageURLs() else { return }

        let addresses = try resolvedAddresses(for: host)
        guard !addresses.isEmpty else {
            throw VisionInputError.urlFetchFailed("host '\(host)' did not resolve")
        }
        if let blocked = addresses.first(where: { $0.isBlockedForRemoteFetch }) {
            throw VisionInputError.blockedPrivateImageURL("\(host) resolved to \(blocked)")
        }
    }

    private static func allowPrivateImageURLs() -> Bool {
        let raw = ProcessInfo.processInfo.environment["TELEMAK_ALLOW_PRIVATE_IMAGE_URLS"] ?? ""
        switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
        case "1", "true", "yes", "on":
            return true
        default:
            return false
        }
    }

    private static func resolvedAddresses(for host: String) throws -> [ResolvedIPAddress] {
        var hints = addrinfo()
        hints.ai_family = AF_UNSPEC
        hints.ai_socktype = SOCK_STREAM

        var result: UnsafeMutablePointer<addrinfo>?
        let status = getaddrinfo(host, nil, &hints, &result)
        guard status == 0, let first = result else {
            let message = String(cString: gai_strerror(status))
            throw VisionInputError.urlFetchFailed("could not resolve host '\(host)': \(message)")
        }
        defer { freeaddrinfo(first) }

        var addresses: [ResolvedIPAddress] = []
        var current: UnsafeMutablePointer<addrinfo>? = first
        while let pointer = current {
            let info = pointer.pointee
            if info.ai_family == AF_INET, let addr = info.ai_addr {
                let sin = addr.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { $0.pointee }
                addresses.append(.v4(UInt32(bigEndian: sin.sin_addr.s_addr)))
            } else if info.ai_family == AF_INET6, let addr = info.ai_addr {
                let sin6 = addr.withMemoryRebound(to: sockaddr_in6.self, capacity: 1) { $0.pointee }
                let bytes = withUnsafeBytes(of: sin6.sin6_addr) { Array($0) }
                addresses.append(.v6(bytes))
            }
            current = info.ai_next
        }
        return addresses
    }

    private enum ImageReference {
        case base64(String)
        case openAIURL(String)
        case remoteURL(String)
    }
}

private final class ImageURLFetchDelegate: NSObject, URLSessionTaskDelegate {
    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        willPerformHTTPRedirection response: HTTPURLResponse,
        newRequest request: URLRequest,
        completionHandler: @escaping @Sendable (URLRequest?) -> Void
    ) {
        guard let url = request.url, (try? VisionInputs.validateRemoteImageURL(url)) != nil else {
            completionHandler(nil)
            return
        }
        completionHandler(request)
    }
}

private enum ResolvedIPAddress: CustomStringConvertible {
    case v4(UInt32)
    case v6([UInt8])

    var isBlockedForRemoteFetch: Bool {
        switch self {
        case .v4(let value):
            return Self.isBlockedIPv4(value)
        case .v6(let bytes):
            if let mapped = Self.ipv4MappedAddress(bytes) {
                return Self.isBlockedIPv4(mapped)
            }
            return isUnspecifiedIPv6(bytes)
                || isLoopbackIPv6(bytes)
                || isLinkLocalIPv6(bytes)
                || isUniqueLocalIPv6(bytes)
        }
    }

    var description: String {
        switch self {
        case .v4(let value):
            return [
                UInt8((value >> 24) & 0xff),
                UInt8((value >> 16) & 0xff),
                UInt8((value >> 8) & 0xff),
                UInt8(value & 0xff),
            ].map(String.init).joined(separator: ".")
        case .v6(let bytes):
            var storage = bytes
            return storage.withUnsafeMutableBytes { rawBuffer in
                guard let base = rawBuffer.baseAddress else { return "<invalid-ipv6>" }
                var output = [CChar](repeating: 0, count: Int(INET6_ADDRSTRLEN))
                inet_ntop(AF_INET6, base, &output, socklen_t(INET6_ADDRSTRLEN))
                return String(cString: output)
            }
        }
    }

    private static func isBlockedIPv4(_ value: UInt32) -> Bool {
        let first = UInt8((value >> 24) & 0xff)
        let second = UInt8((value >> 16) & 0xff)

        return first == 0
            || first == 10
            || first == 127
            || (first == 169 && second == 254)
            || (first == 172 && (16...31).contains(second))
            || (first == 192 && second == 168)
    }

    private static func ipv4MappedAddress(_ bytes: [UInt8]) -> UInt32? {
        guard bytes.count == 16,
              bytes[0..<10].allSatisfy({ $0 == 0 }),
              bytes[10] == 0xff,
              bytes[11] == 0xff
        else {
            return nil
        }
        return (UInt32(bytes[12]) << 24)
            | (UInt32(bytes[13]) << 16)
            | (UInt32(bytes[14]) << 8)
            | UInt32(bytes[15])
    }

    private func isUnspecifiedIPv6(_ bytes: [UInt8]) -> Bool {
        bytes.count == 16 && bytes.allSatisfy { $0 == 0 }
    }

    private func isLoopbackIPv6(_ bytes: [UInt8]) -> Bool {
        bytes.count == 16 && bytes[0..<15].allSatisfy { $0 == 0 } && bytes[15] == 1
    }

    private func isLinkLocalIPv6(_ bytes: [UInt8]) -> Bool {
        bytes.count == 16 && bytes[0] == 0xfe && (bytes[1] & 0xc0) == 0x80
    }

    private func isUniqueLocalIPv6(_ bytes: [UInt8]) -> Bool {
        bytes.count == 16 && (bytes[0] & 0xfe) == 0xfc
    }
}
