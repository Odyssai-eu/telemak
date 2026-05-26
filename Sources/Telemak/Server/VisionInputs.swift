import CoreImage
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

        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 5
        config.timeoutIntervalForResource = 5
        let session = URLSession(configuration: config)
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

    private enum ImageReference {
        case base64(String)
        case openAIURL(String)
        case remoteURL(String)
    }
}
