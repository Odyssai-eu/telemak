import ArgumentParser
import Foundation
import MLXLMCommon
#if canImport(Darwin)
import Darwin
#endif

struct Smoke: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "smoke",
        abstract: "Load a model and generate from a single prompt (offline test)."
    )

    @Argument(help: "Prompt to feed the model.")
    var prompt: String

    @Option(
        name: .long,
        help: "Model id (e.g. mlx-community/Qwen3-4B-4bit) or absolute path to a local model directory."
    )
    var model: String = "mlx-community/Qwen3-4B-4bit"

    @Option(name: .long, help: "Maximum tokens to generate.")
    var maxTokens: Int = 100

    @Option(name: .long, help: "Sampling temperature.")
    var temperature: Float = 0.7

    @Option(name: .long, help: "Override HF_HUB_CACHE for the duration of this run.")
    var hfHubCache: String?

    @Option(name: .long, help: "Optional system prompt.")
    var system: String?

    @Flag(name: .long, help: "Stream tokens to stdout as they arrive.")
    var stream: Bool = false

    func run() async throws {
        if let hfHubCache {
            setenv("HF_HUB_CACHE", hfHubCache, 1)
        }

        FileHandle.standardError.write(Data("→ loading \(model)…\n".utf8))
        let loadStart = Date()
        let container = try await ModelLoader.load(identifier: model)
        let loadElapsed = Date().timeIntervalSince(loadStart)
        FileHandle.standardError.write(Data(String(format: "← loaded in %.1fs\n", loadElapsed).utf8))

        var params = GenerateParameters()
        params.maxTokens = maxTokens
        params.temperature = temperature

        let session = ChatSession(
            container,
            instructions: system,
            generateParameters: params
        )

        let genStart = Date()
        var firstTokenAt: TimeInterval?
        var totalChars = 0

        if stream {
            for try await chunk in session.streamResponse(to: prompt) {
                if firstTokenAt == nil {
                    firstTokenAt = Date().timeIntervalSince(genStart)
                }
                totalChars += chunk.count
                FileHandle.standardOutput.write(Data(chunk.utf8))
            }
            FileHandle.standardOutput.write(Data("\n".utf8))
        } else {
            let response = try await session.respond(to: prompt)
            print(response)
            totalChars = response.count
        }

        let genElapsed = Date().timeIntervalSince(genStart)
        var summary = String(
            format: "\n[generation: %.1fs total", genElapsed
        )
        if let firstTokenAt {
            summary += String(format: ", first token %.2fs", firstTokenAt)
        }
        summary += "]\n"
        FileHandle.standardError.write(Data(summary.utf8))
    }
}
