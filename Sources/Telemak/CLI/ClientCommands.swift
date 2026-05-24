import ArgumentParser
import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// CLI subcommands that talk to a running `telemak serve`.
///
/// `telemak models` — list models available on disk (HF cache + Odysseus dir)
/// `telemak load <id>` — POST /admin/load
/// `telemak unload <id>` | `unload --all`
/// `telemak chat <prompt> [--model <id>]` — one-shot generation via HTTP

struct ModelsCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "models",
        abstract: "List models available on disk (Odysseus models-dir + HF cache)."
    )

    @Option(name: .long, help: "Telemak server URL (default http://127.0.0.1:8003).")
    var server: String = "http://127.0.0.1:8003"

    func run() async throws {
        let url = URL(string: "\(server)/admin/models/available")!
        let (data, _) = try await URLSession.shared.data(from: url)
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let models = json["models"] as? [[String: Any]] else {
            print("(empty)")
            return
        }
        for m in models {
            let id = m["id"] as? String ?? "?"
            let size = m["size_gb"] as? Double ?? 0
            let source = m["source"] as? String ?? "?"
            print(String(format: "%-22s  %7.2f GB  %@", source, size, id))
        }
    }
}

struct LoadCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "load",
        abstract: "Load a model into telemak via /admin/load."
    )

    @Argument(help: "Model id, e.g. mlx-community/Qwen3-0.6B-4bit")
    var modelId: String

    @Option(name: .long, help: "Telemak server URL (default http://127.0.0.1:8003).")
    var server: String = "http://127.0.0.1:8003"

    func run() async throws {
        let url = URL(string: "\(server)/admin/load")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONSerialization.data(withJSONObject: ["model": modelId])
        let (data, response) = try await URLSession.shared.data(for: req)
        let body = String(data: data, encoding: .utf8) ?? ""
        let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
        print("[\(statusCode)] \(body)")
        if statusCode >= 400 { throw ExitCode(1) }
    }
}

struct UnloadCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "unload",
        abstract: "Unload a model via /admin/unload."
    )

    @Argument(help: "Model id, omit if using --all.")
    var modelId: String?

    @Flag(name: .long, help: "Unload every loaded model.")
    var all: Bool = false

    @Option(name: .long, help: "Telemak server URL (default http://127.0.0.1:8003).")
    var server: String = "http://127.0.0.1:8003"

    func run() async throws {
        let urlString = all ? "\(server)/admin/unload?all=true" : "\(server)/admin/unload"
        var req = URLRequest(url: URL(string: urlString)!)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let modelId, !all {
            req.httpBody = try JSONSerialization.data(withJSONObject: ["model": modelId])
        } else {
            req.httpBody = Data("{}".utf8)
        }
        let (data, response) = try await URLSession.shared.data(for: req)
        let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
        print("[\(statusCode)] \(String(data: data, encoding: .utf8) ?? "")")
        if statusCode >= 400 { throw ExitCode(1) }
    }
}

struct ChatCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "chat",
        abstract: "Send a one-shot prompt to telemak (HTTP non-stream)."
    )

    @Argument(help: "Prompt to send.")
    var prompt: String

    @Option(name: .long, help: "Model id (must already be loaded).")
    var model: String

    @Option(name: .long, help: "Telemak server URL (default http://127.0.0.1:8003).")
    var server: String = "http://127.0.0.1:8003"

    @Option(name: .long, help: "Max tokens.")
    var maxTokens: Int = 256

    func run() async throws {
        let url = URL(string: "\(server)/v1/chat/completions")!
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let payload: [String: Any] = [
            "model": model,
            "messages": [["role": "user", "content": prompt]],
            "max_tokens": maxTokens,
            "stream": false,
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: payload)
        let (data, response) = try await URLSession.shared.data(for: req)
        let statusCode = (response as? HTTPURLResponse)?.statusCode ?? 0
        if statusCode >= 400 {
            print("[\(statusCode)] \(String(data: data, encoding: .utf8) ?? "")")
            throw ExitCode(1)
        }
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let message = choices.first?["message"] as? [String: Any],
              let content = message["content"] as? String else {
            print(String(data: data, encoding: .utf8) ?? "")
            return
        }
        print(content)
    }
}
