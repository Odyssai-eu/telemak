import SwiftUI

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: String
    let content: String
}

/// Minimal test chat (B3) — non-streaming, in-memory history, no persistence.
struct ChatView: View {
    let settings: Settings

    @State private var messages: [ChatMessage] = []
    @State private var inputText: String = ""
    @State private var isSending = false
    @State private var loadedModels: [String] = []
    @State private var modelsChecked = false
    @State private var engineDown = false

    private var activeModel: String? { loadedModels.first }

    var body: some View {
        VStack(spacing: 0) {
            if modelsChecked && engineDown {
                ContentUnavailableView {
                    Label("Chat", systemImage: "bolt.slash")
                } description: {
                    Text("Moteur indisponible — vérifiez le service Telemak")
                } actions: {
                    Button("Réessayer") { Task { await refreshModels() } }
                }
            } else if modelsChecked && activeModel == nil {
                ContentUnavailableView {
                    Label("Chat", systemImage: "bubble.left.and.bubble.right")
                } description: {
                    Text("Aucun modèle chargé — chargez-en un dans Models")
                }
            } else {
                messageList
                Divider()
                inputBar
            }
        }
        .task { await refreshModels() }
    }

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(messages) { message in
                        bubble(for: message)
                            .id(message.id)
                    }
                }
                .padding()
            }
            .onChange(of: messages.count) {
                if let last = messages.last {
                    withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                }
            }
        }
    }

    @ViewBuilder
    private func bubble(for message: ChatMessage) -> some View {
        if message.role == "error" {
            Text(message.content)
                .font(.callout)
                .foregroundStyle(.red)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .center)
        } else {
            HStack {
                if message.role == "user" { Spacer(minLength: 60) }
                Text(message.content)
                    .textSelection(.enabled)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(
                        message.role == "user"
                            ? Color.accentColor.opacity(0.2)
                            : Color.secondary.opacity(0.15),
                        in: RoundedRectangle(cornerRadius: 12)
                    )
                if message.role != "user" { Spacer(minLength: 60) }
            }
        }
    }

    private var inputBar: some View {
        HStack(spacing: 8) {
            TextField("Message", text: $inputText, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...5)
                .onSubmit { send() }
                .disabled(isSending)
            Button(action: send) {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.title2)
            }
            .buttonStyle(.plain)
            .disabled(isSending || inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding()
    }

    private func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isSending else { return }
        inputText = ""
        messages.append(ChatMessage(role: "user", content: text))
        isSending = true
        Task { await sendRequest() }
    }

    private func authed(_ url: URL, method: String = "GET", body: Data? = nil, timeout: Double = 4) -> URLRequest {
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.timeoutInterval = timeout
        let key = settings.apiKey
        if !key.isEmpty { req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
        if let body {
            req.httpBody = body
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return req
    }

    @MainActor
    private func refreshModels() async {
        guard let url = settings.endpointURL?.appendingPathComponent("/v1/models") else {
            modelsChecked = true
            return
        }
        do {
            let (data, _) = try await URLSession.shared.data(for: authed(url))
            engineDown = false
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            loadedModels = (json["data"] as? [[String: Any]])?.compactMap { entry in
                guard (entry["x_telemak"] as? [String: Any])?["kind"] as? String != "embedder" else { return nil }
                return entry["id"] as? String
            } ?? []
        } catch {
            loadedModels = []
            engineDown = error is URLError
        }
        modelsChecked = true
    }

    @MainActor
    private func sendRequest() async {
        defer { isSending = false }
        if activeModel == nil { await refreshModels() }
        if engineDown {
            messages.append(ChatMessage(role: "error", content: "Moteur indisponible — vérifiez le service Telemak"))
            return
        }
        guard let model = activeModel,
              let url = settings.endpointURL?.appendingPathComponent("/v1/chat/completions") else {
            messages.append(ChatMessage(role: "error", content: "Aucun modèle chargé — chargez-en un dans Models"))
            return
        }
        // Only the last 20 messages go to the API to keep the prompt bounded.
        let history = messages.suffix(20).filter { $0.role == "user" || $0.role == "assistant" }
        do {
            let body = try JSONSerialization.data(withJSONObject: [
                "model": model,
                "messages": history.map { ["role": $0.role, "content": $0.content] },
                "stream": false,
                "max_tokens": 512,
            ])
            let (data, response) = try await URLSession.shared.data(
                for: authed(url, method: "POST", body: body, timeout: 120))
            let statusCode = (response as? HTTPURLResponse)?.statusCode ?? -1
            if statusCode >= 400 {
                let detail = String(data: data, encoding: .utf8) ?? ""
                messages.append(ChatMessage(role: "error", content: "Erreur HTTP \(statusCode) : \(detail)"))
                return
            }
            guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let choices = json["choices"] as? [[String: Any]],
                  let message = choices.first?["message"] as? [String: Any] else {
                messages.append(ChatMessage(role: "error", content: "Réponse inattendue : \(String(data: data, encoding: .utf8) ?? "")"))
                return
            }
            messages.append(ChatMessage(role: "assistant", content: displayContent(of: message)))
        } catch {
            if let urlError = error as? URLError, urlError.code == .timedOut {
                messages.append(ChatMessage(role: "error", content: "Délai dépassé — moteur occupé ou génération lente (\(error.localizedDescription))"))
            } else if error is URLError {
                messages.append(ChatMessage(role: "error", content: "Moteur indisponible — vérifiez le service Telemak (\(error.localizedDescription))"))
            } else {
                messages.append(ChatMessage(role: "error", content: "Erreur : \(error.localizedDescription)"))
            }
        }
    }

    /// Display fallback for null content — spontaneous tool call or empty response.
    private func displayContent(of message: [String: Any]) -> String {
        if let content = message["content"] as? String { return content }
        let toolCalls = message["tool_calls"] as? [[String: Any]] ?? []
        guard let name = (toolCalls.first?["function"] as? [String: Any])?["name"] as? String,
              !name.isEmpty else {
            return toolCalls.isEmpty ? "[Réponse vide]" : "[Appel d'outil généré]"
        }
        return "[Appel d'outil : \(name)]"
    }
}
