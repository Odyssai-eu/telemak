import Foundation
import SwiftUI
import TelemakVersion

@main
struct TelemakMonitorApp: App {
    @StateObject private var store = MonitorStore()

    var body: some Scene {
        WindowGroup("Telemak Monitor") {
            MonitorDashboard(store: store)
                .frame(minWidth: 920, minHeight: 560)
        }
        .windowResizability(.contentMinSize)
    }
}

struct MonitorNode: Identifiable, Codable, Equatable {
    var id: UUID
    var name: String
    var endpoint: String

    init(id: UUID = UUID(), name: String, endpoint: String) {
        self.id = id
        self.name = name
        self.endpoint = endpoint
    }
}

@MainActor
final class MonitorStore: ObservableObject {
    @Published var nodes: [MonitorNode] = [] {
        didSet { save() }
    }

    private let defaultsKey = "telemakMonitorNodes"

    init() {
        load()
    }

    func add(name: String, endpoint: String) {
        let cleanEndpoint = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleanEndpoint.isEmpty else { return }
        let cleanName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        nodes.append(MonitorNode(
            name: cleanName.isEmpty ? suggestedName(for: cleanEndpoint) : cleanName,
            endpoint: cleanEndpoint
        ))
    }

    func remove(_ node: MonitorNode) {
        nodes.removeAll { $0.id == node.id }
    }

    private func load() {
        guard let data = UserDefaults.standard.data(forKey: defaultsKey),
              let decoded = try? JSONDecoder().decode([MonitorNode].self, from: data)
        else {
            nodes = [
                MonitorNode(name: "max-64", endpoint: "http://192.168.86.50:8003"),
            ]
            return
        }
        nodes = decoded
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(nodes) else { return }
        UserDefaults.standard.set(data, forKey: defaultsKey)
    }

    private func suggestedName(for endpoint: String) -> String {
        URL(string: endpoint)?.host ?? "Telemak"
    }
}

struct NodeSnapshot {
    var isUp = false
    var status = "checking"
    var version: String?
    var modelsLoaded: [String] = []
    var requestsServed = 0
    var avgTokS: Double?
    var memoryUsedGB = 0.0
    var memoryFreeGB = 0.0
    var uptimeSeconds = 0.0
    var activeRequests = 0
    var currentModel: String?
    var currentGeneratedTokens = 0
    var currentTokS: Double?
    var currentPhase = "idle"
    var currentElapsedSeconds: Double?
    var lastError: String?
    var refreshedAt: Date?
}

@MainActor
final class NodePoller: ObservableObject {
    @Published private(set) var snapshot = NodeSnapshot()

    let node: MonitorNode
    private var task: Task<Void, Never>?

    init(node: MonitorNode) {
        self.node = node
        start()
    }

    deinit {
        task?.cancel()
    }

    func start() {
        task?.cancel()
        task = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                await self.refresh()
                try? await Task.sleep(for: .seconds(2))
            }
        }
    }

    func refresh() async {
        guard let baseURL = URL(string: node.endpoint) else {
            snapshot.isUp = false
            snapshot.status = "bad URL"
            snapshot.lastError = nil
            snapshot.refreshedAt = Date()
            return
        }

        do {
            var health = URLRequest(url: baseURL.appendingPathComponent("health"))
            health.timeoutInterval = 3
            let (healthData, healthResponse) = try await URLSession.shared.data(for: health)
            guard (healthResponse as? HTTPURLResponse)?.statusCode == 200 else {
                throw URLError(.badServerResponse)
            }
            let healthJSON = try decodeObject(healthData)

            var next = NodeSnapshot()
            next.isUp = true
            next.status = "running"
            next.version = healthJSON["version"] as? String
            next.modelsLoaded = healthJSON["models_loaded"] as? [String] ?? []
            next.requestsServed = healthJSON["requests_served"] as? Int ?? 0
            next.avgTokS = healthJSON["avg_tok_s_recent"] as? Double
            next.memoryUsedGB = healthJSON["wired_memory_used_gb"] as? Double ?? 0
            next.memoryFreeGB = healthJSON["wired_memory_free_gb"] as? Double ?? 0
            next.uptimeSeconds = healthJSON["uptime_s"] as? Double ?? 0
            next.refreshedAt = Date()

            if let activity = try? await fetchActivity(baseURL: baseURL) {
                next.activeRequests = activity["active_requests"] as? Int ?? 0
                next.currentModel = activity["current_model"] as? String
                next.currentGeneratedTokens = activity["current_generated_tokens"] as? Int ?? 0
                next.currentTokS = activity["current_tok_s"] as? Double
                next.currentPhase = activity["current_phase"] as? String ?? "idle"
                next.currentElapsedSeconds = activity["current_elapsed_s"] as? Double
                next.lastError = activity["last_error"] as? String
            }

            snapshot = next
        } catch {
            snapshot = NodeSnapshot(
                isUp: false,
                status: "unreachable",
                lastError: error.localizedDescription,
                refreshedAt: Date()
            )
        }
    }

    private func fetchActivity(baseURL: URL) async throws -> [String: Any] {
        var request = URLRequest(url: baseURL.appendingPathComponent("admin/activity"))
        request.timeoutInterval = 3
        if endpointIsLocal(baseURL), let apiKey = localAPIKey(), !apiKey.isEmpty {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw URLError(.userAuthenticationRequired)
        }
        return try decodeObject(data)
    }

    private func decodeObject(_ data: Data) throws -> [String: Any] {
        try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    }

    private func endpointIsLocal(_ url: URL) -> Bool {
        guard let host = url.host else { return false }
        return host == "127.0.0.1" || host == "localhost" || host == "::1"
    }

    private func localAPIKey() -> String? {
        let keyFile = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("telemak/api-key.txt")
        return (try? String(contentsOf: keyFile, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

struct MonitorDashboard: View {
    @ObservedObject var store: MonitorStore
    @State private var addingNode = false

    private let columns = [
        GridItem(.adaptive(minimum: 300, maximum: 420), spacing: 14, alignment: .top)
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Telemak Monitor")
                        .font(.title2.weight(.semibold))
                    Text("v\(telemakVersion)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Button {
                    addingNode = true
                } label: {
                    Label("Add Telemak", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
            }
            .padding(18)

            Divider()

            if store.nodes.isEmpty {
                ContentUnavailableView("No Telemak nodes", systemImage: "server.rack", description: Text("Add a Telemak endpoint to start monitoring."))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVGrid(columns: columns, alignment: .leading, spacing: 14) {
                        ForEach(store.nodes) { node in
                            NodeCard(node: node) {
                                store.remove(node)
                            }
                        }
                    }
                    .padding(18)
                }
            }
        }
        .sheet(isPresented: $addingNode) {
            AddNodeSheet(store: store)
        }
    }
}

struct NodeCard: View {
    let node: MonitorNode
    let remove: () -> Void

    @StateObject private var poller: NodePoller

    init(node: MonitorNode, remove: @escaping () -> Void) {
        self.node = node
        self.remove = remove
        _poller = StateObject(wrappedValue: NodePoller(node: node))
    }

    var body: some View {
        let snapshot = poller.snapshot

        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 10) {
                Circle()
                    .fill(snapshot.isUp ? .green : .gray)
                    .frame(width: 10, height: 10)
                    .padding(.top, 5)
                VStack(alignment: .leading, spacing: 2) {
                    Text(node.name)
                        .font(.headline)
                        .lineLimit(1)
                    Text(node.endpoint)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                }
                Spacer()
                Menu {
                    Button("Refresh") {
                        Task { await poller.refresh() }
                    }
                    Button("Remove", role: .destructive, action: remove)
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
            }

            HStack(spacing: 8) {
                pill(snapshot.status, color: snapshot.isUp ? .green : .secondary)
                pill(snapshot.currentPhase, color: snapshot.activeRequests > 0 ? .blue : .secondary)
                if let version = snapshot.version {
                    pill("v\(version)", color: .secondary)
                }
            }

            VStack(alignment: .leading, spacing: 7) {
                row("Active", "\(snapshot.activeRequests)")
                row("Current", snapshot.currentModel ?? "—")
                row("Tokens", "\(snapshot.currentGeneratedTokens)")
                row("Current speed", snapshot.currentTokS.map { String(format: "%.1f tok/s", $0) } ?? "—")
                row("Recent speed", snapshot.avgTokS.map { String(format: "%.1f tok/s", $0) } ?? "—")
                row("Requests", "\(snapshot.requestsServed)")
                row("Memory", String(format: "%.1f GB used · %.1f GB free", snapshot.memoryUsedGB, snapshot.memoryFreeGB))
                row("Uptime", formatUptime(snapshot.uptimeSeconds))
            }

            Divider()

            VStack(alignment: .leading, spacing: 5) {
                Text("Loaded models")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                if snapshot.modelsLoaded.isEmpty {
                    Text("—")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(snapshot.modelsLoaded, id: \.self) { model in
                        Text(model)
                            .font(.system(.caption, design: .monospaced))
                            .lineLimit(1)
                            .truncationMode(.middle)
                            .textSelection(.enabled)
                    }
                }
            }

            if let lastError = snapshot.lastError, !lastError.isEmpty {
                Text(lastError)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
    }

    private func row(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .frame(width: 88, alignment: .leading)
            Text(value)
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
            Spacer(minLength: 0)
        }
        .font(.subheadline)
    }

    private func pill(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .foregroundStyle(color)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(Capsule().fill(.quaternary))
    }

    private func formatUptime(_ seconds: Double) -> String {
        let s = Int(seconds)
        if s < 60 { return "\(s)s" }
        if s < 3600 { return "\(s / 60)m \(s % 60)s" }
        return "\(s / 3600)h \((s % 3600) / 60)m"
    }
}

struct AddNodeSheet: View {
    @ObservedObject var store: MonitorStore
    @Environment(\.dismiss) private var dismiss

    @State private var name = ""
    @State private var endpoint = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Add Telemak")
                .font(.headline)

            VStack(alignment: .leading, spacing: 6) {
                Text("Name")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                TextField("ultra-512", text: $name)
                    .textFieldStyle(.roundedBorder)
            }

            VStack(alignment: .leading, spacing: 6) {
                Text("Endpoint")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                TextField("http://192.168.86.50:8003", text: $endpoint)
                    .textFieldStyle(.roundedBorder)
            }

            HStack {
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Add") {
                    store.add(name: name, endpoint: normalizedEndpoint)
                    dismiss()
                }
                .buttonStyle(.borderedProminent)
                .disabled(normalizedEndpoint.isEmpty)
            }
        }
        .padding(18)
        .frame(width: 420)
    }

    private var normalizedEndpoint: String {
        let raw = endpoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !raw.isEmpty else { return "" }
        if raw.hasPrefix("http://") || raw.hasPrefix("https://") {
            return raw
        }
        return "http://\(raw)"
    }
}
