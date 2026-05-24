import AppKit
import Combine
import Foundation
import SwiftUI

@main
struct TelemakMenuBarApp: App {
    @StateObject private var poller = HealthPoller(serverURL: URL(string: "http://127.0.0.1:8003")!)

    var body: some Scene {
        MenuBarExtra("Telemak", systemImage: poller.statusImageName) {
            MenuBarPopover(poller: poller)
        }
        .menuBarExtraStyle(.window)
    }
}

@MainActor
final class HealthPoller: ObservableObject {
    @Published var status: String = "checking…"
    @Published var modelsLoaded: [String] = []
    @Published var avgTokPerSec: Double?
    @Published var requestsServed: Int = 0
    @Published var memoryUsedGB: Double = 0
    @Published var isUp: Bool = false

    let serverURL: URL
    private var timer: Timer?

    var statusImageName: String {
        isUp ? "circle.fill" : "circle.dotted"
    }

    init(serverURL: URL) {
        self.serverURL = serverURL
        startPolling()
    }

    func startPolling() {
        Task { @MainActor in await self.refresh() }
        timer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in await self.refresh() }
        }
    }

    func refresh() async {
        do {
            let (data, response) = try await URLSession.shared.data(from: serverURL.appendingPathComponent("/health"))
            guard let code = (response as? HTTPURLResponse)?.statusCode, code == 200 else {
                isUp = false
                status = "Down (\((response as? HTTPURLResponse)?.statusCode.description ?? "?"))"
                modelsLoaded = []
                return
            }
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            isUp = true
            status = "Running"
            modelsLoaded = json["models_loaded"] as? [String] ?? []
            avgTokPerSec = json["avg_tok_s_recent"] as? Double
            requestsServed = json["requests_served"] as? Int ?? 0
            memoryUsedGB = json["wired_memory_used_gb"] as? Double ?? 0
        } catch {
            isUp = false
            status = "Down (unreachable)"
            modelsLoaded = []
        }
    }
}

struct MenuBarPopover: View {
    @ObservedObject var poller: HealthPoller

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Circle()
                    .fill(poller.isUp ? Color.green : Color.gray)
                    .frame(width: 10, height: 10)
                Text("Telemak — \(poller.status)")
                    .font(.headline)
            }

            Divider()

            if poller.modelsLoaded.isEmpty {
                Text("No models loaded.")
                    .foregroundColor(.secondary)
                    .font(.subheadline)
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Loaded models")
                        .font(.subheadline.weight(.semibold))
                    ForEach(poller.modelsLoaded, id: \.self) { model in
                        Text("• \(model)")
                            .font(.system(.body, design: .monospaced))
                            .textSelection(.enabled)
                    }
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 4) {
                if let tps = poller.avgTokPerSec {
                    Label(String(format: "%.1f tok/s recent", tps), systemImage: "gauge.high")
                } else {
                    Label("— tok/s", systemImage: "gauge.high")
                        .foregroundColor(.secondary)
                }
                Label("\(poller.requestsServed) requests served", systemImage: "number")
                Label(String(format: "%.2f GB MLX memory", poller.memoryUsedGB), systemImage: "memorychip")
            }
            .font(.subheadline)

            Divider()

            HStack {
                Button("Refresh") {
                    Task { @MainActor in await poller.refresh() }
                }
                Spacer()
                Button("Quit") {
                    NSApplication.shared.terminate(nil)
                }
            }
        }
        .padding(16)
        .frame(width: 320)
    }
}
