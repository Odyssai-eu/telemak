import AppKit
import Combine
import Foundation
import SwiftUI
import TelemakVersion

@main
struct TelemakMenuBarApp: App {
    @StateObject private var settings = Settings()
    @StateObject private var poller: HealthPoller

    init() {
        let s = Settings()
        _settings = StateObject(wrappedValue: s)
        _poller = StateObject(wrappedValue: HealthPoller(settings: s))
        let skipInstaller = ProcessInfo.processInfo.environment["TELEMAK_SKIP_INSTALLER"] == "1"
        if !skipInstaller && TelemakInstaller.isInstalled == false {
            DispatchQueue.main.async {
                InstallerWindowController.shared.show()
            }
        }
    }

    var body: some Scene {
        MenuBarExtra {
            MenuBarPopover(poller: poller, settings: settings)
        } label: {
            // Status icon: filled circle when up, dotted when down. Inherits
            // the menu-bar's dynamic color (so it adapts to light/dark mode).
            HStack(spacing: 4) {
                Image(systemName: poller.isUp ? "circle.fill" : "circle.dotted")
                Text(telemakVersion)
            }
        }
        .menuBarExtraStyle(.window)

        // Settings window (opens via the gear button).
        Window("Telemak Settings", id: "telemak-settings") {
            SettingsView(settings: settings)
                .frame(width: 440, height: 220)
        }
        .windowResizability(.contentSize)
    }
}

@MainActor
final class HealthPoller: ObservableObject {
    @Published var status: String = "checking…"
    @Published var runtimeVersion: String?
    @Published var modelsLoaded: [String] = []
    @Published var avgTokPerSec: Double?
    @Published var requestsServed: Int = 0
    @Published var memoryUsedGB: Double = 0
    @Published var memoryFreeGB: Double = 0
    @Published var uptimeSeconds: Double = 0
    @Published var activeRequests: Int = 0
    @Published var currentModel: String?
    @Published var currentRequestStartedAt: String?
    @Published var currentGeneratedTokens: Int = 0
    @Published var currentTokPerSec: Double?
    @Published var currentPhase: String = "idle"
    @Published var runtimeLastError: String?
    @Published var isUp: Bool = false
    @Published var agentState: LaunchAgentControl.AgentState = .notInstalled
    @Published var lastError: String?

    let settings: Settings
    private var timer: Timer?

    init(settings: Settings) {
        self.settings = settings
        // Re-evaluate agent state immediately + start polling.
        refreshAgentState()
        Task { @MainActor in await self.refresh() }
        timer = Timer.scheduledTimer(withTimeInterval: settings.pollInterval, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                await self.refresh()
                self.refreshAgentState()
            }
        }
    }

    // No deinit needed — Timer is invalidated automatically when the
    // RunLoop tears down on app quit, and HealthPoller lives for the app
    // lifetime as a @StateObject so it never deallocates early.

    func refreshAgentState() {
        agentState = LaunchAgentControl.state()
    }

    func refresh() async {
        guard let url = settings.endpointURL?.appendingPathComponent("/health") else {
            isUp = false
            status = "Bad endpoint URL"
            return
        }
        do {
            var req = URLRequest(url: url)
            req.timeoutInterval = 3.0
            let (data, response) = try await URLSession.shared.data(for: req)
            guard let code = (response as? HTTPURLResponse)?.statusCode, code == 200 else {
                isUp = false
                status = "Down (HTTP \((response as? HTTPURLResponse)?.statusCode ?? -1))"
                modelsLoaded = []
                return
            }
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            isUp = true
            status = "Running"
            runtimeVersion = json["version"] as? String
            modelsLoaded = json["models_loaded"] as? [String] ?? []
            avgTokPerSec = json["avg_tok_s_recent"] as? Double
            requestsServed = json["requests_served"] as? Int ?? 0
            memoryUsedGB = json["wired_memory_used_gb"] as? Double ?? 0
            memoryFreeGB = json["wired_memory_free_gb"] as? Double ?? 0
            uptimeSeconds = json["uptime_s"] as? Double ?? 0
            await refreshActivity()
            lastError = nil
        } catch {
            isUp = false
            status = "Unreachable"
            runtimeVersion = nil
            modelsLoaded = []
            clearActivity()
        }
    }

    private func refreshActivity() async {
        guard let url = settings.endpointURL?.appendingPathComponent("/admin/activity") else {
            clearActivity()
            return
        }
        do {
            var req = URLRequest(url: url)
            req.timeoutInterval = 3.0
            let (data, response) = try await URLSession.shared.data(for: req)
            guard let code = (response as? HTTPURLResponse)?.statusCode, code == 200 else {
                clearActivity()
                return
            }
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            activeRequests = json["active_requests"] as? Int ?? 0
            currentModel = json["current_model"] as? String
            currentRequestStartedAt = json["current_request_started_at"] as? String
            currentGeneratedTokens = json["current_generated_tokens"] as? Int ?? 0
            currentTokPerSec = json["current_tok_s"] as? Double
            currentPhase = json["current_phase"] as? String ?? "idle"
            runtimeLastError = json["last_error"] as? String
        } catch {
            clearActivity()
        }
    }

    private func clearActivity() {
        activeRequests = 0
        currentModel = nil
        currentRequestStartedAt = nil
        currentGeneratedTokens = 0
        currentTokPerSec = nil
        currentPhase = "idle"
        runtimeLastError = nil
    }

    // MARK: - Service control

    func startService() {
        do {
            try LaunchAgentControl.start()
            lastError = nil
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(2))
                refreshAgentState()
                await refresh()
            }
        } catch {
            lastError = error.localizedDescription
        }
    }

    func stopService() {
        do {
            try LaunchAgentControl.stop()
            lastError = nil
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(1))
                refreshAgentState()
                await refresh()
            }
        } catch {
            lastError = error.localizedDescription
        }
    }

    func restartService() {
        do {
            try LaunchAgentControl.kickstart()
            lastError = nil
            Task { @MainActor in
                try? await Task.sleep(for: .seconds(3))
                refreshAgentState()
                await refresh()
            }
        } catch {
            lastError = error.localizedDescription
        }
    }
}

struct MenuBarPopover: View {
    @ObservedObject var poller: HealthPoller
    @ObservedObject var settings: Settings
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            header
            Divider()
            modelsSection
            Divider()
            activitySection
            Divider()
            metricsSection
            if let lastError = poller.lastError {
                Divider()
                Text(lastError)
                    .font(.caption)
                    .foregroundColor(.red)
            }
            Divider()
            actions
        }
        .padding(16)
        .frame(width: 360)
    }

    private var header: some View {
        HStack(spacing: 10) {
            Circle()
                .fill(poller.isUp ? Color.green : Color.gray)
                .frame(width: 10, height: 10)
            VStack(alignment: .leading, spacing: 2) {
                Text("Telemak — \(poller.status)")
                    .font(.headline)
                Text(versionLine)
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
                Text(settings.endpoint)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .textSelection(.enabled)
            }
            Spacer()
            if settings.endpointIsLocal {
                Text(poller.agentState.label)
                    .font(.caption2.weight(.semibold))
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(.quaternary))
            }
        }
    }

    private var versionLine: String {
        if let runtimeVersion = poller.runtimeVersion {
            return "Menubar v\(telemakVersion) · Runtime v\(runtimeVersion)"
        }
        return "Menubar v\(telemakVersion)"
    }

    private var modelsSection: some View {
        Group {
            if poller.modelsLoaded.isEmpty {
                Text("No models loaded")
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
                            .lineLimit(1)
                            .truncationMode(.middle)
                    }
                }
            }
        }
    }

    private var metricsSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            metricRow(
                systemImage: "gauge.high",
                label: poller.avgTokPerSec.map { String(format: "%.1f tok/s recent", $0) } ?? "— tok/s",
                muted: poller.avgTokPerSec == nil
            )
            metricRow(
                systemImage: "number",
                label: "\(poller.requestsServed) requests served",
                muted: false
            )
            metricRow(
                systemImage: "memorychip",
                label: String(format: "MLX wired: %.1f GB used / %.1f GB free", poller.memoryUsedGB, poller.memoryFreeGB),
                muted: false
            )
            metricRow(
                systemImage: "clock",
                label: "Uptime: \(formatUptime(poller.uptimeSeconds))",
                muted: !poller.isUp
            )
        }
        .font(.subheadline)
    }

    private var activitySection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Activity")
                .font(.subheadline.weight(.semibold))
            metricRow(
                systemImage: poller.activeRequests > 0 ? "bolt.fill" : "pause",
                label: "\(poller.activeRequests) active · \(poller.currentPhase)",
                muted: poller.activeRequests == 0
            )
            if let model = poller.currentModel {
                metricRow(
                    systemImage: "cpu",
                    label: model,
                    muted: false
                )
            }
            metricRow(
                systemImage: "textformat.123",
                label: "\(poller.currentGeneratedTokens) generated tokens",
                muted: poller.currentGeneratedTokens == 0
            )
            metricRow(
                systemImage: "speedometer",
                label: poller.currentTokPerSec.map { String(format: "%.1f tok/s current", $0) } ?? "— tok/s current",
                muted: poller.currentTokPerSec == nil
            )
            if let started = poller.currentRequestStartedAt {
                metricRow(
                    systemImage: "clock.badge",
                    label: "Started: \(started)",
                    muted: false
                )
            }
            if let runtimeLastError = poller.runtimeLastError {
                metricRow(
                    systemImage: "exclamationmark.triangle",
                    label: runtimeLastError,
                    muted: false
                )
                .foregroundColor(.red)
            }
        }
        .font(.subheadline)
    }

    private func metricRow(systemImage: String, label: String, muted: Bool) -> some View {
        HStack {
            Image(systemName: systemImage)
                .frame(width: 16)
            Text(label)
                .textSelection(.enabled)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .foregroundColor(muted ? .secondary : .primary)
    }

    private func formatUptime(_ seconds: Double) -> String {
        let s = Int(seconds)
        if s < 60 { return "\(s)s" }
        if s < 3600 { return "\(s/60)m \(s%60)s" }
        return "\(s/3600)h \((s%3600)/60)m"
    }

    private var actions: some View {
        VStack(spacing: 8) {
            HStack(spacing: 8) {
                if settings.endpointIsLocal {
                    switch poller.agentState {
                    case .running:
                        Button(action: poller.stopService) {
                            Label("Stop", systemImage: "stop.fill")
                        }
                        Button(action: poller.restartService) {
                            Label("Restart", systemImage: "arrow.clockwise")
                        }
                    case .stopped, .notInstalled:
                        Button(action: poller.startService) {
                            Label("Start", systemImage: "play.fill")
                        }
                        .disabled(poller.agentState == .notInstalled)
                    }
                }
                Spacer()
                Button {
                    Task { @MainActor in await poller.refresh() }
                } label: {
                    Image(systemName: "arrow.triangle.2.circlepath")
                }
                .help("Refresh now")
            }
            HStack(spacing: 8) {
                Button(action: openDashboard) {
                    Label("Dashboard", systemImage: "rectangle.on.rectangle")
                }
                Button {
                    openWindow(id: "telemak-settings")
                } label: {
                    Image(systemName: "gear")
                }
                .help("Settings")
                Button(action: { InstallerWindowController.shared.show() }) {
                    Image(systemName: "arrow.down.app")
                }
                .help("Install or repair local service")
                Spacer()
                Button(action: { NSApplication.shared.terminate(nil) }) {
                    Text("Quit").foregroundColor(.red)
                }
            }
        }
    }

    private func openDashboard() {
        guard let url = settings.dashboardURL else { return }
        NSWorkspace.shared.open(url)
    }
}

struct SettingsView: View {
    @ObservedObject var settings: Settings

    var body: some View {
        Form {
            Section("Telemak endpoint") {
                TextField("URL", text: $settings.endpoint)
                    .textFieldStyle(.roundedBorder)
                Text("e.g. `http://127.0.0.1:8003` (local) or `http://192.168.86.50:8003` (remote target). Start/Stop only works when local.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Section("Dashboard URL") {
                TextField("URL", text: $settings.dashboard)
                    .textFieldStyle(.roundedBorder)
                Text("Opens in your default browser. Default: Odysseus dashboard.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Section("Poll interval") {
                Slider(value: $settings.pollInterval, in: 1...10, step: 1) {
                    Text("seconds")
                }
                Text("\(Int(settings.pollInterval))s between /health polls. Restart the app to apply.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .formStyle(.grouped)
        .padding(16)
    }
}
