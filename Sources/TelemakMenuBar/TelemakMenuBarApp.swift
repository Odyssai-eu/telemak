import AppKit
import Combine
import Foundation
import SwiftUI
import TelemakVersion

@main
struct TelemakMenuBarApp: App {
    @StateObject private var settings = Settings()
    @StateObject private var engine: EngineController
    @StateObject private var poller: HealthPoller

    init() {
        // ── Headless provisioning ──────────────────────────────────────
        // `Telemak --provision [--models-dir PATH]` runs the full install
        // (CLI + runtime bundles + LaunchAgents) with NO UI and exits.
        // This is how the OdyssAI Configurator drives Telemak end-to-end:
        // one installer for the whole family — Telemak's own first-run
        // window remains only as the standalone fallback.
        let args = CommandLine.arguments
        if args.contains("--provision") {
            var modelsDir: URL? = TelemakInstaller.defaultModelsDir()
            if let i = args.firstIndex(of: "--models-dir"), i + 1 < args.count {
                modelsDir = URL(fileURLWithPath: args[i + 1], isDirectory: true)
            }
            // The Configurator (OdyssAI-X) must pass --models-dir, or a prior
            // config.json/env must exist. No default — abort otherwise.
            guard let modelsDir, !modelsDir.path.isEmpty else {
                FileHandle.standardError.write(
                    Data("[telemak-provision] ERROR: --models-dir is required when provisioning (no models directory configured)\n".utf8))
                exit(1)
            }
            do {
                let log = try TelemakInstaller.install(modelsDir: modelsDir)
                log.forEach { print("[telemak-provision] \($0)") }
                // Detached: init() runs on the main actor and we park the
                // main thread below — an actor-inherited Task would deadlock.
                Task.detached {
                    let ok = await TelemakInstaller.waitForInstallSmoke(timeoutSeconds: 60)
                    print(ok ? "[telemak-provision] smoke OK — serving on :8003"
                             : "[telemak-provision] WARNING: smoke checks did not pass within 60 s")
                    exit(ok ? 0 : 2)
                }
                DispatchSemaphore(value: 0).wait()   // parked until the task exits the process
            } catch {
                FileHandle.standardError.write(
                    Data("[telemak-provision] ERROR: \(error)\n".utf8))
                exit(1)
            }
        }

        let s = Settings()
        _settings = StateObject(wrappedValue: s)
        let e = EngineController(settings: s)
        _engine = StateObject(wrappedValue: e)
        let p = HealthPoller(settings: s, engine: e)
        _poller = StateObject(wrappedValue: p)
        // The app IS Telemak: launch the app, the engine runs; quit the app,
        // the engine stops (EngineController observes willTerminate). No
        // launchd, no installer — the engine is a child process of the app.
        DispatchQueue.main.async { e.start() }
    }

    var body: some Scene {
        MenuBarExtra {
            MenuBarPopover(poller: poller, settings: settings)
        } label: {
            HStack(spacing: 4) {
                TelemakStatusIcon(isUp: poller.isUp)
                    .frame(width: 16, height: 16)
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
final class MonitorWindowController {
    static let shared = MonitorWindowController()

    private var panel: NSPanel?

    private init() {}

    func show(poller: HealthPoller, settings: Settings) {
        let content = MonitorWindow(poller: poller, settings: settings)
        if panel == nil {
            let panel = NSPanel(
                contentRect: NSRect(x: 0, y: 0, width: 380, height: 310),
                styleMask: [.titled, .closable, .resizable, .utilityWindow, .fullSizeContentView],
                backing: .buffered,
                defer: false
            )
            panel.title = "Telemak Monitor"
            panel.isFloatingPanel = true
            panel.level = .floating
            panel.hidesOnDeactivate = false
            panel.isMovableByWindowBackground = true
            panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            panel.minSize = NSSize(width: 320, height: 240)
            panel.setFrameAutosaveName("TelemakMonitorWindow")
            self.panel = panel
        }

        panel?.contentViewController = NSHostingController(rootView: content)
        panel?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

// MARK: - Models window (picker + directory + local load)

@MainActor
final class ModelsWindowController {
    static let shared = ModelsWindowController()
    private var panel: NSPanel?
    private init() {}

    func show(poller: HealthPoller, settings: Settings) {
        let content = ModelsWindow(poller: poller, settings: settings)
        if panel == nil {
            let panel = NSPanel(
                contentRect: NSRect(x: 0, y: 0, width: 440, height: 440),
                styleMask: [.titled, .closable, .resizable, .utilityWindow, .fullSizeContentView],
                backing: .buffered,
                defer: false
            )
            panel.title = "Telemak Models"
            panel.isFloatingPanel = true
            panel.level = .floating
            panel.hidesOnDeactivate = false
            panel.isMovableByWindowBackground = true
            panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
            panel.minSize = NSSize(width: 400, height: 340)
            panel.setFrameAutosaveName("TelemakModelsWindow")
            self.panel = panel
        }
        panel?.contentViewController = NSHostingController(rootView: content)
        panel?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

@MainActor
final class ModelsWindowModel: ObservableObject {
    @Published var available: [String] = []
    @Published var modelsDirUnset = false
    @Published var status: String = ""
    @Published var dirField: String = ""
    let settings: Settings

    init(settings: Settings) {
        self.settings = settings
        Task { await refresh() }
    }

    private func authed(_ url: URL, method: String = "GET", body: Data? = nil) -> URLRequest {
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.timeoutInterval = 4
        let key = settings.apiKey
        if !key.isEmpty { req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
        if let body {
            req.httpBody = body
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        return req
    }

    func refresh() async {
        guard let base = settings.endpointURL else { return }
        do {
            let (data, _) = try await URLSession.shared.data(
                for: authed(base.appendingPathComponent("/admin/models/available")))
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            available = ((json["models"] as? [[String: Any]])?.compactMap { $0["id"] as? String } ?? []).sorted()
            modelsDirUnset = json["models_dir_unset"] as? Bool ?? false
        } catch {
            status = "Could not list models: \(error.localizedDescription)"
        }
    }

    func load(_ id: String) async {
        guard let base = settings.endpointURL else { return }
        status = "Loading \(id)…"
        let body = try? JSONSerialization.data(withJSONObject: ["model": id])
        do {
            let (_, resp) = try await URLSession.shared.data(
                for: authed(base.appendingPathComponent("/admin/load"), method: "POST", body: body))
            let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
            status = code == 200 ? "Loaded \(id)" : "Load failed (HTTP \(code))"
        } catch {
            status = "Load failed: \(error.localizedDescription)"
        }
    }

    func setDir(create: Bool) async {
        guard let base = settings.endpointURL else { return }
        let path = dirField.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !path.isEmpty else { status = "Enter a directory path"; return }
        let body = try? JSONSerialization.data(
            withJSONObject: ["dir": path, "create": create, "managed": false])
        do {
            let (_, resp) = try await URLSession.shared.data(
                for: authed(base.appendingPathComponent("/admin/models-dir"), method: "POST", body: body))
            let code = (resp as? HTTPURLResponse)?.statusCode ?? -1
            if code == 200 { status = "Models directory set"; await refresh() }
            else if code == 409 { status = "Directory doesn't exist — use “Create & set”." }
            else { status = "Set failed (HTTP \(code))" }
        } catch {
            status = "Set failed: \(error.localizedDescription)"
        }
    }
}

struct ModelsWindow: View {
    @ObservedObject var poller: HealthPoller
    @StateObject private var model: ModelsWindowModel

    init(poller: HealthPoller, settings: Settings) {
        self.poller = poller
        _model = StateObject(wrappedValue: ModelsWindowModel(settings: settings))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Models").font(.headline)

            VStack(alignment: .leading, spacing: 6) {
                Text("Models directory").font(.subheadline.weight(.semibold))
                if poller.modelsDirManaged {
                    HStack {
                        Image(systemName: "lock.fill").foregroundColor(.secondary)
                        Text(poller.modelsDir ?? "—")
                            .font(.system(.body, design: .monospaced)).textSelection(.enabled)
                            .lineLimit(1).truncationMode(.middle)
                        Spacer()
                        Text("managed by OdyssAI-X").font(.caption2).foregroundColor(.secondary)
                    }
                } else {
                    HStack {
                        TextField("/path/to/models", text: $model.dirField)
                            .textFieldStyle(.roundedBorder)
                        Button("Set") { Task { await model.setDir(create: false) } }
                        Button("Create & set") { Task { await model.setDir(create: true) } }
                    }
                    Text("Standalone — set where models live on this Mac.")
                        .font(.caption).foregroundColor(.secondary)
                }
            }

            Divider()

            Text("Available").font(.subheadline.weight(.semibold))
            if model.modelsDirUnset {
                Text("Set your models directory above to see available models.")
                    .font(.callout).foregroundColor(.orange)
            } else if model.available.isEmpty {
                Text("No models found under the directory.")
                    .font(.callout).foregroundColor(.secondary)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(model.available, id: \.self) { id in
                            HStack {
                                Text(id).font(.system(.caption, design: .monospaced))
                                    .lineLimit(1).truncationMode(.middle)
                                Spacer()
                                if poller.modelsLoaded.contains(id) {
                                    Text("loaded").font(.caption2).foregroundColor(.green)
                                } else {
                                    Button("Load") { Task { await model.load(id) } }
                                        .controlSize(.small)
                                }
                            }
                        }
                    }
                }
                .frame(maxHeight: 180)
            }

            if !model.status.isEmpty {
                Text(model.status).font(.caption).foregroundColor(.secondary).textSelection(.enabled)
            }
            Spacer(minLength: 0)
            Button("Refresh") { Task { await model.refresh() } }
        }
        .padding(16)
        .frame(minWidth: 400, minHeight: 340)
        .onAppear { if model.dirField.isEmpty { model.dirField = poller.modelsDir ?? "" } }
    }
}

private struct TelemakStatusIcon: View {
    let isUp: Bool

    var body: some View {
        TelemakGlyph()
            .fill(.primary)
            .opacity(isUp ? 1.0 : 0.38)
            .accessibilityLabel(isUp ? "Telemak running" : "Telemak unreachable")
    }
}

private struct TelemakGlyph: Shape {
    func path(in rect: CGRect) -> Path {
        let scale = min(rect.width, rect.height) / 24.0
        let xOffset = rect.midX - 12.0 * scale
        let yOffset = rect.midY - 12.0 * scale

        func point(_ x: CGFloat, _ y: CGFloat) -> CGPoint {
            CGPoint(x: xOffset + x * scale, y: yOffset + y * scale)
        }

        func scaledRect(x: CGFloat, y: CGFloat, width: CGFloat, height: CGFloat) -> CGRect {
            CGRect(
                x: xOffset + x * scale,
                y: yOffset + y * scale,
                width: width * scale,
                height: height * scale
            )
        }

        var path = Path()

        path.move(to: point(19.94, 14.24))
        [
            (18.18, 14.24), (18.18, 12.36), (18.18, 10.49), (19.94, 10.49),
            (19.94, 8.61), (19.94, 6.74), (18.18, 6.74), (18.18, 4.86),
            (16.41, 4.86), (16.41, 2.99), (14.65, 2.99), (14.65, 1.11),
            (12.88, 1.11), (12.88, 2.99), (12.88, 4.86), (12.88, 4.86),
            (12.88, 2.99), (12.88, 1.11), (11.12, 1.11), (11.12, 2.99),
            (11.12, 4.86), (11.12, 4.86), (11.12, 2.99), (11.12, 1.11),
            (9.35, 1.11), (9.35, 2.99), (7.59, 2.99), (7.59, 4.86),
            (5.82, 4.86), (5.82, 6.74), (4.06, 6.74), (4.06, 8.61),
            (4.06, 10.49), (5.82, 10.49), (5.82, 12.36), (5.82, 14.24),
            (4.06, 14.24), (3.94, 14.24), (3.94, 16.11), (4.06, 16.11),
            (5.82, 16.11), (7.59, 16.11), (7.59, 14.24), (7.59, 8.53),
            (9.35, 8.53), (11.12, 8.53), (12.88, 8.53), (14.65, 8.53),
            (16.41, 8.53), (16.41, 14.24), (16.41, 16.11), (18.18, 16.11),
            (19.94, 16.11), (20.23, 16.11), (20.23, 14.24), (19.94, 14.24),
        ].forEach { path.addLine(to: point($0.0, $0.1)) }
        path.closeSubpath()

        [
            (5.84, 16.11, 1.76, 1.88),
            (9.34, 19.48, 1.76, 1.88),
            (7.60, 17.75, 1.76, 1.88),
            (11.12, 21.36, 1.76, 1.88),
            (16.45, 16.11, 1.76, 1.88),
            (12.89, 19.48, 1.76, 1.88),
            (14.69, 17.69, 1.76, 1.88),
            (11.12, 12.29, 1.76, 1.88),
            (11.12, 14.14, 1.76, 1.88),
            (11.12, 17.69, 1.76, 1.88),
            (14.69, 16.11, 1.76, 1.88),
            (7.60, 16.11, 1.76, 1.88),
            (8.49, 10.41, 1.76, 1.88),
            (13.80, 10.41, 1.76, 1.88),
        ].forEach { path.addRect(scaledRect(x: $0.0, y: $0.1, width: $0.2, height: $0.3)) }

        return path
    }
}

struct MonitorWindow: View {
    @ObservedObject var poller: HealthPoller
    @ObservedObject var settings: Settings

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 10) {
                TelemakStatusIcon(isUp: poller.isUp)
                    .frame(width: 22, height: 22)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Telemak")
                        .font(.headline)
                    Text(versionLine)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
                Spacer()
                Text(poller.currentPhase)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(poller.activeRequests > 0 ? .green : .secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Capsule().fill(.quaternary))
            }

            VStack(alignment: .leading, spacing: 8) {
                monitorRow("Endpoint", settings.endpoint)
                monitorRow("Status", poller.status)
                monitorRow("Active", "\(poller.activeRequests)")
                monitorRow("Current", poller.currentModel ?? "—")
                monitorRow("Tokens", "\(poller.currentGeneratedTokens)")
                monitorRow("Current speed", poller.currentTokPerSec.map { String(format: "%.1f tok/s", $0) } ?? "—")
                monitorRow("Recent speed", poller.avgTokPerSec.map { String(format: "%.1f tok/s", $0) } ?? "—")
                monitorRow("Requests", "\(poller.requestsServed)")
                monitorRow("Memory", String(format: "%.1f GB used · %.1f GB free", poller.memoryUsedGB, poller.memoryFreeGB))
                monitorRow("Uptime", formatUptime(poller.uptimeSeconds))
            }

            if let runtimeLastError = poller.runtimeLastError {
                Text(runtimeLastError)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .textSelection(.enabled)
            }

            Spacer(minLength: 0)
        }
        .padding(16)
        .frame(minWidth: 320, minHeight: 240)
        .background(.regularMaterial)
    }

    private var versionLine: String {
        if let runtimeVersion = poller.runtimeVersion {
            return "Menubar v\(telemakVersion) · Runtime v\(runtimeVersion)"
        }
        return "Menubar v\(telemakVersion)"
    }

    private func monitorRow(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(label)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
                .frame(width: 86, alignment: .leading)
            Text(value)
                .font(.system(.body, design: label == "Current" ? .monospaced : .default))
                .lineLimit(1)
                .truncationMode(.middle)
                .textSelection(.enabled)
            Spacer(minLength: 0)
        }
    }

    private func formatUptime(_ seconds: Double) -> String {
        let s = Int(seconds)
        if s < 60 { return "\(s)s" }
        if s < 3600 { return "\(s / 60)m \(s % 60)s" }
        return "\(s / 3600)h \((s % 3600) / 60)m"
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
    @Published var currentElapsedSeconds: Double?
    @Published var currentGeneratedTokens: Int = 0
    @Published var currentTokPerSec: Double?
    @Published var currentPhase: String = "idle"
    @Published var runtimeLastError: String?
    @Published var isUp: Bool = false
    @Published var agentState: LaunchAgentControl.AgentState = .notInstalled
    @Published var lastError: String?
    // Effective models directory (from GET /admin/models-dir). `managed` ⇒
    // OdyssAI-X owns it (read-only in the UI); else standalone (editable).
    @Published var modelsDir: String?
    @Published var modelsDirSource: String = "unset"
    @Published var modelsDirManaged: Bool = false

    let settings: Settings
    let engine: EngineController
    private var timer: Timer?

    init(settings: Settings, engine: EngineController) {
        self.settings = settings
        self.engine = engine
        // Re-evaluate engine state immediately + start polling.
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
        agentState = engine.isRunning ? .running : .stopped
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
            await refreshModelsDir()
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
            let key = settings.apiKey
            if !key.isEmpty {
                req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            }
            let (data, response) = try await URLSession.shared.data(for: req)
            guard let code = (response as? HTTPURLResponse)?.statusCode, code == 200 else {
                clearActivity()
                return
            }
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            activeRequests = json["active_requests"] as? Int ?? 0
            currentModel = json["current_model"] as? String
            currentRequestStartedAt = json["current_request_started_at"] as? String
            currentElapsedSeconds = json["current_elapsed_s"] as? Double
            currentGeneratedTokens = json["current_generated_tokens"] as? Int ?? 0
            currentTokPerSec = json["current_tok_s"] as? Double
            currentPhase = json["current_phase"] as? String ?? "idle"
            runtimeLastError = json["last_error"] as? String
        } catch {
            clearActivity()
        }
    }

    /// Poll the effective models directory (bearer-protected admin route).
    private func refreshModelsDir() async {
        guard let url = settings.endpointURL?.appendingPathComponent("/admin/models-dir") else { return }
        do {
            var req = URLRequest(url: url)
            req.timeoutInterval = 3.0
            let key = settings.apiKey
            if !key.isEmpty {
                req.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
            }
            let (data, response) = try await URLSession.shared.data(for: req)
            guard let code = (response as? HTTPURLResponse)?.statusCode, code == 200 else { return }
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
            modelsDir = json["dir"] as? String
            modelsDirSource = json["source"] as? String ?? "unset"
            modelsDirManaged = json["managed"] as? Bool ?? false
        } catch {
            // Keep last-known values on a transient failure.
        }
    }

    private func clearActivity() {
        activeRequests = 0
        currentModel = nil
        currentRequestStartedAt = nil
        currentElapsedSeconds = nil
        currentGeneratedTokens = 0
        currentTokPerSec = nil
        currentPhase = "idle"
        runtimeLastError = nil
    }

    // MARK: - Service control

    func startService() {
        engine.start()
        lastError = nil
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(2))
            refreshAgentState()
            await refresh()
        }
    }

    func stopService() {
        engine.stop()
        lastError = nil
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(1))
            refreshAgentState()
            await refresh()
        }
    }

    func restartService() {
        engine.restart()
        lastError = nil
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(3))
            refreshAgentState()
            await refresh()
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
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 6) {
                Image(systemName: "folder").frame(width: 16)
                Text(poller.modelsDir ?? "no models directory set")
                    .font(.caption.monospaced())
                    .foregroundColor(poller.modelsDir == nil ? .orange : .secondary)
                    .lineLimit(1).truncationMode(.middle)
                Spacer()
                Text(poller.modelsDirManaged ? "OdyssAI-X" : (poller.modelsDir == nil ? "unset" : "local"))
                    .font(.caption2.weight(.semibold))
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 5).padding(.vertical, 1)
                    .background(Capsule().fill(.quaternary))
            }
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
            if let elapsed = poller.currentElapsedSeconds {
                metricRow(
                    systemImage: "timer",
                    label: String(format: "%.0fs elapsed", elapsed),
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
            if settings.endpointIsLocal {
                Toggle(isOn: $settings.relaunchOnCrash) {
                    Text("Relaunch engine on crash (empty)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .toggleStyle(.switch)
                .controlSize(.mini)
                .help("If the engine crashes, restart it without reloading models — so a model that crashes it can't loop.")
            }
            HStack(spacing: 8) {
                Button {
                    ModelsWindowController.shared.show(poller: poller, settings: settings)
                } label: {
                    Label("Models", systemImage: "shippingbox")
                }
                Button {
                    MonitorWindowController.shared.show(poller: poller, settings: settings)
                } label: {
                    Label("Monitor", systemImage: "macwindow")
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
                Text("e.g. `http://127.0.0.1:8003` (local) or `http://<host>:8003` (remote target). Start/Stop only works when local.")
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
