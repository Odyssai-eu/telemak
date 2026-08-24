import ServiceManagement
import SwiftUI

struct SettingsView: View {
    @ObservedObject var settings: Settings
    @State private var loginItemStatus: SMAppService.Status = .notRegistered

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
            engineDefaultsSection
            lifecycleSection
        }
        .formStyle(.grouped)
        .padding(16)
        .onAppear { reconcileLoginItem() }
    }

    /// Reconcile the stored toggle with the actual SMAppService status:
    /// the system may lose or revoke the registration (OS update, app
    /// reinstall) without the app seeing a toggle flip.
    private func reconcileLoginItem() {
        loginItemStatus = SMAppService.mainApp.status
        switch loginItemStatus {
        case .enabled, .requiresApproval:
            settings.launchAtLogin = true
        default:
            settings.launchAtLogin = false
        }
    }

    // B2 — engine generation defaults. Injected into the local engine's
    // environment at spawn time (EngineController); payloads always win.
    private var engineDefaultsSection: some View {
        Section("Engine defaults") {
            VStack(alignment: .leading, spacing: 4) {
                Slider(value: $settings.defaultTemperature, in: 0...2, step: 0.05) {
                    Text("Temperature")
                }
                Text(String(format: "temperature: %.2f", settings.defaultTemperature))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            VStack(alignment: .leading, spacing: 4) {
                Slider(value: $settings.defaultTopP, in: 0...1, step: 0.05) {
                    Text("Top P")
                }
                Text(String(format: "top_p: %.2f", settings.defaultTopP))
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Stepper(value: $settings.defaultTopK, in: 0...200) {
                Text("top_k: \(settings.defaultTopK == 0 ? "disabled" : String(settings.defaultTopK))")
            }
            Stepper(value: $settings.defaultMaxSessions, in: 1...128) {
                Text("Max sessions: \(settings.defaultMaxSessions)")
            }
            Picker("enable_thinking", selection: $settings.defaultEnableThinking) {
                Text("Template default").tag(0)
                Text("On").tag(1)
                Text("Off").tag(2)
            }
            Text("Applied to the local engine at spawn. Restart the engine to apply.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    // B4 — app + engine lifecycle.
    private var lifecycleSection: some View {
        Section("Lifecycle") {
            VStack(alignment: .leading, spacing: 4) {
                Toggle("Launch Telemak at login", isOn: launchAtLoginBinding)
                    .disabled(!appIsInApplications)
                Text(launchAtLoginCaption)
                    .font(.caption)
                    .foregroundColor(.secondary)
                if loginItemStatus == .requiresApproval {
                    Button("Open System Settings → Login Items") {
                        SMAppService.openSystemSettingsLoginItems()
                    }
                    .buttonStyle(.link)
                    .font(.caption)
                }
            }
            VStack(alignment: .leading, spacing: 4) {
                Toggle("Reload last model on start", isOn: $settings.reloadLastModelOnStart)
                Text("Server reads ~/.telemak/state.json at boot and replays model loads. Disable to start empty. Applies at the next engine start.")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            VStack(alignment: .leading, spacing: 4) {
                Toggle("Relaunch engine on crash", isOn: $settings.relaunchOnCrash)
                Text("Respawn the engine without replaying state.json after an unexpected exit (breaks crash loops).")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
    }

    // SMAppService registers the login item from the running bundle's path;
    // outside /Applications (dev builds) the registration has no durable
    // effect, so surface it instead of a broken toggle. Accept both
    // /Applications and ~/Applications; reject App Translocation paths.
    private var appIsInApplications: Bool {
        let bundlePath = Bundle.main.bundleURL.path
        let homeApplications = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Applications").path
        return bundlePath.hasPrefix("/Applications/")
            || bundlePath.hasPrefix(homeApplications + "/")
    }

    private var launchAtLoginCaption: String {
        if !appIsInApplications {
            return "Requires the app in /Applications or ~/Applications — move Telemak.app there first."
        }
        switch loginItemStatus {
        case .requiresApproval:
            return "Approval required in System Settings → Login Items."
        case .enabled:
            return "Registers the app as a macOS login item."
        case .notRegistered:
            return "Not registered as a login item."
        case .notFound:
            return "Login item not found (may have been removed)."
        @unknown default:
            return "Unknown login item status."
        }
    }

    private var launchAtLoginBinding: Binding<Bool> {
        Binding(
            get: { settings.launchAtLogin },
            set: { enabled in
                if enabled {
                    try? SMAppService.mainApp.register()
                } else {
                    try? SMAppService.mainApp.unregister()
                }
                loginItemStatus = SMAppService.mainApp.status
                switch loginItemStatus {
                case .enabled, .requiresApproval:
                    settings.launchAtLogin = true
                default:
                    settings.launchAtLogin = false
                }
            }
        )
    }
}
