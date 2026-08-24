import Foundation
import SwiftUI

/// Operator-tunable settings, persisted in UserDefaults.
@MainActor
final class Settings: ObservableObject {
    @AppStorage("endpoint") var endpoint: String = "http://127.0.0.1:8003"
    @AppStorage("dashboard") var dashboard: String = "http://localhost:8000/"
    @AppStorage("pollInterval") var pollInterval: Double = 2.0
    /// When on, a crashed local engine is relaunched empty (see EngineController).
    @AppStorage("relaunchOnCrash") var relaunchOnCrash: Bool = true

    // Engine generation defaults (B2). Injected into the child engine's
    // environment at spawn time by EngineController; the server applies them
    // as a base that request payloads can always override. Defaults match
    // GenerateParameters / SessionStore so behaviour is unchanged until the
    // operator touches them.
    @AppStorage("defaultTemperature") var defaultTemperature: Double = 0.6
    @AppStorage("defaultTopP") var defaultTopP: Double = 1.0
    @AppStorage("defaultTopK") var defaultTopK: Int = 0
    @AppStorage("defaultMaxSessions") var defaultMaxSessions: Int = 32
    /// Tri-state so the template default is never forced:
    /// 0 = template default (not set), 1 = on, 2 = off.
    @AppStorage("defaultEnableThinking") var defaultEnableThinking: Int = 0

    // Lifecycle (B4).
    /// Register the app as a macOS login item via SMAppService. Only
    /// effective when the app bundle lives in /Applications.
    @AppStorage("launchAtLogin") var launchAtLogin: Bool = false
    /// Opt-out for the boot-time state replay: when off, the engine is
    /// spawned with `--no-replay` and starts empty instead of replaying
    /// ~/.telemak/state.json. Applies at the next engine start.
    @AppStorage("reloadLastModelOnStart") var reloadLastModelOnStart: Bool = true

    init() {
        if let endpoint = ProcessInfo.processInfo.environment["TELEMAK_ENDPOINT"], !endpoint.isEmpty {
            self.endpoint = endpoint
        }
    }

    var endpointURL: URL? { URL(string: endpoint) }
    var dashboardURL: URL? { URL(string: dashboard) }

    /// Bearer token for the local server. Read from ~/telemak/api-key.txt
    /// which is written (0600) by the installer on every install.
    var apiKey: String {
        let keyFile = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("telemak/api-key.txt")
        return (try? String(contentsOf: keyFile, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    /// `true` when the endpoint points at the local machine — Start/Stop
    /// buttons only make sense for the local LaunchAgent.
    var endpointIsLocal: Bool {
        guard let host = URL(string: endpoint)?.host else { return false }
        return host == "127.0.0.1" || host == "localhost" || host == "::1"
    }
}
