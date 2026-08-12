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
