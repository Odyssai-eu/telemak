import Foundation
import SwiftUI

/// Operator-tunable settings, persisted in UserDefaults.
@MainActor
final class Settings: ObservableObject {
    @AppStorage("endpoint") var endpoint: String = "http://127.0.0.1:8003"
    @AppStorage("dashboard") var dashboard: String = "http://192.168.86.141:8000/"
    @AppStorage("pollInterval") var pollInterval: Double = 2.0

    var endpointURL: URL? { URL(string: endpoint) }
    var dashboardURL: URL? { URL(string: dashboard) }

    /// `true` when the endpoint points at the local machine — Start/Stop
    /// buttons only make sense for the local LaunchAgent.
    var endpointIsLocal: Bool {
        guard let host = URL(string: endpoint)?.host else { return false }
        return host == "127.0.0.1" || host == "localhost" || host == "::1"
    }
}
