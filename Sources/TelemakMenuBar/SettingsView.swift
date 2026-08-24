import SwiftUI

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
            engineDefaultsSection
        }
        .formStyle(.grouped)
        .padding(16)
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
}
