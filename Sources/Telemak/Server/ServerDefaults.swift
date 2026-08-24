import Foundation

/// Engine generation defaults injected by the menu-bar app at spawn time
/// (B2). Read once from the process environment; `nil` means "not set —
/// keep the GenerateParameters / template default". Request payload fields
/// always override these values.
///
/// Env contract (set by EngineController.start):
///   TELEMAK_DEFAULT_TEMPERATURE       → temperature
///   TELEMAK_DEFAULT_TOP_P             → topP
///   TELEMAK_DEFAULT_TOP_K             → topK
///   TELEMAK_DEFAULT_ENABLE_THINKING   → enable_thinking ("1"/"true"/"yes" = on)
enum ServerDefaults {
    private static let env = ProcessInfo.processInfo.environment

    static let temperature: Float? = env["TELEMAK_DEFAULT_TEMPERATURE"].flatMap(Float.init)
    static let topP: Float? = env["TELEMAK_DEFAULT_TOP_P"].flatMap(Float.init)
    static let topK: Int? = env["TELEMAK_DEFAULT_TOP_K"].flatMap(Int.init)
    static let enableThinking: Bool? = {
        guard let raw = env["TELEMAK_DEFAULT_ENABLE_THINKING"]?.lowercased() else { return nil }
        return raw == "1" || raw == "true" || raw == "yes"
    }()
}
