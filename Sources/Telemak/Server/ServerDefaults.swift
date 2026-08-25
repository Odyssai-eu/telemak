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
///   TELEMAK_DEFAULT_ENABLE_MTP         → speculative MTP path — OPT-IN.
///                                        Only "1"/"true"/"yes" enables it.
///                                        ABSENT or any other value = OFF:
///                                        MTP never routes a request, so the
///                                        service boots safe even with a draft
///                                        paired (the MTP GPU/Metal path has
///                                        deadlocked this stack in the field).
enum ServerDefaults {
    private static let env = ProcessInfo.processInfo.environment

    static let temperature: Float? = env["TELEMAK_DEFAULT_TEMPERATURE"].flatMap(Float.init)
    static let topP: Float? = env["TELEMAK_DEFAULT_TOP_P"].flatMap(Float.init)
    static let topK: Int? = env["TELEMAK_DEFAULT_TOP_K"].flatMap(Int.init)
    static let enableThinking: Bool? = {
        guard let raw = env["TELEMAK_DEFAULT_ENABLE_THINKING"]?.lowercased() else { return nil }
        return raw == "1" || raw == "true" || raw == "yes"
    }()
    /// When true, a model with a paired draft MAY be served by the MTP
    /// iterator. Falses out as soon as the env var is absent or not an
    /// explicit "true" value: `mtpDraftIfEligible` then returns nil and the
    /// regular ChatSession path serves the request. MTP is strictly opt-in.
    static let enableMTP: Bool = {
        guard let raw = env["TELEMAK_DEFAULT_ENABLE_MTP"]?.lowercased() else { return false }
        return raw == "1" || raw == "true" || raw == "yes"
    }()
}
