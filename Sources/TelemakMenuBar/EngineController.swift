import AppKit
import Foundation

/// Owns the local Telemak engine as a child process of the menu-bar app —
/// no launchd. The app starts it on launch and stops it on quit, so the
/// engine's lifetime is exactly the app's lifetime: launch the app, the
/// engine runs; quit the app, the engine stops.
///
/// Crash policy: on an *unexpected* exit, if `relaunchOnCrash` is on, the
/// engine is respawned WITHOUT replaying `~/.telemak/state.json`
/// (`serve --no-replay`) — an empty engine. If a specific model is what
/// crashes the engine, replaying it would crash again immediately and loop;
/// starting empty breaks that loop and hands control back to the operator.
@MainActor
final class EngineController: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var lastEvent: String?

    private var process: Process?
    private var stopping = false
    private let settings: Settings

    init(settings: Settings) {
        self.settings = settings
        // The engine dies with the app: stop it when the app quits so it
        // never orphans as a headless process on :8003.
        NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self else { return }
                self.stopping = true
                self.process?.terminate()
            }
        }
    }

    /// The engine binary bundled inside Telemak.app (Contents/Resources/telemak).
    private var engineURL: URL? {
        Bundle.main.resourceURL?.appendingPathComponent("telemak")
    }

    /// Port taken from the configured local endpoint (default 8003).
    private var port: String {
        if let p = URL(string: settings.endpoint)?.port { return String(p) }
        return "8003"
    }

    /// Start the engine. No-op when pointing at a remote endpoint (nothing
    /// to run locally) or when it is already running. `freshNoReplay` starts
    /// empty (post-crash), skipping the state.json replay.
    func start(freshNoReplay: Bool = false) {
        guard settings.endpointIsLocal else { return }
        guard process == nil else { return }
        guard let engineURL, FileManager.default.isExecutableFile(atPath: engineURL.path) else {
            lastEvent = "engine binary not found in app bundle"
            return
        }
        let p = Process()
        p.executableURL = engineURL
        var args = ["serve", "--host", "0.0.0.0", "--port", port]
        // `freshNoReplay` (post-crash) or the operator opted out of the
        // state.json replay — start the engine empty.
        if freshNoReplay || !settings.reloadLastModelOnStart { args.append("--no-replay") }
        p.arguments = args
        // B2 — engine generation defaults: pass the app's @AppStorage values
        // down as TELEMAK_* env vars so the server (ServerDefaults) uses them
        // as a base under request payloads. The child inherits the parent
        // environment plus these overrides.
        var env = ProcessInfo.processInfo.environment
        env["TELEMAK_DEFAULT_TEMPERATURE"] = String(settings.defaultTemperature)
        env["TELEMAK_DEFAULT_TOP_P"] = String(settings.defaultTopP)
        env["TELEMAK_DEFAULT_TOP_K"] = String(settings.defaultTopK)
        env["TELEMAK_MAX_SESSIONS"] = String(settings.defaultMaxSessions)
        if settings.defaultEnableThinking != 0 {
            env["TELEMAK_DEFAULT_ENABLE_THINKING"] =
                settings.defaultEnableThinking == 1 ? "1" : "0"
        } else {
            env["TELEMAK_DEFAULT_ENABLE_THINKING"] = nil
        }
        p.environment = env
        p.terminationHandler = { [weak self] proc in
            let status = proc.terminationStatus
            Task { @MainActor in self?.handleExit(status: status) }
        }
        do {
            try p.run()
            process = p
            stopping = false
            isRunning = true
            lastEvent = freshNoReplay ? "restarted empty after crash" : "started"
        } catch {
            lastEvent = "spawn failed: \(error.localizedDescription)"
            isRunning = false
        }
    }

    /// Stop the engine (graceful SIGTERM). Marks the exit as intentional so
    /// the crash-relaunch policy does not fire.
    func stop() {
        guard let p = process else { return }
        stopping = true
        p.terminate()
    }

    /// Stop then start again (a normal restart — replays state.json).
    func restart() {
        if process != nil {
            stop()
            DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { [weak self] in
                self?.start()
            }
        } else {
            start()
        }
    }

    private func handleExit(status: Int32) {
        process = nil
        isRunning = false
        if stopping {
            stopping = false
            lastEvent = "stopped"
            return
        }
        // Unexpected exit == crash.
        lastEvent = "engine crashed (exit \(status))"
        if settings.relaunchOnCrash {
            start(freshNoReplay: true)
        }
    }
}
