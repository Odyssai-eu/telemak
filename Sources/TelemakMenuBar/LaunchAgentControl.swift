import Foundation

/// Wraps `launchctl` calls to control the local Telemak LaunchAgent.
///
/// Affects ONLY the machine the menu-bar app is running on — no SSH.
/// When pointing at a remote endpoint (e.g. 192.168.86.50:8003), the
/// Start/Stop buttons are disabled.
enum LaunchAgentControl {
    static var label: String {
        let value = ProcessInfo.processInfo.environment["TELEMAK_AGENT_LABEL"] ?? ""
        return value.isEmpty ? "eu.odyssai.telemak" : value
    }
    static var plistURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/LaunchAgents/\(label).plist")
    }

    enum AgentState {
        case running
        case stopped
        case notInstalled

        var label: String {
            switch self {
            case .running: return "Running"
            case .stopped: return "Stopped"
            case .notInstalled: return "Not installed"
            }
        }
    }

    /// Check whether the LaunchAgent is loaded by querying launchctl print.
    /// Returns `.notInstalled` if the plist doesn't exist.
    static func state() -> AgentState {
        if !FileManager.default.fileExists(atPath: plistURL.path) {
            return .notInstalled
        }
        let uid = getuid()
        let result = run(["/bin/launchctl", "print", "gui/\(uid)/\(label)"])
        return result.exitCode == 0 ? .running : .stopped
    }

    /// `launchctl bootstrap gui/<uid> <plist>` — load + start.
    static func start() throws {
        let uid = getuid()
        let result = run(["/bin/launchctl", "bootstrap", "gui/\(uid)", plistURL.path])
        if result.exitCode != 0 {
            throw ControlError.launchctlFailed(verb: "bootstrap", stderr: result.stderr)
        }
    }

    /// `launchctl bootout gui/<uid> <plist>` — stop + unload.
    static func stop() throws {
        let uid = getuid()
        let result = run(["/bin/launchctl", "bootout", "gui/\(uid)", plistURL.path])
        if result.exitCode != 0 {
            throw ControlError.launchctlFailed(verb: "bootout", stderr: result.stderr)
        }
    }

    /// `launchctl kickstart -k gui/<uid>/<label>` — restart in place.
    static func kickstart() throws {
        let uid = getuid()
        let result = run(["/bin/launchctl", "kickstart", "-k", "gui/\(uid)/\(label)"])
        if result.exitCode != 0 {
            throw ControlError.launchctlFailed(verb: "kickstart", stderr: result.stderr)
        }
    }

    // MARK: - internals

    enum ControlError: LocalizedError {
        case launchctlFailed(verb: String, stderr: String)

        var errorDescription: String? {
            switch self {
            case .launchctlFailed(let verb, let stderr):
                return "launchctl \(verb) failed: \(stderr)"
            }
        }
    }

    private static func run(_ args: [String]) -> (exitCode: Int32, stdout: String, stderr: String) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: args[0])
        process.arguments = Array(args.dropFirst())
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return (-1, "", "spawn failed: \(error)")
        }
        let stdout = String(data: outPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let stderr = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        return (process.terminationStatus, stdout, stderr)
    }
}
