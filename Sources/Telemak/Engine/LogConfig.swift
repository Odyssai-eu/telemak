import Foundation
import Logging

/// Centralised LoggingSystem bootstrap. Call exactly once, before any
/// `Logger(...)` is created.
public enum LogConfig {

    /// Read `TELEMAK_LOG_LEVEL` from the environment (defaults to `info`)
    /// and install a `JSONLogHandler` for every Logger instance.
    ///
    /// File output: `~/.telemak/logs/telemak-YYYY-MM-DD.log`, retains the
    /// last 7 daily files.
    public static func bootstrap() {
        let level = parseLevel(ProcessInfo.processInfo.environment["TELEMAK_LOG_LEVEL"]) ?? .info
        let directory = FileManager.default
            .homeDirectoryForCurrentUser
            .appendingPathComponent(".telemak/logs")

        LoggingSystem.bootstrap { label in
            JSONLogHandler(
                label: label,
                directory: directory,
                retainDays: 7,
                level: level
            )
        }
    }

    private static func parseLevel(_ raw: String?) -> Logger.Level? {
        guard let raw = raw?.lowercased(), !raw.isEmpty else { return nil }
        return Logger.Level(rawValue: raw)
    }
}
