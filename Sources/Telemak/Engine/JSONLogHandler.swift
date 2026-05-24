import Foundation
import Logging

/// `LogHandler` that emits one JSON object per log line, simultaneously to
/// stderr (so launchd captures it) and to a daily-rotating file at
/// `~/.telemak/logs/telemak-YYYY-MM-DD.log`.
///
/// The handler keeps the **last 7 daily files** and prunes anything older
/// on rotation. JSON shape:
///
/// ```json
/// {"ts":"2026-05-24T13:17:42.123Z","level":"info","label":"telemak",
///  "msg":"…","metadata":{"k":"v",…},
///  "source":"telemak","file":"App.swift","function":"…","line":17}
/// ```
///
/// Thread-safe — bootstrap once via `LogConfig.bootstrap(...)` and pass
/// the resulting handler factory to `LoggingSystem.bootstrap`.
public final class JSONLogHandler: LogHandler, @unchecked Sendable {

    public var logLevel: Logger.Level
    public var metadata: Logger.Metadata = [:]

    private let label: String
    private let directory: URL
    private let retainDays: Int
    private let lock = NSLock()
    private var currentDay: String = ""
    private var currentHandle: FileHandle?
    private let stderrFH = FileHandle.standardError

    /// Used by the test in LogConfig to suspend file writes during tests.
    private let writesEnabled: Bool

    public init(
        label: String,
        directory: URL,
        retainDays: Int = 7,
        level: Logger.Level = .info,
        writesEnabled: Bool = true
    ) {
        self.label = label
        self.directory = directory
        self.retainDays = retainDays
        self.logLevel = level
        self.writesEnabled = writesEnabled
        if writesEnabled {
            try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        }
    }

    public subscript(metadataKey key: String) -> Logger.Metadata.Value? {
        get { metadata[key] }
        set { metadata[key] = newValue }
    }

    public func log(
        level: Logger.Level,
        message: Logger.Message,
        metadata explicit: Logger.Metadata?,
        source: String,
        file: String,
        function: String,
        line: UInt
    ) {
        let merged = explicit.map { self.metadata.merging($0, uniquingKeysWith: { _, b in b }) } ?? self.metadata
        let line = formatLine(
            level: level,
            message: message.description,
            metadata: merged,
            source: source,
            file: file,
            function: function,
            lineNumber: line
        )

        lock.lock()
        defer { lock.unlock() }

        if let lineData = (line + "\n").data(using: .utf8) {
            // Always emit to stderr — launchd's StandardErrorPath catches it.
            try? stderrFH.write(contentsOf: lineData)

            // File output — optional (off during unit tests).
            if writesEnabled {
                rotateIfNeededLocked()
                try? currentHandle?.write(contentsOf: lineData)
            }
        }
    }

    // MARK: - Rotation (caller holds lock)

    private func rotateIfNeededLocked() {
        let today = Self.dayString(for: Date())
        if today == currentDay, currentHandle != nil {
            return
        }
        try? currentHandle?.close()
        currentHandle = nil
        currentDay = today

        let url = directory.appendingPathComponent("telemak-\(today).log")
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        currentHandle = try? FileHandle(forWritingTo: url)
        _ = try? currentHandle?.seekToEnd()

        pruneOldFiles()
    }

    private func pruneOldFiles() {
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        ) else { return }
        let dailyLogs = files
            .filter { $0.lastPathComponent.hasPrefix("telemak-") && $0.pathExtension == "log" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        let excess = dailyLogs.count - retainDays
        guard excess > 0 else { return }
        for url in dailyLogs.prefix(excess) {
            try? FileManager.default.removeItem(at: url)
        }
    }

    // MARK: - Formatting

    private func formatLine(
        level: Logger.Level,
        message: String,
        metadata: Logger.Metadata,
        source: String,
        file: String,
        function: String,
        lineNumber: UInt
    ) -> String {
        var payload: [String: Any] = [
            "ts": Self.iso8601(date: Date()),
            "level": level.rawValue,
            "label": label,
            "msg": message,
            "source": source,
            "file": (file as NSString).lastPathComponent,
            "function": function,
            "line": lineNumber,
        ]
        if !metadata.isEmpty {
            var dict: [String: Any] = [:]
            for (key, value) in metadata {
                dict[key] = stringify(value)
            }
            payload["metadata"] = dict
        }
        if let data = try? JSONSerialization.data(withJSONObject: payload, options: [.sortedKeys, .withoutEscapingSlashes]),
           let s = String(data: data, encoding: .utf8) {
            return s
        }
        // Fallback if a metadata value isn't JSON-serializable.
        return "{\"ts\":\"\(Self.iso8601(date: Date()))\",\"level\":\"\(level.rawValue)\",\"label\":\"\(label)\",\"msg\":\"<unserializable>\"}"
    }

    private func stringify(_ value: Logger.Metadata.Value) -> Any {
        switch value {
        case .string(let s): return s
        case .stringConvertible(let s): return s.description
        case .dictionary(let d): return d.mapValues { stringify($0) }
        case .array(let a): return a.map { stringify($0) }
        }
    }

    // MARK: - Helpers

    nonisolated(unsafe) private static let isoFormatter: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        f.timeZone = TimeZone(identifier: "UTC")
        return f
    }()

    private static func iso8601(date: Date) -> String {
        // The formatters are thread-safe for `string(from:)` on Darwin — but
        // Swift 6 strict concurrency can't prove that. We guard externally
        // with the handler's NSLock anyway, so concurrent calls are serialised.
        isoFormatter.string(from: date)
    }

    nonisolated(unsafe) private static let dayFormatter: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        f.timeZone = TimeZone(identifier: "UTC")
        f.locale = Locale(identifier: "en_US_POSIX")
        return f
    }()

    private static func dayString(for date: Date) -> String {
        dayFormatter.string(from: date)
    }
}
