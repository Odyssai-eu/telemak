import AppKit
import Foundation
import SwiftUI

@MainActor
final class InstallerWindowController {
    static let shared = InstallerWindowController()

    private var window: NSWindow?

    func show() {
        if let window {
            window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let model = InstallerModel()
        let controller = NSHostingController(rootView: InstallerView(model: model))
        let window = NSWindow(contentViewController: controller)
        window.title = "Telemak Installer"
        window.styleMask = [.titled, .closable, .miniaturizable]
        window.setContentSize(NSSize(width: 620, height: 520))
        window.center()
        window.isReleasedWhenClosed = false
        self.window = window
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }
}

@MainActor
final class InstallerModel: ObservableObject {
    @Published var modelsDir: String
    @Published var status: String
    @Published var details: String = ""
    @Published var isInstalling = false
    @Published var installed: Bool
    @Published var smokeOK = false

    init() {
        self.modelsDir = TelemakInstaller.defaultModelsDir().path
        let alreadyInstalled = TelemakInstaller.isInstalled
        self.installed = alreadyInstalled
        self.status = alreadyInstalled ? "Installed" : "Ready to install"
    }

    func chooseModelsDir() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Use this folder"
        panel.directoryURL = URL(fileURLWithPath: modelsDir, isDirectory: true)
        if panel.runModal() == .OK, let url = panel.url {
            modelsDir = url.path
        }
    }

    func install() {
        isInstalling = true
        smokeOK = false
        status = "Installing..."
        details = ""

        Task {
            do {
                let result = try TelemakInstaller.install(modelsDir: URL(fileURLWithPath: modelsDir, isDirectory: true))
                details = result.joined(separator: "\n")
                status = "Waiting for health check..."
                smokeOK = await TelemakInstaller.waitForHealth(timeoutSeconds: 30)
                installed = TelemakInstaller.isInstalled
                status = smokeOK ? "Installed and running" : "Installed, health check pending"
                if !smokeOK {
                    details += "\nHealth did not answer within 30 seconds. Check Full Disk Access and logs in ~/telemak/."
                }
            } catch {
                status = "Install failed"
                details = error.localizedDescription
            }
            isInstalling = false
        }
    }

    func openFullDiskAccess() {
        TelemakInstaller.openSystemSettings(anchor: "Privacy_AllFiles")
    }

    func openRemovableVolumes() {
        TelemakInstaller.openSystemSettings(anchor: "Privacy_RemovableVolumes")
    }
}

struct InstallerView: View {
    @ObservedObject var model: InstallerModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            header
            modelsDirPicker
            permissions
            installControls
            output
            Spacer()
        }
        .padding(24)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Telemak")
                .font(.largeTitle.weight(.semibold))
            Text("Install the local MLX runtime, LaunchAgents, and menu bar monitor for this Mac.")
                .foregroundColor(.secondary)
        }
    }

    private var modelsDirPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Models directory").font(.headline)
            HStack(spacing: 8) {
                TextField("Models directory", text: $model.modelsDir)
                    .textFieldStyle(.roundedBorder)
                Button("Choose...", action: model.chooseModelsDir)
            }
            Text("Default is /Volumes/models/odysseus when present, otherwise ~/Telemak-Models.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }

    private var permissions: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("macOS permissions").font(.headline)
            Text("If models live on external volumes, allow Telemak in Full Disk Access and Removable Volumes, then restart the service from the menu bar.")
                .font(.callout)
                .foregroundColor(.secondary)
            HStack {
                Button("Open Full Disk Access", action: model.openFullDiskAccess)
                Button("Open Removable Volumes", action: model.openRemovableVolumes)
            }
        }
    }

    private var installControls: some View {
        HStack {
            Button {
                model.install()
            } label: {
                Label(model.installed ? "Repair Install" : "Install", systemImage: "arrow.down.app")
            }
            .keyboardShortcut(.defaultAction)
            .disabled(model.isInstalling)

            ProgressView()
                .controlSize(.small)
                .opacity(model.isInstalling ? 1 : 0)

            Spacer()

            Label(model.status, systemImage: model.smokeOK ? "checkmark.circle.fill" : "info.circle")
                .foregroundColor(model.smokeOK ? .green : .secondary)
        }
    }

    private var output: some View {
        ScrollView {
            Text(model.details.isEmpty ? "No install log yet." : model.details)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
        }
        .frame(minHeight: 130)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .textBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.2)))
    }
}

enum TelemakInstaller {
    static let serverLabel = "eu.odyssai.telemak"
    static let menubarLabel = "eu.odyssai.telemak.menubar"

    static var home: URL { FileManager.default.homeDirectoryForCurrentUser }
    static var telemakRoot: URL { home.appendingPathComponent("telemak", isDirectory: true) }
    static var releaseDir: URL { telemakRoot.appendingPathComponent("Release", isDirectory: true) }
    static var launchAgentsDir: URL { home.appendingPathComponent("Library/LaunchAgents", isDirectory: true) }
    static var serverPlist: URL { launchAgentsDir.appendingPathComponent("\(serverLabel).plist") }
    static var menubarPlist: URL { launchAgentsDir.appendingPathComponent("\(menubarLabel).plist") }

    static var isInstalled: Bool {
        FileManager.default.fileExists(atPath: releaseDir.appendingPathComponent("telemak").path)
            && FileManager.default.fileExists(atPath: serverPlist.path)
    }

    static func defaultModelsDir() -> URL {
        let odysseus = URL(fileURLWithPath: "/Volumes/models/odysseus", isDirectory: true)
        if FileManager.default.fileExists(atPath: odysseus.path) {
            return odysseus
        }
        return home.appendingPathComponent("Telemak-Models", isDirectory: true)
    }

    static func install(modelsDir: URL) throws -> [String] {
        let fm = FileManager.default
        var log: [String] = []
        guard Bundle.main.bundleURL.path.hasPrefix("/Applications/") else {
            throw InstallerError.appMustLiveInApplications
        }
        guard let resources = Bundle.main.resourceURL else {
            throw InstallerError.missingResource("bundle resources")
        }

        try fm.createDirectory(at: releaseDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: launchAgentsDir, withIntermediateDirectories: true)
        try fm.createDirectory(at: modelsDir, withIntermediateDirectories: true)

        try copyResource("telemak", from: resources, to: releaseDir.appendingPathComponent("telemak"))
        try chmodExecutable(releaseDir.appendingPathComponent("telemak"))
        log.append("Installed CLI: \(releaseDir.appendingPathComponent("telemak").path)")

        let rawMenubar = resources.appendingPathComponent("telemak-menubar")
        if fm.fileExists(atPath: rawMenubar.path) {
            try copyItemReplacing(rawMenubar, releaseDir.appendingPathComponent("telemak-menubar"))
            try chmodExecutable(releaseDir.appendingPathComponent("telemak-menubar"))
            log.append("Installed menubar helper: \(releaseDir.appendingPathComponent("telemak-menubar").path)")
        }

        let mlxBundle = resources.appendingPathComponent("mlx-swift_Cmlx.bundle", isDirectory: true)
        if fm.fileExists(atPath: mlxBundle.path) {
            try copyItemReplacing(mlxBundle, releaseDir.appendingPathComponent("mlx-swift_Cmlx.bundle", isDirectory: true))
            log.append("Installed MLX Metal bundle.")
        } else {
            throw InstallerError.missingResource("mlx-swift_Cmlx.bundle")
        }

        let appExecutable = Bundle.main.executableURL?.path ?? "/Applications/Telemak.app/Contents/MacOS/Telemak"
        try writeLaunchAgent(
            serverPlist,
            label: serverLabel,
            programArguments: [
                releaseDir.appendingPathComponent("telemak").path,
                "serve",
                "--host",
                "0.0.0.0",
                "--port",
                "8003",
            ],
            workingDirectory: telemakRoot.path,
            environment: [
                "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
                "TELEMAK_MODELS_DIR": modelsDir.path,
                "TELEMAK_LOG_LEVEL": "info",
            ],
            stdout: telemakRoot.appendingPathComponent("launchd.out").path,
            stderr: telemakRoot.appendingPathComponent("launchd.err").path
        )
        log.append("Wrote \(serverPlist.path)")

        try writeLaunchAgent(
            menubarPlist,
            label: menubarLabel,
            programArguments: [appExecutable],
            workingDirectory: telemakRoot.path,
            environment: [:],
            stdout: telemakRoot.appendingPathComponent("menubar.out").path,
            stderr: telemakRoot.appendingPathComponent("menubar.err").path
        )
        log.append("Wrote \(menubarPlist.path)")

        let uid = getuid()
        _ = try? run(["/bin/launchctl", "bootout", "gui/\(uid)/\(serverLabel)"], allowFailure: true)
        _ = try? run(["/bin/launchctl", "bootout", "gui/\(uid)/\(menubarLabel)"], allowFailure: true)
        try run(["/bin/launchctl", "bootstrap", "gui/\(uid)", serverPlist.path])
        try run(["/bin/launchctl", "bootstrap", "gui/\(uid)", menubarPlist.path])
        try run(["/bin/launchctl", "kickstart", "-k", "gui/\(uid)/\(serverLabel)"])
        log.append("LaunchAgents bootstrapped.")
        return log
    }

    static func waitForHealth(timeoutSeconds: Int) async -> Bool {
        guard let url = URL(string: "http://127.0.0.1:8003/health") else { return false }
        let deadline = Date().addingTimeInterval(TimeInterval(timeoutSeconds))
        while Date() < deadline {
            do {
                var request = URLRequest(url: url)
                request.timeoutInterval = 2
                let (_, response) = try await URLSession.shared.data(for: request)
                if (response as? HTTPURLResponse)?.statusCode == 200 {
                    return true
                }
            } catch {
                try? await Task.sleep(for: .seconds(2))
            }
        }
        return false
    }

    static func openSystemSettings(anchor: String) {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)")!
        NSWorkspace.shared.open(url)
    }

    private static func copyResource(_ name: String, from resources: URL, to destination: URL) throws {
        let source = resources.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: source.path) else {
            throw InstallerError.missingResource(name)
        }
        try copyItemReplacing(source, destination)
    }

    private static func copyItemReplacing(_ source: URL, _ destination: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            try fm.removeItem(at: destination)
        }
        try fm.copyItem(at: source, to: destination)
    }

    private static func chmodExecutable(_ url: URL) throws {
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
    }

    private static func writeLaunchAgent(
        _ url: URL,
        label: String,
        programArguments: [String],
        workingDirectory: String,
        environment: [String: String],
        stdout: String,
        stderr: String
    ) throws {
        var plist: [String: Any] = [
            "Label": label,
            "ProgramArguments": programArguments,
            "WorkingDirectory": workingDirectory,
            "RunAtLoad": true,
            "KeepAlive": true,
            "StandardOutPath": stdout,
            "StandardErrorPath": stderr,
        ]
        if !environment.isEmpty {
            plist["EnvironmentVariables"] = environment
        }
        let data = try PropertyListSerialization.data(fromPropertyList: plist, format: .xml, options: 0)
        try data.write(to: url, options: .atomic)
    }

    @discardableResult
    private static func run(_ args: [String], allowFailure: Bool = false) throws -> String {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: args[0])
        process.arguments = Array(args.dropFirst())
        let outPipe = Pipe()
        let errPipe = Pipe()
        process.standardOutput = outPipe
        process.standardError = errPipe
        try process.run()
        process.waitUntilExit()
        let stdout = String(data: outPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let stderr = String(data: errPipe.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        if process.terminationStatus != 0, !allowFailure {
            throw InstallerError.commandFailed(args.joined(separator: " "), stderr)
        }
        return stdout
    }

    enum InstallerError: LocalizedError {
        case appMustLiveInApplications
        case missingResource(String)
        case commandFailed(String, String)

        var errorDescription: String? {
            switch self {
            case .appMustLiveInApplications:
                return "Move Telemak.app to /Applications before installing. LaunchAgents must point to a stable app path."
            case .missingResource(let name):
                return "Missing bundled resource: \(name)"
            case .commandFailed(let command, let stderr):
                return "\(command) failed: \(stderr)"
            }
        }
    }
}
