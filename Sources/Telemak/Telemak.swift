import ArgumentParser

let telemakVersion = "0.1.0"

@main
struct Telemak: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "telemak",
        abstract: "Native macOS HTTP runtime for MLX inference.",
        version: telemakVersion,
        subcommands: [Serve.self, Smoke.self],
        defaultSubcommand: Serve.self
    )
}
