import ArgumentParser
import TelemakVersion

@main
struct Telemak: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "telemak",
        abstract: "Native macOS HTTP runtime for MLX inference.",
        version: telemakVersion,
        subcommands: [
            Serve.self,
            Smoke.self,
            ModelsCommand.self,
            LoadCommand.self,
            UnloadCommand.self,
            ChatCommand.self,
        ],
        defaultSubcommand: Serve.self
    )
}
