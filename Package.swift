// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "Telemak",
    platforms: [
        .macOS(.v14),
    ],
    products: [
        .library(name: "TelemakVersion", targets: ["TelemakVersion"]),
        .library(name: "TelemakMTP", targets: ["TelemakMTP"]),
        .executable(name: "telemak", targets: ["Telemak"]),
        .executable(name: "telemak-menubar", targets: ["TelemakMenuBar"]),
    ],
    dependencies: [
        // Odyssai-eu fork — adds Qwen35 hidden-state access plus
        // targetVerify / rollbackSpeculativeCache for V2 MTP speculative
        // decoding. Tracks upstream main otherwise.
        .package(url: "https://github.com/Odyssai-eu/mlx-swift-lm", branch: "feat/v2-mtp-ssm-rollback"),
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
        .package(url: "https://github.com/hummingbird-project/hummingbird", from: "2.24.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.5.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .target(name: "TelemakVersion"),
        .target(
            name: "TelemakMTP",
            dependencies: [
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .executableTarget(
            name: "Telemak",
            dependencies: [
                "TelemakVersion",
                "TelemakMTP",
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ]
        ),
        .executableTarget(
            name: "TelemakMenuBar",
            dependencies: ["TelemakVersion"]
        ),
        .testTarget(
            name: "TelemakTests",
            dependencies: [
                "TelemakMTP",
                .product(name: "HummingbirdTesting", package: "hummingbird"),
            ]
        ),
    ]
)
