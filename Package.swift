// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "MLXStudio",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "MLXStudioApp", targets: ["MLXStudioApp"])
    ],
    targets: [
        .executableTarget(
            name: "MLXStudioApp",
            resources: [
                .copy("Resources/models.json")
            ]
        ),
        .testTarget(
            name: "MLXStudioAppTests",
            dependencies: ["MLXStudioApp"]
        )
    ]
)
