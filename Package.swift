// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "DecorrelationStretch",
    platforms: [.macOS(.v13), .iOS(.v16)],
    products: [
        .library(name: "DecorrelationStretch", targets: ["DecorrelationStretch"]),
    ],
    targets: [
        .target(
            name: "DecorrelationStretch",
            resources: [.process("Shaders")]
        ),
        .testTarget(
            name: "DecorrelationStretchTests",
            dependencies: ["DecorrelationStretch"]
        ),
    ]
)
