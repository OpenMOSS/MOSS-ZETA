import Foundation
import CoreGraphics
import ImageIO

func fail(_ message: String) -> Never {
    FileHandle.standardError.write("\(message)\n".data(using: .utf8)!)
    exit(1)
}

let args = Array(CommandLine.arguments.dropFirst())
if args.count != 2 {
    fail("Usage: macos-capture.swift <cgDisplayId> <outputPath>")
}

guard let displayIdValue = UInt32(args[0]) else {
    fail("Invalid CGDirectDisplayID: \(args[0])")
}

let displayId = CGDirectDisplayID(displayIdValue)
let outputUrl = URL(fileURLWithPath: args[1])

guard let image = CGDisplayCreateImage(displayId) else {
    fail("CGDisplayCreateImage failed for display \(displayIdValue). Check Screen Recording permission.")
}

guard let destination = CGImageDestinationCreateWithURL(outputUrl as CFURL, "public.png" as CFString, 1, nil) else {
    fail("Could not create PNG destination: \(args[1])")
}

CGImageDestinationAddImage(destination, image, nil)

if !CGImageDestinationFinalize(destination) {
    fail("Could not write PNG: \(args[1])")
}
