import Foundation
import CoreGraphics

let maxDisplays: UInt32 = 32
var displayIds = [CGDirectDisplayID](repeating: 0, count: Int(maxDisplays))
var displayCount: UInt32 = 0
let error = CGGetActiveDisplayList(maxDisplays, &displayIds, &displayCount)

if error != .success {
    FileHandle.standardError.write("CGGetActiveDisplayList failed: \(error.rawValue)\n".data(using: .utf8)!)
    exit(1)
}

let displays = displayIds.prefix(Int(displayCount)).map { displayId -> [String: Any] in
    let bounds = CGDisplayBounds(displayId)
    return [
        "cgDisplayId": Int(displayId),
        "left": Double(bounds.origin.x),
        "top": Double(bounds.origin.y),
        "width": Double(bounds.size.width),
        "height": Double(bounds.size.height),
        "primary": CGDisplayIsMain(displayId) != 0
    ]
}.sorted { left, right in
    let leftPrimary = (left["primary"] as? Bool) == true
    let rightPrimary = (right["primary"] as? Bool) == true
    if leftPrimary != rightPrimary {
        return leftPrimary
    }
    let leftX = left["left"] as? Double ?? 0
    let rightX = right["left"] as? Double ?? 0
    if leftX != rightX {
        return leftX < rightX
    }
    let leftY = left["top"] as? Double ?? 0
    let rightY = right["top"] as? Double ?? 0
    return leftY < rightY
}.enumerated().map { index, display -> [String: Any] in
    var item = display
    let displayId = display["cgDisplayId"] as? Int ?? 0
    let primary = (display["primary"] as? Bool) == true
    item["id"] = String(displayId)
    item["sourceId"] = displayId
    item["name"] = primary ? "Main Display" : "Display \(index + 1)"
    return item
}

let data = try JSONSerialization.data(withJSONObject: displays, options: [])
FileHandle.standardOutput.write(data)
FileHandle.standardOutput.write(Data([0x0a]))
