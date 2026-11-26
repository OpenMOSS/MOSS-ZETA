import Foundation
import CoreGraphics

enum HelperError: Error, CustomStringConvertible {
    case invalidMessage
    case unsupportedCommand(String)
    case unsupportedKey(String)
    case unsupportedButton(String)
    case cursorUnavailable

    var description: String {
        switch self {
        case .invalidMessage:
            return "Invalid JSON message."
        case .unsupportedCommand(let command):
            return "Unsupported input command: \(command)"
        case .unsupportedKey(let key):
            return "Unsupported key: \(key)"
        case .unsupportedButton(let button):
            return "Unsupported mouse button: \(button)"
        case .cursorUnavailable:
            return "Unable to read cursor position."
        }
    }
}

let keyCodes: [String: CGKeyCode] = [
    "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7,
    "c": 8, "v": 9, "b": 11, "q": 12, "w": 13, "e": 14, "r": 15,
    "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22,
    "5": 23, "equal": 24, "9": 25, "7": 26, "minus": 27, "8": 28,
    "0": 29, "rightbracket": 30, "o": 31, "u": 32, "leftbracket": 33,
    "i": 34, "p": 35, "enter": 36, "return": 36, "l": 37, "j": 38,
    "quote": 39, "apostrophe": 39, "k": 40, "semicolon": 41, "backslash": 42,
    "comma": 43, "slash": 44, "n": 45, "m": 46, "period": 47, "tab": 48,
    "space": 49, "backquote": 50, "grave": 50, "backspace": 51, "delete": 117,
    "del": 117, "escape": 53, "esc": 53, "command": 55, "cmd": 55, "meta": 55,
    "shift": 56, "capslock": 57, "alt": 58, "option": 58, "ctrl": 59,
    "control": 59, "rightshift": 60, "rightalt": 61, "rightctrl": 62,
    "f1": 122, "f2": 120, "f3": 99, "f4": 118, "f5": 96, "f6": 97,
    "f7": 98, "f8": 100, "f9": 101, "f10": 109, "f11": 103, "f12": 111,
    "home": 115, "pageup": 116, "pagedown": 121, "end": 119, "insert": 114, "help": 114,
    "left": 123, "right": 124, "down": 125, "up": 126,
    "numpad0": 82, "numpad1": 83, "numpad2": 84, "numpad3": 85, "numpad4": 86,
    "numpad5": 87, "numpad6": 88, "numpad7": 89, "numpad8": 91, "numpad9": 92,
    "multiply": 67, "add": 69, "subtract": 78, "decimal": 65, "divide": 75
]

func writeJson(_ object: [String: Any]) {
    do {
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data([0x0a]))
    } catch {
        FileHandle.standardError.write("JSON write failed: \(error)\n".data(using: .utf8)!)
    }
}

func number(_ value: Any?, fallback: Double = 0) -> Double {
    if let value = value as? NSNumber {
        return value.doubleValue
    }
    if let value = value as? String, let parsed = Double(value) {
        return parsed
    }
    return fallback
}

func string(_ value: Any?, fallback: String = "") -> String {
    if let value = value as? String {
        return value
    }
    return fallback
}

func cursorPosition() throws -> CGPoint {
    guard let event = CGEvent(source: nil) else {
        throw HelperError.cursorUnavailable
    }
    return event.location
}

func mouseButton(_ button: String) throws -> CGMouseButton {
    switch button.lowercased() {
    case "left":
        return .left
    case "right":
        return .right
    case "middle":
        return .center
    default:
        throw HelperError.unsupportedButton(button)
    }
}

func mouseEventType(button: CGMouseButton, down: Bool) -> CGEventType {
    if button == .left {
        return down ? .leftMouseDown : .leftMouseUp
    }
    if button == .right {
        return down ? .rightMouseDown : .rightMouseUp
    }
    return down ? .otherMouseDown : .otherMouseUp
}

func postMouseMove(dx: Double, dy: Double) throws {
    let current = try cursorPosition()
    let target = CGPoint(x: current.x + dx, y: current.y + dy)
    CGEvent(mouseEventSource: nil, mouseType: .mouseMoved, mouseCursorPosition: target, mouseButton: .left)?
        .post(tap: .cghidEventTap)
}

func postMouseButton(button: String, down: Bool) throws {
    let cgButton = try mouseButton(button)
    let current = try cursorPosition()
    CGEvent(mouseEventSource: nil, mouseType: mouseEventType(button: cgButton, down: down), mouseCursorPosition: current, mouseButton: cgButton)?
        .post(tap: .cghidEventTap)
}

func postKey(key: String, down: Bool) throws {
    let normalized = key.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    guard let code = keyCodes[normalized] else {
        throw HelperError.unsupportedKey(key)
    }
    CGEvent(keyboardEventSource: nil, virtualKey: code, keyDown: down)?.post(tap: .cghidEventTap)
}

func postWheel(amount: Double) {
    let clicks = Int32(max(-20, min(20, amount / 120.0)))
    if clicks != 0 {
        CGEvent(scrollWheelEvent2Source: nil, units: .line, wheelCount: 1, wheel1: clicks, wheel2: 0, wheel3: 0)?
            .post(tap: .cghidEventTap)
    }
}

writeJson(["event": "ready", "backend": "macos-swift-coregraphics"])

while let line = readLine() {
    if line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        continue
    }

    var id: Any = NSNull()

    do {
        guard
            let data = line.data(using: .utf8),
            let message = try JSONSerialization.jsonObject(with: data, options: []) as? [String: Any],
            let type = message["type"] as? String
        else {
            throw HelperError.invalidMessage
        }

        id = message["id"] ?? NSNull()

        switch type {
        case "get_cursor":
            let point = try cursorPosition()
            writeJson(["id": id, "ok": true, "x": point.x, "y": point.y])
        case "key_down":
            try postKey(key: string(message["key"]), down: true)
            writeJson(["id": id, "ok": true])
        case "key_up":
            try postKey(key: string(message["key"]), down: false)
            writeJson(["id": id, "ok": true])
        case "mouse_move":
            try postMouseMove(dx: number(message["dx"]), dy: number(message["dy"]))
            writeJson(["id": id, "ok": true])
        case "mouse_down":
            try postMouseButton(button: string(message["button"], fallback: "left"), down: true)
            writeJson(["id": id, "ok": true])
        case "mouse_up":
            try postMouseButton(button: string(message["button"], fallback: "left"), down: false)
            writeJson(["id": id, "ok": true])
        case "wheel":
            postWheel(amount: number(message["amount"]))
            writeJson(["id": id, "ok": true])
        default:
            throw HelperError.unsupportedCommand(type)
        }
    } catch {
        writeJson(["id": id, "ok": false, "error": String(describing: error)])
    }
}
