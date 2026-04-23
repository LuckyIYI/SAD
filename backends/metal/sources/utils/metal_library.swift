import Metal
import Foundation

func loadMetalLibrary(device: MTLDevice) -> MTLLibrary? {
    if let defaultLib = device.makeDefaultLibrary() {
        return defaultLib
    }

    var candidates = [
        URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
            .appendingPathComponent("sad.metallib")
    ]
    if let executablePath = Bundle.main.executablePath {
        candidates.append(
            URL(fileURLWithPath: executablePath)
                .deletingLastPathComponent()
                .appendingPathComponent("sad.metallib")
        )
    }

    for libPath in candidates {
        do {
            return try device.makeLibrary(URL: libPath)
        } catch {
            continue
        }
    }

    print("Failed to load sad.metallib from current or executable directory")
    return nil
}
