import Foundation
import CoreGraphics
import ImageIO

func loadImage(_ path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        return nil
    }
    return image
}

func checkForArtifacts(_ img: CGImage) {
    guard let data = img.dataProvider?.data else {
        print("No data")
        return
    }

    let ptr = CFDataGetBytePtr(data)!
    let width = img.width
    let height = img.height
    let bytesPerPixel = img.bitsPerPixel / 8

    var blackPixels = 0
    var whitePixels = 0
    var grayPixels = 0
    var normalPixels = 0

    for y in 0..<height {
        for x in 0..<width {
            let offset = (y * width + x) * bytesPerPixel
            let r = Int(ptr[offset])
            let g = Int(ptr[offset + 1])
            let b = Int(ptr[offset + 2])

            let avg = (r + g + b) / 3
            let variance = abs(r - avg) + abs(g - avg) + abs(b - avg)

            if avg < 10 && variance < 5 {
                blackPixels += 1
            } else if avg > 245 && variance < 5 {
                whitePixels += 1
            } else if variance < 5 && avg > 100 && avg < 180 {
                grayPixels += 1
            } else {
                normalPixels += 1
            }
        }
    }

    let total = width * height
    print("Image: \(width)x\(height)")
    print("Black pixels: \(blackPixels) (\(Float(blackPixels)*100/Float(total))%)")
    print("White pixels: \(whitePixels) (\(Float(whitePixels)*100/Float(total))%)")
    print("Gray pixels: \(grayPixels) (\(Float(grayPixels)*100/Float(total))%)")
    print("Normal pixels: \(normalPixels) (\(Float(normalPixels)*100/Float(total))%)")
}

guard CommandLine.arguments.count == 2 else {
    print("Usage: check_artifacts <image>")
    exit(1)
}

let path = CommandLine.arguments[1]
guard let img = loadImage(path) else {
    print("Failed to load image")
    exit(1)
}

checkForArtifacts(img)
