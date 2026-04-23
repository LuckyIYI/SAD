import Foundation
import CoreGraphics
import ImageIO

func loadImage(_ path: String) -> CGImage? {
    let url = URL(fileURLWithPath: path)
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        print("Failed to load: \(path)")
        return nil
    }
    return image
}

func computePSNR(_ img1: CGImage, _ img2: CGImage) -> Float {
    guard img1.width == img2.width && img1.height == img2.height else {
        print("Images have different sizes!")
        return 0
    }

    let width = img1.width
    let height = img1.height

    // Create pixel buffers
    guard let data1 = img1.dataProvider?.data,
          let data2 = img2.dataProvider?.data else {
        return 0
    }

    let ptr1 = CFDataGetBytePtr(data1)
    let ptr2 = CFDataGetBytePtr(data2)

    var mse: Float = 0.0
    let bytesPerPixel = img1.bitsPerPixel / 8

    for y in 0..<height {
        for x in 0..<width {
            let offset = (y * width + x) * bytesPerPixel

            // Read RGB values (assuming RGBA format)
            let r1 = Float(ptr1![offset]) / 255.0
            let g1 = Float(ptr1![offset + 1]) / 255.0
            let b1 = Float(ptr1![offset + 2]) / 255.0

            let r2 = Float(ptr2![offset]) / 255.0
            let g2 = Float(ptr2![offset + 1]) / 255.0
            let b2 = Float(ptr2![offset + 2]) / 255.0

            let dr = r1 - r2
            let dg = g1 - g2
            let db = b1 - b2

            mse += (dr * dr + dg * dg + db * db) / 3.0
        }
    }

    mse /= Float(width * height)

    if mse < 1e-10 {
        return 100.0
    }

    return 20.0 * log10(1.0 / sqrt(mse))
}

guard CommandLine.arguments.count == 3 else {
    print("Usage: verify_psnr <image1> <image2>")
    exit(1)
}

let path1 = CommandLine.arguments[1]
let path2 = CommandLine.arguments[2]

guard let img1 = loadImage(path1),
      let img2 = loadImage(path2) else {
    print("Failed to load images")
    exit(1)
}

let psnr = computePSNR(img1, img2)
print(String(format: "PSNR: %.2f dB", psnr))
