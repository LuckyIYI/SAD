import Foundation
import Metal
import MetalPerformanceShaders
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

// Load image from file (downscale with MPS if needed).
func loadImage(path: String, maxDimension: Int, device: MTLDevice) -> (CGImage, Int, Int)? {
    let url = URL(fileURLWithPath: path)
    guard let imageSource = CGImageSourceCreateWithURL(url as CFURL, nil),
          let originalImage = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
        return nil
    }

    let originalWidth = originalImage.width
    let originalHeight = originalImage.height
    let maxDim = max(originalWidth, originalHeight)
    if maxDim <= maxDimension {
        return (originalImage, originalWidth, originalHeight)
    }

    let scale = Float(maxDimension) / Float(maxDim)
    let newWidth = Int(Float(originalWidth) * scale)
    let newHeight = Int(Float(originalHeight) * scale)

    print("Downscaling from \(originalWidth)x\(originalHeight) to \(newWidth)x\(newHeight)")

    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm,
        width: originalWidth,
        height: originalHeight,
        mipmapped: false)
    textureDescriptor.usage = [.shaderRead, .shaderWrite]

    guard let sourceTexture = device.makeTexture(descriptor: textureDescriptor) else {
        return nil
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bytesPerRow = originalWidth * 4
    var rawData = [UInt8](repeating: 0, count: originalHeight * bytesPerRow)

    guard let context = CGContext(data: &rawData,
                                  width: originalWidth,
                                  height: originalHeight,
                                  bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
        return nil
    }

    context.draw(originalImage, in: CGRect(x: 0, y: 0, width: originalWidth, height: originalHeight))

    sourceTexture.replace(region: MTLRegionMake2D(0, 0, originalWidth, originalHeight),
                         mipmapLevel: 0,
                         withBytes: &rawData,
                         bytesPerRow: bytesPerRow)

    let destDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba8Unorm,
        width: newWidth,
        height: newHeight,
        mipmapped: false)
    destDescriptor.usage = [.shaderRead, .shaderWrite]

    guard let destTexture = device.makeTexture(descriptor: destDescriptor),
          let commandQueue = device.makeCommandQueue(),
          let commandBuffer = commandQueue.makeCommandBuffer() else {
        return nil
    }

    let scaler = MPSImageLanczosScale(device: device)
    scaler.encode(commandBuffer: commandBuffer, sourceTexture: sourceTexture, destinationTexture: destTexture)

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    var downscaledData = [UInt8](repeating: 0, count: newHeight * newWidth * 4)
    destTexture.getBytes(&downscaledData,
                        bytesPerRow: newWidth * 4,
                        from: MTLRegionMake2D(0, 0, newWidth, newHeight),
                        mipmapLevel: 0)

    guard let downscaledContext = CGContext(data: &downscaledData,
                                           width: newWidth,
                                           height: newHeight,
                                           bitsPerComponent: 8,
                                           bytesPerRow: newWidth * 4,
                                           space: colorSpace,
                                           bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
          let downscaledImage = downscaledContext.makeImage() else {
        return nil
    }

    return (downscaledImage, newWidth, newHeight)
}

// Convert CGImage to Metal texture.
func createTextureFromImage(_ image: CGImage, device: MTLDevice) -> MTLTexture? {
    let width = image.width
    let height = image.height

    let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
        pixelFormat: .rgba32Float,
        width: width,
        height: height,
        mipmapped: false)
    textureDescriptor.usage = [.shaderRead, .shaderWrite]

    guard let texture = device.makeTexture(descriptor: textureDescriptor) else {
        return nil
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    let bytesPerPixel = 4
    let bytesPerRow = bytesPerPixel * width
    var rawData = [UInt8](repeating: 0, count: height * bytesPerRow)

    guard let context = CGContext(data: &rawData,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else {
        return nil
    }

    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

    var floatData = [Float](repeating: 0, count: width * height * 4)
    for i in 0..<(width * height) {
        floatData[i * 4 + 0] = Float(rawData[i * 4 + 0]) / 255.0
        floatData[i * 4 + 1] = Float(rawData[i * 4 + 1]) / 255.0
        floatData[i * 4 + 2] = Float(rawData[i * 4 + 2]) / 255.0
        floatData[i * 4 + 3] = 1.0
    }

    let region = MTLRegionMake2D(0, 0, width, height)
    texture.replace(region: region, mipmapLevel: 0, withBytes: floatData,
                    bytesPerRow: width * 4 * MemoryLayout<Float>.stride)

    return texture
}

// Save texture to PNG.
func saveTexture(_ texture: MTLTexture, path: String) {
    let width = texture.width
    let height = texture.height
    let bytesPerRow = width * 4 * MemoryLayout<Float>.stride

    var floatData = [Float](repeating: 0, count: width * height * 4)
    let region = MTLRegionMake2D(0, 0, width, height)
    texture.getBytes(&floatData, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

    var byteData = [UInt8](repeating: 0, count: width * height * 4)
    for i in 0..<(width * height * 4) {
        let value = floatData[i]
        let finiteValue = value.isFinite ? value : 0.0
        let clamped = min(max(finiteValue * 255.0, 0.0), 255.0)
        byteData[i] = UInt8(clamped)
    }

    let colorSpace = CGColorSpaceCreateDeviceRGB()
    guard let context = CGContext(data: &byteData,
                                  width: width,
                                  height: height,
                                  bitsPerComponent: 8,
                                  bytesPerRow: width * 4,
                                  space: colorSpace,
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
          let cgImage = context.makeImage() else {
        return
    }

    let url = URL(fileURLWithPath: path)
    guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
        return
    }

    CGImageDestinationAddImage(destination, cgImage, nil)
    CGImageDestinationFinalize(destination)
}
