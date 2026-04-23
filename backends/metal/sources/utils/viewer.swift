import Foundation
import MetalKit
import AppKit

final class TextureViewRenderer: NSObject, MTKViewDelegate {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private let sampler: MTLSamplerState
    var texture: MTLTexture?

    init(device: MTLDevice, library: MTLLibrary, colorPixelFormat: MTLPixelFormat) {
        self.device = device
        self.commandQueue = device.makeCommandQueue()!

        let samplerDesc = MTLSamplerDescriptor()
        samplerDesc.minFilter = .linear
        samplerDesc.magFilter = .linear
        samplerDesc.sAddressMode = .clampToEdge
        samplerDesc.tAddressMode = .clampToEdge
        self.sampler = device.makeSamplerState(descriptor: samplerDesc)!

        let pipelineDesc = MTLRenderPipelineDescriptor()
        pipelineDesc.vertexFunction = library.makeFunction(name: "viewerVertex")
        pipelineDesc.fragmentFunction = library.makeFunction(name: "viewerFragment")
        pipelineDesc.colorAttachments[0].pixelFormat = colorPixelFormat
        self.pipeline = try! device.makeRenderPipelineState(descriptor: pipelineDesc)

        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable,
              let passDesc = view.currentRenderPassDescriptor,
              let texture = texture else {
            return
        }

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDesc) else {
            return
        }
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(texture, index: 0)
        encoder.setFragmentSamplerState(sampler, index: 0)
        encoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        encoder.endEncoding()
        commandBuffer.present(drawable)
        commandBuffer.commit()
    }
}

final class LiveViewer {
    private let window: NSWindow
    private let imageView: MTKView
    private let idsView: MTKView
    private let imageRenderer: TextureViewRenderer
    private let idsRenderer: TextureViewRenderer

    init(device: MTLDevice, library: MTLLibrary, width: Int, height: Int) {
        let windowWidth = min(CGFloat(width * 2), 1400)
        let windowHeight = min(CGFloat(height), 900)
        window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: windowWidth, height: windowHeight),
            styleMask: [.titled, .closable, .resizable],
            backing: .buffered,
            defer: false)
        window.title = "Training Viewer"

        imageView = MTKView(frame: .zero, device: device)
        idsView = MTKView(frame: .zero, device: device)
        for view in [imageView, idsView] {
            view.enableSetNeedsDisplay = true
            view.isPaused = true
            view.framebufferOnly = false
            view.colorPixelFormat = .bgra8Unorm
        }

        imageRenderer = TextureViewRenderer(device: device, library: library, colorPixelFormat: imageView.colorPixelFormat)
        idsRenderer = TextureViewRenderer(device: device, library: library, colorPixelFormat: idsView.colorPixelFormat)
        imageView.delegate = imageRenderer
        idsView.delegate = idsRenderer

        let stack = NSStackView(views: [imageView, idsView])
        stack.orientation = .horizontal
        stack.distribution = .fillEqually
        stack.translatesAutoresizingMaskIntoConstraints = false

        window.contentView?.addSubview(stack)
        if let content = window.contentView {
            NSLayoutConstraint.activate([
                stack.leadingAnchor.constraint(equalTo: content.leadingAnchor),
                stack.trailingAnchor.constraint(equalTo: content.trailingAnchor),
                stack.topAnchor.constraint(equalTo: content.topAnchor),
                stack.bottomAnchor.constraint(equalTo: content.bottomAnchor)
            ])
        }
    }

    func show() {
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func close() {
        window.close()
    }

    func update(image: MTLTexture, ids: MTLTexture) {
        DispatchQueue.main.async {
            self.imageRenderer.texture = image
            self.idsRenderer.texture = ids
            self.imageView.draw()
            self.idsView.draw()
        }
    }
}
