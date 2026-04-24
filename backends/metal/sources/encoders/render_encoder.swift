import Metal

/// Parameters for tau heatmap visualization
struct TauHeatmapParams {
    var minTau: Float
    var meanTau: Float
    var maxTau: Float
    var dotRadius: Float
}

/// Encoder for rendering Voronoi diagrams
final class RenderEncoder {
    private let renderPipeline: MTLComputePipelineState
    private let coloringPipeline: MTLComputePipelineState
    private let tauHeatmapPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let renderFunc = library.makeFunction(name: "renderVoronoi"),
              let coloringFunc = library.makeFunction(name: "renderVoronoiColoring"),
              let tauHeatmapFunc = library.makeFunction(name: "renderCentroidsTauHeatmap") else {
            throw NSError(domain: "RenderEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing render kernels"])
        }
        self.renderPipeline = try device.makeComputePipelineState(function: renderFunc)
        self.coloringPipeline = try device.makeComputePipelineState(function: coloringFunc)
        self.tauHeatmapPipeline = try device.makeComputePipelineState(function: tauHeatmapFunc)
    }

    /// Render Voronoi with site colors (actual output)
    func encodeRender(cand0: MTLTexture, cand1: MTLTexture, output: MTLTexture,
                      sitesBuffer: MTLBuffer, invScaleSq: Float, siteCount: UInt32,
                      in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Render Voronoi"
        encoder.setComputePipelineState(renderPipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setTexture(cand1, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var invScaleSqVar = invScaleSq
        var siteCountVar = siteCount
        encoder.setBytes(&invScaleSqVar, length: MemoryLayout<Float>.stride, index: 1)
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 2)

        dispatchGrid(encoder: encoder, texture: output)
        encoder.endEncoding()
    }

    /// Render Voronoi with hashed cell colors (visualization)
    func encodeColoring(cand0: MTLTexture, cand1: MTLTexture, output: MTLTexture,
                        sitesBuffer: MTLBuffer, invScaleSq: Float, siteCount: UInt32,
                        in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Render Voronoi Coloring"
        encoder.setComputePipelineState(coloringPipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setTexture(cand1, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var invScaleSqVar = invScaleSq
        var siteCountVar = siteCount
        encoder.setBytes(&invScaleSqVar, length: MemoryLayout<Float>.stride, index: 1)
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 2)

        dispatchGrid(encoder: encoder, texture: output)
        encoder.endEncoding()
    }

    /// Render tau heatmap (visualization)
    func encodeTauHeatmap(cand0: MTLTexture, cand1: MTLTexture, output: MTLTexture,
                          sitesBuffer: MTLBuffer, params: TauHeatmapParams, siteCount: UInt32,
                          in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Render Tau Heatmap"
        encoder.setComputePipelineState(tauHeatmapPipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setTexture(cand1, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var paramsVar = params
        var siteCountVar = siteCount
        encoder.setBytes(&paramsVar, length: MemoryLayout<TauHeatmapParams>.stride, index: 1)
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 2)

        dispatchGrid(encoder: encoder, texture: output)
        encoder.endEncoding()
    }

    private func dispatchGrid(encoder: MTLComputeCommandEncoder, texture: MTLTexture) {
        let width = texture.width
        let height = texture.height
        let threadGroupSize = MTLSize(width: 16, height: 16, depth: 1)
        let threadGroups = MTLSize(
            width: (width + 15) / 16,
            height: (height + 15) / 16,
            depth: 1)
        encoder.dispatchThreadgroups(threadGroups, threadsPerThreadgroup: threadGroupSize)
    }
}
