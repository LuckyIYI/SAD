import Metal

final class PackedRenderEncoder {
    private let renderPipeline: MTLComputePipelineState
    private let coloringPipeline: MTLComputePipelineState
    private let centroidsPipeline: MTLComputePipelineState
    private let tauHeatmapPipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        guard let renderFunc = library.makeFunction(name: "renderVoronoiPacked"),
              let coloringFunc = library.makeFunction(name: "renderVoronoiColoringPacked"),
              let centroidsFunc = library.makeFunction(name: "renderVoronoiCentroidsPacked"),
              let tauHeatmapFunc = library.makeFunction(name: "renderCentroidsTauHeatmapPacked") else {
            throw NSError(domain: "PackedRenderEncoder", code: 1,
                         userInfo: [NSLocalizedDescriptionKey: "Missing packed render kernels"])
        }
        self.renderPipeline = try device.makeComputePipelineState(function: renderFunc)
        self.coloringPipeline = try device.makeComputePipelineState(function: coloringFunc)
        self.centroidsPipeline = try device.makeComputePipelineState(function: centroidsFunc)
        self.tauHeatmapPipeline = try device.makeComputePipelineState(function: tauHeatmapFunc)
    }

    func encodeRender(cand0: MTLTexture, cand1: MTLTexture, output: MTLTexture,
                      sitesBuffer: MTLBuffer, quant: PackedSiteQuant,
                      invScaleSq: Float, siteCount: UInt32,
                      in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Render Voronoi Packed"
        encoder.setComputePipelineState(renderPipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setTexture(cand1, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var quantVar = quant
        var invScaleSqVar = invScaleSq
        var siteCountVar = siteCount
        encoder.setBytes(&quantVar, length: MemoryLayout<PackedSiteQuant>.stride, index: 1)
        encoder.setBytes(&invScaleSqVar, length: MemoryLayout<Float>.stride, index: 2)
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 3)

        dispatchGrid(encoder: encoder, texture: output)
        encoder.endEncoding()
    }

    func encodeColoring(cand0: MTLTexture, cand1: MTLTexture, output: MTLTexture,
                        sitesBuffer: MTLBuffer, quant: PackedSiteQuant,
                        invScaleSq: Float, siteCount: UInt32,
                        in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Render Voronoi Coloring Packed"
        encoder.setComputePipelineState(coloringPipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setTexture(cand1, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var quantVar = quant
        var invScaleSqVar = invScaleSq
        var siteCountVar = siteCount
        encoder.setBytes(&quantVar, length: MemoryLayout<PackedSiteQuant>.stride, index: 1)
        encoder.setBytes(&invScaleSqVar, length: MemoryLayout<Float>.stride, index: 2)
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 3)

        dispatchGrid(encoder: encoder, texture: output)
        encoder.endEncoding()
    }

    func encodeCentroids(cand0: MTLTexture, cand1: MTLTexture, output: MTLTexture,
                         sitesBuffer: MTLBuffer, quant: PackedSiteQuant,
                         dotRadius: Float, siteCount: UInt32,
                         in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Render Centroids Packed"
        encoder.setComputePipelineState(centroidsPipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setTexture(cand1, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var quantVar = quant
        var dotRadiusVar = dotRadius
        var siteCountVar = siteCount
        encoder.setBytes(&quantVar, length: MemoryLayout<PackedSiteQuant>.stride, index: 1)
        encoder.setBytes(&dotRadiusVar, length: MemoryLayout<Float>.stride, index: 2)
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 3)

        dispatchGrid(encoder: encoder, texture: output)
        encoder.endEncoding()
    }

    func encodeTauHeatmap(cand0: MTLTexture, cand1: MTLTexture, output: MTLTexture,
                          sitesBuffer: MTLBuffer, quant: PackedSiteQuant,
                          params: TauHeatmapParams, siteCount: UInt32,
                          in commandBuffer: MTLCommandBuffer) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.label = "Render Tau Heatmap Packed"
        encoder.setComputePipelineState(tauHeatmapPipeline)
        encoder.setTexture(cand0, index: 0)
        encoder.setTexture(cand1, index: 1)
        encoder.setTexture(output, index: 2)
        encoder.setBuffer(sitesBuffer, offset: 0, index: 0)

        var quantVar = quant
        var paramsVar = params
        var siteCountVar = siteCount
        encoder.setBytes(&quantVar, length: MemoryLayout<PackedSiteQuant>.stride, index: 1)
        encoder.setBytes(&paramsVar, length: MemoryLayout<TauHeatmapParams>.stride, index: 2)
        encoder.setBytes(&siteCountVar, length: MemoryLayout<UInt32>.stride, index: 3)

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
