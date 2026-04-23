import Metal

final class RadixSortUInt2 {
    private let device: MTLDevice

    private let histogramPipeline: MTLComputePipelineState
    private let scanBlocksPipeline: MTLComputePipelineState
    private let scanBlockSumsPipeline: MTLComputePipelineState
    private let applyOffsetsPipeline: MTLComputePipelineState
    private let scatterPipeline: MTLComputePipelineState

    private let paddedCount: Int
    private let gridSize: Int
    private let histLength: Int
    private let histBlocks: Int

    private let histFlat: MTLBuffer
    private let blockSums: MTLBuffer
    private let scratch: MTLBuffer

    init(device: MTLDevice, library: MTLLibrary, paddedCount: Int) throws {
        self.device = device
        self.paddedCount = paddedCount

        guard let histFn = library.makeFunction(name: "radixHistogramUInt2"),
              let scanBlocksFn = library.makeFunction(name: "radixScanHistogramBlocks"),
              let scanSumsFn = library.makeFunction(name: "radixExclusiveScanBlockSums"),
              let applyFn = library.makeFunction(name: "radixApplyOffsets"),
              let scatterFn = library.makeFunction(name: "radixScatterUInt2")
        else {
            throw NSError(domain: "RadixSortUInt2", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing radix sort kernels in metallib"])
        }

        self.histogramPipeline = try device.makeComputePipelineState(function: histFn)
        self.scanBlocksPipeline = try device.makeComputePipelineState(function: scanBlocksFn)
        self.scanBlockSumsPipeline = try device.makeComputePipelineState(function: scanSumsFn)
        self.applyOffsetsPipeline = try device.makeComputePipelineState(function: applyFn)
        self.scatterPipeline = try device.makeComputePipelineState(function: scatterFn)

        let elementsPerBlock = 256 * 4
        self.gridSize = (paddedCount + elementsPerBlock - 1) / elementsPerBlock
        self.histLength = 256 * gridSize
        self.histBlocks = (histLength + 256 - 1) / 256

        // Histogram layout: 256 bins x gridSize blocks
        self.histFlat = device.makeBuffer(length: histLength * MemoryLayout<UInt32>.stride, options: .storageModeShared)!
        self.blockSums = device.makeBuffer(length: histBlocks * MemoryLayout<UInt32>.stride, options: .storageModeShared)!

        // Ping-pong scratch for passes
        self.scratch = device.makeBuffer(length: paddedCount * MemoryLayout<SIMD2<UInt32>>.stride, options: .storageModeShared)!
    }

    /// Sorts `paddedCount` `uint2` values by `.x` key (ascending), stable per pass.
    /// `maxKeyExclusive` is used only to choose #passes (2 for <= 16-bit keys, else 4).
    func encode(data: MTLBuffer, maxKeyExclusive: UInt32, in commandBuffer: MTLCommandBuffer) {
        precondition(data.length >= paddedCount * MemoryLayout<SIMD2<UInt32>>.stride)

        let passes: Int = (maxKeyExclusive <= (1 << 16)) ? 2 : 4

        var input = data
        var output = scratch

        var paddedCountU = UInt32(paddedCount)

        for pass in 0..<passes {
            var shift = UInt32(pass * 8)

            // 1) Histogram
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.label = "Radix Histogram (pass \(pass))"
                enc.setComputePipelineState(histogramPipeline)
                enc.setBuffer(input, offset: 0, index: 0)
                enc.setBuffer(histFlat, offset: 0, index: 1)
                enc.setBytes(&paddedCountU, length: 4, index: 2)
                enc.setBytes(&shift, length: 4, index: 3)
                enc.dispatchThreadgroups(
                    MTLSize(width: gridSize, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                enc.endEncoding()
            }

            // 2) Scan histogram blocks (in-place in histFlat)
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.label = "Radix Scan Blocks (pass \(pass))"
                enc.setComputePipelineState(scanBlocksPipeline)
                enc.setBuffer(histFlat, offset: 0, index: 0)
                enc.setBuffer(blockSums, offset: 0, index: 1)
                enc.setBytes(&paddedCountU, length: 4, index: 2)
                enc.dispatchThreadgroups(
                    MTLSize(width: histBlocks, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                enc.endEncoding()
            }

            // 3) Exclusive scan block sums (single TG)
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.label = "Radix Scan BlockSums (pass \(pass))"
                enc.setComputePipelineState(scanBlockSumsPipeline)
                enc.setBuffer(blockSums, offset: 0, index: 0)
                enc.setBytes(&paddedCountU, length: 4, index: 1)
                enc.dispatchThreads(
                    MTLSize(width: 256, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                enc.endEncoding()
            }

            // 4) Apply offsets to histFlat (in-place)
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.label = "Radix Apply Offsets (pass \(pass))"
                enc.setComputePipelineState(applyOffsetsPipeline)
                enc.setBuffer(histFlat, offset: 0, index: 0)
                enc.setBuffer(blockSums, offset: 0, index: 1)
                enc.setBytes(&paddedCountU, length: 4, index: 2)
                enc.dispatchThreadgroups(
                    MTLSize(width: histBlocks, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                enc.endEncoding()
            }

            // 5) Scatter
            if let enc = commandBuffer.makeComputeCommandEncoder() {
                enc.label = "Radix Scatter (pass \(pass))"
                enc.setComputePipelineState(scatterPipeline)
                enc.setBuffer(input, offset: 0, index: 0)
                enc.setBuffer(output, offset: 0, index: 1)
                enc.setBuffer(histFlat, offset: 0, index: 2)
                enc.setBytes(&paddedCountU, length: 4, index: 3)
                enc.setBytes(&shift, length: 4, index: 4)
                enc.dispatchThreadgroups(
                    MTLSize(width: gridSize, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
                )
                enc.endEncoding()
            }

            swap(&input, &output)
        }

        // Even number of passes => result is back in `data`.
        // If we ever run odd passes, we'd need a blit copy here.
        precondition(passes % 2 == 0, "RadixSortUInt2 currently assumes an even number of passes")
    }
}
