import Metal
import Foundation

func nextPow2(_ n: Int) -> Int {
    if n <= 1 { return 1 }
    var v = n - 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    #if arch(arm64) || arch(x86_64)
    if MemoryLayout<Int>.size == 8 { v |= v >> 32 }
    #endif
    return v + 1
}

struct SeededRNG {
    private var state: UInt64

    init(seed: UInt64) { self.state = seed != 0 ? seed : 0x1234_5678_9abc_def0 }

    mutating func nextU32() -> UInt32 {
        var x = state
        x ^= x >> 12
        x ^= x << 25
        x ^= x >> 27
        state = x
        let y = x &* 2685821657736338717
        return UInt32(truncatingIfNeeded: y >> 32)
    }

    mutating func nextFloat01() -> Float {
        let u = nextU32() & 0x00ff_ffff
        return Float(u) / Float(0x0100_0000)
    }
}

func runRadixSortTest(device: MTLDevice, library: MTLLibrary, keyMaxExclusive: UInt32, count: Int) {
    print("\n--- Testing RadixSortUInt2 (maxKeyExclusive=\(keyMaxExclusive), count=\(count)) ---")

    // Generate data: (key, originalIndex). With a stable sort by key, originalIndex is increasing within equal keys.
    var rng = SeededRNG(seed: 0xC0FFEE)
    var data = [SIMD2<UInt32>]()
    data.reserveCapacity(count)
    for i in 0..<count {
        let key: UInt32 = (keyMaxExclusive == UInt32.max)
            ? rng.nextU32()
            : (rng.nextU32() % max(UInt32(1), keyMaxExclusive))
        data.append(SIMD2(key, UInt32(i)))
    }

    let paddedCount = nextPow2(count)
    while data.count < paddedCount {
        data.append(SIMD2(UInt32.max, UInt32.max))
    }

    let buffer = device.makeBuffer(
        bytes: data,
        length: MemoryLayout<SIMD2<UInt32>>.stride * paddedCount,
        options: .storageModeShared
    )!

    let radix = try! RadixSortUInt2(device: device, library: library, paddedCount: paddedCount)

    guard let queue = device.makeCommandQueue(),
          let cb = queue.makeCommandBuffer() else {
        fatalError("Failed to create command buffer")
    }

    radix.encode(data: buffer, maxKeyExclusive: keyMaxExclusive, in: cb)
    cb.commit()
    cb.waitUntilCompleted()

    let gpu = Array(UnsafeBufferPointer(
        start: buffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: paddedCount),
        count: paddedCount
    ))

    // CPU reference: sort by key, then by originalIndex (tie-breaker to define the expected stable order).
    let cpu = data
        .prefix(count)
        .sorted { a, b in
            if a.x != b.x { return a.x < b.x }
            return a.y < b.y
        }

    // Verify sorted and stable.
    var ok = true
    for i in 1..<count {
        if gpu[i].x < gpu[i - 1].x {
            print("ERROR: Not sorted by key at \(i): \(gpu[i - 1].x) > \(gpu[i].x)")
            ok = false
            break
        }
        if gpu[i].x == gpu[i - 1].x && gpu[i].y < gpu[i - 1].y {
            print("ERROR: Not stable within key=\(gpu[i].x) at \(i): \(gpu[i - 1].y) > \(gpu[i].y)")
            ok = false
            break
        }
    }

    if ok {
        for i in 0..<count {
            if gpu[i] != cpu[i] {
                print("ERROR: GPU != CPU at \(i): gpu=(\(gpu[i].x),\(gpu[i].y)) cpu=(\(cpu[i].x),\(cpu[i].y))")
                ok = false
                break
            }
        }
    }

    if ok {
        print("✓ RadixSortUInt2 matches CPU and is stable")
    } else {
        print("✗ RadixSortUInt2 FAILED")
    }
}

// MARK: - Additional Edge Case Tests

/// Test with very small counts (edge cases for the algorithm)
func runSmallCountTests(device: MTLDevice, library: MTLLibrary) {
    print("\n--- Testing RadixSortUInt2 with small counts ---")

    for count in [1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 257] {
        let ok = testRadixSort(device: device, library: library, count: count, keyMax: UInt32.max, seed: UInt64(count))
        if !ok {
            print("✗ FAILED at count=\(count)")
            return
        }
    }
    print("✓ All small count tests passed")
}

/// Test with identical keys (stress test for stability)
func runIdenticalKeyTests(device: MTLDevice, library: MTLLibrary) {
    print("\n--- Testing RadixSortUInt2 with identical keys (stability) ---")

    for count in [256, 1024, 4096] {
        for keyVal in [UInt32(0), UInt32(1), UInt32(255), UInt32(256), UInt32(65535), UInt32(65536), UInt32.max - 1] {
            let ok = testRadixSortIdenticalKeys(device: device, library: library, count: count, keyValue: keyVal)
            if !ok {
                print("✗ FAILED at count=\(count), key=\(keyVal)")
                return
            }
        }
    }
    print("✓ All identical key tests passed")
}

/// Test with keys at specific boundaries (8-bit, 16-bit boundaries)
func runBoundaryKeyTests(device: MTLDevice, library: MTLLibrary) {
    print("\n--- Testing RadixSortUInt2 with boundary keys ---")

    // Test keys clustered around specific boundaries
    let boundaries: [(String, ClosedRange<UInt32>)] = [
        ("around 255-256", 250...260),
        ("around 65535-65536", 65530...65542),
        ("around 16777215-16777216", 16777210...16777222),
        ("low values only", 0...10),
        ("high values", (UInt32.max - 100)...UInt32.max),
    ]

    for (name, range) in boundaries {
        let ok = testRadixSortKeyRange(device: device, library: library, count: 4096, keyRange: range, seed: 0xDEAD)
        if !ok {
            print("✗ FAILED for \(name)")
            return
        }
    }
    print("✓ All boundary key tests passed")
}

/// Test with already sorted input
func runSortedInputTests(device: MTLDevice, library: MTLLibrary) {
    print("\n--- Testing RadixSortUInt2 with sorted/reverse input ---")

    for count in [256, 1024, 2048] {
        // Already sorted
        var ok = testRadixSortSortedInput(device: device, library: library, count: count, ascending: true)
        if !ok {
            print("✗ FAILED for sorted input, count=\(count)")
            return
        }

        // Reverse sorted
        ok = testRadixSortSortedInput(device: device, library: library, count: count, ascending: false)
        if !ok {
            print("✗ FAILED for reverse sorted input, count=\(count)")
            return
        }
    }
    print("✓ All sorted input tests passed")
}

/// Test with various non-power-of-2 counts
func runNonPow2CountTests(device: MTLDevice, library: MTLLibrary) {
    print("\n--- Testing RadixSortUInt2 with non-power-of-2 counts ---")

    // Various non-power-of-2 counts including edge cases around block boundaries
    let counts = [
        100, 200, 300, 500, 700, 900,
        1000, 1023, 1025, 1100,
        2000, 2047, 2049, 3000,
        4000, 4095, 4097, 5000,
        10000, 16383, 16385
    ]

    for count in counts {
        let ok = testRadixSort(device: device, library: library, count: count, keyMax: UInt32.max, seed: UInt64(count) ^ 0xCAFE)
        if !ok {
            print("✗ FAILED at count=\(count)")
            return
        }
    }
    print("✓ All non-power-of-2 count tests passed")
}

/// Test with specific patterns that might break the algorithm
func runPatternTests(device: MTLDevice, library: MTLLibrary) {
    print("\n--- Testing RadixSortUInt2 with specific patterns ---")

    // Test 1: Alternating high/low keys
    var ok = testRadixSortPattern(device: device, library: library, count: 1024, pattern: .alternating)
    if !ok {
        print("✗ FAILED for alternating pattern")
        return
    }

    // Test 2: Keys with only certain bits set
    ok = testRadixSortPattern(device: device, library: library, count: 1024, pattern: .lowBytesOnly)
    if !ok {
        print("✗ FAILED for lowBytesOnly pattern")
        return
    }

    ok = testRadixSortPattern(device: device, library: library, count: 1024, pattern: .highBytesOnly)
    if !ok {
        print("✗ FAILED for highBytesOnly pattern")
        return
    }

    // Test 3: Keys that differ only in specific bytes
    ok = testRadixSortPattern(device: device, library: library, count: 1024, pattern: .sameHighBytes)
    if !ok {
        print("✗ FAILED for sameHighBytes pattern")
        return
    }

    print("✓ All pattern tests passed")
}

/// Test with counts that exercise different numbers of grid blocks
func runGridSizeTests(device: MTLDevice, library: MTLLibrary) {
    print("\n--- Testing RadixSortUInt2 with various grid sizes ---")

    // elementsPerBlock = 1024, so test around block boundaries
    let counts = [
        512,    // < 1 block
        1024,   // exactly 1 block
        1025,   // 1 block + 1
        2048,   // exactly 2 blocks
        2049,   // 2 blocks + 1
        3072,   // exactly 3 blocks
        4096,   // exactly 4 blocks
        5000,   // 4+ blocks
        8192,   // 8 blocks
        10240,  // 10 blocks
    ]

    for count in counts {
        let ok = testRadixSort(device: device, library: library, count: count, keyMax: UInt32.max, seed: UInt64(count) ^ 0xBEEF)
        if !ok {
            print("✗ FAILED at count=\(count) (gridSize related)")
            return
        }
    }
    print("✓ All grid size tests passed")
}

/// Test realistic pipeline scenario: repeated sorts with millions of items
/// This mimics the main.swift usage where sortDataCount = numPixels * 8
func runPipelineSimulationTests(device: MTLDevice, library: MTLLibrary) {
    print("\n--- Testing RadixSortUInt2 pipeline simulation ---")

    // Simulate different image sizes
    // numPixels * 8 for each
    let imageSizes: [(name: String, pixels: Int, sites: Int)] = [
        ("256x256 image, 1k sites", 256 * 256, 1000),
        ("512x512 image, 4k sites", 512 * 512, 4000),
        ("1024x1024 image, 8k sites", 1024 * 1024, 8000),
        ("2k image, 16k sites", 2048 * 2048, 16000),
    ]

    for (name, pixels, sites) in imageSizes {
        let sortDataCount = pixels * 8
        print("  Testing: \(name) (sortData=\(sortDataCount), sites=\(sites))")

        let ok = testPipelineSort(
            device: device,
            library: library,
            sortDataCount: sortDataCount,
            numSites: sites,
            iterations: 5  // Simulate multiple training iterations
        )
        if !ok {
            print("✗ FAILED for \(name)")
            return
        }
    }
    print("✓ All pipeline simulation tests passed")
}

/// Simulate the actual pipeline: sort -> use -> sort -> use pattern
func testPipelineSort(device: MTLDevice, library: MTLLibrary, sortDataCount: Int, numSites: Int, iterations: Int) -> Bool {
    let paddedCount = nextPow2(sortDataCount)

    // Create radix sorter (reused across iterations like in main.swift)
    let radix: RadixSortUInt2
    do {
        radix = try RadixSortUInt2(device: device, library: library, paddedCount: paddedCount)
    } catch {
        print("    ERROR: Failed to create RadixSortUInt2: \(error)")
        return false
    }

    let buffer = device.makeBuffer(
        length: MemoryLayout<SIMD2<UInt32>>.stride * paddedCount,
        options: .storageModeShared
    )!

    guard let queue = device.makeCommandQueue() else {
        print("    ERROR: Failed to create command queue")
        return false
    }

    var rng = SeededRNG(seed: 0xF1E1D)

    for iter in 0..<iterations {
        // Generate data for this iteration
        var data = [SIMD2<UInt32>]()
        data.reserveCapacity(sortDataCount)

        for i in 0..<sortDataCount {
            // Key is siteID (0..<numSites), like in the main pipeline
            let key = rng.nextU32() % UInt32(numSites)
            data.append(SIMD2(key, UInt32(i)))
        }

        // Pad
        while data.count < paddedCount {
            data.append(SIMD2(UInt32.max, UInt32.max))
        }

        // Copy to buffer
        let ptr = buffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: paddedCount)
        for i in 0..<paddedCount {
            ptr[i] = data[i]
        }

        // Sort
        guard let cb = queue.makeCommandBuffer() else {
            print("    ERROR: Failed to create command buffer at iteration \(iter)")
            return false
        }

        radix.encode(data: buffer, maxKeyExclusive: UInt32(numSites), in: cb)
        cb.commit()
        cb.waitUntilCompleted()

        // Verify
        let gpu = Array(UnsafeBufferPointer(start: ptr, count: paddedCount))

        // CPU reference
        let cpu = data.prefix(sortDataCount).sorted { a, b in
            if a.x != b.x { return a.x < b.x }
            return a.y < b.y
        }

        // Check sorted
        for i in 1..<sortDataCount {
            if gpu[i].x < gpu[i - 1].x {
                print("    ERROR iter \(iter): Not sorted at \(i): \(gpu[i - 1].x) > \(gpu[i].x)")
                return false
            }
        }

        // Check matches CPU
        for i in 0..<min(10, sortDataCount) {  // Just check first 10 for speed
            if gpu[i] != cpu[i] {
                print("    ERROR iter \(iter): GPU != CPU at \(i)")
                return false
            }
        }
    }

    return true
}

/// Test single-pass vs multi-pass behavior
func runPassCountTests(device: MTLDevice, library: MTLLibrary) {
    print("\n--- Testing RadixSortUInt2 pass count edge cases ---")

    // Test 2-pass (16-bit keys) vs 4-pass (32-bit keys) boundary
    let testCases: [(maxKey: UInt32, desc: String)] = [
        (UInt32(1 << 15), "15-bit keys (2 passes)"),
        (UInt32(1 << 16), "exactly 16-bit keys (2 passes)"),
        (UInt32(1 << 16) + 1, "16-bit + 1 (4 passes)"),
        (UInt32(1 << 17), "17-bit keys (4 passes)"),
        (UInt32(1 << 24), "24-bit keys (4 passes)"),
        (UInt32.max, "32-bit keys (4 passes)"),
    ]

    for (maxKey, desc) in testCases {
        let ok = testRadixSort(device: device, library: library, count: 4096, keyMax: maxKey, seed: UInt64(maxKey) ^ 0xABCD)
        if !ok {
            print("✗ FAILED for \(desc)")
            return
        }
    }
    print("✓ All pass count tests passed")
}

/// Detailed diagnostic test - print intermediate state
func runDiagnosticTest(device: MTLDevice, library: MTLLibrary) {
    print("\n--- Diagnostic test (count=16, small keys) ---")

    let count = 16
    var rng = SeededRNG(seed: 0xD1A6)

    var data = [SIMD2<UInt32>]()
    for i in 0..<count {
        let key = rng.nextU32() % 10  // Small keys 0-9
        data.append(SIMD2(key, UInt32(i)))
    }

    print("Input:")
    for i in 0..<count {
        print("  [\(i)] key=\(data[i].x), idx=\(data[i].y)")
    }

    let paddedCount = nextPow2(count)
    while data.count < paddedCount {
        data.append(SIMD2(UInt32.max, UInt32.max))
    }

    let buffer = device.makeBuffer(
        bytes: data,
        length: MemoryLayout<SIMD2<UInt32>>.stride * paddedCount,
        options: .storageModeShared
    )!

    let radix = try! RadixSortUInt2(device: device, library: library, paddedCount: paddedCount)

    guard let queue = device.makeCommandQueue(),
          let cb = queue.makeCommandBuffer() else {
        print("ERROR: Failed to create command buffer")
        return
    }

    radix.encode(data: buffer, maxKeyExclusive: 10, in: cb)
    cb.commit()
    cb.waitUntilCompleted()

    let gpu = Array(UnsafeBufferPointer(
        start: buffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: paddedCount),
        count: paddedCount
    ))

    print("\nOutput:")
    for i in 0..<count {
        print("  [\(i)] key=\(gpu[i].x), idx=\(gpu[i].y)")
    }

    // CPU reference
    let cpu = data.prefix(count).sorted { a, b in
        if a.x != b.x { return a.x < b.x }
        return a.y < b.y
    }

    print("\nExpected:")
    for i in 0..<count {
        print("  [\(i)] key=\(cpu[i].x), idx=\(cpu[i].y)")
    }

    var ok = true
    for i in 0..<count {
        if gpu[i] != cpu[i] {
            print("\nMISMATCH at \(i): got (\(gpu[i].x),\(gpu[i].y)) expected (\(cpu[i].x),\(cpu[i].y))")
            ok = false
        }
    }

    if ok {
        print("\n✓ Diagnostic test passed")
    } else {
        print("\n✗ Diagnostic test FAILED")
    }
}

/// Test with larger count that fills at least one full block (1024 elements)
func runFullBlockDiagnostic(device: MTLDevice, library: MTLLibrary) {
    print("\n--- Full block diagnostic test (count=1024) ---")

    let count = 1024  // Exactly one full block
    var rng = SeededRNG(seed: 0xFB10C)

    var data = [SIMD2<UInt32>]()
    for i in 0..<count {
        let key = rng.nextU32() % 100  // Keys 0-99
        data.append(SIMD2(key, UInt32(i)))
    }

    let paddedCount = nextPow2(count)

    let buffer = device.makeBuffer(
        bytes: data,
        length: MemoryLayout<SIMD2<UInt32>>.stride * paddedCount,
        options: .storageModeShared
    )!

    // Also create a copy to check if buffer is modified
    let bufferCopy = device.makeBuffer(
        bytes: data,
        length: MemoryLayout<SIMD2<UInt32>>.stride * paddedCount,
        options: .storageModeShared
    )!

    let radix = try! RadixSortUInt2(device: device, library: library, paddedCount: paddedCount)

    guard let queue = device.makeCommandQueue(),
          let cb = queue.makeCommandBuffer() else {
        print("ERROR: Failed to create command buffer")
        return
    }

    radix.encode(data: buffer, maxKeyExclusive: 100, in: cb)
    cb.commit()
    cb.waitUntilCompleted()

    let gpu = Array(UnsafeBufferPointer(
        start: buffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: paddedCount),
        count: paddedCount
    ))

    let original = Array(UnsafeBufferPointer(
        start: bufferCopy.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: paddedCount),
        count: paddedCount
    ))

    // Check if buffer was modified at all
    var modified = false
    for i in 0..<count {
        if gpu[i] != original[i] {
            modified = true
            break
        }
    }

    if !modified {
        print("  WARNING: Buffer was NOT modified by sort!")
    }

    // Check first 20 elements
    print("\nFirst 20 output elements:")
    for i in 0..<20 {
        print("  [\(i)] key=\(gpu[i].x), idx=\(gpu[i].y)")
    }

    // CPU reference
    let cpu = data.sorted { a, b in
        if a.x != b.x { return a.x < b.x }
        return a.y < b.y
    }

    print("\nFirst 20 expected elements:")
    for i in 0..<20 {
        print("  [\(i)] key=\(cpu[i].x), idx=\(cpu[i].y)")
    }

    // Check sorted
    var sortErrors = 0
    for i in 1..<count {
        if gpu[i].x < gpu[i - 1].x {
            sortErrors += 1
            if sortErrors <= 5 {
                print("  Sort error at \(i): \(gpu[i - 1].x) > \(gpu[i].x)")
            }
        }
    }

    if sortErrors > 0 {
        print("  Total sort errors: \(sortErrors)")
        print("✗ Full block test FAILED")
    } else {
        // Check matches CPU
        var matchErrors = 0
        for i in 0..<count {
            if gpu[i] != cpu[i] {
                matchErrors += 1
            }
        }
        if matchErrors > 0 {
            print("  GPU != CPU in \(matchErrors) positions")
            print("✗ Full block test FAILED (stability issue)")
        } else {
            print("✓ Full block test passed")
        }
    }
}

/// Test to check if identical key test actually uses all the sort machinery
func runIdenticalKeyDiagnostic(device: MTLDevice, library: MTLLibrary) {
    print("\n--- Identical key diagnostic ---")

    let count = 256
    let keyValue: UInt32 = 42

    var data = [SIMD2<UInt32>]()
    for i in 0..<count {
        data.append(SIMD2(keyValue, UInt32(i)))
    }

    let paddedCount = nextPow2(count)

    let buffer = device.makeBuffer(
        bytes: data,
        length: MemoryLayout<SIMD2<UInt32>>.stride * paddedCount,
        options: .storageModeShared
    )!

    let radix = try! RadixSortUInt2(device: device, library: library, paddedCount: paddedCount)

    guard let queue = device.makeCommandQueue(),
          let cb = queue.makeCommandBuffer() else {
        print("ERROR: Failed to create command buffer")
        return
    }

    // Use 4 passes to test full sort (43 > 2^16 is false, so this uses 2 passes)
    radix.encode(data: buffer, maxKeyExclusive: 100, in: cb)
    cb.commit()
    cb.waitUntilCompleted()

    let gpu = Array(UnsafeBufferPointer(
        start: buffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: paddedCount),
        count: paddedCount
    ))

    print("First 10 output elements (all should have key=42, indices 0-9):")
    for i in 0..<10 {
        print("  [\(i)] key=\(gpu[i].x), idx=\(gpu[i].y)")
    }

    var ok = true
    for i in 0..<count {
        if gpu[i].x != keyValue || gpu[i].y != UInt32(i) {
            print("  ERROR at \(i): expected (\(keyValue),\(i)) got (\(gpu[i].x),\(gpu[i].y))")
            ok = false
            break
        }
    }

    if ok {
        print("✓ Identical key diagnostic passed (stability preserved)")
    } else {
        print("✗ Identical key diagnostic FAILED")
    }
}

// MARK: - Test Helper Functions

enum TestPattern {
    case alternating
    case lowBytesOnly
    case highBytesOnly
    case sameHighBytes
}

func testRadixSort(device: MTLDevice, library: MTLLibrary, count: Int, keyMax: UInt32, seed: UInt64) -> Bool {
    var rng = SeededRNG(seed: seed)
    var data = [SIMD2<UInt32>]()
    data.reserveCapacity(count)

    for i in 0..<count {
        let key: UInt32 = (keyMax == UInt32.max)
            ? rng.nextU32()
            : (rng.nextU32() % max(UInt32(1), keyMax))
        data.append(SIMD2(key, UInt32(i)))
    }

    return runSortAndVerify(device: device, library: library, data: data, count: count, maxKey: keyMax)
}

func testRadixSortIdenticalKeys(device: MTLDevice, library: MTLLibrary, count: Int, keyValue: UInt32) -> Bool {
    var data = [SIMD2<UInt32>]()
    data.reserveCapacity(count)

    for i in 0..<count {
        data.append(SIMD2(keyValue, UInt32(i)))
    }

    // Use 4 passes for any key that might have high bits
    let maxKey = keyValue < (1 << 16) ? keyValue + 1 : UInt32.max
    return runSortAndVerify(device: device, library: library, data: data, count: count, maxKey: maxKey)
}

func testRadixSortKeyRange(device: MTLDevice, library: MTLLibrary, count: Int, keyRange: ClosedRange<UInt32>, seed: UInt64) -> Bool {
    var rng = SeededRNG(seed: seed)
    var data = [SIMD2<UInt32>]()
    data.reserveCapacity(count)

    let rangeSize = UInt32(keyRange.count)
    for i in 0..<count {
        let offset = rng.nextU32() % rangeSize
        let key = keyRange.lowerBound + offset
        data.append(SIMD2(key, UInt32(i)))
    }

    // If all keys fit in 16 bits, use 2 passes; otherwise 4
    let maxKey = keyRange.upperBound < (1 << 16) ? keyRange.upperBound + 1 : UInt32.max
    return runSortAndVerify(device: device, library: library, data: data, count: count, maxKey: maxKey)
}

func testRadixSortSortedInput(device: MTLDevice, library: MTLLibrary, count: Int, ascending: Bool) -> Bool {
    var data = [SIMD2<UInt32>]()
    data.reserveCapacity(count)

    for i in 0..<count {
        let key: UInt32 = ascending ? UInt32(i) : UInt32(count - 1 - i)
        data.append(SIMD2(key, UInt32(i)))
    }

    let maxKey: UInt32 = count <= 65536 ? UInt32(count) : UInt32.max
    return runSortAndVerify(device: device, library: library, data: data, count: count, maxKey: maxKey)
}

func testRadixSortPattern(device: MTLDevice, library: MTLLibrary, count: Int, pattern: TestPattern) -> Bool {
    var rng = SeededRNG(seed: 0xDA77E8)
    var data = [SIMD2<UInt32>]()
    data.reserveCapacity(count)

    for i in 0..<count {
        let key: UInt32
        switch pattern {
        case .alternating:
            key = (i % 2 == 0) ? rng.nextU32() : (UInt32.max - rng.nextU32())
        case .lowBytesOnly:
            key = rng.nextU32() & 0x0000FFFF
        case .highBytesOnly:
            key = rng.nextU32() & 0xFFFF0000
        case .sameHighBytes:
            key = 0xABCD0000 | (rng.nextU32() & 0x0000FFFF)
        }
        data.append(SIMD2(key, UInt32(i)))
    }

    return runSortAndVerify(device: device, library: library, data: data, count: count, maxKey: UInt32.max)
}

func runSortAndVerify(device: MTLDevice, library: MTLLibrary, data: [SIMD2<UInt32>], count: Int, maxKey: UInt32) -> Bool {
    let paddedCount = nextPow2(count)
    var paddedData = data
    while paddedData.count < paddedCount {
        paddedData.append(SIMD2(UInt32.max, UInt32.max))
    }

    let buffer = device.makeBuffer(
        bytes: paddedData,
        length: MemoryLayout<SIMD2<UInt32>>.stride * paddedCount,
        options: .storageModeShared
    )!

    let radix: RadixSortUInt2
    do {
        radix = try RadixSortUInt2(device: device, library: library, paddedCount: paddedCount)
    } catch {
        print("ERROR: Failed to create RadixSortUInt2: \(error)")
        return false
    }

    guard let queue = device.makeCommandQueue(),
          let cb = queue.makeCommandBuffer() else {
        print("ERROR: Failed to create command buffer")
        return false
    }

    radix.encode(data: buffer, maxKeyExclusive: maxKey, in: cb)
    cb.commit()
    cb.waitUntilCompleted()

    let gpu = Array(UnsafeBufferPointer(
        start: buffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: paddedCount),
        count: paddedCount
    ))

    // CPU reference: stable sort by key
    let cpu = data.sorted { a, b in
        if a.x != b.x { return a.x < b.x }
        return a.y < b.y
    }

    // Verify sorted and stable
    for i in 1..<count {
        if gpu[i].x < gpu[i - 1].x {
            print("  ERROR: Not sorted by key at \(i): \(gpu[i - 1].x) > \(gpu[i].x)")
            return false
        }
        if gpu[i].x == gpu[i - 1].x && gpu[i].y < gpu[i - 1].y {
            print("  ERROR: Not stable within key=\(gpu[i].x) at \(i): \(gpu[i - 1].y) > \(gpu[i].y)")
            return false
        }
    }

    // Verify exact match with CPU
    for i in 0..<count {
        if gpu[i] != cpu[i] {
            print("  ERROR: GPU != CPU at \(i): gpu=(\(gpu[i].x),\(gpu[i].y)) cpu=(\(cpu[i].x),\(cpu[i].y))")
            return false
        }
    }

    return true
}

func runRadixPairsFloatKeyTest(device: MTLDevice, library: MTLLibrary, count: Int) {
    print("\n--- Testing RadixSortUInt2 on float-derived pair keys (count=\(count)) ---")

    var rng = SeededRNG(seed: 0xBADC0DE)
    var data = [SIMD2<UInt32>]()
    data.reserveCapacity(count)

    // Build float scores spanning many exponents; key = 0xffffffff - bitPattern(score)
    for i in 0..<count {
        let u = Double(rng.nextFloat01())
        let exp = u * 20.0 - 10.0
        let score = pow(2.0, exp) * Double(rng.nextFloat01())
        let f = Float(score)
        let key = UInt32.max &- f.bitPattern
        data.append(SIMD2(key, UInt32(i)))
    }

    let paddedCount = nextPow2(count)
    while data.count < paddedCount {
        data.append(SIMD2(UInt32.max, UInt32.max))
    }

    let buffer = device.makeBuffer(
        bytes: data,
        length: MemoryLayout<SIMD2<UInt32>>.stride * paddedCount,
        options: .storageModeShared
    )!

    let radix = try! RadixSortUInt2(device: device, library: library, paddedCount: paddedCount)

    guard let queue = device.makeCommandQueue(),
          let cb = queue.makeCommandBuffer() else {
        fatalError("Failed to create command buffer")
    }

    radix.encode(data: buffer, maxKeyExclusive: UInt32.max, in: cb)
    cb.commit()
    cb.waitUntilCompleted()

    let gpu = Array(UnsafeBufferPointer(
        start: buffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: paddedCount),
        count: paddedCount
    ))

    let cpu = data
        .prefix(count)
        .sorted { a, b in
            if a.x != b.x { return a.x < b.x }
            return a.y < b.y
        }

    var ok = true
    for i in 1..<count {
        if gpu[i].x < gpu[i - 1].x {
            print("ERROR: Not sorted by key at \(i): \(gpu[i - 1].x) > \(gpu[i].x)")
            ok = false
            break
        }
        if gpu[i].x == gpu[i - 1].x && gpu[i].y < gpu[i - 1].y {
            print("ERROR: Not stable within key=\(gpu[i].x) at \(i): \(gpu[i - 1].y) > \(gpu[i].y)")
            ok = false
            break
        }
    }

    if ok {
        for i in 0..<count {
            if gpu[i] != cpu[i] {
                print("ERROR: GPU != CPU at \(i): gpu=(\(gpu[i].x),\(gpu[i].y)) cpu=(\(cpu[i].x),\(cpu[i].y))")
                ok = false
                break
            }
        }
    }

    if ok {
        print("✓ Float-derived key pair sort matches CPU")
    } else {
        print("✗ Float-derived key pair sort FAILED")
    }
}


// Entry point
@main
struct TestProgram {
    static func main() {
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal not supported")
            return
        }

        print("Using device: \(device.name)")

        let library: MTLLibrary
        if let defaultLib = device.makeDefaultLibrary() {
            library = defaultLib
        } else {
            let libPath = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
                .appendingPathComponent("sad.metallib")
            do {
                library = try device.makeLibrary(URL: libPath)
            } catch {
                print("Failed to load Metal library: \(error)")
                return
            }
        }

        // Original tests
        runRadixSortTest(device: device, library: library, keyMaxExclusive: 10_000, count: 4096)
        runRadixSortTest(device: device, library: library, keyMaxExclusive: UInt32.max, count: 4096)
        runRadixPairsFloatKeyTest(device: device, library: library, count: 4096)

        // New comprehensive edge case tests
        runSmallCountTests(device: device, library: library)
        runIdenticalKeyTests(device: device, library: library)
        runBoundaryKeyTests(device: device, library: library)
        runSortedInputTests(device: device, library: library)
        runNonPow2CountTests(device: device, library: library)
        runPatternTests(device: device, library: library)
        runGridSizeTests(device: device, library: library)

        // Pass count edge cases (2 vs 4 passes)
        runPassCountTests(device: device, library: library)

        // Diagnostic tests for debugging
        runDiagnosticTest(device: device, library: library)
        runFullBlockDiagnostic(device: device, library: library)
        runIdenticalKeyDiagnostic(device: device, library: library)

        // Realistic pipeline simulation (large counts, repeated sorts)
        runPipelineSimulationTests(device: device, library: library)
    }
}
