func hilbertBitsForSize(width: Int, height: Int) -> Int {
    let maxDim = max(width, height)
    var n = 1
    var bits = 0
    while n < maxDim {
        n <<= 1
        bits += 1
    }
    return max(bits, 1)
}
