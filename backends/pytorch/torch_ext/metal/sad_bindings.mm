#include <torch/torch.h>
#include <torch/mps.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>
#include <dispatch/dispatch.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#ifdef EMBEDDED_METALLIB_HEADER
#include EMBEDDED_METALLIB_HEADER
#else
#error "EMBEDDED_METALLIB_HEADER not defined"
#endif

static inline id<MTLBuffer> getMTLBufferStorage(const torch::Tensor tensor) {
    return __builtin_bit_cast(id<MTLBuffer>, tensor.storage().data());
}

static id<MTLDevice> g_device = nil;
static id<MTLLibrary> g_library = nil;
static id<MTLCommandQueue> g_queue = nil;

static void ensureMetalInitialized() {
    if (g_device != nil) {
        return;
    }
    g_device = MTLCreateSystemDefaultDevice();
    TORCH_CHECK(g_device != nil, "Failed to create Metal device");

    NSError *error = nil;
    g_library = EMBEDDED_METALLIB_NAMESPACE::createLibrary(g_device, &error);
    TORCH_CHECK(g_library != nil,
                "Failed to create Metal library: ",
                error ? error.localizedDescription.UTF8String : "unknown error");
}

static id<MTLCommandQueue> getCommandQueue() {
    if (g_queue == nil) {
        g_queue = [g_device newCommandQueue];
        TORCH_CHECK(g_queue != nil, "Failed to create Metal command queue");
    }
    return g_queue;
}

static id<MTLComputePipelineState> getPipeline(NSString *name) {
    static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *cache = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        cache = [NSMutableDictionary new];
    });

    id<MTLComputePipelineState> pso = cache[name];
    if (pso != nil) {
        return pso;
    }

    id<MTLFunction> kernel = [g_library newFunctionWithName:name];
    TORCH_CHECK(kernel != nil, "Failed to get kernel: ", name.UTF8String);

    NSError *error = nil;
    pso = [g_device newComputePipelineStateWithFunction:kernel error:&error];
    TORCH_CHECK(pso != nil, "Failed to create PSO: ", error.localizedDescription.UTF8String);
    cache[name] = pso;
    return pso;
}

static inline id<MTLTexture> textureFromTensor(torch::Tensor tensor,
                                               MTLPixelFormat format,
                                               int64_t width,
                                               int64_t height,
                                               int64_t channels) {
    TORCH_CHECK(tensor.device().is_mps(), "tensor must be MPS");
    TORCH_CHECK(tensor.is_contiguous(), "tensor must be contiguous");
    TORCH_CHECK(width > 0 && height > 0, "invalid texture size");
    TORCH_CHECK(channels > 0, "invalid channel count");

    id<MTLBuffer> buffer = getMTLBufferStorage(tensor);
    TORCH_CHECK(buffer != nil, "tensor buffer is null");

    MTLTextureDescriptor *desc =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                           width:(NSUInteger)width
                                                          height:(NSUInteger)height
                                                       mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    desc.storageMode = MTLStorageModeShared;

    NSUInteger bytesPerRow = (NSUInteger)(width * channels * tensor.element_size());
    NSUInteger offset = (NSUInteger)(tensor.storage_offset() * tensor.element_size());
    id<MTLTexture> texture = [buffer newTextureWithDescriptor:desc
                                                      offset:offset
                                                 bytesPerRow:bytesPerRow];
    TORCH_CHECK(texture != nil, "Failed to create texture view from tensor buffer");
    return texture;
}

static inline void dispatchThreads2D(id<MTLComputeCommandEncoder> encoder,
                                     int64_t width,
                                     int64_t height,
                                     int64_t tgW = 16,
                                     int64_t tgH = 16) {
    MTLSize gridSize = MTLSizeMake(width, height, 1);
    MTLSize threadgroupSize = MTLSizeMake(tgW, tgH, 1);
    [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
}

static inline void dispatchThreadgroups1D(id<MTLComputeCommandEncoder> encoder,
                                          int64_t count,
                                          int64_t tg = 256) {
    MTLSize threadsPerThreadgroup = MTLSizeMake(tg, 1, 1);
    MTLSize threadgroups = MTLSizeMake((count + tg - 1) / tg, 1, 1);
    [encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threadsPerThreadgroup];
}

// Render Voronoi using buffer-based Metal kernel.
torch::Tensor renderVoronoi(
    torch::Tensor cand0,
    torch::Tensor cand1,
    torch::Tensor sites,
    double inv_scale_sq,
    int64_t width,
    int64_t height)
{
    @autoreleasepool {
        ensureMetalInitialized();

        TORCH_CHECK(cand0.device().is_mps(), "cand0 must be MPS tensor");
        TORCH_CHECK(cand1.device().is_mps(), "cand1 must be MPS tensor");
        TORCH_CHECK(sites.device().is_mps(), "sites must be MPS tensor");
        TORCH_CHECK(cand0.scalar_type() == torch::kInt32, "cand0 must be int32");
        TORCH_CHECK(cand1.scalar_type() == torch::kInt32, "cand1 must be int32");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
        TORCH_CHECK(cand1.dim() == 2 && cand1.size(1) == 4, "cand1 must be [M,4]");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
        TORCH_CHECK(width > 0 && height > 0, "width/height must be positive");
        TORCH_CHECK(cand0.size(0) == width * height,
                    "cand0 size must match width*height");
        TORCH_CHECK(cand1.size(0) == width * height,
                    "cand1 size must match width*height");

        cand0 = cand0.contiguous();
        cand1 = cand1.contiguous();
        sites = sites.contiguous();

        auto output = torch::zeros({height, width, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));

        id<MTLComputePipelineState> pso = getPipeline(@"renderVoronoiBuffer");

        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");

            [encoder setComputePipelineState:pso];

            [encoder setBuffer:getMTLBufferStorage(cand0) offset:cand0.storage_offset() * cand0.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(cand1) offset:cand1.storage_offset() * cand1.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(output) offset:output.storage_offset() * output.element_size() atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:3];

            float inv_scale_sq_f = static_cast<float>(inv_scale_sq);
            [encoder setBytes:&inv_scale_sq_f length:sizeof(float) atIndex:4];
            uint32_t siteCount = static_cast<uint32_t>(sites.size(0));
            [encoder setBytes:&siteCount length:sizeof(uint32_t) atIndex:5];
            uint32_t width_u32 = static_cast<uint32_t>(width);
            uint32_t height_u32 = static_cast<uint32_t>(height);
            [encoder setBytes:&width_u32 length:sizeof(uint32_t) atIndex:6];
            [encoder setBytes:&height_u32 length:sizeof(uint32_t) atIndex:7];

            MTLSize gridSize = MTLSizeMake(width, height, 1);
            MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            torch::mps::commit();
        });

        return output;
    }
}

torch::Tensor renderVoronoiPadded(
    torch::Tensor cand0,
    torch::Tensor cand1,
    torch::Tensor sites,
    double inv_scale_sq,
    int64_t width,
    int64_t height,
    int64_t cand_width,
    int64_t cand_height)
{
    @autoreleasepool {
        ensureMetalInitialized();

        TORCH_CHECK(cand0.device().is_mps(), "cand0 must be MPS tensor");
        TORCH_CHECK(cand1.device().is_mps(), "cand1 must be MPS tensor");
        TORCH_CHECK(sites.device().is_mps(), "sites must be MPS tensor");
        TORCH_CHECK(cand0.scalar_type() == torch::kInt32, "cand0 must be int32");
        TORCH_CHECK(cand1.scalar_type() == torch::kInt32, "cand1 must be int32");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
        TORCH_CHECK(cand1.dim() == 2 && cand1.size(1) == 4, "cand1 must be [M,4]");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        TORCH_CHECK(width > 0 && height > 0, "width/height must be positive");
        TORCH_CHECK(cand_width > 0 && cand_height > 0, "invalid cand size");
        TORCH_CHECK(cand0.size(0) == cand_width * cand_height,
                    "cand0 size must match cand_width*cand_height");
        TORCH_CHECK(cand1.size(0) == cand_width * cand_height,
                    "cand1 size must match cand_width*cand_height");

        cand0 = cand0.contiguous();
        cand1 = cand1.contiguous();
        sites = sites.contiguous();

        auto output = torch::zeros({height, width, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));

        id<MTLTexture> cand0Tex = textureFromTensor(cand0, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);
        id<MTLTexture> cand1Tex = textureFromTensor(cand1, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);
        id<MTLTexture> outTex = textureFromTensor(output, MTLPixelFormatRGBA32Float, width, height, 4);

        id<MTLComputePipelineState> pso = getPipeline(@"renderVoronoi");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");

            [encoder setComputePipelineState:pso];
            [encoder setTexture:cand0Tex atIndex:0];
            [encoder setTexture:cand1Tex atIndex:1];
            [encoder setTexture:outTex atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            float inv_scale_sq_f = static_cast<float>(inv_scale_sq);
            [encoder setBytes:&inv_scale_sq_f length:sizeof(float) atIndex:1];
            uint32_t siteCount = static_cast<uint32_t>(sites.size(0));
            [encoder setBytes:&siteCount length:sizeof(uint32_t) atIndex:2];

            MTLSize gridSize = MTLSizeMake(width, height, 1);
            MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];

            torch::mps::commit();
        });

        return output;
    }
}

// Backward is not implemented for Metal (training uses custom kernels).
torch::Tensor renderVoronoiBackward(
    torch::Tensor cand0,
    torch::Tensor cand1,
    torch::Tensor sites,
    torch::Tensor grad_output,
    double inv_scale_sq,
    int64_t width,
    int64_t height) {
    @autoreleasepool {
        ensureMetalInitialized();

        TORCH_CHECK(cand0.device().is_mps(), "cand0 must be MPS tensor");
        TORCH_CHECK(cand1.device().is_mps(), "cand1 must be MPS tensor");
        TORCH_CHECK(sites.device().is_mps(), "sites must be MPS tensor");
        TORCH_CHECK(grad_output.device().is_mps(), "grad_output must be MPS tensor");
        TORCH_CHECK(cand0.scalar_type() == torch::kInt32, "cand0 must be int32");
        TORCH_CHECK(cand1.scalar_type() == torch::kInt32, "cand1 must be int32");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(grad_output.scalar_type() == torch::kFloat32, "grad_output must be float32");
        TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
        TORCH_CHECK(cand1.dim() == 2 && cand1.size(1) == 4, "cand1 must be [M,4]");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 10, "sites must be [N,10]");
        TORCH_CHECK(grad_output.dim() == 3 && grad_output.size(2) == 3,
                    "grad_output must be [H,W,3]");
        TORCH_CHECK(width > 0 && height > 0, "width/height must be positive");
        TORCH_CHECK(cand0.size(0) == width * height,
                    "cand0 size must match width*height");
        TORCH_CHECK(cand1.size(0) == width * height,
                    "cand1 size must match width*height");
        TORCH_CHECK(grad_output.size(0) == height && grad_output.size(1) == width,
                    "grad_output size must match height/width");

        cand0 = cand0.contiguous();
        cand1 = cand1.contiguous();
        sites = sites.contiguous();
        grad_output = grad_output.contiguous();

        auto grad_sites = torch::zeros({sites.size(0), 10},
                                       torch::TensorOptions().dtype(torch::kFloat32).device(torch::kMPS));

        id<MTLComputePipelineState> pso = getPipeline(@"renderVoronoiBackwardBuffer");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");

            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(cand0) offset:cand0.storage_offset() * cand0.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(cand1) offset:cand1.storage_offset() * cand1.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(grad_output) offset:grad_output.storage_offset() * grad_output.element_size() atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(grad_sites) offset:grad_sites.storage_offset() * grad_sites.element_size() atIndex:3];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:4];

            float inv_scale_sq_f = static_cast<float>(inv_scale_sq);
            uint32_t siteCount = static_cast<uint32_t>(sites.size(0));
            uint32_t width_u32 = static_cast<uint32_t>(width);
            uint32_t height_u32 = static_cast<uint32_t>(height);
            [encoder setBytes:&inv_scale_sq_f length:sizeof(float) atIndex:5];
            [encoder setBytes:&siteCount length:sizeof(uint32_t) atIndex:6];
            [encoder setBytes:&width_u32 length:sizeof(uint32_t) atIndex:7];
            [encoder setBytes:&height_u32 length:sizeof(uint32_t) atIndex:8];

            dispatchThreads2D(encoder, width, height, 16, 16);
            [encoder endEncoding];

            torch::mps::commit();
        });

        return grad_sites;
    }
}

std::tuple<torch::Tensor, torch::Tensor> initCandidates(int64_t width, int64_t height, int64_t siteCount, int64_t seed) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(width > 0 && height > 0, "invalid texture size");
        TORCH_CHECK(siteCount > 0, "siteCount must be positive");

        auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kMPS);
        auto cand0 = torch::zeros({height * width, 4}, options).contiguous();
        auto cand1 = torch::zeros({height * width, 4}, options).contiguous();

        id<MTLTexture> cand0Tex = textureFromTensor(cand0, MTLPixelFormatRGBA32Uint, width, height, 4);
        id<MTLTexture> cand1Tex = textureFromTensor(cand1, MTLPixelFormatRGBA32Uint, width, height, 4);

        id<MTLComputePipelineState> pso = getPipeline(@"initCandidates");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");

            [encoder setComputePipelineState:pso];
            [encoder setTexture:cand0Tex atIndex:0];
            [encoder setTexture:cand1Tex atIndex:1];

            uint32_t siteCountU = static_cast<uint32_t>(siteCount);
            uint32_t seedU = static_cast<uint32_t>(seed);
            bool perPixel = (siteCount == width * height);
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:0];
            [encoder setBytes:&seedU length:sizeof(uint32_t) atIndex:1];
            [encoder setBytes:&perPixel length:sizeof(bool) atIndex:2];

            dispatchThreads2D(encoder, width, height, 16, 16);
            [encoder endEncoding];

            torch::mps::commit();
        });

        return std::make_tuple(cand0, cand1);
    }
}

std::tuple<torch::Tensor, torch::Tensor> vptPass(
    torch::Tensor cand0,
    torch::Tensor cand1,
    torch::Tensor sites,
    double inv_scale_sq,
    int64_t width,
    int64_t height,
    int64_t jump,
    int64_t radius_probes,
    double radius_scale,
    int64_t inject_count,
    int64_t seed) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(cand0.device().is_mps(), "cand0 must be MPS tensor");
        TORCH_CHECK(cand1.device().is_mps(), "cand1 must be MPS tensor");
        TORCH_CHECK(sites.device().is_mps(), "sites must be MPS tensor");
        TORCH_CHECK(cand0.scalar_type() == torch::kInt32, "cand0 must be int32");
        TORCH_CHECK(cand1.scalar_type() == torch::kInt32, "cand1 must be int32");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(width > 0 && height > 0, "invalid texture size");
        TORCH_CHECK(cand0.numel() == width * height * 4, "cand0 size mismatch");
        TORCH_CHECK(cand1.numel() == width * height * 4, "cand1 size mismatch");
        TORCH_CHECK(sites.dim() == 2 && (sites.size(1) == 10 || sites.size(1) == 12),
                    "sites must be [N,10] or [N,12]");

        cand0 = cand0.contiguous();
        cand1 = cand1.contiguous();
        sites = sites.contiguous();

        auto out0 = torch::zeros_like(cand0);
        auto out1 = torch::zeros_like(cand1);

        id<MTLTexture> cand0In = textureFromTensor(cand0, MTLPixelFormatRGBA32Uint, width, height, 4);
        id<MTLTexture> cand1In = textureFromTensor(cand1, MTLPixelFormatRGBA32Uint, width, height, 4);
        id<MTLTexture> cand0Out = textureFromTensor(out0, MTLPixelFormatRGBA32Uint, width, height, 4);
        id<MTLTexture> cand1Out = textureFromTensor(out1, MTLPixelFormatRGBA32Uint, width, height, 4);

        id<MTLBuffer> sitesBuffer = getMTLBufferStorage(sites);

        NSUInteger packedStride = sizeof(uint16_t) * 8; // 2 * half4
        id<MTLBuffer> packedSites = [g_device newBufferWithLength:(packedStride * sites.size(0))
                                                          options:MTLResourceStorageModeShared];
        TORCH_CHECK(packedSites != nil, "Failed to allocate packed sites buffer");

        static id<MTLBuffer> dummyBuffer = nil;
        if (dummyBuffer == nil) {
            dummyBuffer = [g_device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        }

        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            // Pack sites
            NSString *packKernel = (sites.size(1) == 10) ? @"packCandidateSitesRaw" : @"packCandidateSites";
            id<MTLComputePipelineState> packPSO = getPipeline(packKernel);
            id<MTLComputeCommandEncoder> packEnc = [cmdBuf computeCommandEncoder];
            [packEnc setComputePipelineState:packPSO];
            [packEnc setBuffer:sitesBuffer offset:sites.storage_offset() * sites.element_size() atIndex:0];
            [packEnc setBuffer:packedSites offset:0 atIndex:1];
            uint32_t siteCountU = static_cast<uint32_t>(sites.size(0));
            [packEnc setBytes:&siteCountU length:sizeof(uint32_t) atIndex:2];
            dispatchThreadgroups1D(packEnc, sites.size(0), 256);
            [packEnc endEncoding];

            // Update candidates (single pass)
            id<MTLComputePipelineState> updatePSO = getPipeline(@"updateCandidatesCompact");
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            [encoder setComputePipelineState:updatePSO];
            [encoder setTexture:cand0In atIndex:0];
            [encoder setTexture:cand1In atIndex:1];
            [encoder setTexture:cand0Out atIndex:2];
            [encoder setTexture:cand1Out atIndex:3];
            [encoder setBuffer:packedSites offset:0 atIndex:0];
            [encoder setBuffer:dummyBuffer offset:0 atIndex:8];
            [encoder setBuffer:dummyBuffer offset:0 atIndex:9];

            uint32_t siteCountVar = static_cast<uint32_t>(sites.size(0));
            uint32_t step = (static_cast<uint32_t>(jump) << 16);
            float invScale = static_cast<float>(inv_scale_sq);
            uint32_t stepHigh = 0;
            float radiusScale = static_cast<float>(radius_scale);
            uint32_t radiusProbes = static_cast<uint32_t>(radius_probes);
            uint32_t injectCount = static_cast<uint32_t>(inject_count);
            uint32_t hilbertProbe = 0;
            uint32_t hilbertWindow = 0;
            uint32_t candDownscale = 1;
            uint32_t targetWidth = static_cast<uint32_t>(width);
            uint32_t targetHeight = static_cast<uint32_t>(height);

            [encoder setBytes:&siteCountVar length:sizeof(uint32_t) atIndex:1];
            [encoder setBytes:&step length:sizeof(uint32_t) atIndex:2];
            [encoder setBytes:&invScale length:sizeof(float) atIndex:3];
            [encoder setBytes:&stepHigh length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&radiusScale length:sizeof(float) atIndex:5];
            [encoder setBytes:&radiusProbes length:sizeof(uint32_t) atIndex:6];
            [encoder setBytes:&injectCount length:sizeof(uint32_t) atIndex:7];
            [encoder setBytes:&hilbertProbe length:sizeof(uint32_t) atIndex:10];
            [encoder setBytes:&hilbertWindow length:sizeof(uint32_t) atIndex:11];
            [encoder setBytes:&candDownscale length:sizeof(uint32_t) atIndex:12];
            [encoder setBytes:&targetWidth length:sizeof(uint32_t) atIndex:13];
            [encoder setBytes:&targetHeight length:sizeof(uint32_t) atIndex:14];

            dispatchThreads2D(encoder, width, height, 16, 16);
            [encoder endEncoding];

            torch::mps::commit();
        });

        return std::make_tuple(out0, out1);
    }
}

// Clear a float buffer (first count elements).
torch::Tensor clearBuffer(torch::Tensor buffer, int64_t count) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(buffer.device().is_mps(), "buffer must be MPS tensor");
        TORCH_CHECK(buffer.scalar_type() == torch::kFloat32, "buffer must be float32");
        buffer = buffer.contiguous();
        int64_t numel = buffer.numel();
        if (count <= 0 || count > numel) {
            count = numel;
        }

        id<MTLComputePipelineState> pso = getPipeline(@"clearAtomicBuffer");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(buffer) offset:buffer.storage_offset() * buffer.element_size() atIndex:0];
            uint32_t countU = static_cast<uint32_t>(count);
            [encoder setBytes:&countU length:sizeof(uint32_t) atIndex:1];
            dispatchThreadgroups1D(encoder, count, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return buffer;
    }
}

torch::Tensor packCandidateSites(torch::Tensor sites, torch::Tensor packed, int64_t site_count) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(sites.device().is_mps(), "sites must be MPS tensor");
        TORCH_CHECK(packed.device().is_mps(), "packed must be MPS tensor");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(packed.scalar_type() == torch::kFloat16, "packed must be float16");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12] padded layout");
        TORCH_CHECK(packed.dim() == 2 && packed.size(1) == 8, "packed must be [N,8] half layout");
        sites = sites.contiguous();
        packed = packed.contiguous();
        if (site_count <= 0) {
            site_count = sites.size(0);
        }

        id<MTLComputePipelineState> pso = getPipeline(@"packCandidateSites");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(packed) offset:packed.storage_offset() * packed.element_size() atIndex:1];
            uint32_t siteCountU = static_cast<uint32_t>(site_count);
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:2];
            dispatchThreadgroups1D(encoder, site_count, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return packed;
    }
}

torch::Tensor updateCandidatesCompact(torch::Tensor cand0_in, torch::Tensor cand1_in,
                                      torch::Tensor cand0_out, torch::Tensor cand1_out,
                                      torch::Tensor packed_sites, torch::Tensor hilbert_order,
                                      torch::Tensor hilbert_pos, double inv_scale_sq,
                                      int64_t site_count, int64_t step, int64_t step_high,
                                      double radius_scale, int64_t radius_probes, int64_t inject_count,
                                      int64_t hilbert_probes, int64_t hilbert_window,
                                      int64_t cand_downscale, int64_t target_width, int64_t target_height,
                                      int64_t cand_width, int64_t cand_height) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(cand0_in.device().is_mps() && cand1_in.device().is_mps(), "cand inputs must be MPS tensors");
        TORCH_CHECK(cand0_out.device().is_mps() && cand1_out.device().is_mps(), "cand outputs must be MPS tensors");
        TORCH_CHECK(packed_sites.device().is_mps(), "packed_sites must be MPS tensor");
        TORCH_CHECK(cand0_in.scalar_type() == torch::kInt32, "cand0_in must be int32");
        TORCH_CHECK(cand1_in.scalar_type() == torch::kInt32, "cand1_in must be int32");
        TORCH_CHECK(cand0_out.scalar_type() == torch::kInt32, "cand0_out must be int32");
        TORCH_CHECK(cand1_out.scalar_type() == torch::kInt32, "cand1_out must be int32");
        TORCH_CHECK(packed_sites.scalar_type() == torch::kFloat16, "packed_sites must be float16");
        TORCH_CHECK(cand0_in.dim() == 2 && cand0_in.size(1) == 4, "cand0_in must be [M,4]");
        TORCH_CHECK(cand1_in.dim() == 2 && cand1_in.size(1) == 4, "cand1_in must be [M,4]");
        TORCH_CHECK(cand0_out.dim() == 2 && cand0_out.size(1) == 4, "cand0_out must be [M,4]");
        TORCH_CHECK(cand1_out.dim() == 2 && cand1_out.size(1) == 4, "cand1_out must be [M,4]");
        TORCH_CHECK(cand_width > 0 && cand_height > 0, "invalid cand size");
        TORCH_CHECK(cand0_in.size(0) == cand_width * cand_height, "cand0_in size mismatch");
        TORCH_CHECK(cand1_in.size(0) == cand_width * cand_height, "cand1_in size mismatch");
        TORCH_CHECK(cand0_out.size(0) == cand_width * cand_height, "cand0_out size mismatch");
        TORCH_CHECK(cand1_out.size(0) == cand_width * cand_height, "cand1_out size mismatch");

        cand0_in = cand0_in.contiguous();
        cand1_in = cand1_in.contiguous();
        cand0_out = cand0_out.contiguous();
        cand1_out = cand1_out.contiguous();
        packed_sites = packed_sites.contiguous();
        if (hilbert_order.defined()) hilbert_order = hilbert_order.contiguous();
        if (hilbert_pos.defined()) hilbert_pos = hilbert_pos.contiguous();

        id<MTLTexture> cand0In = textureFromTensor(cand0_in, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);
        id<MTLTexture> cand1In = textureFromTensor(cand1_in, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);
        id<MTLTexture> cand0Out = textureFromTensor(cand0_out, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);
        id<MTLTexture> cand1Out = textureFromTensor(cand1_out, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);

        static id<MTLBuffer> dummyBuffer = nil;
        if (dummyBuffer == nil) {
            dummyBuffer = [g_device newBufferWithLength:4 options:MTLResourceStorageModeShared];
        }

        id<MTLComputePipelineState> pso = getPipeline(@"updateCandidatesCompact");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setTexture:cand0In atIndex:0];
            [encoder setTexture:cand1In atIndex:1];
            [encoder setTexture:cand0Out atIndex:2];
            [encoder setTexture:cand1Out atIndex:3];
            [encoder setBuffer:getMTLBufferStorage(packed_sites) offset:packed_sites.storage_offset() * packed_sites.element_size() atIndex:0];

            id<MTLBuffer> hilbertOrderBuf = hilbert_order.defined() ? getMTLBufferStorage(hilbert_order) : dummyBuffer;
            id<MTLBuffer> hilbertPosBuf = hilbert_pos.defined() ? getMTLBufferStorage(hilbert_pos) : dummyBuffer;
            [encoder setBuffer:hilbertOrderBuf offset:hilbert_order.defined() ? hilbert_order.storage_offset() * hilbert_order.element_size() : 0 atIndex:8];
            [encoder setBuffer:hilbertPosBuf offset:hilbert_pos.defined() ? hilbert_pos.storage_offset() * hilbert_pos.element_size() : 0 atIndex:9];

            uint32_t siteCountU = static_cast<uint32_t>(site_count);
            uint32_t stepU = static_cast<uint32_t>(step);
            uint32_t stepHighU = static_cast<uint32_t>(step_high);
            float invScale = static_cast<float>(inv_scale_sq);
            float radiusScale = static_cast<float>(radius_scale);
            uint32_t radiusProbesU = static_cast<uint32_t>(radius_probes);
            uint32_t injectCountU = static_cast<uint32_t>(inject_count);
            uint32_t hilbertProbeU = static_cast<uint32_t>(hilbert_probes);
            uint32_t hilbertWindowU = static_cast<uint32_t>(hilbert_window);
            uint32_t candDownU = static_cast<uint32_t>(cand_downscale);
            uint32_t targetWU = static_cast<uint32_t>(target_width);
            uint32_t targetHU = static_cast<uint32_t>(target_height);

            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:1];
            [encoder setBytes:&stepU length:sizeof(uint32_t) atIndex:2];
            [encoder setBytes:&invScale length:sizeof(float) atIndex:3];
            [encoder setBytes:&stepHighU length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&radiusScale length:sizeof(float) atIndex:5];
            [encoder setBytes:&radiusProbesU length:sizeof(uint32_t) atIndex:6];
            [encoder setBytes:&injectCountU length:sizeof(uint32_t) atIndex:7];
            [encoder setBytes:&hilbertProbeU length:sizeof(uint32_t) atIndex:10];
            [encoder setBytes:&hilbertWindowU length:sizeof(uint32_t) atIndex:11];
            [encoder setBytes:&candDownU length:sizeof(uint32_t) atIndex:12];
            [encoder setBytes:&targetWU length:sizeof(uint32_t) atIndex:13];
            [encoder setBytes:&targetHU length:sizeof(uint32_t) atIndex:14];

            dispatchThreads2D(encoder, cand_width, cand_height, 16, 16);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return cand0_out;
    }
}

torch::Tensor jfaClear(torch::Tensor cand0, torch::Tensor cand1, int64_t cand_width, int64_t cand_height) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(cand0.device().is_mps() && cand1.device().is_mps(), "cand tensors must be MPS");
        TORCH_CHECK(cand0.scalar_type() == torch::kInt32 && cand1.scalar_type() == torch::kInt32, "cand tensors must be int32");
        TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
        TORCH_CHECK(cand1.dim() == 2 && cand1.size(1) == 4, "cand1 must be [M,4]");
        TORCH_CHECK(cand_width > 0 && cand_height > 0, "invalid cand size");
        TORCH_CHECK(cand0.size(0) == cand_width * cand_height, "cand0 size mismatch");
        TORCH_CHECK(cand1.size(0) == cand_width * cand_height, "cand1 size mismatch");
        cand0 = cand0.contiguous();
        cand1 = cand1.contiguous();

        id<MTLTexture> cand0Tex = textureFromTensor(cand0, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);
        id<MTLTexture> cand1Tex = textureFromTensor(cand1, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);

        id<MTLComputePipelineState> pso = getPipeline(@"jfaClearCandidates");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setTexture:cand0Tex atIndex:0];
            [encoder setTexture:cand1Tex atIndex:1];
            dispatchThreads2D(encoder, cand_width, cand_height, 16, 16);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return cand0;
    }
}

torch::Tensor jfaSeed(torch::Tensor cand0, torch::Tensor sites, int64_t site_count,
                      int64_t cand_downscale, int64_t cand_width, int64_t cand_height) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(cand0.device().is_mps(), "cand0 must be MPS tensor");
        TORCH_CHECK(sites.device().is_mps(), "sites must be MPS tensor");
        TORCH_CHECK(cand0.scalar_type() == torch::kInt32, "cand0 must be int32");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        TORCH_CHECK(cand_width > 0 && cand_height > 0, "invalid cand size");
        TORCH_CHECK(cand0.size(0) == cand_width * cand_height, "cand0 size mismatch");
        cand0 = cand0.contiguous();
        sites = sites.contiguous();
        if (site_count <= 0) {
            site_count = sites.size(0);
        }

        id<MTLTexture> cand0Tex = textureFromTensor(cand0, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);

        id<MTLComputePipelineState> pso = getPipeline(@"jfaSeed");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setTexture:cand0Tex atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            uint32_t siteCountU = static_cast<uint32_t>(site_count);
            uint32_t candDownU = static_cast<uint32_t>(cand_downscale);
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:1];
            [encoder setBytes:&candDownU length:sizeof(uint32_t) atIndex:2];
            dispatchThreadgroups1D(encoder, site_count, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return cand0;
    }
}

torch::Tensor jfaFlood(torch::Tensor cand0_in, torch::Tensor cand0_out, torch::Tensor sites,
                       double inv_scale_sq, int64_t site_count, int64_t step_size,
                       int64_t cand_downscale, int64_t cand_width, int64_t cand_height) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(cand0_in.device().is_mps() && cand0_out.device().is_mps(), "cand tensors must be MPS");
        TORCH_CHECK(sites.device().is_mps(), "sites must be MPS tensor");
        TORCH_CHECK(cand0_in.scalar_type() == torch::kInt32 && cand0_out.scalar_type() == torch::kInt32, "cand tensors must be int32");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(cand0_in.dim() == 2 && cand0_in.size(1) == 4, "cand0_in must be [M,4]");
        TORCH_CHECK(cand0_out.dim() == 2 && cand0_out.size(1) == 4, "cand0_out must be [M,4]");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        TORCH_CHECK(cand_width > 0 && cand_height > 0, "invalid cand size");
        TORCH_CHECK(cand0_in.size(0) == cand_width * cand_height, "cand0_in size mismatch");
        TORCH_CHECK(cand0_out.size(0) == cand_width * cand_height, "cand0_out size mismatch");
        cand0_in = cand0_in.contiguous();
        cand0_out = cand0_out.contiguous();
        sites = sites.contiguous();
        if (site_count <= 0) {
            site_count = sites.size(0);
        }

        id<MTLTexture> cand0In = textureFromTensor(cand0_in, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);
        id<MTLTexture> cand0Out = textureFromTensor(cand0_out, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);

        id<MTLComputePipelineState> pso = getPipeline(@"jfaFlood");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setTexture:cand0In atIndex:0];
            [encoder setTexture:cand0Out atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            uint32_t siteCountU = static_cast<uint32_t>(site_count);
            uint32_t stepU = static_cast<uint32_t>(step_size);
            float invScale = static_cast<float>(inv_scale_sq);
            uint32_t candDownU = static_cast<uint32_t>(cand_downscale);
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:1];
            [encoder setBytes:&stepU length:sizeof(uint32_t) atIndex:2];
            [encoder setBytes:&invScale length:sizeof(float) atIndex:3];
            [encoder setBytes:&candDownU length:sizeof(uint32_t) atIndex:4];
            dispatchThreads2D(encoder, cand_width, cand_height, 16, 16);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return cand0_out;
    }
}

torch::Tensor computeGradientsTiled(torch::Tensor cand0, torch::Tensor cand1,
                                    torch::Tensor target, torch::Tensor rendered, torch::Tensor mask,
                                    torch::Tensor sites,
                                    torch::Tensor grad_pos_x, torch::Tensor grad_pos_y,
                                    torch::Tensor grad_log_tau, torch::Tensor grad_radius,
                                    torch::Tensor grad_color_r, torch::Tensor grad_color_g, torch::Tensor grad_color_b,
                                    torch::Tensor grad_dir_x, torch::Tensor grad_dir_y, torch::Tensor grad_log_aniso,
                                    torch::Tensor removal_delta,
                                    double inv_scale_sq, int64_t site_count,
                                    int64_t compute_removal, double ssim_weight,
                                    int64_t cand_width, int64_t cand_height) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(cand0.device().is_mps() && cand1.device().is_mps(), "cand tensors must be MPS");
        TORCH_CHECK(target.device().is_mps() && rendered.device().is_mps() && mask.device().is_mps(),
                    "target/rendered/mask must be MPS");
        TORCH_CHECK(sites.device().is_mps(), "sites must be MPS");
        TORCH_CHECK(cand0.scalar_type() == torch::kInt32 && cand1.scalar_type() == torch::kInt32, "cand tensors must be int32");
        TORCH_CHECK(target.scalar_type() == torch::kFloat32 && rendered.scalar_type() == torch::kFloat32 &&
                    mask.scalar_type() == torch::kFloat32, "target/rendered/mask must be float32");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        TORCH_CHECK(target.dim() == 3 && target.size(2) == 4, "target must be [H,W,4]");
        TORCH_CHECK(rendered.dim() == 3 && rendered.size(2) == 4, "rendered must be [H,W,4]");
        TORCH_CHECK(mask.dim() == 3 && mask.size(2) == 4, "mask must be [H,W,4]");
        TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
        TORCH_CHECK(cand1.dim() == 2 && cand1.size(1) == 4, "cand1 must be [M,4]");
        TORCH_CHECK(cand_width > 0 && cand_height > 0, "invalid cand size");
        TORCH_CHECK(cand0.size(0) == cand_width * cand_height, "cand0 size mismatch");
        TORCH_CHECK(cand1.size(0) == cand_width * cand_height, "cand1 size mismatch");

        cand0 = cand0.contiguous();
        cand1 = cand1.contiguous();
        target = target.contiguous();
        rendered = rendered.contiguous();
        mask = mask.contiguous();
        sites = sites.contiguous();
        grad_pos_x = grad_pos_x.contiguous();
        grad_pos_y = grad_pos_y.contiguous();
        grad_log_tau = grad_log_tau.contiguous();
        grad_radius = grad_radius.contiguous();
        grad_color_r = grad_color_r.contiguous();
        grad_color_g = grad_color_g.contiguous();
        grad_color_b = grad_color_b.contiguous();
        grad_dir_x = grad_dir_x.contiguous();
        grad_dir_y = grad_dir_y.contiguous();
        grad_log_aniso = grad_log_aniso.contiguous();
        removal_delta = removal_delta.contiguous();

        int64_t height = target.size(0);
        int64_t width = target.size(1);
        if (site_count <= 0) {
            site_count = sites.size(0);
        }

        id<MTLTexture> cand0Tex = textureFromTensor(cand0, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);
        id<MTLTexture> cand1Tex = textureFromTensor(cand1, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);
        id<MTLTexture> targetTex = textureFromTensor(target, MTLPixelFormatRGBA32Float, width, height, 4);
        id<MTLTexture> renderedTex = textureFromTensor(rendered, MTLPixelFormatRGBA32Float, width, height, 4);
        id<MTLTexture> maskTex = textureFromTensor(mask, MTLPixelFormatRGBA32Float, width, height, 4);

        id<MTLComputePipelineState> pso = getPipeline(@"computeGradientsTiled");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setTexture:cand0Tex atIndex:0];
            [encoder setTexture:cand1Tex atIndex:1];
            [encoder setTexture:targetTex atIndex:2];
            [encoder setTexture:renderedTex atIndex:3];
            [encoder setTexture:maskTex atIndex:4];
            [encoder setBuffer:getMTLBufferStorage(grad_pos_x) offset:grad_pos_x.storage_offset() * grad_pos_x.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(grad_pos_y) offset:grad_pos_y.storage_offset() * grad_pos_y.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(grad_log_tau) offset:grad_log_tau.storage_offset() * grad_log_tau.element_size() atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(grad_radius) offset:grad_radius.storage_offset() * grad_radius.element_size() atIndex:3];
            [encoder setBuffer:getMTLBufferStorage(grad_color_r) offset:grad_color_r.storage_offset() * grad_color_r.element_size() atIndex:4];
            [encoder setBuffer:getMTLBufferStorage(grad_color_g) offset:grad_color_g.storage_offset() * grad_color_g.element_size() atIndex:5];
            [encoder setBuffer:getMTLBufferStorage(grad_color_b) offset:grad_color_b.storage_offset() * grad_color_b.element_size() atIndex:6];
            [encoder setBuffer:getMTLBufferStorage(grad_dir_x) offset:grad_dir_x.storage_offset() * grad_dir_x.element_size() atIndex:7];
            [encoder setBuffer:getMTLBufferStorage(grad_dir_y) offset:grad_dir_y.storage_offset() * grad_dir_y.element_size() atIndex:8];
            [encoder setBuffer:getMTLBufferStorage(grad_log_aniso) offset:grad_log_aniso.storage_offset() * grad_log_aniso.element_size() atIndex:9];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:10];
            float invScale = static_cast<float>(inv_scale_sq);
            uint32_t siteCountU = static_cast<uint32_t>(site_count);
            [encoder setBytes:&invScale length:sizeof(float) atIndex:11];
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:12];
            [encoder setBuffer:getMTLBufferStorage(removal_delta) offset:removal_delta.storage_offset() * removal_delta.element_size() atIndex:13];
            uint32_t computeRemovalU = static_cast<uint32_t>(compute_removal);
            [encoder setBytes:&computeRemovalU length:sizeof(uint32_t) atIndex:14];
            float ssimW = static_cast<float>(ssim_weight);
            [encoder setBytes:&ssimW length:sizeof(float) atIndex:15];

            const int tileHashSize = 256;
            size_t keyMem = tileHashSize * sizeof(uint32_t);
            size_t gradMem = tileHashSize * sizeof(int32_t);
            [encoder setThreadgroupMemoryLength:keyMem atIndex:0];
            for (int i = 0; i < 10; ++i) {
                [encoder setThreadgroupMemoryLength:gradMem atIndex:1 + i];
            }
            [encoder setThreadgroupMemoryLength:gradMem atIndex:11];

            MTLSize tgs = MTLSizeMake(16, 16, 1);
            MTLSize tg = MTLSizeMake((width + 15) / 16, (height + 15) / 16, 1);
            [encoder dispatchThreadgroups:tg threadsPerThreadgroup:tgs];
            [encoder endEncoding];
            torch::mps::commit();
        });

        return removal_delta;
    }
}

torch::Tensor tauDiffuse(torch::Tensor cand0, torch::Tensor cand1, torch::Tensor sites,
                         torch::Tensor grad_raw, torch::Tensor grad_in, torch::Tensor grad_out,
                         int64_t site_count, double lambda, int64_t cand_downscale,
                         int64_t cand_width, int64_t cand_height) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(cand0.device().is_mps() && cand1.device().is_mps(), "cand tensors must be MPS");
        TORCH_CHECK(sites.device().is_mps(), "sites must be MPS");
        TORCH_CHECK(grad_raw.device().is_mps() && grad_in.device().is_mps() && grad_out.device().is_mps(),
                    "grad tensors must be MPS");
        TORCH_CHECK(cand0.scalar_type() == torch::kInt32 && cand1.scalar_type() == torch::kInt32, "cand tensors must be int32");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(grad_raw.scalar_type() == torch::kFloat32, "grad tensors must be float32");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
        TORCH_CHECK(cand1.dim() == 2 && cand1.size(1) == 4, "cand1 must be [M,4]");
        TORCH_CHECK(cand_width > 0 && cand_height > 0, "invalid cand size");
        TORCH_CHECK(cand0.size(0) == cand_width * cand_height, "cand0 size mismatch");
        TORCH_CHECK(cand1.size(0) == cand_width * cand_height, "cand1 size mismatch");
        cand0 = cand0.contiguous();
        cand1 = cand1.contiguous();
        sites = sites.contiguous();
        grad_raw = grad_raw.contiguous();
        grad_in = grad_in.contiguous();
        grad_out = grad_out.contiguous();
        if (site_count <= 0) {
            site_count = sites.size(0);
        }

        id<MTLTexture> cand0Tex = textureFromTensor(cand0, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);
        id<MTLTexture> cand1Tex = textureFromTensor(cand1, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);

        id<MTLComputePipelineState> pso = getPipeline(@"diffuseTauGradientsAtSite");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setTexture:cand0Tex atIndex:0];
            [encoder setTexture:cand1Tex atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(grad_raw) offset:grad_raw.storage_offset() * grad_raw.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(grad_in) offset:grad_in.storage_offset() * grad_in.element_size() atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(grad_out) offset:grad_out.storage_offset() * grad_out.element_size() atIndex:3];
            uint32_t siteCountU = static_cast<uint32_t>(site_count);
            float lambdaF = static_cast<float>(lambda);
            uint32_t candDownU = static_cast<uint32_t>(cand_downscale);
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&lambdaF length:sizeof(float) atIndex:5];
            [encoder setBytes:&candDownU length:sizeof(uint32_t) atIndex:6];
            dispatchThreadgroups1D(encoder, site_count, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return grad_out;
    }
}

torch::Tensor adamUpdate(torch::Tensor sites, torch::Tensor adam,
                         torch::Tensor grad_pos_x, torch::Tensor grad_pos_y,
                         torch::Tensor grad_log_tau, torch::Tensor grad_radius,
                         torch::Tensor grad_color_r, torch::Tensor grad_color_g, torch::Tensor grad_color_b,
                         torch::Tensor grad_dir_x, torch::Tensor grad_dir_y, torch::Tensor grad_log_aniso,
                         double lr_pos, double lr_tau, double lr_radius, double lr_color,
                         double lr_dir, double lr_aniso, double beta1, double beta2, double eps,
                         int64_t t, int64_t width, int64_t height) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(sites.device().is_mps(), "sites must be MPS");
        TORCH_CHECK(adam.device().is_mps(), "adam must be MPS");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(adam.scalar_type() == torch::kFloat32, "adam must be float32");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        sites = sites.contiguous();
        adam = adam.contiguous();
        grad_pos_x = grad_pos_x.contiguous();
        grad_pos_y = grad_pos_y.contiguous();
        grad_log_tau = grad_log_tau.contiguous();
        grad_radius = grad_radius.contiguous();
        grad_color_r = grad_color_r.contiguous();
        grad_color_g = grad_color_g.contiguous();
        grad_color_b = grad_color_b.contiguous();
        grad_dir_x = grad_dir_x.contiguous();
        grad_dir_y = grad_dir_y.contiguous();
        grad_log_aniso = grad_log_aniso.contiguous();

        int64_t site_count = sites.size(0);
        id<MTLComputePipelineState> pso = getPipeline(@"adamUpdate");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(adam) offset:adam.storage_offset() * adam.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(grad_pos_x) offset:grad_pos_x.storage_offset() * grad_pos_x.element_size() atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(grad_pos_y) offset:grad_pos_y.storage_offset() * grad_pos_y.element_size() atIndex:3];
            [encoder setBuffer:getMTLBufferStorage(grad_log_tau) offset:grad_log_tau.storage_offset() * grad_log_tau.element_size() atIndex:4];
            [encoder setBuffer:getMTLBufferStorage(grad_radius) offset:grad_radius.storage_offset() * grad_radius.element_size() atIndex:5];
            [encoder setBuffer:getMTLBufferStorage(grad_color_r) offset:grad_color_r.storage_offset() * grad_color_r.element_size() atIndex:6];
            [encoder setBuffer:getMTLBufferStorage(grad_color_g) offset:grad_color_g.storage_offset() * grad_color_g.element_size() atIndex:7];
            [encoder setBuffer:getMTLBufferStorage(grad_color_b) offset:grad_color_b.storage_offset() * grad_color_b.element_size() atIndex:8];
            [encoder setBuffer:getMTLBufferStorage(grad_dir_x) offset:grad_dir_x.storage_offset() * grad_dir_x.element_size() atIndex:9];
            [encoder setBuffer:getMTLBufferStorage(grad_dir_y) offset:grad_dir_y.storage_offset() * grad_dir_y.element_size() atIndex:10];
            [encoder setBuffer:getMTLBufferStorage(grad_log_aniso) offset:grad_log_aniso.storage_offset() * grad_log_aniso.element_size() atIndex:11];

            float lrPosF = static_cast<float>(lr_pos);
            float lrTauF = static_cast<float>(lr_tau);
            float lrRadiusF = static_cast<float>(lr_radius);
            float lrColorF = static_cast<float>(lr_color);
            float lrDirF = static_cast<float>(lr_dir);
            float lrAnisoF = static_cast<float>(lr_aniso);
            float beta1F = static_cast<float>(beta1);
            float beta2F = static_cast<float>(beta2);
            float epsF = static_cast<float>(eps);
            uint32_t tU = static_cast<uint32_t>(t);
            uint32_t wU = static_cast<uint32_t>(width);
            uint32_t hU = static_cast<uint32_t>(height);

            [encoder setBytes:&lrPosF length:sizeof(float) atIndex:12];
            [encoder setBytes:&lrTauF length:sizeof(float) atIndex:13];
            [encoder setBytes:&lrRadiusF length:sizeof(float) atIndex:14];
            [encoder setBytes:&lrColorF length:sizeof(float) atIndex:15];
            [encoder setBytes:&lrDirF length:sizeof(float) atIndex:16];
            [encoder setBytes:&lrAnisoF length:sizeof(float) atIndex:17];
            [encoder setBytes:&beta1F length:sizeof(float) atIndex:18];
            [encoder setBytes:&beta2F length:sizeof(float) atIndex:19];
            [encoder setBytes:&epsF length:sizeof(float) atIndex:20];
            [encoder setBytes:&tU length:sizeof(uint32_t) atIndex:21];
            [encoder setBytes:&wU length:sizeof(uint32_t) atIndex:22];
            [encoder setBytes:&hU length:sizeof(uint32_t) atIndex:23];

            dispatchThreadgroups1D(encoder, site_count, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return sites;
    }
}

torch::Tensor computeSiteStatsTiled(torch::Tensor cand0, torch::Tensor cand1,
                                    torch::Tensor target, torch::Tensor mask, torch::Tensor sites,
                                    torch::Tensor mass, torch::Tensor energy,
                                    torch::Tensor err_w, torch::Tensor err_wx, torch::Tensor err_wy,
                                    torch::Tensor err_wxx, torch::Tensor err_wxy, torch::Tensor err_wyy,
                                    double inv_scale_sq, int64_t site_count,
                                    int64_t cand_width, int64_t cand_height) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(cand0.device().is_mps() && cand1.device().is_mps(), "cand tensors must be MPS");
        TORCH_CHECK(target.device().is_mps() && mask.device().is_mps(), "target/mask must be MPS");
        TORCH_CHECK(sites.device().is_mps(), "sites must be MPS");
        TORCH_CHECK(cand0.scalar_type() == torch::kInt32 && cand1.scalar_type() == torch::kInt32, "cand tensors must be int32");
        TORCH_CHECK(target.scalar_type() == torch::kFloat32 && mask.scalar_type() == torch::kFloat32, "target/mask must be float32");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        TORCH_CHECK(target.dim() == 3 && target.size(2) == 4, "target must be [H,W,4]");
        TORCH_CHECK(mask.dim() == 3 && mask.size(2) == 4, "mask must be [H,W,4]");
        TORCH_CHECK(cand0.dim() == 2 && cand0.size(1) == 4, "cand0 must be [M,4]");
        TORCH_CHECK(cand1.dim() == 2 && cand1.size(1) == 4, "cand1 must be [M,4]");
        TORCH_CHECK(cand_width > 0 && cand_height > 0, "invalid cand size");
        TORCH_CHECK(cand0.size(0) == cand_width * cand_height, "cand0 size mismatch");
        TORCH_CHECK(cand1.size(0) == cand_width * cand_height, "cand1 size mismatch");

        cand0 = cand0.contiguous();
        cand1 = cand1.contiguous();
        target = target.contiguous();
        mask = mask.contiguous();
        sites = sites.contiguous();
        mass = mass.contiguous();
        energy = energy.contiguous();
        err_w = err_w.contiguous();
        err_wx = err_wx.contiguous();
        err_wy = err_wy.contiguous();
        err_wxx = err_wxx.contiguous();
        err_wxy = err_wxy.contiguous();
        err_wyy = err_wyy.contiguous();

        int64_t height = target.size(0);
        int64_t width = target.size(1);
        if (site_count <= 0) {
            site_count = sites.size(0);
        }

        id<MTLTexture> cand0Tex = textureFromTensor(cand0, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);
        id<MTLTexture> cand1Tex = textureFromTensor(cand1, MTLPixelFormatRGBA32Uint, cand_width, cand_height, 4);
        id<MTLTexture> targetTex = textureFromTensor(target, MTLPixelFormatRGBA32Float, width, height, 4);
        id<MTLTexture> maskTex = textureFromTensor(mask, MTLPixelFormatRGBA32Float, width, height, 4);

        id<MTLComputePipelineState> pso = getPipeline(@"computeSiteStatsTiled");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setTexture:cand0Tex atIndex:0];
            [encoder setTexture:cand1Tex atIndex:1];
            [encoder setTexture:targetTex atIndex:2];
            [encoder setTexture:maskTex atIndex:3];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            float invScale = static_cast<float>(inv_scale_sq);
            uint32_t siteCountU = static_cast<uint32_t>(site_count);
            [encoder setBytes:&invScale length:sizeof(float) atIndex:1];
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(mass) offset:mass.storage_offset() * mass.element_size() atIndex:3];
            [encoder setBuffer:getMTLBufferStorage(energy) offset:energy.storage_offset() * energy.element_size() atIndex:4];
            [encoder setBuffer:getMTLBufferStorage(err_w) offset:err_w.storage_offset() * err_w.element_size() atIndex:5];
            [encoder setBuffer:getMTLBufferStorage(err_wx) offset:err_wx.storage_offset() * err_wx.element_size() atIndex:6];
            [encoder setBuffer:getMTLBufferStorage(err_wy) offset:err_wy.storage_offset() * err_wy.element_size() atIndex:7];
            [encoder setBuffer:getMTLBufferStorage(err_wxx) offset:err_wxx.storage_offset() * err_wxx.element_size() atIndex:8];
            [encoder setBuffer:getMTLBufferStorage(err_wxy) offset:err_wxy.storage_offset() * err_wxy.element_size() atIndex:9];
            [encoder setBuffer:getMTLBufferStorage(err_wyy) offset:err_wyy.storage_offset() * err_wyy.element_size() atIndex:10];

            const int tileHashSize = 256;
            size_t keyMem = tileHashSize * sizeof(uint32_t);
            size_t statMem = tileHashSize * sizeof(int32_t);
            [encoder setThreadgroupMemoryLength:keyMem atIndex:0];
            for (int i = 0; i < 8; ++i) {
                [encoder setThreadgroupMemoryLength:statMem atIndex:1 + i];
            }

            MTLSize tgs = MTLSizeMake(16, 16, 1);
            MTLSize tg = MTLSizeMake((width + 15) / 16, (height + 15) / 16, 1);
            [encoder dispatchThreadgroups:tg threadsPerThreadgroup:tgs];
            [encoder endEncoding];
            torch::mps::commit();
        });

        return mass;
    }
}

torch::Tensor computeDensifyScorePairs(torch::Tensor sites, torch::Tensor mass, torch::Tensor energy,
                                       torch::Tensor pairs, int64_t site_count,
                                       double min_mass, double score_alpha, int64_t pair_count) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(sites.device().is_mps() && mass.device().is_mps() && energy.device().is_mps() && pairs.device().is_mps(),
                    "tensors must be MPS");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(mass.scalar_type() == torch::kFloat32 && energy.scalar_type() == torch::kFloat32, "mass/energy must be float32");
        TORCH_CHECK(pairs.scalar_type() == torch::kInt32, "pairs must be int32");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        TORCH_CHECK(pairs.dim() == 2 && pairs.size(1) == 2, "pairs must be [M,2]");
        sites = sites.contiguous();
        mass = mass.contiguous();
        energy = energy.contiguous();
        pairs = pairs.contiguous();
        if (site_count <= 0) {
            site_count = sites.size(0);
        }
        if (pair_count <= 0) {
            pair_count = pairs.size(0);
        }

        id<MTLComputePipelineState> pso = getPipeline(@"computeDensifyScorePairs");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(mass) offset:mass.storage_offset() * mass.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(energy) offset:energy.storage_offset() * energy.element_size() atIndex:2];
            [encoder setBuffer:getMTLBufferStorage(pairs) offset:pairs.storage_offset() * pairs.element_size() atIndex:3];
            uint32_t siteCountU = static_cast<uint32_t>(site_count);
            float minMassF = static_cast<float>(min_mass);
            float scoreAlphaF = static_cast<float>(score_alpha);
            uint32_t pairCountU = static_cast<uint32_t>(pair_count);
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&minMassF length:sizeof(float) atIndex:5];
            [encoder setBytes:&scoreAlphaF length:sizeof(float) atIndex:6];
            [encoder setBytes:&pairCountU length:sizeof(uint32_t) atIndex:7];
            dispatchThreadgroups1D(encoder, pair_count, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return pairs;
    }
}

torch::Tensor computePruneScorePairs(torch::Tensor sites, torch::Tensor removal_delta, torch::Tensor pairs,
                                     int64_t site_count, double delta_norm, int64_t pair_count) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(sites.device().is_mps() && removal_delta.device().is_mps() && pairs.device().is_mps(),
                    "tensors must be MPS");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(removal_delta.scalar_type() == torch::kFloat32, "removal_delta must be float32");
        TORCH_CHECK(pairs.scalar_type() == torch::kInt32, "pairs must be int32");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        TORCH_CHECK(pairs.dim() == 2 && pairs.size(1) == 2, "pairs must be [M,2]");
        sites = sites.contiguous();
        removal_delta = removal_delta.contiguous();
        pairs = pairs.contiguous();
        if (site_count <= 0) {
            site_count = sites.size(0);
        }
        if (pair_count <= 0) {
            pair_count = pairs.size(0);
        }

        id<MTLComputePipelineState> pso = getPipeline(@"computePruneScorePairs");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(removal_delta) offset:removal_delta.storage_offset() * removal_delta.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(pairs) offset:pairs.storage_offset() * pairs.element_size() atIndex:2];
            uint32_t siteCountU = static_cast<uint32_t>(site_count);
            float deltaNormF = static_cast<float>(delta_norm);
            uint32_t pairCountU = static_cast<uint32_t>(pair_count);
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&deltaNormF length:sizeof(float) atIndex:4];
            [encoder setBytes:&pairCountU length:sizeof(uint32_t) atIndex:5];
            dispatchThreadgroups1D(encoder, pair_count, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return pairs;
    }
}

torch::Tensor writeSplitIndices(torch::Tensor pairs, torch::Tensor indices, int64_t num_to_split) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(pairs.device().is_mps() && indices.device().is_mps(), "tensors must be MPS");
        TORCH_CHECK(pairs.scalar_type() == torch::kInt32 && indices.scalar_type() == torch::kInt32, "tensors must be int32");
        TORCH_CHECK(pairs.dim() == 2 && pairs.size(1) == 2, "pairs must be [M,2]");
        TORCH_CHECK(indices.dim() == 1, "indices must be [K]");
        pairs = pairs.contiguous();
        indices = indices.contiguous();
        if (num_to_split <= 0) {
            return indices;
        }

        id<MTLComputePipelineState> pso = getPipeline(@"writeSplitIndicesFromSorted");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(pairs) offset:pairs.storage_offset() * pairs.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(indices) offset:indices.storage_offset() * indices.element_size() atIndex:1];
            uint32_t countU = static_cast<uint32_t>(num_to_split);
            [encoder setBytes:&countU length:sizeof(uint32_t) atIndex:2];
            dispatchThreadgroups1D(encoder, num_to_split, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return indices;
    }
}

torch::Tensor splitSites(torch::Tensor sites, torch::Tensor adam, torch::Tensor split_indices,
                         torch::Tensor mass, torch::Tensor err_w, torch::Tensor err_wx, torch::Tensor err_wy,
                         torch::Tensor err_wxx, torch::Tensor err_wxy, torch::Tensor err_wyy,
                         int64_t current_site_count, int64_t num_to_split, torch::Tensor target) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(sites.device().is_mps() && adam.device().is_mps() && split_indices.device().is_mps(),
                    "tensors must be MPS");
        TORCH_CHECK(target.device().is_mps(), "target must be MPS");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32 && adam.scalar_type() == torch::kFloat32, "sites/adam must be float32");
        TORCH_CHECK(split_indices.scalar_type() == torch::kInt32, "split_indices must be int32");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        TORCH_CHECK(adam.dim() == 2, "adam must be [N,24]");
        TORCH_CHECK(target.dim() == 3 && target.size(2) == 4, "target must be [H,W,4]");
        sites = sites.contiguous();
        adam = adam.contiguous();
        split_indices = split_indices.contiguous();
        mass = mass.contiguous();
        err_w = err_w.contiguous();
        err_wx = err_wx.contiguous();
        err_wy = err_wy.contiguous();
        err_wxx = err_wxx.contiguous();
        err_wxy = err_wxy.contiguous();
        err_wyy = err_wyy.contiguous();
        target = target.contiguous();
        if (num_to_split <= 0) {
            return sites;
        }

        int64_t height = target.size(0);
        int64_t width = target.size(1);

        id<MTLTexture> targetTex = textureFromTensor(target, MTLPixelFormatRGBA32Float, width, height, 4);

        id<MTLComputePipelineState> pso = getPipeline(@"splitSites");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(adam) offset:adam.storage_offset() * adam.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(split_indices) offset:split_indices.storage_offset() * split_indices.element_size() atIndex:2];
            uint32_t numSplitU = static_cast<uint32_t>(num_to_split);
            [encoder setBytes:&numSplitU length:sizeof(uint32_t) atIndex:3];
            [encoder setBuffer:getMTLBufferStorage(mass) offset:mass.storage_offset() * mass.element_size() atIndex:4];
            [encoder setBuffer:getMTLBufferStorage(err_w) offset:err_w.storage_offset() * err_w.element_size() atIndex:5];
            [encoder setBuffer:getMTLBufferStorage(err_wx) offset:err_wx.storage_offset() * err_wx.element_size() atIndex:6];
            [encoder setBuffer:getMTLBufferStorage(err_wy) offset:err_wy.storage_offset() * err_wy.element_size() atIndex:7];
            [encoder setBuffer:getMTLBufferStorage(err_wxx) offset:err_wxx.storage_offset() * err_wxx.element_size() atIndex:8];
            [encoder setBuffer:getMTLBufferStorage(err_wxy) offset:err_wxy.storage_offset() * err_wxy.element_size() atIndex:9];
            [encoder setBuffer:getMTLBufferStorage(err_wyy) offset:err_wyy.storage_offset() * err_wyy.element_size() atIndex:10];
            uint32_t siteCountU = static_cast<uint32_t>(current_site_count);
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:11];
            [encoder setTexture:targetTex atIndex:0];
            dispatchThreadgroups1D(encoder, num_to_split, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return sites;
    }
}

torch::Tensor pruneSites(torch::Tensor sites, torch::Tensor indices, int64_t count) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(sites.device().is_mps() && indices.device().is_mps(), "tensors must be MPS");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        TORCH_CHECK(indices.dim() == 1, "indices must be [K]");
        sites = sites.contiguous();
        indices = indices.contiguous();
        if (count <= 0) {
            return sites;
        }

        id<MTLComputePipelineState> pso = getPipeline(@"pruneSitesByIndex");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(indices) offset:indices.storage_offset() * indices.element_size() atIndex:1];
            uint32_t countU = static_cast<uint32_t>(count);
            [encoder setBytes:&countU length:sizeof(uint32_t) atIndex:2];
            dispatchThreadgroups1D(encoder, count, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return sites;
    }
}

torch::Tensor buildHilbertPairs(torch::Tensor sites, torch::Tensor pairs,
                                int64_t site_count, int64_t padded_count,
                                int64_t width, int64_t height, int64_t bits) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(sites.device().is_mps() && pairs.device().is_mps(), "tensors must be MPS");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(pairs.scalar_type() == torch::kInt32, "pairs must be int32");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        TORCH_CHECK(pairs.dim() == 2 && pairs.size(1) == 2, "pairs must be [M,2]");
        sites = sites.contiguous();
        pairs = pairs.contiguous();
        if (site_count <= 0) {
            site_count = sites.size(0);
        }
        if (padded_count <= 0) {
            padded_count = pairs.size(0);
        }

        id<MTLComputePipelineState> pso = getPipeline(@"buildHilbertPairs");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(pairs) offset:pairs.storage_offset() * pairs.element_size() atIndex:1];
            uint32_t siteCountU = static_cast<uint32_t>(site_count);
            uint32_t paddedCountU = static_cast<uint32_t>(padded_count);
            uint32_t wU = static_cast<uint32_t>(width);
            uint32_t hU = static_cast<uint32_t>(height);
            uint32_t bitsU = static_cast<uint32_t>(bits);
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:2];
            [encoder setBytes:&paddedCountU length:sizeof(uint32_t) atIndex:3];
            [encoder setBytes:&wU length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&hU length:sizeof(uint32_t) atIndex:5];
            [encoder setBytes:&bitsU length:sizeof(uint32_t) atIndex:6];
            dispatchThreadgroups1D(encoder, padded_count, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return pairs;
    }
}

torch::Tensor writeHilbertOrder(torch::Tensor pairs, torch::Tensor order, torch::Tensor pos, int64_t site_count) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(pairs.device().is_mps() && order.device().is_mps() && pos.device().is_mps(), "tensors must be MPS");
        TORCH_CHECK(pairs.scalar_type() == torch::kInt32, "pairs must be int32");
        TORCH_CHECK(order.scalar_type() == torch::kInt32 && pos.scalar_type() == torch::kInt32, "order/pos must be int32");
        TORCH_CHECK(pairs.dim() == 2 && pairs.size(1) == 2, "pairs must be [M,2]");
        TORCH_CHECK(order.dim() == 1 && pos.dim() == 1, "order/pos must be [N]");
        pairs = pairs.contiguous();
        order = order.contiguous();
        pos = pos.contiguous();
        if (site_count <= 0) {
            site_count = order.size(0);
        }

        id<MTLComputePipelineState> pso = getPipeline(@"writeHilbertOrder");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(pairs) offset:pairs.storage_offset() * pairs.element_size() atIndex:0];
            [encoder setBuffer:getMTLBufferStorage(order) offset:order.storage_offset() * order.element_size() atIndex:1];
            [encoder setBuffer:getMTLBufferStorage(pos) offset:pos.storage_offset() * pos.element_size() atIndex:2];
            uint32_t siteCountU = static_cast<uint32_t>(site_count);
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:3];
            dispatchThreadgroups1D(encoder, site_count, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return order;
    }
}

torch::Tensor initGradientWeighted(torch::Tensor sites, torch::Tensor target, torch::Tensor mask,
                                   int64_t site_count, double init_log_tau, double init_radius,
                                   double gradient_alpha) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(sites.device().is_mps() && target.device().is_mps() && mask.device().is_mps(),
                    "sites/target/mask must be MPS");
        TORCH_CHECK(sites.scalar_type() == torch::kFloat32, "sites must be float32");
        TORCH_CHECK(target.scalar_type() == torch::kFloat32 && mask.scalar_type() == torch::kFloat32,
                    "target/mask must be float32");
        TORCH_CHECK(sites.dim() == 2 && sites.size(1) == 12, "sites must be [N,12]");
        TORCH_CHECK(target.dim() == 3 && target.size(2) == 4, "target must be [H,W,4]");
        TORCH_CHECK(mask.dim() == 3 && mask.size(2) == 4, "mask must be [H,W,4]");
        sites = sites.contiguous();
        target = target.contiguous();
        mask = mask.contiguous();
        if (site_count <= 0) {
            site_count = sites.size(0);
        }

        int64_t height = target.size(0);
        int64_t width = target.size(1);
        id<MTLTexture> targetTex = textureFromTensor(target, MTLPixelFormatRGBA32Float, width, height, 4);
        id<MTLTexture> maskTex = textureFromTensor(mask, MTLPixelFormatRGBA32Float, width, height, 4);

        id<MTLBuffer> seedBuffer = [g_device newBufferWithLength:sizeof(uint32_t)
                                                         options:MTLResourceStorageModeShared];
        TORCH_CHECK(seedBuffer != nil, "Failed to allocate init seed buffer");
        memset([seedBuffer contents], 0, sizeof(uint32_t));

        id<MTLComputePipelineState> pso = getPipeline(@"initGradientWeighted");
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
            TORCH_CHECK(encoder != nil, "Failed to create compute encoder");
            [encoder setComputePipelineState:pso];
            [encoder setBuffer:getMTLBufferStorage(sites) offset:sites.storage_offset() * sites.element_size() atIndex:0];
            uint32_t siteCountU = static_cast<uint32_t>(site_count);
            [encoder setBytes:&siteCountU length:sizeof(uint32_t) atIndex:1];
            [encoder setBuffer:seedBuffer offset:0 atIndex:2];
            [encoder setTexture:targetTex atIndex:0];
            [encoder setTexture:maskTex atIndex:1];
            float gradThreshold = 0.01f * static_cast<float>(gradient_alpha);
            uint32_t maxAttempts = 256;
            float initLogTauF = static_cast<float>(init_log_tau);
            float initRadiusF = static_cast<float>(init_radius);
            [encoder setBytes:&gradThreshold length:sizeof(float) atIndex:3];
            [encoder setBytes:&maxAttempts length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&initLogTauF length:sizeof(float) atIndex:5];
            [encoder setBytes:&initRadiusF length:sizeof(float) atIndex:6];
            dispatchThreadgroups1D(encoder, site_count, 256);
            [encoder endEncoding];
            torch::mps::commit();
        });

        return sites;
    }
}

// --- Radix sort (uint2) ---
@interface RadixSortUInt2 : NSObject
- (instancetype)initWithDevice:(id<MTLDevice>)device paddedCount:(int)paddedCount;
- (void)encode:(id<MTLBuffer>)data maxKeyExclusive:(uint32_t)maxKeyExclusive commandBuffer:(id<MTLCommandBuffer>)commandBuffer;
@property (nonatomic, readonly) int paddedCount;
@end

torch::Tensor radixSortPairs(torch::Tensor pairs, int64_t max_key_exclusive) {
    @autoreleasepool {
        ensureMetalInitialized();
        TORCH_CHECK(pairs.device().is_mps(), "pairs must be MPS tensor");
        TORCH_CHECK(pairs.scalar_type() == torch::kInt32, "pairs must be int32");
        TORCH_CHECK(pairs.dim() == 2 && pairs.size(1) == 2, "pairs must be [N,2]");
        pairs = pairs.contiguous();
        int64_t count = pairs.size(0);
        TORCH_CHECK(count > 0, "pairs must be non-empty");

        RadixSortUInt2 *sorter = [[RadixSortUInt2 alloc] initWithDevice:g_device paddedCount:(int)count];
        id<MTLCommandBuffer> cmdBuf = torch::mps::get_command_buffer();
        TORCH_CHECK(cmdBuf != nil, "Failed to get command buffer");

        dispatch_queue_t serialQueue = torch::mps::get_dispatch_queue();
        dispatch_sync(serialQueue, ^() {
            [sorter encode:getMTLBufferStorage(pairs)
               maxKeyExclusive:(uint32_t)max_key_exclusive
             commandBuffer:cmdBuf];
            torch::mps::commit();
        });

        return pairs;
    }
}

@implementation RadixSortUInt2 {
    id<MTLDevice> _device;
    id<MTLComputePipelineState> _histogramPipeline;
    id<MTLComputePipelineState> _scanBlocksPipeline;
    id<MTLComputePipelineState> _scanBlockSumsPipeline;
    id<MTLComputePipelineState> _applyOffsetsPipeline;
    id<MTLComputePipelineState> _scatterPipeline;
    int _paddedCount;
    int _gridSize;
    int _histLength;
    int _histBlocks;
    id<MTLBuffer> _histFlat;
    id<MTLBuffer> _blockSums;
    id<MTLBuffer> _scratch;
}

- (instancetype)initWithDevice:(id<MTLDevice>)device paddedCount:(int)paddedCount {
    self = [super init];
    if (self) {
        _device = device;
        _paddedCount = paddedCount;
        _histogramPipeline = getPipeline(@"radixHistogramUInt2");
        _scanBlocksPipeline = getPipeline(@"radixScanHistogramBlocks");
        _scanBlockSumsPipeline = getPipeline(@"radixExclusiveScanBlockSums");
        _applyOffsetsPipeline = getPipeline(@"radixApplyOffsets");
        _scatterPipeline = getPipeline(@"radixScatterUInt2");

        int elementsPerBlock = 256 * 4;
        _gridSize = (_paddedCount + elementsPerBlock - 1) / elementsPerBlock;
        _histLength = 256 * _gridSize;
        _histBlocks = (_histLength + 255) / 256;

        _histFlat = [_device newBufferWithLength:_histLength * sizeof(uint32_t)
                                        options:MTLResourceStorageModeShared];
        _blockSums = [_device newBufferWithLength:_histBlocks * sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
        _scratch = [_device newBufferWithLength:_paddedCount * sizeof(uint32_t) * 2
                                        options:MTLResourceStorageModeShared];
    }
    return self;
}

- (int)paddedCount {
    return _paddedCount;
}

- (void)encode:(id<MTLBuffer>)data maxKeyExclusive:(uint32_t)maxKeyExclusive commandBuffer:(id<MTLCommandBuffer>)commandBuffer {
    int passes = (maxKeyExclusive <= (1u << 16)) ? 2 : 4;
    id<MTLBuffer> input = data;
    id<MTLBuffer> output = _scratch;
    uint32_t paddedCountU = (uint32_t)_paddedCount;

    for (int pass = 0; pass < passes; ++pass) {
        uint32_t shift = (uint32_t)(pass * 8);

        // Histogram
        if (id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder]) {
            [enc setComputePipelineState:_histogramPipeline];
            [enc setBuffer:input offset:0 atIndex:0];
            [enc setBuffer:_histFlat offset:0 atIndex:1];
            [enc setBytes:&paddedCountU length:sizeof(uint32_t) atIndex:2];
            [enc setBytes:&shift length:sizeof(uint32_t) atIndex:3];
            MTLSize tg = MTLSizeMake(_gridSize, 1, 1);
            MTLSize tgs = MTLSizeMake(256, 1, 1);
            [enc dispatchThreadgroups:tg threadsPerThreadgroup:tgs];
            [enc endEncoding];
        }

        // Scan blocks
        if (id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder]) {
            [enc setComputePipelineState:_scanBlocksPipeline];
            [enc setBuffer:_histFlat offset:0 atIndex:0];
            [enc setBuffer:_blockSums offset:0 atIndex:1];
            [enc setBytes:&paddedCountU length:sizeof(uint32_t) atIndex:2];
            MTLSize tg = MTLSizeMake(_histBlocks, 1, 1);
            MTLSize tgs = MTLSizeMake(256, 1, 1);
            [enc dispatchThreadgroups:tg threadsPerThreadgroup:tgs];
            [enc endEncoding];
        }

        // Scan block sums
        if (id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder]) {
            [enc setComputePipelineState:_scanBlockSumsPipeline];
            [enc setBuffer:_blockSums offset:0 atIndex:0];
            [enc setBytes:&paddedCountU length:sizeof(uint32_t) atIndex:1];
            MTLSize threads = MTLSizeMake(256, 1, 1);
            [enc dispatchThreads:threads threadsPerThreadgroup:threads];
            [enc endEncoding];
        }

        // Apply offsets
        if (id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder]) {
            [enc setComputePipelineState:_applyOffsetsPipeline];
            [enc setBuffer:_histFlat offset:0 atIndex:0];
            [enc setBuffer:_blockSums offset:0 atIndex:1];
            [enc setBytes:&paddedCountU length:sizeof(uint32_t) atIndex:2];
            MTLSize tg = MTLSizeMake(_histBlocks, 1, 1);
            MTLSize tgs = MTLSizeMake(256, 1, 1);
            [enc dispatchThreadgroups:tg threadsPerThreadgroup:tgs];
            [enc endEncoding];
        }

        // Scatter
        if (id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder]) {
            [enc setComputePipelineState:_scatterPipeline];
            [enc setBuffer:input offset:0 atIndex:0];
            [enc setBuffer:output offset:0 atIndex:1];
            [enc setBuffer:_histFlat offset:0 atIndex:2];
            [enc setBytes:&paddedCountU length:sizeof(uint32_t) atIndex:3];
            [enc setBytes:&shift length:sizeof(uint32_t) atIndex:4];
            MTLSize tg = MTLSizeMake(_gridSize, 1, 1);
            MTLSize tgs = MTLSizeMake(256, 1, 1);
            [enc dispatchThreadgroups:tg threadsPerThreadgroup:tgs];
            [enc endEncoding];
        }

        std::swap(input, output);
    }

    TORCH_CHECK(passes % 2 == 0, "Radix sort expects even passes");
}
@end

static inline uint32_t jumpStepForIndex(uint32_t stepIndex, int width, int height) {
    uint32_t maxDim = (uint32_t)std::max(width, height);
    uint32_t pow2 = 1;
    while (pow2 < maxDim) {
        pow2 <<= 1;
    }
    if (pow2 <= 1) {
        return 1;
    }
    uint32_t stages = 0;
    uint32_t tmp = pow2;
    while (tmp > 1) {
        tmp >>= 1;
        stages += 1;
    }
    uint32_t stage = (stages > 0) ? (stepIndex >= stages ? (stages - 1) : stepIndex) : 0;
    uint32_t step = pow2 >> (stage + 1);
    return std::max(step, 1u);
}

static inline uint32_t packJumpStep(uint32_t stepIndex, int width, int height) {
    uint32_t jumpStep = std::min(jumpStepForIndex(stepIndex, width, height), 0xffffu);
    return (jumpStep << 16) | (stepIndex & 0xffffu);
}

static inline int hilbertBitsForSize(int width, int height) {
    int maxDim = std::max(width, height);
    int n = 1;
    int bits = 0;
    while (n < maxDim) {
        n <<= 1;
        bits += 1;
    }
    return std::max(bits, 1);
}

static void updateHilbertResources(id<MTLCommandBuffer> commandBuffer,
                                   id<MTLBuffer> sitesBuffer,
                                   int siteCount,
                                   int width,
                                   int height,
                                   id<MTLBuffer> pairsBuffer,
                                   id<MTLBuffer> orderBuffer,
                                   id<MTLBuffer> posBuffer,
                                   RadixSortUInt2 *sorter,
                                   int paddedCount) {
    if (siteCount <= 0) {
        return;
    }

    uint32_t bits = (uint32_t)hilbertBitsForSize(width, height);
    uint32_t paddedCountU = (uint32_t)paddedCount;
    uint32_t siteCountU = (uint32_t)siteCount;
    uint32_t widthU = (uint32_t)width;
    uint32_t heightU = (uint32_t)height;

    // Build pairs
    if (id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder]) {
        id<MTLComputePipelineState> pso = getPipeline(@"buildHilbertPairs");
        [enc setComputePipelineState:pso];
        [enc setBuffer:sitesBuffer offset:0 atIndex:0];
        [enc setBuffer:pairsBuffer offset:0 atIndex:1];
        [enc setBytes:&siteCountU length:sizeof(uint32_t) atIndex:2];
        [enc setBytes:&paddedCountU length:sizeof(uint32_t) atIndex:3];
        [enc setBytes:&widthU length:sizeof(uint32_t) atIndex:4];
        [enc setBytes:&heightU length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&bits length:sizeof(uint32_t) atIndex:6];
        dispatchThreadgroups1D(enc, paddedCount, 256);
        [enc endEncoding];
    }

    uint32_t maxKeyExclusive = bits >= 16 ? 0xffffffffu : (1u << (bits * 2));
    [sorter encode:pairsBuffer maxKeyExclusive:maxKeyExclusive commandBuffer:commandBuffer];

    // Write order/pos
    if (id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder]) {
        id<MTLComputePipelineState> pso = getPipeline(@"writeHilbertOrder");
        [enc setComputePipelineState:pso];
        [enc setBuffer:pairsBuffer offset:0 atIndex:0];
        [enc setBuffer:orderBuffer offset:0 atIndex:1];
        [enc setBuffer:posBuffer offset:0 atIndex:2];
        [enc setBytes:&siteCountU length:sizeof(uint32_t) atIndex:3];
        dispatchThreadgroups1D(enc, siteCount, 256);
        [enc endEncoding];
    }
}
