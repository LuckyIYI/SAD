#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "json.hpp"
#include "sad_common.cuh"

using json = nlohmann::json;
namespace fs = std::filesystem;

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)              \
                      << " (" << #call << ") at " << __FILE__ << ":"            \
                      << __LINE__ << std::endl;                                  \
            std::exit(1);                                                        \
        }                                                                        \
    } while (0)

static void cudaSync(const char* label) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA sync error after " << label << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

static float measureCuda(cudaStream_t stream, const std::function<void()>& fn) {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));
    fn();
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms;
}

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    size_t count = 0;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept {
        ptr = other.ptr;
        count = other.count;
        other.ptr = nullptr;
        other.count = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) {
                cudaFree(ptr);
            }
            ptr = other.ptr;
            count = other.count;
            other.ptr = nullptr;
            other.count = 0;
        }
        return *this;
    }

    ~DeviceBuffer() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    void alloc(size_t n) {
        count = n;
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
};

struct Image {
    int width = 0;
    int height = 0;
    std::vector<float3> pixels;
};

struct Mask {
    int width = 0;
    int height = 0;
    std::vector<float> values;
    float sum = 0.0f;
};

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::string toLower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}


static Image loadImage(const std::string& path, int maxDim) {
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* data = stbi_load(path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        std::cerr << "Failed to load image: " << path << std::endl;
        std::exit(1);
    }

    int outW = width;
    int outH = height;
    std::vector<stbi_uc> resized;
    const stbi_uc* src = data;

    if (maxDim > 0 && std::max(width, height) > maxDim) {
        float scale = static_cast<float>(maxDim) / static_cast<float>(std::max(width, height));
        outW = std::max(1, static_cast<int>(std::round(width * scale)));
        outH = std::max(1, static_cast<int>(std::round(height * scale)));
        resized.resize(static_cast<size_t>(outW) * outH * 3);
        stbir_resize_uint8_srgb(data, width, height, 0,
                                resized.data(), outW, outH, 0, STBIR_RGB);
        src = resized.data();
    }

    Image img;
    img.width = outW;
    img.height = outH;
    img.pixels.resize(static_cast<size_t>(outW) * outH);

    for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
            size_t idx = static_cast<size_t>(y) * outW + x;
            size_t srcIdx = idx * 3;
            float r = src[srcIdx] / 255.0f;
            float g = src[srcIdx + 1] / 255.0f;
            float b = src[srcIdx + 2] / 255.0f;
            img.pixels[idx] = make_float3(r, g, b);
        }
    }

    stbi_image_free(data);
    return img;
}

static Mask loadMask(const std::string& path, int targetW, int targetH) {
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* data = stbi_load(path.c_str(), &width, &height, &channels, 1);
    if (!data) {
        std::cerr << "Failed to load mask: " << path << std::endl;
        std::exit(1);
    }

    std::vector<stbi_uc> resized;
    const stbi_uc* src = data;
    if (width != targetW || height != targetH) {
        resized.resize(static_cast<size_t>(targetW) * targetH);
        stbir_resize_uint8_linear(data, width, height, 0,
                                  resized.data(), targetW, targetH, 0, STBIR_1CHANNEL);
        src = resized.data();
        width = targetW;
        height = targetH;
    }

    Mask mask;
    mask.width = width;
    mask.height = height;
    mask.values.resize(static_cast<size_t>(width) * height, 0.0f);
    mask.sum = 0.0f;
    for (int i = 0; i < width * height; ++i) {
        float v = src[i] / 255.0f;
        mask.values[static_cast<size_t>(i)] = v;
        if (v > 0.0f) {
            mask.sum += 1.0f;
        }
    }

    stbi_image_free(data);
    return mask;
}

static Mask defaultMask(int width, int height) {
    Mask mask;
    mask.width = width;
    mask.height = height;
    mask.values.resize(static_cast<size_t>(width) * height, 1.0f);
    mask.sum = static_cast<float>(width * height);
    return mask;
}

static bool saveImage(const std::string& path, const std::vector<float3>& pixels,
                      int width, int height) {
    std::vector<unsigned char> out(static_cast<size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            size_t idx = static_cast<size_t>(y) * width + x;
            float3 c = pixels[idx];
            float r = std::clamp(c.x, 0.0f, 1.0f);
            float g = std::clamp(c.y, 0.0f, 1.0f);
            float b = std::clamp(c.z, 0.0f, 1.0f);
            size_t o = idx * 3;
            out[o] = static_cast<unsigned char>(std::round(r * 255.0f));
            out[o + 1] = static_cast<unsigned char>(std::round(g * 255.0f));
            out[o + 2] = static_cast<unsigned char>(std::round(b * 255.0f));
        }
    }
    int ok = stbi_write_png(path.c_str(), width, height, 3, out.data(), width * 3);
    return ok != 0;
}

struct LoadedSites {
    std::vector<Site> sites;
    int width = 0;
    int height = 0;
    bool hasSize = false;
};

static bool parseImageSize(const std::string& line, int& width, int& height) {
    std::string cleaned;
    cleaned.reserve(line.size());
    for (char c : line) {
        if ((c >= '0' && c <= '9') || c == '-') {
            cleaned.push_back(c);
        } else {
            cleaned.push_back(' ');
        }
    }
    std::istringstream iss(cleaned);
    int w = 0;
    int h = 0;
    if (iss >> w >> h) {
        width = w;
        height = h;
        return true;
    }
    return false;
}

static LoadedSites loadSitesFromTXT(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open sites TXT: " << path << std::endl;
        std::exit(1);
    }

    LoadedSites out;
    std::string line;
    int headerW = 0;
    int headerH = 0;

    while (std::getline(file, line)) {
        std::string trimmed = trim(line);
        if (trimmed.empty()) {
            continue;
        }
        if (trimmed[0] == '#') {
            std::string lower = toLower(trimmed);
            if (lower.find("image size") != std::string::npos) {
                parseImageSize(trimmed, headerW, headerH);
            }
            continue;
        }

        std::istringstream iss(trimmed);
        std::vector<float> vals;
        float v = 0.0f;
        while (iss >> v) {
            vals.push_back(v);
        }
        if (vals.size() != 7 && vals.size() != 10) {
            continue;
        }

        Site site = {};
        site.position = make_float2(vals[0], vals[1]);
        site_set_color(site, make_float3(vals[2], vals[3], vals[4]));
        site.log_tau = vals[5];
        site.radius = vals[6];
        site_set_aniso_dir(site, make_float2(1.0f, 0.0f));
        site.log_aniso = 0.0f;
        if (vals.size() == 10) {
            site_set_aniso_dir(site, make_float2(vals[7], vals[8]));
            site.log_aniso = vals[9];
        }
        out.sites.push_back(site);
    }

    if (out.sites.empty()) {
        std::cerr << "No sites found in TXT: " << path << std::endl;
        std::exit(1);
    }

    if (headerW > 0 && headerH > 0) {
        out.width = headerW;
        out.height = headerH;
        out.hasSize = true;
    }

    return out;
}

static LoadedSites loadSitesFromJSON(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open sites JSON: " << path << std::endl;
        std::exit(1);
    }

    json j;
    file >> j;

    LoadedSites out;
    out.width = j.at("image_width").get<int>();
    out.height = j.at("image_height").get<int>();
    out.hasSize = true;

    const auto& sitesArr = j.at("sites");
    for (const auto& s : sitesArr) {
        auto pos = s.at("pos");
        auto color = s.at("color");
        float log_tau = s.at("log_tau").get<float>();
        float radius = 0.0f;
        if (s.contains("radius")) {
            radius = s.at("radius").get<float>();
        } else if (s.contains("radius_sq")) {
            radius = s.at("radius_sq").get<float>();
        }
        float log_aniso = s.value("log_aniso", 0.0f);
        float dir_x = 1.0f;
        float dir_y = 0.0f;
        if (s.contains("aniso_dir")) {
            auto dir = s.at("aniso_dir");
            if (dir.size() > 0) dir_x = dir[0].get<float>();
            if (dir.size() > 1) dir_y = dir[1].get<float>();
        }

        Site site = {};
        site.position = make_float2(pos[0].get<float>(), pos[1].get<float>());
        site_set_color(site, make_float3(color[0].get<float>(), color[1].get<float>(), color[2].get<float>()));
        site.log_tau = log_tau;
        site.radius = radius;
        site_set_aniso_dir(site, make_float2(dir_x, dir_y));
        site.log_aniso = log_aniso;
        out.sites.push_back(site);
    }

    if (out.sites.empty()) {
        std::cerr << "No sites found in JSON: " << path << std::endl;
        std::exit(1);
    }
    return out;
}

static LoadedSites loadSites(const std::string& path) {
    std::string ext = toLower(fs::path(path).extension().string());
    if (ext == ".json") {
        return loadSitesFromJSON(path);
    }
    return loadSitesFromTXT(path);
}

static void writeSitesTXT(const std::vector<Site>& sites, int width, int height, const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        std::cerr << "Failed to write sites: " << path << std::endl;
        return;
    }

    out << "# SAD Sites (position_x, position_y, color_r, color_g, color_b, log_tau, radius, aniso_dir_x, aniso_dir_y, log_aniso)\n";
    out << "# Image size: " << width << " " << height << "\n";
    out << "# Total sites: " << sites.size() << "\n";

    out << std::setprecision(9);
    for (const auto& site : sites) {
        float3 col = site_color(site);
        float2 dir = site_aniso_dir(site);
        out << site.position.x << " " << site.position.y << " "
            << col.x << " " << col.y << " " << col.z << " "
            << site.log_tau << " " << site.radius << " "
            << dir.x << " " << dir.y << " " << site.log_aniso << "\n";
    }
}

struct Defaults {
    int nSites = 65536;
    int maxSites = 70000;
    int iterations = 2000;
    float targetBpp = -1.0f;
    float prunePercentile = 0.033f;
    int pruneStart = 100;
    int pruneFreq = 40;
    int pruneEnd = 1600;
    bool pruneDuringDensify = true;
    int densifyStart = 20;
    int densifyFreq = 20;
    int densifyEnd = 1500;
    float densifyPercentile = 0.01f;
    float densifyScoreAlpha = 0.7f;
    int candUpdateFreq = 1;
    int candUpdatePasses = 1;
    int candDownscale = 1;
    float candRadiusScale = 64.0f;
    uint32_t candRadiusProbes = 0;
    uint32_t candInjectCount = 16;
    uint32_t candHilbertWindow = 0;
    uint32_t candHilbertProbes = 0;
    float lrPosBase = 0.05f;
    float lrTauBase = 0.02f;
    float lrRadiusBase = 0.02f;
    float lrColorBase = 0.02f;
    float lrDirBase = 0.005f;
    float lrAnisoBase = 0.05f;
    float initLogTau = std::nanf("");
    float initRadius = std::nanf("");
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    int maxDim = 4096;
};

static Defaults loadDefaults() {
    Defaults defaults;
    std::ifstream file("training_config.json");
    if (!file) {
        std::cerr << "Missing training_config.json (required)." << std::endl;
        std::exit(1);
    }
    json j;
    file >> j;

    auto setInt = [&](const char* key, int& out) {
        if (j.contains(key)) out = j.at(key).get<int>();
    };
    auto setFloat = [&](const char* key, float& out) {
        if (j.contains(key)) out = j.at(key).get<float>();
    };

    setInt("DEFAULT_SITES", defaults.nSites);
    setInt("DEFAULT_MAX_SITES", defaults.maxSites);
    setInt("DEFAULT_ITERS", defaults.iterations);
    setFloat("DEFAULT_TARGET_BPP", defaults.targetBpp);
    setFloat("PRUNE_PERCENTILE", defaults.prunePercentile);
    if (j.contains("PRUNE_DURING_DENSIFY")) defaults.pruneDuringDensify = j.at("PRUNE_DURING_DENSIFY").get<bool>();
    setInt("PRUNE_START", defaults.pruneStart);
    setInt("PRUNE_FREQ", defaults.pruneFreq);
    setInt("PRUNE_END", defaults.pruneEnd);
    setInt("DENSIFY_START", defaults.densifyStart);
    setInt("DENSIFY_FREQ", defaults.densifyFreq);
    setInt("DENSIFY_END", defaults.densifyEnd);
    setFloat("DENSIFY_PERCENTILE", defaults.densifyPercentile);
    setFloat("DENSIFY_SCORE_ALPHA", defaults.densifyScoreAlpha);
    setInt("CAND_UPDATE_FREQ", defaults.candUpdateFreq);
    setInt("CAND_UPDATE_PASSES", defaults.candUpdatePasses);
    setInt("CAND_DOWNSCALE", defaults.candDownscale);
    setFloat("CAND_RADIUS_SCALE", defaults.candRadiusScale);
    if (j.contains("CAND_RADIUS_PROBES")) defaults.candRadiusProbes = j.at("CAND_RADIUS_PROBES").get<uint32_t>();
    if (j.contains("CAND_INJECT_COUNT")) defaults.candInjectCount = j.at("CAND_INJECT_COUNT").get<uint32_t>();
    if (j.contains("CAND_HILBERT_WINDOW")) defaults.candHilbertWindow = j.at("CAND_HILBERT_WINDOW").get<uint32_t>();
    if (j.contains("CAND_HILBERT_PROBES")) defaults.candHilbertProbes = j.at("CAND_HILBERT_PROBES").get<uint32_t>();
    setFloat("LR_POS_BASE", defaults.lrPosBase);
    setFloat("LR_TAU_BASE", defaults.lrTauBase);
    setFloat("LR_RADIUS_BASE", defaults.lrRadiusBase);
    setFloat("LR_COLOR_BASE", defaults.lrColorBase);
    setFloat("LR_DIR_BASE", defaults.lrDirBase);
    setFloat("LR_ANISO_BASE", defaults.lrAnisoBase);
    setFloat("INIT_LOG_TAU", defaults.initLogTau);
    setFloat("INIT_RADIUS", defaults.initRadius);
    setFloat("BETA1", defaults.beta1);
    setFloat("BETA2", defaults.beta2);
    setFloat("EPS", defaults.eps);
    setInt("MAX_DIM", defaults.maxDim);

    if (std::isnan(defaults.initLogTau) || std::isnan(defaults.initRadius)) {
        std::cerr << "INIT_LOG_TAU and INIT_RADIUS must be set in training_config.json" << std::endl;
        std::exit(1);
    }

    return defaults;
}

enum class InitMode {
    GradientWeighted,
    PerPixel,
    FromSites
};


struct TrainingOptions {
    std::string targetPath;
    std::string maskPath;
    int iterations = 0;
    bool hasTargetBpp = false;
    float targetBpp = 0.0f;
    InitMode initMode = InitMode::GradientWeighted;
    std::string initSitesPath;
    int nSites = 0;
    float initGradientAlpha = 1.0f;
    float initLogTau = 0.0f;
    float initRadius = 0.0f;
    int maxDim = 0;
    float prunePercentile = 0.0f;
    int pruneStart = 0;
    int pruneFreq = 1;
    int pruneEnd = 0;
    bool pruneDuringDensify = false;
    bool densifyEnabled = true;
    int densifyStart = 0;
    int densifyFreq = 1;
    int densifyEnd = 0;
    float densifyPercentile = 0.0f;
    float densifyScoreAlpha = 0.0f;
    int maxSites = 0;
    int candUpdateFreq = 1;
    int candUpdatePasses = 1;
    int candDownscale = 1;
    float candRadiusScale = 64.0f;
    uint32_t candRadiusProbes = 0;
    uint32_t candInjectCount = 16;
    uint32_t candHilbertWindow = 0;
    uint32_t candHilbertProbes = 0;
    float lrScale = 1.0f;
    float lrPosBase = 0.0f;
    float lrTauBase = 0.0f;
    float lrRadiusBase = 0.0f;
    float lrColorBase = 0.0f;
    float lrDirBase = 0.0f;
    float lrAnisoBase = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    float ssimWeight = 0.0f;
    bool ssimMetric = false;
    int logFreq = 1000;
    std::string outputDir = "results";
};

struct RenderOptions {
    std::string sitesPath;
    std::string outputPath;
    std::string renderTargetPath;
    int widthOverride = 0;
    int heightOverride = 0;
    int candidatePasses = 16;
    bool useJFA = true;
    int jfaRounds = 1;
    float candRadiusScale = 64.0f;
    uint32_t candRadiusProbes = 0;
    uint32_t candInjectCount = 16;
    int candDownscale = 1;
    uint32_t candHilbertWindow = 0;
    uint32_t candHilbertProbes = 0;
};

static TrainingOptions makeTrainingOptions(const Defaults& defaults) {
    TrainingOptions opts;
    opts.iterations = defaults.iterations;
    if (defaults.targetBpp > 0.0f) {
        opts.hasTargetBpp = true;
        opts.targetBpp = defaults.targetBpp;
    }
    opts.nSites = defaults.nSites;
    opts.initGradientAlpha = 1.0f;
    opts.initLogTau = defaults.initLogTau;
    opts.initRadius = defaults.initRadius;
    opts.maxDim = defaults.maxDim;
    opts.prunePercentile = defaults.prunePercentile;
    opts.pruneStart = defaults.pruneStart;
    opts.pruneFreq = defaults.pruneFreq;
    opts.pruneEnd = defaults.pruneEnd;
    opts.pruneDuringDensify = defaults.pruneDuringDensify;
    opts.densifyEnabled = true;
    opts.densifyStart = defaults.densifyStart;
    opts.densifyFreq = defaults.densifyFreq;
    opts.densifyEnd = defaults.densifyEnd;
    opts.densifyPercentile = defaults.densifyPercentile;
    opts.densifyScoreAlpha = defaults.densifyScoreAlpha;
    opts.maxSites = defaults.maxSites;
    opts.candUpdateFreq = defaults.candUpdateFreq;
    opts.candUpdatePasses = defaults.candUpdatePasses;
    opts.candDownscale = defaults.candDownscale;
    opts.candRadiusScale = defaults.candRadiusScale;
    opts.candRadiusProbes = defaults.candRadiusProbes;
    opts.candInjectCount = defaults.candInjectCount;
    opts.candHilbertWindow = defaults.candHilbertWindow;
    opts.candHilbertProbes = defaults.candHilbertProbes;
    opts.lrPosBase = defaults.lrPosBase;
    opts.lrTauBase = defaults.lrTauBase;
    opts.lrRadiusBase = defaults.lrRadiusBase;
    opts.lrColorBase = defaults.lrColorBase;
    opts.lrDirBase = defaults.lrDirBase;
    opts.lrAnisoBase = defaults.lrAnisoBase;
    opts.beta1 = defaults.beta1;
    opts.beta2 = defaults.beta2;
    opts.eps = defaults.eps;
    return opts;
}

static RenderOptions makeRenderOptions(const Defaults& defaults, const std::string& sitesPath) {
    RenderOptions opts;
    opts.sitesPath = sitesPath;
    opts.candidatePasses = 16;
    opts.useJFA = true;
    opts.jfaRounds = 1;
    opts.candRadiusScale = defaults.candRadiusScale;
    opts.candRadiusProbes = defaults.candRadiusProbes;
    opts.candInjectCount = defaults.candInjectCount;
    opts.candDownscale = defaults.candDownscale;
    opts.candHilbertWindow = defaults.candHilbertWindow;
    opts.candHilbertProbes = defaults.candHilbertProbes;
    return opts;
}

enum class RunMode {
    Train,
    Render
};

struct RunConfig {
    RunMode mode = RunMode::Train;
    TrainingOptions train;
    RenderOptions render;
};

static const char* initModeName(InitMode mode) {
    switch (mode) {
        case InitMode::FromSites:
            return "sites";
        case InitMode::PerPixel:
            return "per-pixel";
        case InitMode::GradientWeighted:
            return "gradient";
    }
    return "unknown";
}

static std::string fixedFloat(float value, int precision) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

static void printUsage(const Defaults& defaults) {
    std::cout << "Usage:\n";
    std::cout << "  sad_cuda <image_path> [options]\n";
    std::cout << "  sad_cuda --render <sites.txt> [options]\n\n";
    std::cout << "Common options:\n";
    std::cout << "  --help, -h            Show this help\n";
    std::cout << "  --out-dir <path>      Output directory (default: results)\n\n";
    std::cout << "Training options:\n";
    std::cout << "  --sites N             Number of sites (default: " << defaults.nSites << ")\n";
    std::cout << "  --iters N             Iterations (default: " << defaults.iterations << ")\n";
    std::cout << "  --target-bpp F         Target bits-per-pixel\n";
    std::cout << "  --mask <path>          Optional mask image (white=keep, black=ignore)\n";
    std::cout << "  --init-per-pixel       Per-pixel initialization\n";
    std::cout << "  --init-gradient        Gradient-weighted init (default)\n";
    std::cout << "  --init-from-sites <p>  Initialize from sites file\n";
    std::cout << "  --densify              Enable densification\n";
    std::cout << "  --prune-during-densify Allow pruning during densification\n";
    std::cout << "  --prune-start N        Prune start (default: " << defaults.pruneStart << ")\n";
    std::cout << "  --prune-end N          Prune end (default: " << defaults.pruneEnd << ")\n";
    std::cout << "  --prune-freq N         Prune frequency (default: " << defaults.pruneFreq << ")\n";
    std::cout << "  --prune-percentile F   Prune percentile (default: " << defaults.prunePercentile << ")\n";
    std::cout << "  --densify-start N      Densify start (default: " << defaults.densifyStart << ")\n";
    std::cout << "  --densify-end N        Densify end (default: " << defaults.densifyEnd << ")\n";
    std::cout << "  --densify-freq N       Densify frequency (default: " << defaults.densifyFreq << ")\n";
    std::cout << "  --densify-percentile F Densify percentile (default: " << defaults.densifyPercentile << ")\n";
    std::cout << "  --densify-score-alpha F Densify score alpha (default: " << defaults.densifyScoreAlpha << ")\n";
    std::cout << "  --cand-freq N          Candidate update freq (default: " << defaults.candUpdateFreq << ")\n";
    std::cout << "  --cand-passes N        Candidate update passes (default: " << defaults.candUpdatePasses << ")\n";
    std::cout << "  --cand-downscale N     Downscale candidate textures by N (default: " << defaults.candDownscale << ")\n";
    std::cout << "  --cand-hilbert-window N Hilbert window for VPT candidates (default: " << defaults.candHilbertWindow << ")\n";
    std::cout << "  --cand-hilbert-probes N Hilbert probes for VPT candidates (default: " << defaults.candHilbertProbes << ")\n";
    std::cout << "  --cand-inject N       Candidate random injects (default: " << defaults.candInjectCount << ")\n";
    std::cout << "  --max-sites N          Max sites (default: " << defaults.maxSites << ")\n";
    std::cout << "  --ssim                 Log SSIM\n";
    std::cout << "  --ssim-weight F        SSIM weight (default: 0)\n";
    std::cout << "  --log-freq N           Log frequency (default: 1000)\n\n";
    std::cout << "Render options:\n";
    std::cout << "  --out <path>           Output PNG path\n";
    std::cout << "  --width N              Required for TXT sites if no header\n";
    std::cout << "  --height N             Required for TXT sites if no header\n";
    std::cout << "  --render-cand-passes N Candidate passes (default: 16)\n";
    std::cout << "  --render-no-jfa         Disable JFA pre-pass\n";
    std::cout << "  --render-jfa-rounds N  JFA rounds (default: 1)\n";
    std::cout << "  --render-cand-downscale N Downscale candidate textures by N (default: " << defaults.candDownscale << ")\n";
    std::cout << "  --render-hilbert-window N Hilbert window for VPT candidates (default: " << defaults.candHilbertWindow << ")\n";
    std::cout << "  --render-hilbert-probes N Hilbert probes for VPT candidates (default: " << defaults.candHilbertProbes << ")\n";
    std::cout << "  --render-cand-inject N Candidate random injects (default: " << defaults.candInjectCount << ")\n";
    std::cout << "  --render-target <path> Optional target image for PSNR\n";
}

static int requireInt(const std::vector<std::string>& args, size_t& i, const std::string& flag) {
    if (i + 1 >= args.size()) {
        std::cerr << "Missing value for " << flag << std::endl;
        std::exit(1);
    }
    return std::stoi(args[++i]);
}

static float requireFloat(const std::vector<std::string>& args, size_t& i, const std::string& flag) {
    if (i + 1 >= args.size()) {
        std::cerr << "Missing value for " << flag << std::endl;
        std::exit(1);
    }
    return std::stof(args[++i]);
}

static std::string requireString(const std::vector<std::string>& args, size_t& i, const std::string& flag) {
    if (i + 1 >= args.size()) {
        std::cerr << "Missing value for " << flag << std::endl;
        std::exit(1);
    }
    return args[++i];
}

static RunConfig parseArgs(int argc, char** argv, const Defaults& defaults) {
    std::vector<std::string> args;
    args.reserve(static_cast<size_t>(argc));
    for (int i = 0; i < argc; ++i) {
        args.emplace_back(argv[i]);
    }

    RunConfig config;
    config.train = makeTrainingOptions(defaults);

    for (size_t i = 1; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(defaults);
            std::exit(0);
        }
        if (arg == "--render") {
            config.mode = RunMode::Render;
            config.render = makeRenderOptions(defaults, requireString(args, i, arg));
            continue;
        }

        if (arg.rfind("--", 0) == 0) {
            if (arg == "--sites") {
                config.train.nSites = requireInt(args, i, arg);
            } else if (arg == "--iters") {
                config.train.iterations = requireInt(args, i, arg);
            } else if (arg == "--target-bpp") {
                config.train.hasTargetBpp = true;
                config.train.targetBpp = requireFloat(args, i, arg);
            } else if (arg == "--mask") {
                config.train.maskPath = requireString(args, i, arg);
            } else if (arg == "--init-per-pixel") {
                config.train.initMode = InitMode::PerPixel;
            } else if (arg == "--init-gradient") {
                config.train.initMode = InitMode::GradientWeighted;
            } else if (arg == "--init-from-sites") {
                config.train.initMode = InitMode::FromSites;
                config.train.initSitesPath = requireString(args, i, arg);
            } else if (arg == "--init-gradient-alpha") {
                config.train.initGradientAlpha = requireFloat(args, i, arg);
            } else if (arg == "--densify") {
                config.train.densifyEnabled = true;
            } else if (arg == "--densify-start") {
                config.train.densifyStart = requireInt(args, i, arg);
            } else if (arg == "--densify-end") {
                config.train.densifyEnd = requireInt(args, i, arg);
            } else if (arg == "--densify-freq") {
                config.train.densifyFreq = requireInt(args, i, arg);
            } else if (arg == "--densify-percentile") {
                config.train.densifyPercentile = requireFloat(args, i, arg);
            } else if (arg == "--densify-score-alpha") {
                config.train.densifyScoreAlpha = requireFloat(args, i, arg);
            } else if (arg == "--prune-during-densify") {
                config.train.pruneDuringDensify = true;
            } else if (arg == "--prune-start") {
                config.train.pruneStart = requireInt(args, i, arg);
            } else if (arg == "--prune-end") {
                config.train.pruneEnd = requireInt(args, i, arg);
            } else if (arg == "--prune-freq") {
                config.train.pruneFreq = requireInt(args, i, arg);
            } else if (arg == "--prune-percentile") {
                config.train.prunePercentile = requireFloat(args, i, arg);
            } else if (arg == "--cand-freq") {
                config.train.candUpdateFreq = requireInt(args, i, arg);
            } else if (arg == "--cand-passes") {
                config.train.candUpdatePasses = requireInt(args, i, arg);
            } else if (arg == "--cand-downscale") {
                config.train.candDownscale = requireInt(args, i, arg);
            } else if (arg == "--cand-radius-scale") {
                config.train.candRadiusScale = requireFloat(args, i, arg);
            } else if (arg == "--cand-radius-probes") {
                config.train.candRadiusProbes = static_cast<uint32_t>(requireInt(args, i, arg));
            } else if (arg == "--cand-inject-count" || arg == "--cand-inject") {
                config.train.candInjectCount = static_cast<uint32_t>(requireInt(args, i, arg));
            } else if (arg == "--cand-hilbert-window") {
                config.train.candHilbertWindow = static_cast<uint32_t>(requireInt(args, i, arg));
            } else if (arg == "--cand-hilbert-probes") {
                config.train.candHilbertProbes = static_cast<uint32_t>(requireInt(args, i, arg));
            } else if (arg == "--max-sites") {
                config.train.maxSites = requireInt(args, i, arg);
            } else if (arg == "--lr") {
                config.train.lrScale = requireFloat(args, i, arg);
            } else if (arg == "--ssim") {
                config.train.ssimMetric = true;
            } else if (arg == "--ssim-weight") {
                config.train.ssimWeight = requireFloat(args, i, arg);
            } else if (arg == "--log-freq") {
                config.train.logFreq = std::max(1, requireInt(args, i, arg));
            } else if (arg == "--out-dir") {
                config.train.outputDir = requireString(args, i, arg);
            } else if (arg == "--out") {
                config.render.outputPath = requireString(args, i, arg);
            } else if (arg == "--width") {
                config.render.widthOverride = requireInt(args, i, arg);
            } else if (arg == "--height") {
                config.render.heightOverride = requireInt(args, i, arg);
            } else if (arg == "--render-cand-passes") {
                config.render.candidatePasses = requireInt(args, i, arg);
            } else if (arg == "--render-hilbert-window") {
                config.render.candHilbertWindow = static_cast<uint32_t>(requireInt(args, i, arg));
            } else if (arg == "--render-hilbert-probes") {
                config.render.candHilbertProbes = static_cast<uint32_t>(requireInt(args, i, arg));
            } else if (arg == "--render-cand-inject") {
                config.render.candInjectCount = static_cast<uint32_t>(requireInt(args, i, arg));
            } else if (arg == "--render-target") {
                config.render.renderTargetPath = requireString(args, i, arg);
            } else if (arg == "--render-no-jfa") {
                config.render.useJFA = false;
            } else if (arg == "--render-jfa-rounds") {
                config.render.jfaRounds = requireInt(args, i, arg);
            } else if (arg == "--render-cand-downscale") {
                config.render.candDownscale = requireInt(args, i, arg);
            } else {
                std::cerr << "Unknown flag: " << arg << std::endl;
                std::exit(1);
            }
            continue;
        }

        if (config.mode == RunMode::Train) {
            if (config.train.targetPath.empty()) {
                config.train.targetPath = arg;
            } else {
                std::cerr << "Unexpected argument: " << arg << std::endl;
                std::exit(1);
            }
        }
    }

    if (config.mode == RunMode::Render) {
        if (config.render.sitesPath.empty()) {
            std::cerr << "--render requires a sites path" << std::endl;
            std::exit(1);
        }
    } else if (config.train.targetPath.empty()) {
        std::cerr << "Missing input image path" << std::endl;
        std::exit(1);
    }

    return config;
}

struct BppSolveResult {
    float densifyPercentile = 0.0f;
    float prunePercentile = 0.0f;
    int finalSites = 0;
    float achievedBpp = 0.0f;
};

static int simulateFinalSites(
    int initSites,
    int maxSites,
    int iters,
    bool densifyEnabled,
    int densifyStart,
    int densifyEnd,
    int densifyFreq,
    float densifyPercentile,
    bool pruneDuringDensify,
    int pruneStart,
    int pruneEnd,
    int pruneFreq,
    float prunePercentile,
    int maxSplitIndices
) {
    int actualSites = initSites;
    int activeEstimate = initSites;
    int maxSitesClamped = std::max(maxSites, initSites);

    int effectivePruneStart = pruneStart;
    if (densifyEnabled && !pruneDuringDensify && pruneStart < densifyEnd) {
        effectivePruneStart = densifyEnd;
    }

    if (iters <= 0) {
        return activeEstimate;
    }

    for (int it = 0; it < iters; ++it) {
        if (densifyEnabled && densifyPercentile > 0.0f &&
            it >= densifyStart && it <= densifyEnd &&
            it % std::max(1, densifyFreq) == 0 &&
            actualSites < maxSitesClamped) {
            int desired = static_cast<int>(activeEstimate * densifyPercentile);
            int available = maxSitesClamped - actualSites;
            int numToSplit = std::min(desired, std::min(available, maxSplitIndices));
            if (numToSplit > 0) {
                actualSites += numToSplit;
                activeEstimate += numToSplit;
            }
        }

        if (prunePercentile > 0.0f &&
            it >= effectivePruneStart &&
            it < pruneEnd &&
            it % std::max(1, pruneFreq) == 0) {
            int desired = static_cast<int>(activeEstimate * prunePercentile);
            int numToPrune = std::min(desired, maxSplitIndices);
            if (numToPrune > 0) {
                activeEstimate = std::max(0, activeEstimate - numToPrune);
            }
        }
    }

    return activeEstimate;
}

static BppSolveResult solveTargetBpp(
    float targetBpp,
    int width,
    int height,
    int initSites,
    int maxSites,
    int iters,
    bool densifyEnabled,
    int densifyStart,
    int densifyEnd,
    int densifyFreq,
    float baseDensify,
    bool pruneDuringDensify,
    int pruneStart,
    int pruneEnd,
    int pruneFreq,
    float basePrune,
    int maxSplitIndices
) {
    const float bitsPerSite = 16.0f * 8.0f;
    int targetSites = std::max(1, static_cast<int>(std::round(targetBpp * float(width * height) / bitsPerSite)));
    float maxBase = std::max(baseDensify, basePrune);

    if (maxBase <= 0.0f) {
        int finalSites = simulateFinalSites(
            initSites, maxSites, iters,
            densifyEnabled, densifyStart, densifyEnd, densifyFreq,
            0.0f, pruneDuringDensify,
            pruneStart, pruneEnd, pruneFreq,
            0.0f, maxSplitIndices
        );
        float achievedBpp = float(finalSites) * bitsPerSite / float(width * height);
        return {0.0f, 0.0f, finalSites, achievedBpp};
    }

    const float maxPct = 0.95f;
    float sMax = maxPct / maxBase;
    if (sMax > 50.0f) {
        sMax = 50.0f;
    }

    auto evalSites = [&](float scale) {
        float densify = densifyEnabled ? std::min(maxPct, baseDensify * scale) : 0.0f;
        float prune = std::min(maxPct, basePrune * scale);
        return simulateFinalSites(
            initSites, maxSites, iters,
            densifyEnabled, densifyStart, densifyEnd, densifyFreq,
            densify, pruneDuringDensify,
            pruneStart, pruneEnd, pruneFreq,
            prune, maxSplitIndices
        );
    };

    float bestScale = 0.0f;
    int bestSites = evalSites(0.0f);
    int bestErr = std::abs(bestSites - targetSites);
    const int samples = 80;

    for (int i = 0; i <= samples; ++i) {
        float s = sMax * float(i) / float(samples);
        int sites = evalSites(s);
        int err = std::abs(sites - targetSites);
        if (err < bestErr) {
            bestErr = err;
            bestScale = s;
            bestSites = sites;
        }
    }

    float step = sMax / float(samples);
    for (int i = 0; i < 20; ++i) {
        bool improved = false;
        float s0 = bestScale - step;
        float s1 = bestScale + step;
        if (s0 >= 0.0f) {
            int sites = evalSites(s0);
            int err = std::abs(sites - targetSites);
            if (err < bestErr) {
                bestErr = err;
                bestScale = s0;
                bestSites = sites;
                improved = true;
            }
        }
        if (s1 <= sMax) {
            int sites = evalSites(s1);
            int err = std::abs(sites - targetSites);
            if (err < bestErr) {
                bestErr = err;
                bestScale = s1;
                bestSites = sites;
                improved = true;
            }
        }
        if (!improved) {
            step *= 0.5f;
        }
    }

    float densify = densifyEnabled ? std::min(maxPct, baseDensify * bestScale) : 0.0f;
    float prune = std::min(maxPct, basePrune * bestScale);
    float achievedBpp = float(bestSites) * bitsPerSite / float(width * height);

    return {densify, prune, bestSites, achievedBpp};
}

struct SiteCapacityPlan {
    bool densifyEnabled = false;
    bool needsPairs = false;
    bool needsPrune = false;
    int maxSitesCapacity = 0;
    int requestedCapacity = 0;
    int bufferCapacity = 0;
    int scorePairsCount = 0;
    int maxSplitIndicesCapacity = 0;
};

static SiteCapacityPlan planSiteCapacity(const TrainingOptions& options, int initialSiteCount, int numPixels) {
    SiteCapacityPlan plan;
    plan.needsPairs = options.densifyEnabled || options.prunePercentile > 0.0f;
    plan.needsPrune = options.prunePercentile > 0.0f;

    if (options.maxSites > 0) {
        plan.maxSitesCapacity = options.maxSites;
    } else if (options.densifyEnabled) {
        plan.maxSitesCapacity = std::min(numPixels * 2, std::max(options.nSites * 8, 8192));
    } else {
        plan.maxSitesCapacity = numPixels * 2;
    }

    plan.requestedCapacity = options.densifyEnabled ? plan.maxSitesCapacity : initialSiteCount;
    plan.bufferCapacity = std::max(initialSiteCount, plan.requestedCapacity);
    plan.scorePairsCount = plan.needsPairs ? std::max(1, plan.bufferCapacity) : 0;
    plan.maxSplitIndicesCapacity = 65536;
    plan.densifyEnabled = options.densifyEnabled;
    return plan;
}

static void printTrainingOverview(const TrainingOptions& options,
                                  int width,
                                  int height,
                                  int actualSites,
                                  int activeSites,
                                  const SiteCapacityPlan& plan) {
    std::cout << "Training | backend=cuda"
              << " | image=" << width << "x" << height
              << " | sites=" << activeSites << "/" << actualSites
              << " | iters=" << options.iterations
              << " | log-freq=" << options.logFreq
              << " | mask=" << (options.maskPath.empty() ? "no" : "yes")
              << " | out=" << options.outputDir;
    if (options.hasTargetBpp && options.targetBpp > 0.0f) {
        std::cout << " | target-bpp=" << fixedFloat(options.targetBpp, 3);
    }
    std::cout << std::endl;

    std::cout << "Schedule | init=" << initModeName(options.initMode)
              << " | densify=" << (options.densifyEnabled ? ("on cap=" + std::to_string(plan.maxSitesCapacity)) : "off")
              << " | prune=";
    if (options.prunePercentile > 0.0f) {
        int pruneEnd = options.pruneEnd > 0 ? options.pruneEnd : (options.iterations - 1);
        std::cout << "on " << fixedFloat(options.prunePercentile, 3)
                  << " @" << options.pruneStart << "-" << pruneEnd
                  << "/" << std::max(1, options.pruneFreq);
    } else {
        std::cout << "off";
    }
    std::cout << " | cand=freq " << options.candUpdateFreq
              << ", passes " << options.candUpdatePasses
              << ", downscale " << options.candDownscale << "x";
    if (options.candHilbertProbes > 0 && options.candHilbertWindow > 0) {
        std::cout << " hilbert=" << options.candHilbertProbes << "x" << options.candHilbertWindow;
    }
    std::cout << std::endl;
}

struct CandidateUpdatePlan {
    bool shouldUpdate = false;
    int passes = 0;
};

static CandidateUpdatePlan candidateUpdatePlan(int iter, const TrainingOptions& options,
                                               int effectivePruneStart) {
    if (options.initMode == InitMode::PerPixel) {
        if (iter < effectivePruneStart) {
            return {false, 0};
        }
    }

    bool shouldUpdate = options.candUpdateFreq > 0
        ? (iter % options.candUpdateFreq == 0)
        : false;
    return {shouldUpdate, shouldUpdate ? std::max(1, options.candUpdatePasses) : 0};
}

static uint32_t jumpStepForIndex(uint32_t stepIndex, int width, int height) {
    uint32_t maxDim = static_cast<uint32_t>(std::max(width, height));
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
    uint32_t stage = 0;
    if (stages > 0) {
        stage = stepIndex >= stages ? (stages - 1) : stepIndex;
    }
    uint32_t step = pow2 >> (stage + 1);
    return std::max(step, 1u);
}

static uint32_t packJumpStep(uint32_t stepIndex, int width, int height) {
    uint32_t jumpStep = std::min(jumpStepForIndex(stepIndex, width, height), 0xffffu);
    return (jumpStep << 16) | (stepIndex & 0xffffu);
}

extern "C" {
void launchRenderVoronoi(
    const uint32_t* cand0,
    const uint32_t* cand1,
    float3* output,
    const Site* sites,
    float inv_scale_sq,
    uint32_t siteCount,
    int width,
    int height,
    int candWidth,
    int candHeight,
    cudaStream_t stream);

void launchInitCandidates(
    uint32_t* cand0,
    uint32_t* cand1,
    uint32_t siteCount,
    uint32_t seed,
    bool perPixelMode,
    int width,
    int height,
    cudaStream_t stream);

void launchClearCandidates(
    uint32_t* cand0,
    uint32_t* cand1,
    int width,
    int height,
    cudaStream_t stream);

void launchJFASeed(
    uint32_t* cand0,
    const Site* sites,
    uint32_t siteCount,
    int width,
    int height,
    int candDownscale,
    cudaStream_t stream);

void launchJFAFlood(
    const uint32_t* inCand0,
    uint32_t* outCand0,
    const Site* sites,
    uint32_t siteCount,
    uint32_t stepSize,
    float inv_scale_sq,
    int width,
    int height,
    int candDownscale,
    int targetWidth,
    int targetHeight,
    cudaStream_t stream);

void launchPackCandidateSites(
    const Site* sites,
    PackedCandidateSite* packed,
    uint32_t siteCount,
    cudaStream_t stream);

void launchUpdateCandidates(
    const uint32_t* inCand0,
    const uint32_t* inCand1,
    uint32_t* outCand0,
    uint32_t* outCand1,
    const PackedCandidateSite* sites,
    uint32_t siteCount,
    uint32_t step,
    float inv_scale_sq,
    uint32_t stepHigh,
    float radiusScale,
    uint32_t radiusProbes,
    uint32_t injectCount,
    const uint32_t* hilbertOrder,
    const uint32_t* hilbertPos,
    uint32_t hilbertProbeCount,
    uint32_t hilbertWindow,
    int candDownscale,
    int targetWidth,
    int targetHeight,
    int width,
    int height,
    cudaStream_t stream);

void launchComputeGradientsTiled(
    const uint32_t* cand0,
    const uint32_t* cand1,
    const float3* target,
    const float3* rendered,
    const float* mask,
    float* grad_pos_x, float* grad_pos_y,
    float* grad_log_tau, float* grad_radius,
    float* grad_color_r, float* grad_color_g, float* grad_color_b,
    float* grad_dir_x, float* grad_dir_y, float* grad_log_aniso,
    const Site* sites,
    float inv_scale_sq,
    uint32_t siteCount,
    float* removal_delta,
    uint32_t computeRemoval,
    float ssim_weight,
    int width,
    int height,
    int candWidth,
    int candHeight,
    cudaStream_t stream);

void launchAdamUpdate(
    Site* sites,
    AdamState* adam,
    const float* grad_pos_x, const float* grad_pos_y,
    const float* grad_log_tau, const float* grad_radius,
    const float* grad_color_r, const float* grad_color_g, const float* grad_color_b,
    const float* grad_dir_x, const float* grad_dir_y, const float* grad_log_aniso,
    float lr_pos, float lr_tau, float lr_radius,
    float lr_color, float lr_dir, float lr_aniso,
    float beta1, float beta2, float eps,
    uint32_t t,
    uint32_t siteCount,
    int width,
    int height,
    cudaStream_t stream);

void launchClearGradients(
    float* grad_pos_x, float* grad_pos_y,
    float* grad_log_tau, float* grad_radius,
    float* grad_color_r, float* grad_color_g, float* grad_color_b,
    float* grad_dir_x, float* grad_dir_y, float* grad_log_aniso,
    uint32_t siteCount,
    cudaStream_t stream);

void launchClearBuffer(
    float* buffer,
    uint32_t count,
    cudaStream_t stream);

void launchComputePSNR(
    const float3* rendered,
    const float3* target,
    const float* mask,
    float* mse_accum,
    int width,
    int height,
    cudaStream_t stream);

void launchComputeSSIM(
    const float3* rendered,
    const float3* target,
    const float* mask,
    float* ssim_accum,
    int width,
    int height,
    cudaStream_t stream);

void launchCountActiveSites(
    const Site* sites,
    uint32_t* count,
    uint32_t siteCount,
    cudaStream_t stream);

void launchComputeSiteStatsSimple(
    const uint32_t* cand0,
    const uint32_t* cand1,
    const float3* target,
    const float* mask,
    const Site* sites,
    float inv_scale_sq,
    uint32_t siteCount,
    float* mass,
    float* energy,
    float* err_w, float* err_wx, float* err_wy,
    float* err_wxx, float* err_wxy, float* err_wyy,
    int width,
    int height,
    int candWidth,
    int candHeight,
    cudaStream_t stream);

void launchInitGradientWeighted(
    Site* sites,
    uint32_t numSites,
    uint32_t* seedCounter,
    const float3* target,
    const float* mask,
    float gradThreshold,
    uint32_t maxAttempts,
    float init_log_tau,
    float init_radius,
    int width,
    int height,
    cudaStream_t stream);

void launchSplitSites(
    Site* sites,
    AdamState* adam,
    const uint32_t* splitIndices,
    uint32_t numToSplit,
    const float* mass,
    const float* err_w, const float* err_wx, const float* err_wy,
    const float* err_wxx, const float* err_wxy, const float* err_wyy,
    uint32_t currentSiteCount,
    const float3* target,
    int width,
    int height,
    cudaStream_t stream);

void launchComputeDensifyScorePairs(
    const Site* sites,
    const float* mass,
    const float* energy,
    uint2* pairs,
    uint32_t siteCount,
    float minMass,
    float scoreAlpha,
    uint32_t pairCount,
    cudaStream_t stream);

void launchComputePruneScorePairs(
    const Site* sites,
    const float* removal_delta,
    uint2* pairs,
    uint32_t siteCount,
    float deltaNorm,
    uint32_t pairCount,
    cudaStream_t stream);

void launchWriteSplitIndicesFromSorted(
    const uint2* sortedPairs,
    uint32_t* splitIndices,
    uint32_t numToWrite,
    cudaStream_t stream);

void launchPruneSitesByIndex(
    Site* sites,
    const uint32_t* indices,
    uint32_t count,
    cudaStream_t stream);

void launchRadixSortUInt2(
    uint2* data,
    uint2* scratch,
    uint32_t* histFlat,
    uint32_t* blockSums,
    uint32_t paddedCount,
    uint32_t maxKeyExclusive,
    cudaStream_t stream);

void launchTauDiffuse(
    const uint32_t* cand0, const uint32_t* cand1,
    const Site* sites, const float* grad_raw,
    const float* grad_in, float* grad_out,
    uint32_t siteCount, float lambda,
    int width, int height, int candDownscale, cudaStream_t stream);
} // extern "C"

static void updateCandidates(
    uint32_t*& cand0A,
    uint32_t*& cand1A,
    uint32_t*& cand0B,
    uint32_t*& cand1B,
    const PackedCandidateSite* sites,
    uint32_t siteCount,
    int candWidth,
    int candHeight,
    int targetWidth,
    int targetHeight,
    int candDownscale,
    float invScaleSq,
    float radiusScale,
    uint32_t radiusProbes,
    uint32_t injectCount,
    const uint32_t* hilbertOrder,
    const uint32_t* hilbertPos,
    uint32_t hilbertProbeCount,
    uint32_t hilbertWindow,
    int passes,
    uint32_t& jumpPassIndex,
    cudaStream_t stream) {
    for (int i = 0; i < passes; ++i) {
        uint32_t step = packJumpStep(jumpPassIndex, candWidth, candHeight);
        uint32_t stepHigh = jumpPassIndex >> 16;
        launchUpdateCandidates(
            cand0A, cand1A, cand0B, cand1B,
            sites, siteCount,
            step, invScaleSq, stepHigh,
            radiusScale, radiusProbes, injectCount,
            hilbertOrder, hilbertPos, hilbertProbeCount, hilbertWindow,
            candDownscale, targetWidth, targetHeight,
            candWidth, candHeight, stream
        );
        jumpPassIndex += 1;
        std::swap(cand0A, cand0B);
        std::swap(cand1A, cand1B);
    }
}

struct HilbertBuffers {
    DeviceBuffer<uint32_t> order;
    DeviceBuffer<uint32_t> pos;
    DeviceBuffer<uint2> pairs;
    DeviceBuffer<uint2> scratch;
    DeviceBuffer<uint32_t> hist;
    DeviceBuffer<uint32_t> blockSums;
    uint32_t paddedCount = 0;
    uint32_t siteCount = 0;
    bool ready = false;
};

static HilbertBuffers makeHilbertBuffers(int siteCapacity) {
    HilbertBuffers buffers;
    const uint32_t radixBlock = 1024u;
    uint32_t siteCap = static_cast<uint32_t>(std::max(1, siteCapacity));
    buffers.order.alloc(siteCap);
    buffers.pos.alloc(siteCap);

    uint32_t paddedCount = (siteCap + radixBlock - 1) / radixBlock * radixBlock;
    buffers.paddedCount = paddedCount;
    buffers.pairs.alloc(paddedCount);
    buffers.scratch.alloc(paddedCount);

    uint32_t gridSize = (paddedCount + radixBlock - 1) / radixBlock;
    uint32_t histLength = gridSize * 256u;
    uint32_t histBlocks = (histLength + 255u) / 256u;
    buffers.hist.alloc(std::max(1u, histLength));
    buffers.blockSums.alloc(std::max(1u, histBlocks));
    return buffers;
}

static int hilbertBitsForSize(int width, int height) {
    int maxDim = std::max(width, height);
    int n = 1;
    int bits = 0;
    while (n < maxDim) {
        n <<= 1;
        bits += 1;
    }
    return std::max(bits, 1);
}

__device__ __forceinline__ uint32_t hilbertIndexDevice(uint32_t x, uint32_t y, int bits) {
    uint32_t xi = x;
    uint32_t yi = y;
    uint32_t index = 0;
    uint32_t mask = (bits >= 32) ? 0xFFFFFFFFu : ((1u << bits) - 1u);
    for (int i = bits - 1; i >= 0; --i) {
        uint32_t shift = static_cast<uint32_t>(i);
        uint32_t rx = (xi >> shift) & 1u;
        uint32_t ry = (yi >> shift) & 1u;
        uint32_t d = (3u * rx) ^ ry;
        index |= d << (2u * shift);
        if (ry == 0u) {
            if (rx == 1u) {
                xi = mask - xi;
                yi = mask - yi;
            }
            uint32_t tmp = xi;
            xi = yi;
            yi = tmp;
        }
    }
    return index;
}

__global__ void computeHilbertPairsKernel(
    const Site* __restrict__ sites,
    uint2* __restrict__ pairs,
    uint32_t siteCount,
    uint32_t paddedCount,
    int width,
    int height,
    int bits
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= paddedCount) return;
    if (idx < siteCount) {
        float2 pos = sites[idx].position;
        int px = static_cast<int>(pos.x);
        int py = static_cast<int>(pos.y);
        px = px < 0 ? 0 : (px >= width ? (width - 1) : px);
        py = py < 0 ? 0 : (py >= height ? (height - 1) : py);
        uint32_t key = hilbertIndexDevice(static_cast<uint32_t>(px),
                                          static_cast<uint32_t>(py),
                                          bits);
        pairs[idx] = make_uint2(key, idx);
    } else {
        pairs[idx] = make_uint2(0xFFFFFFFFu, 0u);
    }
}

__global__ void writeHilbertOrderKernel(
    const uint2* __restrict__ sortedPairs,
    uint32_t* __restrict__ order,
    uint32_t* __restrict__ pos,
    uint32_t siteCount
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= siteCount) return;
    uint32_t siteIdx = sortedPairs[idx].y;
    order[idx] = siteIdx;
    pos[siteIdx] = idx;
}

static void updateHilbertBuffersGPU(HilbertBuffers& buffers,
                                    const Site* sites,
                                    uint32_t siteCount,
                                    int width,
                                    int height,
                                    cudaStream_t stream) {
    if (siteCount == 0) return;
    int bits = hilbertBitsForSize(width, height);
    uint32_t paddedCount = (siteCount + 1023u) / 1024u * 1024u;
    paddedCount = std::min(paddedCount, std::max(1u, buffers.paddedCount));

    int threads = 256;
    int blocks = static_cast<int>((paddedCount + threads - 1) / threads);
    computeHilbertPairsKernel<<<blocks, threads, 0, stream>>>(
        sites, buffers.pairs.ptr, siteCount, paddedCount, width, height, bits);

    uint32_t maxKeyExclusive = (bits >= 16) ? 0xFFFFFFFFu : (1u << (2 * bits));
    launchRadixSortUInt2(buffers.pairs.ptr,
                         buffers.scratch.ptr,
                         buffers.hist.ptr,
                         buffers.blockSums.ptr,
                         paddedCount,
                         maxKeyExclusive,
                         stream);

    int writeBlocks = static_cast<int>((siteCount + threads - 1) / threads);
    writeHilbertOrderKernel<<<writeBlocks, threads, 0, stream>>>(
        buffers.pairs.ptr, buffers.order.ptr, buffers.pos.ptr, siteCount);
}

static void runJFA(
    uint32_t* cand0,
    uint32_t* cand1,
    const Site* sites,
    uint32_t siteCount,
    float invScaleSq,
    int candWidth,
    int candHeight,
    int targetWidth,
    int targetHeight,
    int candDownscale,
    cudaStream_t stream) {
    launchClearCandidates(cand0, cand1, candWidth, candHeight, stream);
    launchJFASeed(cand0, sites, siteCount, candWidth, candHeight, candDownscale, stream);

    int maxDim = std::max(candWidth, candHeight);
    int step = 1;
    int numPasses = 0;
    while (step < maxDim) {
        step <<= 1;
        numPasses += 1;
    }

    int stepSize = step / 2;
    uint32_t* in = cand0;
    uint32_t* out = cand1;

    while (stepSize >= 1) {
        launchJFAFlood(in, out, sites, siteCount,
                       static_cast<uint32_t>(stepSize), invScaleSq,
                       candWidth, candHeight, candDownscale, targetWidth, targetHeight, stream);
        std::swap(in, out);
        stepSize /= 2;
    }

    if (in != cand0) {
        size_t bytes = static_cast<size_t>(candWidth) * candHeight * 4 * sizeof(uint32_t);
        CUDA_CHECK(cudaMemcpyAsync(cand0, in, bytes, cudaMemcpyDeviceToDevice, stream));
    }
}

static std::vector<Site> filterActiveSites(const std::vector<Site>& sites) {
    std::vector<Site> active;
    active.reserve(sites.size());
    for (const auto& site : sites) {
        if (site.position.x >= 0.0f) {
            active.push_back(site);
        }
    }
    return active;
}

static int countActiveSitesHost(const std::vector<Site>& sites) {
    int count = 0;
    for (const auto& site : sites) {
        if (site.position.x >= 0.0f) {
            count++;
        }
    }
    return count;
}

static void renderVoronoiFromSites(const RenderOptions& options) {
    LoadedSites loaded = loadSites(options.sitesPath);
    int width = loaded.hasSize ? loaded.width : options.widthOverride;
    int height = loaded.hasSize ? loaded.height : options.heightOverride;
    if (width <= 0 || height <= 0) {
        std::cerr << "TXT sites require --width and --height or a header line with image size." << std::endl;
        std::exit(1);
    }

    std::cout << "Loaded " << loaded.sites.size() << " sites" << std::endl;
    std::cout << "Render size: " << width << "x" << height << std::endl;

    int candDownscale = std::max(1, options.candDownscale);
    int candWidth = std::max(1, (width + candDownscale - 1) / candDownscale);
    int candHeight = std::max(1, (height + candDownscale - 1) / candDownscale);

    DeviceBuffer<Site> d_sites;
    d_sites.alloc(loaded.sites.size());
    CUDA_CHECK(cudaMemcpy(d_sites.ptr, loaded.sites.data(), loaded.sites.size() * sizeof(Site), cudaMemcpyHostToDevice));

    DeviceBuffer<PackedCandidateSite> packedCandidates;
    packedCandidates.alloc(loaded.sites.size());

    size_t candCount = static_cast<size_t>(candWidth) * candHeight * 4;
    DeviceBuffer<uint32_t> cand0A;
    DeviceBuffer<uint32_t> cand1A;
    DeviceBuffer<uint32_t> cand0B;
    DeviceBuffer<uint32_t> cand1B;
    cand0A.alloc(candCount);
    cand1A.alloc(candCount);
    cand0B.alloc(candCount);
    cand1B.alloc(candCount);

    DeviceBuffer<float3> d_render;
    d_render.alloc(static_cast<size_t>(width) * height);

    cudaStream_t stream = 0;
    bool usesVptHilbert = options.candHilbertProbes > 0
        && options.candHilbertWindow > 0;
    HilbertBuffers vptHilbert;
    if (usesVptHilbert) {
        vptHilbert = makeHilbertBuffers(static_cast<int>(loaded.sites.size()));
    }

    float invScaleSq = 1.0f / (float(std::max(width, height)) * float(std::max(width, height)));
    uint32_t jumpPassIndex = 0;
    uint32_t seed = 12345u;

    cudaEvent_t candStart = nullptr;
    cudaEvent_t candStop = nullptr;
    cudaEvent_t renderStart = nullptr;
    cudaEvent_t renderStop = nullptr;
    CUDA_CHECK(cudaEventCreate(&candStart));
    CUDA_CHECK(cudaEventCreate(&candStop));
    CUDA_CHECK(cudaEventCreate(&renderStart));
    CUDA_CHECK(cudaEventCreate(&renderStop));

    float candMs = 0.0f;
    float renderMs = 0.0f;

    if (!options.useJFA) {
        launchInitCandidates(cand0A.ptr, cand1A.ptr,
                             static_cast<uint32_t>(loaded.sites.size()), seed,
                             false, candWidth, candHeight, stream);
        launchJFASeed(cand0A.ptr, d_sites.ptr,
                      static_cast<uint32_t>(loaded.sites.size()),
                      candWidth, candHeight, candDownscale, stream);
    }

    int passes = std::max(0, options.candidatePasses);
    int rounds = options.useJFA ? std::max(1, options.jfaRounds) : 1;
    int basePasses = rounds > 0 ? (passes / rounds) : 0;
    int remainder = rounds > 0 ? (passes % rounds) : 0;

    launchPackCandidateSites(d_sites.ptr, packedCandidates.ptr,
                             static_cast<uint32_t>(loaded.sites.size()), stream);

    const uint32_t* hilbertOrder = nullptr;
    const uint32_t* hilbertPos = nullptr;
    uint32_t hilbertProbes = 0;
    uint32_t hilbertWindow = 0;
    if (usesVptHilbert) {
        if (!vptHilbert.ready || vptHilbert.siteCount != loaded.sites.size()) {
            updateHilbertBuffersGPU(vptHilbert, d_sites.ptr,
                                    static_cast<uint32_t>(loaded.sites.size()),
                                    width, height, stream);
            vptHilbert.ready = true;
            vptHilbert.siteCount = static_cast<uint32_t>(loaded.sites.size());
        }
        hilbertOrder = vptHilbert.order.ptr;
        hilbertPos = vptHilbert.pos.ptr;
        hilbertProbes = options.candHilbertProbes;
        hilbertWindow = options.candHilbertWindow;
    }

    CUDA_CHECK(cudaEventRecord(candStart, stream));
    for (int round = 0; round < rounds; ++round) {
        if (options.useJFA) {
            runJFA(cand0A.ptr, cand1A.ptr, d_sites.ptr,
                   static_cast<uint32_t>(loaded.sites.size()),
                   invScaleSq, candWidth, candHeight, width, height, candDownscale, stream);
        }

        int passesThis = basePasses + (round < remainder ? 1 : 0);
        if (passesThis > 0) {
            updateCandidates(cand0A.ptr, cand1A.ptr, cand0B.ptr, cand1B.ptr,
                             packedCandidates.ptr, static_cast<uint32_t>(loaded.sites.size()),
                             candWidth, candHeight, width, height, candDownscale, invScaleSq,
                             options.candRadiusScale, options.candRadiusProbes,
                             options.candInjectCount,
                             hilbertOrder, hilbertPos, hilbertProbes, hilbertWindow,
                             passesThis,
                             jumpPassIndex, stream);
        }
    }
    CUDA_CHECK(cudaEventRecord(candStop, stream));
    CUDA_CHECK(cudaEventSynchronize(candStop));
    CUDA_CHECK(cudaEventElapsedTime(&candMs, candStart, candStop));

    CUDA_CHECK(cudaEventRecord(renderStart, stream));
    launchRenderVoronoi(cand0A.ptr, cand1A.ptr, d_render.ptr,
                        d_sites.ptr, invScaleSq,
                        static_cast<uint32_t>(loaded.sites.size()),
                        width, height, candWidth, candHeight, stream);
    CUDA_CHECK(cudaEventRecord(renderStop, stream));
    CUDA_CHECK(cudaEventSynchronize(renderStop));
    CUDA_CHECK(cudaEventElapsedTime(&renderMs, renderStart, renderStop));

    if (!options.renderTargetPath.empty()) {
        Image target = loadImage(options.renderTargetPath, std::max(width, height));
        if (target.width != width || target.height != height) {
            std::cerr << "Render target size mismatch; skipping PSNR." << std::endl;
        } else {
            Mask mask = defaultMask(width, height);
            DeviceBuffer<float3> d_target;
            DeviceBuffer<float> d_mask;
            DeviceBuffer<float> d_mse;
            d_target.alloc(static_cast<size_t>(width) * height);
            d_mask.alloc(static_cast<size_t>(width) * height);
            d_mse.alloc(1);
            CUDA_CHECK(cudaMemcpy(d_target.ptr, target.pixels.data(),
                                  target.pixels.size() * sizeof(float3), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_mask.ptr, mask.values.data(),
                                  mask.values.size() * sizeof(float), cudaMemcpyHostToDevice));
            launchComputePSNR(d_render.ptr, d_target.ptr, d_mask.ptr, d_mse.ptr, width, height, stream);
            cudaSync("renderPSNR");

            float mseSum = 0.0f;
            CUDA_CHECK(cudaMemcpy(&mseSum, d_mse.ptr, sizeof(float), cudaMemcpyDeviceToHost));
            float denom = std::max(1.0f, mask.sum) * 3.0f;
            float mse = mseSum / denom;
            float psnr = mse > 0.0f ? 20.0f * log10f(1.0f / sqrtf(mse)) : 100.0f;
            std::cout << std::fixed << std::setprecision(2)
                      << "Render PSNR: " << psnr << " dB" << std::endl;
        }
    }

    std::cout << "Candidate build: " << std::fixed << std::setprecision(3)
              << candMs << " ms | Render: " << renderMs << " ms" << std::endl;

    std::vector<float3> hostRender(static_cast<size_t>(width) * height);
    CUDA_CHECK(cudaMemcpy(hostRender.data(), d_render.ptr,
                          hostRender.size() * sizeof(float3),
                          cudaMemcpyDeviceToHost));

    fs::path base = fs::path(options.sitesPath).replace_extension("");
    std::string outPath = options.outputPath.empty()
        ? (base.string() + "_render.png")
        : options.outputPath;

    if (!saveImage(outPath, hostRender, width, height)) {
        std::cerr << "Failed to save: " << outPath << std::endl;
    } else {
        std::cout << "Saved: " << outPath << std::endl;
    }

    CUDA_CHECK(cudaEventDestroy(candStart));
    CUDA_CHECK(cudaEventDestroy(candStop));
    CUDA_CHECK(cudaEventDestroy(renderStart));
    CUDA_CHECK(cudaEventDestroy(renderStop));
}

static void trainVoronoi(const TrainingOptions& input) {
    TrainingOptions options = input;

    Image target = loadImage(options.targetPath, options.maxDim);
    int width = target.width;
    int height = target.height;
    int numPixels = width * height;
    Mask mask = options.maskPath.empty()
        ? defaultMask(width, height)
        : loadMask(options.maskPath, width, height);

    if (options.initMode == InitMode::PerPixel) {
        options.nSites = numPixels;
    }

    int candDownscale = std::max(1, options.candDownscale);
    int candWidth = std::max(1, (width + candDownscale - 1) / candDownscale);
    int candHeight = std::max(1, (height + candDownscale - 1) / candDownscale);

    if (options.hasTargetBpp && options.targetBpp > 0.0f) {
        int pruneEnd = options.pruneEnd > 0 ? options.pruneEnd : (options.iterations - 1);
        int maxSitesCapacity;
        if (options.maxSites > 0) {
            maxSitesCapacity = options.maxSites;
        } else if (options.densifyEnabled) {
            maxSitesCapacity = std::min(numPixels * 2, std::max(options.nSites * 8, 8192));
        } else {
            maxSitesCapacity = numPixels * 2;
        }
        int maxSites = maxSitesCapacity;
        BppSolveResult solve = solveTargetBpp(
            options.targetBpp, width, height,
            options.nSites, maxSites,
            options.iterations, options.densifyEnabled,
            options.densifyStart, options.densifyEnd,
            std::max(1, options.densifyFreq),
            options.densifyPercentile,
            options.pruneDuringDensify,
            options.pruneStart, pruneEnd,
            std::max(1, options.pruneFreq),
            options.prunePercentile,
            65536
        );
        options.densifyPercentile = solve.densifyPercentile;
        options.prunePercentile = solve.prunePercentile;
    }

    std::vector<Site> hostSites;
    if (options.initMode == InitMode::FromSites) {
        if (options.initSitesPath.empty()) {
            std::cerr << "--init-from-sites requires a path" << std::endl;
            std::exit(1);
        }
        LoadedSites loaded = loadSites(options.initSitesPath);
        hostSites = loaded.sites;
        if (loaded.hasSize && (loaded.width != width || loaded.height != height)) {
            float scaleX = float(width) / float(loaded.width);
            float scaleY = float(height) / float(loaded.height);
            for (auto& site : hostSites) {
                site.position.x *= scaleX;
                site.position.y *= scaleY;
            }
        }
    } else if (options.initMode == InitMode::PerPixel) {
        hostSites.reserve(static_cast<size_t>(numPixels));
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                size_t idx = static_cast<size_t>(y) * width + x;
                float3 color = target.pixels[idx];
                Site site = {};
                site.position = make_float2(float(x), float(y));
                site.log_tau = options.initLogTau;
                site.radius = options.initRadius;
                site_set_color(site, color);
                site_set_aniso_dir(site, make_float2(1.0f, 0.0f));
                site.log_aniso = 0.0f;
                hostSites.push_back(site);
            }
        }
    } else {
        hostSites.resize(static_cast<size_t>(options.nSites));
    }

    int actualNSites = static_cast<int>(hostSites.size());
    SiteCapacityPlan plan = planSiteCapacity(options, actualNSites, numPixels);

    DeviceBuffer<float3> d_target;
    d_target.alloc(static_cast<size_t>(numPixels));
    CUDA_CHECK(cudaMemcpy(d_target.ptr, target.pixels.data(),
                          target.pixels.size() * sizeof(float3),
                          cudaMemcpyHostToDevice));

    DeviceBuffer<float> d_mask;
    d_mask.alloc(static_cast<size_t>(numPixels));
    CUDA_CHECK(cudaMemcpy(d_mask.ptr, mask.values.data(),
                          mask.values.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    DeviceBuffer<Site> d_sites;
    d_sites.alloc(plan.bufferCapacity);
    DeviceBuffer<AdamState> d_adam;
    d_adam.alloc(plan.bufferCapacity);
    CUDA_CHECK(cudaMemset(d_adam.ptr, 0, plan.bufferCapacity * sizeof(AdamState)));

    DeviceBuffer<float> grad_pos_x, grad_pos_y, grad_log_tau, grad_radius;
    DeviceBuffer<float> grad_color_r, grad_color_g, grad_color_b;
    DeviceBuffer<float> grad_dir_x, grad_dir_y, grad_log_aniso;
    grad_pos_x.alloc(plan.bufferCapacity);
    grad_pos_y.alloc(plan.bufferCapacity);
    grad_log_tau.alloc(plan.bufferCapacity);
    grad_radius.alloc(plan.bufferCapacity);
    grad_color_r.alloc(plan.bufferCapacity);
    grad_color_g.alloc(plan.bufferCapacity);
    grad_color_b.alloc(plan.bufferCapacity);
    grad_dir_x.alloc(plan.bufferCapacity);
    grad_dir_y.alloc(plan.bufferCapacity);
    grad_log_aniso.alloc(plan.bufferCapacity);

    CUDA_CHECK(cudaMemset(grad_pos_x.ptr, 0, plan.bufferCapacity * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_pos_y.ptr, 0, plan.bufferCapacity * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_log_tau.ptr, 0, plan.bufferCapacity * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_radius.ptr, 0, plan.bufferCapacity * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_color_r.ptr, 0, plan.bufferCapacity * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_color_g.ptr, 0, plan.bufferCapacity * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_color_b.ptr, 0, plan.bufferCapacity * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_dir_x.ptr, 0, plan.bufferCapacity * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_dir_y.ptr, 0, plan.bufferCapacity * sizeof(float)));
    CUDA_CHECK(cudaMemset(grad_log_aniso.ptr, 0, plan.bufferCapacity * sizeof(float)));

    DeviceBuffer<float> tauGradRaw;
    DeviceBuffer<float> tauGradTmp;
    tauGradRaw.alloc(plan.bufferCapacity);
    tauGradTmp.alloc(plan.bufferCapacity);

    DeviceBuffer<float> removalDelta;
    removalDelta.alloc(plan.bufferCapacity);
    CUDA_CHECK(cudaMemset(removalDelta.ptr, 0, plan.bufferCapacity * sizeof(float)));

    DeviceBuffer<float> mass, energy, err_w, err_wx, err_wy, err_wxx, err_wxy, err_wyy;
    if (plan.densifyEnabled) {
        mass.alloc(plan.bufferCapacity);
        energy.alloc(plan.bufferCapacity);
        err_w.alloc(plan.bufferCapacity);
        err_wx.alloc(plan.bufferCapacity);
        err_wy.alloc(plan.bufferCapacity);
        err_wxx.alloc(plan.bufferCapacity);
        err_wxy.alloc(plan.bufferCapacity);
        err_wyy.alloc(plan.bufferCapacity);
    }

    DeviceBuffer<uint2> scorePairs;
    DeviceBuffer<uint32_t> splitIndices;
    DeviceBuffer<uint32_t> pruneIndices;
    if (plan.needsPairs) {
        scorePairs.alloc(plan.scorePairsCount);
    }
    if (plan.densifyEnabled) {
        splitIndices.alloc(plan.maxSplitIndicesCapacity);
    }
    if (plan.needsPrune) {
        pruneIndices.alloc(plan.maxSplitIndicesCapacity);
    }

    DeviceBuffer<uint2> sortScratch;
    DeviceBuffer<uint32_t> sortHist;
    DeviceBuffer<uint32_t> sortBlockSums;
    if (plan.needsPairs) {
        const uint32_t paddedCount = static_cast<uint32_t>(plan.scorePairsCount);
        const uint32_t elementsPerBlock = 256u * 4u;
        uint32_t gridSize = (paddedCount + elementsPerBlock - 1) / elementsPerBlock;
        uint32_t histLength = 256u * gridSize;
        uint32_t histBlocks = (histLength + 256u - 1) / 256u;

        sortScratch.alloc(paddedCount);
        sortHist.alloc(histLength);
        sortBlockSums.alloc(histBlocks);
    }

    size_t candCount = static_cast<size_t>(candWidth) * candHeight * 4;
    DeviceBuffer<uint32_t> cand0A, cand1A, cand0B, cand1B;
    cand0A.alloc(candCount);
    cand1A.alloc(candCount);
    cand0B.alloc(candCount);
    cand1B.alloc(candCount);

    DeviceBuffer<PackedCandidateSite> packedCandidates;
    packedCandidates.alloc(plan.bufferCapacity);

    bool usesVptHilbert = options.candHilbertProbes > 0
        && options.candHilbertWindow > 0;
    HilbertBuffers vptHilbert;
    if (usesVptHilbert) {
        vptHilbert = makeHilbertBuffers(plan.bufferCapacity);
    }

    DeviceBuffer<float3> d_render;
    d_render.alloc(static_cast<size_t>(numPixels));

    DeviceBuffer<float> d_mse;
    d_mse.alloc(1);
    DeviceBuffer<float> d_ssim;
    if (options.ssimMetric) {
        d_ssim.alloc(1);
    }

    DeviceBuffer<uint32_t> d_activeCount;
    d_activeCount.alloc(1);

    // Main stream and async candidate stream for candidate updates
    cudaStream_t stream = nullptr;
    cudaStream_t candStream = nullptr;
    cudaEvent_t candDone = nullptr;
    cudaEvent_t sitesReady = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&candStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&candDone, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&sitesReady, cudaEventDisableTiming));

    if (options.initMode == InitMode::GradientWeighted) {
        DeviceBuffer<uint32_t> seedCounter;
        seedCounter.alloc(1);
        CUDA_CHECK(cudaMemset(seedCounter.ptr, 0, sizeof(uint32_t)));

        float gradThreshold = 0.01f * options.initGradientAlpha;
        launchInitGradientWeighted(
            d_sites.ptr, static_cast<uint32_t>(options.nSites),
            seedCounter.ptr, d_target.ptr, d_mask.ptr,
            gradThreshold, 256,
            options.initLogTau, options.initRadius,
            width, height, stream
        );
        cudaSync("initGradientWeighted");

        // Kernel now sets tau/radius with gradient adjustment - no CPU override needed
        actualNSites = options.nSites;
    } else {
        CUDA_CHECK(cudaMemcpy(d_sites.ptr, hostSites.data(),
                              hostSites.size() * sizeof(Site),
                              cudaMemcpyHostToDevice));
    }

    int activeSitesEstimate = countActiveSitesHost(hostSites);
    printTrainingOverview(options, width, height, actualNSites, activeSitesEstimate, plan);

    uint32_t nSitesU = static_cast<uint32_t>(actualNSites);
    uint32_t seed = 0u;
    launchInitCandidates(cand0A.ptr, cand1A.ptr, nSitesU, seed,
                         options.initMode == InitMode::PerPixel,
                         candWidth, candHeight, stream);
    launchJFASeed(cand0A.ptr, d_sites.ptr, nSitesU,
                  candWidth, candHeight, candDownscale, stream);
    cudaSync("initCandidates");
    CUDA_CHECK(cudaEventRecord(sitesReady, stream));

    std::cout << "Logs | Iter | PSNR | Active | speed | elapsed" << std::endl;
    auto startTime = std::chrono::steady_clock::now();

    float invScaleSq = 1.0f / (float(std::max(width, height)) * float(std::max(width, height)));
    float deltaNormPerPixel = 1.0f / float(std::max(1.0f, mask.sum));
    int tauDiffusePasses = 4;
    float tauDiffuseLambda = 0.05f;

    float lrPos = options.lrPosBase * options.lrScale;
    float lrTau = options.lrTauBase * options.lrScale;
    float lrRadius = options.lrRadiusBase * options.lrScale;
    float lrColor = options.lrColorBase * options.lrScale;
    float lrDirBase = options.lrDirBase * options.lrScale;
    float lrAnisoBase = options.lrAnisoBase * options.lrScale;

    int effectivePruneStart = options.pruneStart;
    if (options.densifyEnabled && !options.pruneDuringDensify && options.pruneStart < options.densifyEnd) {
        effectivePruneStart = options.densifyEnd;
    }
    int effectivePruneEnd = options.pruneEnd > 0 ? options.pruneEnd : (options.iterations - 1);

    uint32_t jumpPassIndex = 0;
    float bestPSNR = 0.0f;
    float finalPSNR = 0.0f;
    float bestSSIM = 0.0f;
    float finalSSIM = 0.0f;
    // For async candidate updates: track which buffers hold current data
    uint32_t* candAsync0A = cand0A.ptr;
    uint32_t* candAsync1A = cand1A.ptr;
    uint32_t* candAsync0B = cand0B.ptr;
    uint32_t* candAsync1B = cand1B.ptr;
    bool pendingCandSwap = false;
    auto applyPendingCandidates = [&]() {
        if (!pendingCandSwap) {
            return;
        }
        CUDA_CHECK(cudaStreamWaitEvent(stream, candDone, 0));
        cand0A.ptr = candAsync0A;
        cand1A.ptr = candAsync1A;
        cand0B.ptr = candAsync0B;
        cand1B.ptr = candAsync1B;
        pendingCandSwap = false;
    };

    for (int iter = 0; iter < options.iterations; ++iter) {
        CandidateUpdatePlan cup = candidateUpdatePlan(iter, options, effectivePruneStart);
        bool shouldUpdateCandidates = cup.shouldUpdate;
        int candidatePasses = cup.passes;
        int desiredSplits = std::max(0, static_cast<int>(activeSitesEstimate * options.densifyPercentile));
        bool shouldDensify = options.densifyEnabled &&
            iter >= options.densifyStart &&
            iter <= options.densifyEnd &&
            (iter % std::max(1, options.densifyFreq) == 0) &&
            actualNSites < plan.bufferCapacity &&
            desiredSplits > 0;
        bool shouldUpdateSynchronously = shouldDensify;

        // Candidate updates read the mutable site buffer on a side stream. Keep
        // them async in the common case, but synchronize on densify iterations so
        // topology changes use the same updated candidate field as other backends.
        if (shouldUpdateCandidates && candidatePasses > 0) {
            const uint32_t* hilbertOrder = nullptr;
            const uint32_t* hilbertPos = nullptr;
            uint32_t hilbertProbes = 0;
            uint32_t hilbertWindow = 0;
            cudaStream_t updateStream = shouldUpdateSynchronously ? stream : candStream;
            if (!shouldUpdateSynchronously) {
                CUDA_CHECK(cudaStreamWaitEvent(candStream, sitesReady, 0));
            } else {
                applyPendingCandidates();
            }
            if (usesVptHilbert) {
                if (!vptHilbert.ready || vptHilbert.siteCount != nSitesU) {
                    updateHilbertBuffersGPU(vptHilbert, d_sites.ptr, nSitesU, width, height, updateStream);
                    vptHilbert.ready = true;
                    vptHilbert.siteCount = nSitesU;
                }
                hilbertOrder = vptHilbert.order.ptr;
                hilbertPos = vptHilbert.pos.ptr;
                hilbertProbes = options.candHilbertProbes;
                hilbertWindow = options.candHilbertWindow;
            }
            launchPackCandidateSites(d_sites.ptr, packedCandidates.ptr, nSitesU, updateStream);
            // Use local copies so swaps don't affect main pointers during async execution
            uint32_t* localCand0A = shouldUpdateSynchronously ? cand0A.ptr : candAsync0A;
            uint32_t* localCand1A = shouldUpdateSynchronously ? cand1A.ptr : candAsync1A;
            uint32_t* localCand0B = shouldUpdateSynchronously ? cand0B.ptr : candAsync0B;
            uint32_t* localCand1B = shouldUpdateSynchronously ? cand1B.ptr : candAsync1B;
            updateCandidates(localCand0A, localCand1A, localCand0B, localCand1B,
                             packedCandidates.ptr, nSitesU,
                             candWidth, candHeight, width, height, candDownscale, invScaleSq,
                             options.candRadiusScale,
                             options.candRadiusProbes,
                             options.candInjectCount,
                             hilbertOrder, hilbertPos, hilbertProbes, hilbertWindow,
                             candidatePasses,
                             jumpPassIndex,
                             updateStream);
            // After updateCandidates swaps internally, final data is in localCand0A/1A
            if (shouldUpdateSynchronously) {
                cand0A.ptr = localCand0A;
                cand1A.ptr = localCand1A;
                cand0B.ptr = localCand0B;
                cand1B.ptr = localCand1B;
                candAsync0A = cand0A.ptr;
                candAsync1A = cand1A.ptr;
                candAsync0B = cand0B.ptr;
                candAsync1B = cand1B.ptr;
            } else {
                candAsync0A = localCand0A;
                candAsync1A = localCand1A;
                candAsync0B = localCand0B;
                candAsync1B = localCand1B;
                CUDA_CHECK(cudaEventRecord(candDone, candStream));
                pendingCandSwap = true;
            }
        }

        if (shouldDensify && plan.densifyEnabled && plan.needsPairs) {
            // Clear stats buffers before computing (matching Metal algorithm)
            launchClearBuffer(mass.ptr, nSitesU, stream);
            launchClearBuffer(energy.ptr, nSitesU, stream);
            launchClearBuffer(err_w.ptr, nSitesU, stream);
            launchClearBuffer(err_wx.ptr, nSitesU, stream);
            launchClearBuffer(err_wy.ptr, nSitesU, stream);
            launchClearBuffer(err_wxx.ptr, nSitesU, stream);
            launchClearBuffer(err_wxy.ptr, nSitesU, stream);
            launchClearBuffer(err_wyy.ptr, nSitesU, stream);

            launchComputeSiteStatsSimple(
                cand0A.ptr, cand1A.ptr,
                d_target.ptr, d_mask.ptr, d_sites.ptr,
                invScaleSq, nSitesU,
                mass.ptr, energy.ptr,
                err_w.ptr, err_wx.ptr, err_wy.ptr,
                err_wxx.ptr, err_wxy.ptr, err_wyy.ptr,
                width, height, candWidth, candHeight, stream
            );
            cudaSync("computeSiteStatsSimple");

            launchComputeDensifyScorePairs(
                d_sites.ptr, mass.ptr, energy.ptr,
                scorePairs.ptr, nSitesU,
                1.0f, options.densifyScoreAlpha,
                static_cast<uint32_t>(plan.scorePairsCount), stream
            );
            launchRadixSortUInt2(
                scorePairs.ptr, sortScratch.ptr,
                sortHist.ptr, sortBlockSums.ptr,
                static_cast<uint32_t>(plan.scorePairsCount),
                0xffffffffu, stream
            );
            cudaSync("densifySort");

            if (desiredSplits > 0 && actualNSites < plan.bufferCapacity) {
                int available = plan.bufferCapacity - actualNSites;
                int numToSplit = std::min(desiredSplits, std::min(available, plan.maxSplitIndicesCapacity));
                if (numToSplit > 0) {
                    launchWriteSplitIndicesFromSorted(
                        scorePairs.ptr, splitIndices.ptr,
                        static_cast<uint32_t>(numToSplit), stream
                    );
                    launchSplitSites(
                        d_sites.ptr, d_adam.ptr,
                        splitIndices.ptr, static_cast<uint32_t>(numToSplit),
                        mass.ptr,
                        err_w.ptr, err_wx.ptr, err_wy.ptr,
                        err_wxx.ptr, err_wxy.ptr, err_wyy.ptr,
                        static_cast<uint32_t>(actualNSites),
                        d_target.ptr,
                        width, height, stream
                    );
                    cudaSync("splitSites");
                    actualNSites += numToSplit;
                    nSitesU = static_cast<uint32_t>(actualNSites);
                    activeSitesEstimate += numToSplit;
                }
            }
        }

        bool shouldPrune = options.prunePercentile > 0.0f &&
            iter >= effectivePruneStart &&
            iter < effectivePruneEnd &&
            (iter % std::max(1, options.pruneFreq) == 0) &&
            plan.needsPairs && plan.needsPrune;

        if (shouldPrune) {
            launchClearBuffer(removalDelta.ptr, nSitesU, stream);
        }

        // Note: gradients are cleared by Adam after consumption (see adam.cu)
        // No need to clear here - Adam from previous iteration already zeroed them

        // Render before gradients if SSIM loss is used (matching Metal algorithm)
        bool needsLossRender = options.ssimWeight > 0.0f;
        if (needsLossRender) {
            launchRenderVoronoi(
                cand0A.ptr, cand1A.ptr,
                d_render.ptr,
                d_sites.ptr, invScaleSq,
                nSitesU,
                width, height, candWidth, candHeight, stream
            );
        }

        launchComputeGradientsTiled(
            cand0A.ptr, cand1A.ptr,
            d_target.ptr, d_render.ptr, d_mask.ptr,
            grad_pos_x.ptr, grad_pos_y.ptr,
            grad_log_tau.ptr, grad_radius.ptr,
            grad_color_r.ptr, grad_color_g.ptr, grad_color_b.ptr,
            grad_dir_x.ptr, grad_dir_y.ptr, grad_log_aniso.ptr,
            d_sites.ptr, invScaleSq, nSitesU,
            removalDelta.ptr,
            shouldPrune ? 1u : 0u,
            options.ssimWeight,
            width, height, candWidth, candHeight, stream
        );

        if (tauDiffusePasses > 0 && tauDiffuseLambda > 0.0f) {
            // Copy grad_log_tau to tauGradRaw (save original for fixed reference)
            CUDA_CHECK(cudaMemcpyAsync(tauGradRaw.ptr, grad_log_tau.ptr,
                                       nSitesU * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream));

            float blend = float(iter) / float(std::max(1, options.iterations));
            float lambda = tauDiffuseLambda * (0.1f + 0.9f * blend);

            applyPendingCandidates();

            // Start from grad_log_tau (not tauGradRaw), matching Metal algorithm
            float* currentIn = grad_log_tau.ptr;
            float* currentOut = tauGradTmp.ptr;
            for (int pass = 0; pass < tauDiffusePasses; ++pass) {
                launchTauDiffuse(
                    cand0A.ptr, cand1A.ptr,  // Use updated candidates after sync
                    d_sites.ptr,
                    tauGradRaw.ptr,  // Fixed reference (original gradients)
                    currentIn,        // Input to diffuse
                    currentOut,       // Output buffer
                    nSitesU, lambda,
                    candWidth, candHeight, candDownscale, stream
                );
                std::swap(currentIn, currentOut);
            }
            // After even number of passes (4), result is back in grad_log_tau
            // No writeback needed - diffusion modifies grad_log_tau in-place
        } else if (pendingCandSwap) {
            // If no tau diffusion, wait before Adam to ensure candidates complete
            applyPendingCandidates();
        }

        float lrDir = lrDirBase;
        float lrAniso = lrAnisoBase;

        launchAdamUpdate(
            d_sites.ptr, d_adam.ptr,
            grad_pos_x.ptr, grad_pos_y.ptr,
            grad_log_tau.ptr, grad_radius.ptr,
            grad_color_r.ptr, grad_color_g.ptr, grad_color_b.ptr,
            grad_dir_x.ptr, grad_dir_y.ptr, grad_log_aniso.ptr,
            lrPos, lrTau, lrRadius, lrColor, lrDir, lrAniso,
            options.beta1, options.beta2, options.eps,
            static_cast<uint32_t>(iter + 1),
            nSitesU,
            width, height, stream
        );

        // Clear gradients after Adam consumes them (matching Metal algorithm)
        launchClearGradients(
            grad_pos_x.ptr, grad_pos_y.ptr,
            grad_log_tau.ptr, grad_radius.ptr,
            grad_color_r.ptr, grad_color_g.ptr, grad_color_b.ptr,
            grad_dir_x.ptr, grad_dir_y.ptr, grad_log_aniso.ptr,
            nSitesU, stream
        );

        if (shouldPrune) {
            launchComputePruneScorePairs(
                d_sites.ptr, removalDelta.ptr,
                scorePairs.ptr, nSitesU,
                deltaNormPerPixel,
                static_cast<uint32_t>(plan.scorePairsCount), stream
            );
            launchRadixSortUInt2(
                scorePairs.ptr, sortScratch.ptr,
                sortHist.ptr, sortBlockSums.ptr,
                static_cast<uint32_t>(plan.scorePairsCount),
                0xffffffffu, stream
            );
            cudaSync("pruneSort");

            int desiredPrunes = std::max(0, static_cast<int>(activeSitesEstimate * options.prunePercentile));
            int numToPrune = std::min(desiredPrunes, plan.maxSplitIndicesCapacity);
            if (numToPrune > 0) {
                launchWriteSplitIndicesFromSorted(
                    scorePairs.ptr, pruneIndices.ptr,
                    static_cast<uint32_t>(numToPrune), stream
                );
                launchPruneSitesByIndex(
                    d_sites.ptr, pruneIndices.ptr,
                    static_cast<uint32_t>(numToPrune), stream
                );
                activeSitesEstimate = std::max(0, activeSitesEstimate - numToPrune);
            }
        }
        CUDA_CHECK(cudaEventRecord(sitesReady, stream));

        bool shouldLog = (iter % options.logFreq == 0) || (iter == options.iterations - 1);
        if (shouldLog) {
            launchRenderVoronoi(
                cand0A.ptr, cand1A.ptr,
                d_render.ptr,
                d_sites.ptr, invScaleSq,
                nSitesU,
                width, height, candWidth, candHeight, stream
            );
            launchComputePSNR(d_render.ptr, d_target.ptr, d_mask.ptr, d_mse.ptr, width, height, stream);
            if (options.ssimMetric) {
                launchComputeSSIM(d_render.ptr, d_target.ptr, d_mask.ptr, d_ssim.ptr, width, height, stream);
            }
            launchCountActiveSites(d_sites.ptr, d_activeCount.ptr, nSitesU, stream);
            cudaSync("logMetrics");

            float mseSum = 0.0f;
            CUDA_CHECK(cudaMemcpy(&mseSum, d_mse.ptr, sizeof(float), cudaMemcpyDeviceToHost));
            float mse = mseSum / float(std::max(1.0f, mask.sum) * 3.0f);
            float psnr = mse > 0.0f ? 20.0f * log10f(1.0f / std::sqrt(mse)) : 100.0f;

            float ssim = 0.0f;
            if (options.ssimMetric) {
                float ssimSum = 0.0f;
                CUDA_CHECK(cudaMemcpy(&ssimSum, d_ssim.ptr, sizeof(float), cudaMemcpyDeviceToHost));
                ssim = ssimSum / float(std::max(1.0f, mask.sum));
            }

            uint32_t activeCount = 0;
            CUDA_CHECK(cudaMemcpy(&activeCount, d_activeCount.ptr, sizeof(uint32_t), cudaMemcpyDeviceToHost));

            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            float itPerSec = elapsed > 0.0f ? float(iter + 1) / elapsed : 0.0f;

            if (options.ssimMetric) {
                std::cout << "Iter " << std::setw(4) << iter
                          << " | PSNR: " << std::fixed << std::setprecision(2) << psnr
                          << " dB | SSIM: " << std::setprecision(4) << ssim
                          << " | Active: " << activeCount << "/" << actualNSites
                          << " | " << std::setprecision(1) << itPerSec << " it/s"
                          << " | " << elapsed << "s" << std::endl;
            } else {
                std::cout << "Iter " << std::setw(4) << iter
                          << " | PSNR: " << std::fixed << std::setprecision(2) << psnr
                          << " dB | Active: " << activeCount << "/" << actualNSites
                          << " | " << std::setprecision(1) << itPerSec << " it/s"
                          << " | " << elapsed << "s" << std::endl;
            }

            finalPSNR = psnr;
            bestPSNR = std::max(bestPSNR, psnr);
            if (options.ssimMetric) {
                finalSSIM = ssim;
                bestSSIM = std::max(bestSSIM, ssim);
            }
        }
    }

    auto trainEndTime = std::chrono::steady_clock::now();
    float trainTime = std::chrono::duration<float>(trainEndTime - startTime).count();
    applyPendingCandidates();

    launchRenderVoronoi(
        cand0A.ptr, cand1A.ptr,
        d_render.ptr,
        d_sites.ptr, invScaleSq,
        nSitesU,
        width, height, candWidth, candHeight, stream
    );
    cudaSync("finalRender");

    std::vector<float3> hostRender(static_cast<size_t>(numPixels));
    CUDA_CHECK(cudaMemcpy(hostRender.data(), d_render.ptr,
                          hostRender.size() * sizeof(float3),
                          cudaMemcpyDeviceToHost));

    fs::create_directories(options.outputDir);
    std::string base = fs::path(options.targetPath).stem().string();
    std::string imagePath = (fs::path(options.outputDir) / (base + ".png")).string();
    std::string sitesPath = (fs::path(options.outputDir) / (base + "_sites.txt")).string();

    if (saveImage(imagePath, hostRender, width, height)) {
        std::cout << "Saved: " << imagePath << std::endl;
    } else {
        std::cerr << "Failed to save: " << imagePath << std::endl;
    }

    std::vector<Site> hostFinal(static_cast<size_t>(actualNSites));
    CUDA_CHECK(cudaMemcpy(hostFinal.data(), d_sites.ptr,
                          hostFinal.size() * sizeof(Site),
                          cudaMemcpyDeviceToHost));
    std::vector<Site> activeSites = filterActiveSites(hostFinal);
    writeSitesTXT(activeSites, width, height, sitesPath);
    std::cout << "Saved: " << sitesPath << std::endl;

    std::cout << "Final PSNR: " << finalPSNR << " dB (best " << bestPSNR << ")" << std::endl;
    if (options.ssimMetric) {
        std::cout << "Final SSIM: " << finalSSIM << " (best " << bestSSIM << ")" << std::endl;
    }
    std::cout << std::fixed << std::setprecision(2)
              << "Training time: " << trainTime << " s" << std::endl;
    auto totalEndTime = std::chrono::steady_clock::now();
    float totalTime = std::chrono::duration<float>(totalEndTime - startTime).count();
    std::cout << "Total time: " << totalTime << " s" << std::endl;

    // Cleanup streams
    CUDA_CHECK(cudaEventDestroy(candDone));
    CUDA_CHECK(cudaEventDestroy(sitesReady));
    CUDA_CHECK(cudaStreamDestroy(candStream));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

int main(int argc, char** argv) {
    Defaults defaults = loadDefaults();
    RunConfig config = parseArgs(argc, argv, defaults);

    if (config.mode == RunMode::Render) {
        renderVoronoiFromSites(config.render);
        return 0;
    }

    trainVoronoi(config.train);
    return 0;
}
