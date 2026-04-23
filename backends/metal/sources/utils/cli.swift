import Foundation

enum RunMode {
    case render(RenderOptions)
    case train(TrainingOptions)
}

struct Defaults {
    var nSites: Int = 65536
    var nIterations: Int = 2000
    var prunePercentile: Float = 0.01
    var pruneStartIter: Int = 100
    var pruneFreq: Int = 20
    var pruneEndIter: Int? = 3600
    var densifyStart: Int = 20
    var densifyFreq: Int = 20
    var densifyEnd: Int = 3500
    var densifyPercentile: Float = 0.01
    var densifyScoreAlpha: Float = 0.7
    var candUpdateFreq: Int = 1
    var candUpdatePasses: Int = 1
    var candDownscale: Int = 1
    var candRadiusScale: Float = 64.0
    var candRadiusProbes: UInt32 = 0
    var candInjectCount: UInt32 = 16
    var candHilbertWindow: UInt32 = 0
    var candHilbertProbes: UInt32 = 0
    var lrPosBase: Float = 0.05
    var lrTauBase: Float = 0.01
    var lrRadiusBase: Float = 0.02
    var lrColorBase: Float = 0.02
    var lrDirBase: Float = 0.02
    var lrAnisoBase: Float = 0.02
    var initLogTau: Float = Float.nan
    var initRadius: Float = Float.nan
    var beta1: Float = 0.9
    var beta2: Float = 0.999
    var eps: Float = 1e-8
    var maxDim: Int = 2048
    var maxSites: Int = 70000
    var targetBpp: Float = -1.0
    var pruneDuringDensify: Bool = true
}

private func resolveTrainingConfigPath() -> String? {
    let env = ProcessInfo.processInfo.environment
    let fileManager = FileManager.default

    var candidates: [String] = []
    if let envPath = env["CONFIG_PATH"], !envPath.isEmpty {
        candidates.append(envPath)
    }
    candidates.append((fileManager.currentDirectoryPath as NSString).appendingPathComponent("training_config.json"))
    if let executable = Bundle.main.executablePath {
        let repoRoot = URL(fileURLWithPath: executable)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("training_config.json")
            .path
        candidates.append(repoRoot)
    }

    for path in candidates {
        if fileManager.fileExists(atPath: path) {
            return path
        }
    }
    return nil
}

func loadDefaults() -> Defaults {
    guard let configPath = resolveTrainingConfigPath(),
          let config = loadTrainingConfig(path: configPath) else {
        fatalError("Missing training_config.json (required).")
    }

    var defaults = Defaults()
    if let value = config.DEFAULT_SITES { defaults.nSites = value }
    if let value = config.DEFAULT_ITERS { defaults.nIterations = value }
    if let value = config.DEFAULT_MAX_SITES { defaults.maxSites = value }
    if let value = config.DEFAULT_TARGET_BPP { defaults.targetBpp = value }
    if let value = config.PRUNE_PERCENTILE { defaults.prunePercentile = value }
    if let value = config.PRUNE_DURING_DENSIFY { defaults.pruneDuringDensify = value }
    if let value = config.PRUNE_START { defaults.pruneStartIter = value }
    if let value = config.PRUNE_FREQ { defaults.pruneFreq = value }
    if let value = config.PRUNE_END { defaults.pruneEndIter = value }
    if let value = config.DENSIFY_START { defaults.densifyStart = value }
    if let value = config.DENSIFY_FREQ { defaults.densifyFreq = value }
    if let value = config.DENSIFY_END { defaults.densifyEnd = value }
    if let value = config.DENSIFY_PERCENTILE { defaults.densifyPercentile = value }
    if let value = config.DENSIFY_SCORE_ALPHA { defaults.densifyScoreAlpha = value }
    if let value = config.CAND_UPDATE_FREQ { defaults.candUpdateFreq = value }
    if let value = config.CAND_UPDATE_PASSES { defaults.candUpdatePasses = value }
    if let value = config.CAND_DOWNSCALE { defaults.candDownscale = value }
    if let value = config.CAND_RADIUS_SCALE { defaults.candRadiusScale = value }
    if let value = config.CAND_RADIUS_PROBES { defaults.candRadiusProbes = UInt32(max(0, value)) }
    if let value = config.CAND_INJECT_COUNT { defaults.candInjectCount = UInt32(max(0, value)) }
    if let value = config.CAND_HILBERT_WINDOW { defaults.candHilbertWindow = UInt32(max(0, value)) }
    if let value = config.CAND_HILBERT_PROBES { defaults.candHilbertProbes = UInt32(max(0, value)) }
    if let value = config.LR_POS_BASE { defaults.lrPosBase = value }
    if let value = config.LR_TAU_BASE { defaults.lrTauBase = value }
    if let value = config.LR_RADIUS_BASE { defaults.lrRadiusBase = value }
    if let value = config.LR_COLOR_BASE { defaults.lrColorBase = value }
    if let value = config.LR_DIR_BASE { defaults.lrDirBase = value }
    if let value = config.LR_ANISO_BASE { defaults.lrAnisoBase = value }
    if let value = config.INIT_LOG_TAU { defaults.initLogTau = value }
    if let value = config.INIT_RADIUS { defaults.initRadius = value }
    if let value = config.BETA1 { defaults.beta1 = value }
    if let value = config.BETA2 { defaults.beta2 = value }
    if let value = config.EPS { defaults.eps = value }
    if let value = config.MAX_DIM { defaults.maxDim = value }

    if defaults.initLogTau.isNaN || defaults.initRadius.isNaN {
        fatalError("INIT_LOG_TAU and INIT_RADIUS must be set in training_config.json")
    }

    return defaults
}

func makeTrainingOptions(defaults: Defaults) -> TrainingOptions {
    return TrainingOptions(
        targetPath: "",
        maskPath: nil,
        iterations: defaults.nIterations,
        targetBpp: defaults.targetBpp > 0.0 ? defaults.targetBpp : nil,
        initMode: .gradientWeighted,
        nSites: defaults.nSites,
        initGradientAlpha: 1.0,
        initLogTau: defaults.initLogTau,
        initRadius: defaults.initRadius,
        maxDim: defaults.maxDim,
        prunePercentile: defaults.prunePercentile,
        pruneStartIter: defaults.pruneStartIter,
        pruneFreq: defaults.pruneFreq,
        pruneEndIter: defaults.pruneEndIter,
        pruneDuringDensify: defaults.pruneDuringDensify,
        densifyEnabled: true,
        densifyStart: defaults.densifyStart,
        densifyFreq: defaults.densifyFreq,
        densifyEnd: defaults.densifyEnd,
        densifyPercentile: defaults.densifyPercentile,
        densifyScoreAlpha: defaults.densifyScoreAlpha,
        maxSites: defaults.maxSites,
        candUpdateFreq: defaults.candUpdateFreq,
        candUpdatePasses: defaults.candUpdatePasses,
        candDownscale: defaults.candDownscale,
        candRadiusScale: defaults.candRadiusScale,
        candRadiusProbes: defaults.candRadiusProbes,
        candInjectCount: defaults.candInjectCount,
        candHilbertWindow: defaults.candHilbertWindow,
        candHilbertProbes: defaults.candHilbertProbes,
        exportCandPasses: nil,
        lrScale: 1.0,
        lrPosBase: defaults.lrPosBase,
        lrTauBase: defaults.lrTauBase,
        lrRadiusBase: defaults.lrRadiusBase,
        lrColorBase: defaults.lrColorBase,
        lrDirBase: defaults.lrDirBase,
        lrAnisoBase: defaults.lrAnisoBase,
        beta1: defaults.beta1,
        beta2: defaults.beta2,
        eps: defaults.eps,
        ssimWeight: 0.0,
        ssimMetric: false,
        showViewer: false,
        viewerFreq: 10,
        logFreq: 1000,
        traceFrame: nil,
        outputDir: "results",
        exportNeighbors: false,
        packedPSNR: false
    )
}

func makeRenderOptions(defaults: Defaults, sitesPath: String) -> RenderOptions {
    return RenderOptions(
        sitesPath: sitesPath,
        outputPath: nil,
        outputVoronoiPath: nil,
        renderTargetPath: nil,
        widthOverride: nil,
        heightOverride: nil,
        candidatePasses: 16,
        useJFA: true,
        jfaRounds: 1,
        candRadiusScale: defaults.candRadiusScale,
        candRadiusProbes: defaults.candRadiusProbes,
        candInjectCount: defaults.candInjectCount,
        candDownscale: defaults.candDownscale,
        candHilbertWindow: defaults.candHilbertWindow,
        candHilbertProbes: defaults.candHilbertProbes,
        usePackedInference: false
    )
}

func printUsage(defaults: Defaults) {
    let pruneEndDefault = defaults.pruneEndIter ?? (defaults.nIterations - 1)
    let prunePercentStr = String(format: "%.3f", defaults.prunePercentile)
    let densifyPercentStr = String(format: "%.3f", defaults.densifyPercentile)
    let densifyAlphaStr = String(format: "%.3f", defaults.densifyScoreAlpha)
    print("Usage:")
    print("  ./run.sh --backend metal <image_path> [options]")
    print("  ./run.sh --backend metal --render <sites.(json|txt)> [options]")
    print("  --help, -h            Show this help message")
    print("\nInitialization Modes:")
    print("  MODE 1: Per-pixel")
    print("    --init-per-pixel     1 site per pixel, exact colors, hard blending")
    print("                         Best quality, automatic site count")
    print("  MODE 2: Gradient-weighted (default)")
    print("    --init-gradient      Sample N sites from gradient-weighted distribution")
    print("                         Requires --sites N (e.g. 500-1000)")
    print("    --init-gradient-alpha F    Gradient threshold scale (accept if grad > 0.01*alpha) (default: 1.0)")
    print("  MODE 3: From previous sites")
    print("    --init-from-sites <path>   Initialize from existing sites JSON file")
    print("                               Useful for video frame-by-frame training")
    print("\nTraining Options:")
    print("  --sites N              Number of sites for gradient init (default: \(defaults.nSites))")
    print("  --iters N              Number of iterations (default: \(defaults.nIterations))")
    print("  --target-bpp F         Solve densify/prune percentiles to target bits-per-pixel (16 bytes/site)")
    print("  --prune-start N        Start pruning at iteration N (default: \(defaults.pruneStartIter))")
    print("  --prune-end N          Stop pruning at iteration N (default: \(pruneEndDefault))")
    print("  --prune-freq N         Prune every N iterations (default: \(defaults.pruneFreq))")
    print("  --prune-percentile F   Prune bottom fraction by delta loss [0.0-1.0] (default: \(prunePercentStr))")
    print("  --prune-during-densify Allow pruning while densifying (default: \(defaults.pruneDuringDensify ? "on" : "off"))")
    print("  --cand-freq N          Candidate update frequency (default: \(defaults.candUpdateFreq))")
    print("  --cand-passes N        Candidate update passes per update (default: \(defaults.candUpdatePasses))")
    print("  --cand-downscale N     Downscale candidate textures by N (default: \(defaults.candDownscale))")
    print("  --cand-radius-scale F  Candidate probe radius scale (default: \(defaults.candRadiusScale))")
    print("  --cand-radius-probes N Candidate probe count (default: \(defaults.candRadiusProbes))")
    print("  --cand-inject N        Candidate inject count (default: \(defaults.candInjectCount))")
    print("  --cand-hilbert-window N Hilbert window for VPT candidates (default: \(defaults.candHilbertWindow))")
    print("  --cand-hilbert-probes N Hilbert probes for VPT candidates (default: \(defaults.candHilbertProbes))")
    print("  --lr F                 Learning rate scale multiplier (default: 1.0)")
    print("  --lr-tau F             Learning rate for tau (default: \(defaults.lrTauBase))")
    print("  --lr-radius F          Learning rate for radius (default: \(defaults.lrRadiusBase))")
    print("  --lr-dir F             Learning rate for direction (default: \(defaults.lrDirBase))")
    print("  --lr-aniso F           Learning rate for anisotropy (default: \(defaults.lrAnisoBase))")
    print("  --init-tau F           Initial log tau value (default: \(defaults.initLogTau))")
    print("  --ssim-weight F        Weight for SSIM loss (loss = MSE + w*(1-SSIM)) (default: 0.0)")
    print("  --ssim                 Log SSIM metric during training")
    print("  --packed-psnr          Re-render with packed inference and report PSNR (default: off)")
    print("  --mask <path>          Optional mask image (white=keep, black=ignore)")
    print("\nDensification Options:")
    print("  --densify              Enable densification (split high-error sites up to --max-sites)")
    print("  --densify-start N      Start densifying at iteration N (default: \(defaults.densifyStart))")
    print("  --densify-freq N       Densify every N iterations (default: \(defaults.densifyFreq))")
    print("  --densify-end N        Stop densifying at iteration N (default: \(defaults.densifyEnd))")
    print("  --densify-percentile F Split top fraction by score [0.0-1.0] (default: \(densifyPercentStr))")
    print("  --densify-score-alpha F Densify score normalization exponent (default: \(densifyAlphaStr))")
    print("  --max-sites N          Hard cap on total sites (default: \(defaults.maxSites))")
    print("\nRender Options:")
    print("  --render <path>        Render from exported sites (JSON preferred)")
    print("  --out <path>           Output PNG path (default: <sites>_render.png)")
    print("  --render-cells         Also write hashed cell visualization (<sites>_cells.png)")
    print("  --render-cand-passes N Candidate update passes before rendering (default: 16)")
    print("  --render-no-jfa        Disable JFA pre-pass for render (default: on)")
    print("  --render-jfa-rounds N  Number of JFA rounds in render (default: 1)")
    print("  --render-cand-downscale N Downscale candidate textures by N (default: \(defaults.candDownscale))")
    print("  --render-cand-radius-scale F Candidate probe radius scale (default: \(defaults.candRadiusScale))")
    print("  --render-cand-radius-probes N Candidate probe count (default: \(defaults.candRadiusProbes))")
    print("  --render-cand-inject N Candidate inject count (default: \(defaults.candInjectCount))")
    print("  --render-hilbert-window N Hilbert window for VPT candidates (default: \(defaults.candHilbertWindow))")
    print("  --render-hilbert-probes N Hilbert probes for VPT candidates (default: \(defaults.candHilbertProbes))")
    print("  --render-target <path> Optional target image for PSNR report")
    print("  --render-packed        Use packed inference format for rendering (default: off)")
    print("  --show-viewer          Show a live MTKView preview window during training")
    print("  --viewer-freq N        Viewer update frequency in iterations (default: 10)")
    print("  --log-freq N           PSNR/SSIM logging frequency in iterations (default: 1000)")
    print("  --out-dir <path>       Output directory for all results (default: results/)")
    print("  --export-neighbors     Compute and export neighbor graph in sites.json (default: off)")
    print("  --trace-frame N        Capture Metal GPU trace at iteration N (.gputrace file)")
    print("  --width N              Required for TXT render (no size in TXT)")
    print("  --height N             Required for TXT render (no size in TXT)")
    print("  --export-cand-passes N Candidate update passes for final export (default: cand-passes)")
    print("\nExamples:")
    print("  # Default (recommended): gradient init + densification:")
    print("  ./run.sh image.jpg")
    print("  # Per-pixel mode (best quality):")
    print("  ./run.sh image.jpg --init-per-pixel --iters 600")
    print("  # Gradient-weighted init + densification (adaptive):")
    print("  ./run.sh image.jpg --sites 256 --init-gradient --densify --max-sites 8192 --iters 1000")
    print("\nNote: Images larger than \(defaults.maxDim)x\(defaults.maxDim) are automatically downsampled")
}

func parseArguments(_ args: [String], defaults: Defaults) -> RunMode {
    if args.count <= 1 {
        printUsage(defaults: defaults)
        exit(1)
    }

    if args[1] == "--help" || args[1] == "-h" {
        printUsage(defaults: defaults)
        exit(0)
    }

    var training = makeTrainingOptions(defaults: defaults)
    var renderVoronoiFlag = false
    var mode: RunMode?
    var i = 1

    if args[i] == "--render" {
        if args.count < i + 2 {
            print("Error: --render requires a sites file path")
            exit(1)
        }
        let sitesPath = args[i + 1]
        mode = .render(makeRenderOptions(defaults: defaults, sitesPath: sitesPath))
        i += 2
    } else if args[i].hasSuffix("_sites.json") || args[i].hasSuffix("_sites.txt") {
        let sitesPath = args[i]
        mode = .render(makeRenderOptions(defaults: defaults, sitesPath: sitesPath))
        i += 1
    } else {
        training.targetPath = args[i]
        mode = .train(training)
        i += 1
    }

    while i < args.count {
        switch args[i] {
        case "--out":
            if i + 1 < args.count {
                if case .render(var opts) = mode {
                    opts.outputPath = args[i + 1]
                    mode = .render(opts)
                    i += 2
                } else {
                    print("Error: --out is only valid in render mode (use --out-dir for training)")
                    exit(1)
                }
            } else {
                print("Error: --out requires a file path")
                exit(1)
            }
        case "--render-cells", "--render-voronoi":
            renderVoronoiFlag = true
            i += 1
        case "--render-no-jfa":
            if case .render(var opts) = mode {
                opts.useJFA = false
                mode = .render(opts)
            }
            i += 1
        case "--render-jfa-rounds":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                if case .render(var opts) = mode {
                    opts.jfaRounds = max(1, val)
                    mode = .render(opts)
                }
                i += 2
            } else {
                print("Error: --render-jfa-rounds requires an integer argument")
                exit(1)
            }
        case "--render-cand-passes":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                if case .render(var opts) = mode {
                    opts.candidatePasses = max(0, val)
                    mode = .render(opts)
                }
                i += 2
            } else {
                print("Error: --render-cand-passes requires an integer argument")
                exit(1)
            }
        case "--render-cand-downscale":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                if case .render(var opts) = mode {
                    opts.candDownscale = max(1, val)
                    mode = .render(opts)
                }
                i += 2
            } else {
                print("Error: --render-cand-downscale requires an integer argument")
                exit(1)
            }
        case "--render-cand-radius-scale":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                if case .render(var opts) = mode {
                    opts.candRadiusScale = val
                    mode = .render(opts)
                }
                i += 2
            } else {
                print("Error: --render-cand-radius-scale requires a float argument")
                exit(1)
            }
        case "--render-cand-radius-probes":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                if case .render(var opts) = mode {
                    opts.candRadiusProbes = UInt32(max(0, val))
                    mode = .render(opts)
                }
                i += 2
            } else {
                print("Error: --render-cand-radius-probes requires an integer argument")
                exit(1)
            }
        case "--render-cand-inject":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                if case .render(var opts) = mode {
                    opts.candInjectCount = UInt32(max(0, val))
                    mode = .render(opts)
                }
                i += 2
            } else {
                print("Error: --render-cand-inject requires an integer argument")
                exit(1)
            }
        case "--render-hilbert-window":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                if case .render(var opts) = mode {
                    opts.candHilbertWindow = UInt32(max(0, val))
                    mode = .render(opts)
                }
                i += 2
            } else {
                print("Error: --render-hilbert-window requires an integer argument")
                exit(1)
            }
        case "--render-hilbert-probes":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                if case .render(var opts) = mode {
                    opts.candHilbertProbes = UInt32(max(0, val))
                    mode = .render(opts)
                }
                i += 2
            } else {
                print("Error: --render-hilbert-probes requires an integer argument")
                exit(1)
            }
        case "--render-target":
            if i + 1 < args.count {
                if case .render(var opts) = mode {
                    opts.renderTargetPath = args[i + 1]
                    mode = .render(opts)
                }
                i += 2
            } else {
                print("Error: --render-target requires a file path")
                exit(1)
            }
        case "--render-packed":
            if case .render(var opts) = mode {
                opts.usePackedInference = true
                mode = .render(opts)
                i += 1
            } else {
                print("Error: --render-packed is only valid in render mode")
                exit(1)
            }
        case "--width":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                if case .render(var opts) = mode {
                    opts.widthOverride = val
                    mode = .render(opts)
                }
                i += 2
            } else {
                print("Error: --width requires an integer argument")
                exit(1)
            }
        case "--height":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                if case .render(var opts) = mode {
                    opts.heightOverride = val
                    mode = .render(opts)
                }
                i += 2
            } else {
                print("Error: --height requires an integer argument")
                exit(1)
            }
        case "--show-viewer":
            training.showViewer = true
            mode = .train(training)
            i += 1
        case "--viewer-freq":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.viewerFreq = max(1, val)
                mode = .train(training)
                i += 2
            } else {
                print("Error: --viewer-freq requires an integer argument")
                exit(1)
            }
        case "--log-freq":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.logFreq = max(1, val)
                mode = .train(training)
                i += 2
            } else {
                print("Error: --log-freq requires an integer argument")
                exit(1)
            }
        case "--out-dir":
            if i + 1 < args.count {
                training.outputDir = args[i + 1]
                mode = .train(training)
                i += 2
            } else {
                print("Error: --out-dir requires a directory path")
                exit(1)
            }
        case "--mask":
            if i + 1 < args.count {
                if case .render = mode {
                    print("Error: --mask is only valid in training mode")
                    exit(1)
                }
                training.maskPath = args[i + 1]
                mode = .train(training)
                i += 2
            } else {
                print("Error: --mask requires a file path")
                exit(1)
            }
        case "--export-neighbors":
            training.exportNeighbors = true
            mode = .train(training)
            i += 1
        case "--packed-psnr":
            training.packedPSNR = true
            mode = .train(training)
            i += 1
        case "--trace-frame":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.traceFrame = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --trace-frame requires an integer argument")
                exit(1)
            }
        case "--sites":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.nSites = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --sites requires an integer argument")
                exit(1)
            }
        case "--iters":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.iterations = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --iters requires an integer argument")
                exit(1)
            }
        case "--target-bpp":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.targetBpp = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --target-bpp requires a float argument")
                exit(1)
            }
        case "--prune-start":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.pruneStartIter = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --prune-start requires an integer argument")
                exit(1)
            }
        case "--prune-freq":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.pruneFreq = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --prune-freq requires an integer argument")
                exit(1)
            }
        case "--prune-end":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.pruneEndIter = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --prune-end requires an integer argument")
                exit(1)
            }
        case "--prune-percentile":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.prunePercentile = max(0.0, min(1.0, val))
                mode = .train(training)
                i += 2
            } else {
                print("Error: --prune-percentile requires a float argument (0.0-1.0)")
                exit(1)
            }
        case "--prune-during-densify":
            training.pruneDuringDensify = true
            mode = .train(training)
            i += 1
        case "--init-per-pixel":
            training.initMode = .perPixel
            mode = .train(training)
            i += 1
        case "--init-gradient":
            training.initMode = .gradientWeighted
            mode = .train(training)
            i += 1
        case "--init-gradient-alpha":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.initGradientAlpha = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --init-gradient-alpha requires a float argument")
                exit(1)
            }
        case "--init-from-sites":
            if i + 1 < args.count {
                training.initMode = .fromSites(args[i + 1])
                mode = .train(training)
                i += 2
            } else {
                print("Error: --init-from-sites requires a file path")
                exit(1)
            }
        case "--densify":
            training.densifyEnabled = true
            mode = .train(training)
            i += 1
        case "--densify-start":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.densifyStart = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --densify-start requires an integer argument")
                exit(1)
            }
        case "--densify-freq":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.densifyFreq = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --densify-freq requires an integer argument")
                exit(1)
            }
        case "--densify-end":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.densifyEnd = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --densify-end requires an integer argument")
                exit(1)
            }
        case "--densify-percentile":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.densifyPercentile = max(0.0, min(1.0, val))
                mode = .train(training)
                i += 2
            } else {
                print("Error: --densify-percentile requires a float argument (0.0-1.0)")
                exit(1)
            }
        case "--densify-score-alpha":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.densifyScoreAlpha = max(0.0, min(1.0, val))
                mode = .train(training)
                i += 2
            } else {
                print("Error: --densify-score-alpha requires a float argument (0.0-1.0)")
                exit(1)
            }
        case "--max-sites":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.maxSites = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --max-sites requires an integer argument")
                exit(1)
            }
        case "--cand-freq":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.candUpdateFreq = max(1, val)
                mode = .train(training)
                i += 2
            } else {
                print("Error: --cand-freq requires an integer argument")
                exit(1)
            }
        case "--cand-passes":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.candUpdatePasses = max(1, val)
                mode = .train(training)
                i += 2
            } else {
                print("Error: --cand-passes requires an integer argument")
                exit(1)
            }
        case "--cand-downscale":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.candDownscale = max(1, val)
                mode = .train(training)
                i += 2
            } else {
                print("Error: --cand-downscale requires an integer argument")
                exit(1)
            }
        case "--cand-radius-scale":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.candRadiusScale = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --cand-radius-scale requires a float argument")
                exit(1)
            }
        case "--cand-radius-probes":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.candRadiusProbes = UInt32(max(0, val))
                mode = .train(training)
                i += 2
            } else {
                print("Error: --cand-radius-probes requires an integer argument")
                exit(1)
            }
        case "--cand-inject":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.candInjectCount = UInt32(max(0, val))
                mode = .train(training)
                i += 2
            } else {
                print("Error: --cand-inject requires an integer argument")
                exit(1)
            }
        case "--cand-hilbert-window":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.candHilbertWindow = UInt32(max(0, val))
                mode = .train(training)
                i += 2
            } else {
                print("Error: --cand-hilbert-window requires an integer argument")
                exit(1)
            }
        case "--cand-hilbert-probes":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.candHilbertProbes = UInt32(max(0, val))
                mode = .train(training)
                i += 2
            } else {
                print("Error: --cand-hilbert-probes requires an integer argument")
                exit(1)
            }
        case "--export-cand-passes":
            if i + 1 < args.count, let val = Int(args[i + 1]) {
                training.exportCandPasses = max(0, val)
                mode = .train(training)
                i += 2
            } else {
                print("Error: --export-cand-passes requires an integer argument")
                exit(1)
            }
        case "--lr":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.lrScale = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --lr requires a float argument")
                exit(1)
            }
        case "--lr-tau":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.lrTauBase = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --lr-tau requires a float argument")
                exit(1)
            }
        case "--lr-radius":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.lrRadiusBase = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --lr-radius requires a float argument")
                exit(1)
            }
        case "--lr-dir":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.lrDirBase = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --lr-dir requires a float argument")
                exit(1)
            }
        case "--lr-aniso":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.lrAnisoBase = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --lr-aniso requires a float argument")
                exit(1)
            }
        case "--init-tau":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.initLogTau = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --init-tau requires a float argument")
                exit(1)
            }
        case "--ssim-weight":
            if i + 1 < args.count, let val = Float(args[i + 1]) {
                training.ssimWeight = val
                mode = .train(training)
                i += 2
            } else {
                print("Error: --ssim-weight requires a float argument")
                exit(1)
            }
        case "--ssim":
            training.ssimMetric = true
            mode = .train(training)
            i += 1
        default:
            print("Unknown option: \(args[i])")
            printUsage(defaults: defaults)
            exit(1)
        }
    }

    switch mode {
    case .render(var opts):
        if renderVoronoiFlag {
            let base = (opts.sitesPath as NSString).deletingPathExtension
            opts.outputVoronoiPath = base + "_cells.png"
        }
        return .render(opts)
    case .train(let opts):
        return .train(opts)
    case .none:
        printUsage(defaults: defaults)
        exit(1)
    }
}
