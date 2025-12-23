// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "ccv",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .tvOS(.v16),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "ccv",
            targets: ["C_swiftpm_ccv"]),
        .library(
            name: "nnc",
            targets: ["C_nnc"]),
        .library(
            name: "sfmt",
            targets: ["C_swiftpm_sfmt"]),
        .library(
            name: "lib_nnc_mps_compat",
            targets: ["lib_nnc_mps_compat"]),
    ],
    targets: [
        // SFMT - SIMD-oriented Fast Mersenne Twister
        .target(
            name: "C_sfmt",
            path: "lib/3rdparty/sfmt",
            exclude: [
                "SFMT-alti.h",  // Altivec (PowerPC) - not available on ARM
                "SFMT-sse2.h",  // SSE2 (x86) - not available on ARM
            ],
            sources: ["SFMT.c"],
            publicHeadersPath: ".",
            cSettings: [
                .headerSearchPath("."),
                .unsafeFlags(["-w"])
            ]
        ),

        // dSFMT - Double-precision SFMT (internal dependency)
        .target(
            name: "dSFMT",
            path: "lib/3rdparty/dsfmt",
            sources: ["dSFMT.c"],
            publicHeadersPath: ".",
            cSettings: [
                .unsafeFlags(["-w"])
            ]
        ),

        // KissFFT (internal dependency)
        .target(
            name: "kissfft",
            path: "lib/3rdparty/kissfft",
            publicHeadersPath: ".",
            cSettings: [
                .unsafeFlags(["-w"])
            ]
        ),

        // SipHash (internal dependency)
        .target(
            name: "siphash",
            path: "lib/3rdparty/siphash",
            sources: ["siphash24.c"],
            publicHeadersPath: ".",
            cSettings: [
                .unsafeFlags(["-w"])
            ]
        ),

        // Main CCV library
        .target(
            name: "C_swiftpm_ccv",
            dependencies: ["C_swiftpm_sfmt", "dSFMT", "kissfft", "siphash"],
            path: "lib",
            exclude: [
                "3rdparty",
                "nnc",
                "cuda",
                "inc",
                "io",
                "configure",
                "configure.ac",
                "makefile",
                "config.mk",
                "scheme.mk",
                ".ycm_extra_conf.py",
            ],
            publicHeadersPath: ".",
            cSettings: [
                .headerSearchPath("."),
                .define("HAVE_CBLAS"),
                .define("HAVE_PTHREAD"),
                .define("HAVE_ACCELERATE_FRAMEWORK"),
                .define("USE_DISPATCH"),
                .unsafeFlags(["-fblocks", "-w"])
            ],
            linkerSettings: [
                .linkedFramework("Accelerate")
            ]
        ),

        // NNC - Neural Network Collection
        .target(
            name: "C_nnc",
            dependencies: ["C_swiftpm_ccv", "C_sfmt", "dSFMT"],
            path: "lib/nnc",
            exclude: [
                // Exclude CUDA/GPU files (not needed for MPS)
                "gpu",
                // Exclude build artifacts and makefiles
                "mps",
                "makefile",
                "cmd/makefile",
                "cmd/config.mk",
                "mps/makefile",
                "mps/.ycm_extra_conf.py",
                "mfa/makefile",
                // Exclude CUDA command implementations
                "cmd/adam/gpu",
                "cmd/blas/gpu",
                "cmd/comm/gpu",
                "cmd/compare/gpu",
                "cmd/compression/gpu",
                "cmd/convolution/gpu",
                "cmd/dropout/gpu",
                "cmd/ew/gpu",
                "cmd/gelu/gpu",
                "cmd/index/gpu",
                "cmd/isnan/gpu",
                "cmd/lamb/gpu",
                "cmd/leaky_relu/gpu",
                "cmd/loss/gpu",
                "cmd/nms/gpu",
                "cmd/norm/gpu",
                "cmd/pad/gpu",
                "cmd/partition/gpu",
                "cmd/pool/gpu",
                "cmd/rand/gpu",
                "cmd/reduce/gpu",
                "cmd/relu/gpu",
                "cmd/rmsprop/gpu",
                "cmd/rnn/gpu",
                "cmd/roi/gpu",
                "cmd/scaled_dot_product_attention/gpu",
                "cmd/scatter_add/gpu",
                "cmd/sgd/gpu",
                "cmd/sigmoid/gpu",
                "cmd/sigmoid_loss/gpu",
                "cmd/softmax/gpu",
                "cmd/softmax_loss/gpu",
                "cmd/sort/gpu",
                "cmd/swish/gpu",
                "cmd/tanh/gpu",
                "cmd/unique_consecutive/gpu",
                "cmd/upsample/gpu",
                "cmd/util/gpu",
            ],
            publicHeadersPath: ".",
            cSettings: [
                .headerSearchPath("."),
                .headerSearchPath(".."),
                .headerSearchPath("cmd"),
                .headerSearchPath("mfa"),
                .headerSearchPath("mfa/3rdparty/metal-cpp"),
                .headerSearchPath("mfa/v2"),
                .define("HAVE_CBLAS"),
                .define("HAVE_PTHREAD"),
                .define("HAVE_ACCELERATE_FRAMEWORK"),
                .define("USE_DISPATCH"),
                .define("HAVE_MPS"),
                .unsafeFlags(["-fblocks", "-fno-objc-arc", "-w"])
            ],
            cxxSettings: [
                .headerSearchPath("."),
                .headerSearchPath(".."),
                .headerSearchPath("mfa"),
                .headerSearchPath("mfa/3rdparty/metal-cpp"),
                .headerSearchPath("mfa/v2"),
                .define("HAVE_MPS"),
                .unsafeFlags(["-std=c++17", "-w"])
            ],
            linkerSettings: [
                .linkedFramework("Accelerate"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("MetalPerformanceShadersGraph"),
                .linkedFramework("Foundation"),
            ]
        ),
        // MPS Compatibility layer for NNC
        .target(
            name: "lib_nnc_mps_compat",
            dependencies: ["C_nnc"],
            path: "lib/nnc/mps",
            exclude: [
                "makefile",
                ".ycm_extra_conf.py",
                ".dep.mk",
            ],
            sources: [
                "ccv_nnc_mps.m",
                "ccv_nnc_palettize.m",
            ],
            publicHeadersPath: ".",
            cSettings: [
                .headerSearchPath("."),
                .headerSearchPath(".."),
                .headerSearchPath("../.."),
                .headerSearchPath("../cmd"),
                .headerSearchPath("../mfa"),
                .headerSearchPath("../mfa/3rdparty/metal-cpp"),
                .define("HAVE_CBLAS"),
                .define("HAVE_PTHREAD"),
                .define("HAVE_ACCELERATE_FRAMEWORK"),
                .define("USE_DISPATCH"),
                .define("HAVE_MPS"),
                .unsafeFlags(["-fblocks", "-fno-objc-arc", "-w"])
            ],
            linkerSettings: [
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShaders"),
                .linkedFramework("MetalPerformanceShadersGraph"),
                .linkedFramework("Foundation"),
            ]
        ),
    ],
    cLanguageStandard: .gnu11,
    cxxLanguageStandard: .cxx17
)
