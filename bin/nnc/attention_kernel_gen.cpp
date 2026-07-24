extern "C" {
#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <sys/time.h>
#include <ctype.h>
}
#include "nnc/mfa/kernels/ShaderCache.hpp"
#include "nnc/mfa/kernels/AttentionDescriptor.hpp"
#include "nnc/mfa/kernels/AttentionKernelDescriptor.hpp"
#include "nnc/mfa/kernels/AttentionKernel.hpp"
#include "3rdparty/dsfmt/dSFMT.h"
#include <fstream>
#include <iostream>

static std::string cacheState(AttentionOperands<bool> cacheStates) noexcept {
	std::string output;
	for (uint16_t i = 0; i < AttentionOperand::size(); i++) {
		AttentionOperand operand((AttentionOperand::Value)i);
		std::optional<bool> value = cacheStates[operand];
		if (value.value_or(false)) {
			output += operand.name();
		}
	}
	return output;
}

static std::string toLower(std::string s) noexcept {
	std::transform(s.begin(), s.end(), s.begin(),
		[](unsigned char c){ return std::tolower(c); });
	return s;
}

static bool writeMetalFile(const std::string& directory, const std::string& file, const std::string& source) {
	std::ofstream output(directory + "/" + file + ".metal", std::ios::binary);
	if (!output) {
		std::cerr << "cannot open " << directory << "/" << file << ".metal" << std::endl;
		return false;
	}
	output << "#include <metal_stdlib>\n\n";
	output << (source.size() > 0 && source[0] == '\n' ? source.substr(1) : source);
	output << "\n";
	return (bool)output;
}

static std::string forwardVariantSuffix(bool isCausal, bool masked, bool isVarlen, bool attentionSinks, bool slidingWindow) {
	std::string output;
	if (isCausal || masked || isVarlen)
		output = std::string("_causal") + std::to_string(isCausal) + "_masked" + std::to_string(masked) + "_varlen" + std::to_string(isVarlen);
	if (slidingWindow)
		output += "_sliding1";
	if (attentionSinks)
		output += "_sinks1";
	return output;
}

static AttentionOperands<GEMMOperandPrecision> createMemoryPrecisions(AttentionKernelType type, bool lowPrecisionInputs, bool lowPrecisionIntermediates, bool isBF16) noexcept {
  AttentionOperands<GEMMOperandPrecision> memoryPrecisions;
  
  if (lowPrecisionInputs) {
    if (isBF16) {
      memoryPrecisions[AttentionOperand::Q] = GEMMOperandPrecision::BF16;
      memoryPrecisions[AttentionOperand::K] = GEMMOperandPrecision::BF16;
      memoryPrecisions[AttentionOperand::V] = GEMMOperandPrecision::BF16;
      memoryPrecisions[AttentionOperand::dO] = GEMMOperandPrecision::BF16;
    } else {
      memoryPrecisions[AttentionOperand::Q] = GEMMOperandPrecision::FP16;
      memoryPrecisions[AttentionOperand::K] = GEMMOperandPrecision::FP16;
      memoryPrecisions[AttentionOperand::V] = GEMMOperandPrecision::FP16;
      memoryPrecisions[AttentionOperand::dO] = GEMMOperandPrecision::FP16;
    }
  } else {
    memoryPrecisions[AttentionOperand::Q] = GEMMOperandPrecision::FP32;
    memoryPrecisions[AttentionOperand::K] = GEMMOperandPrecision::FP32;
    memoryPrecisions[AttentionOperand::V] = GEMMOperandPrecision::FP32;
    memoryPrecisions[AttentionOperand::dO] = GEMMOperandPrecision::FP32;
  }
  
  // Rounding error. In the test that reported these errors, the average
  // magnitude of any scalar was typically 1.0 to 10.0.
  //
  //   | FP32 | FP16/BF16   |
  // - | ---- | ----------- |
  // L | 2e-5 | 7e-3 (FP16) |
  // D | 2e-5 | 1e-1 (BF16) |
  //
  // Although the error in D is relatively large (1e-1), it does not impact
  // the error of the final outputs (O/dV/dK/dQ). For example, the error of
  // O/dV/dK/dQ is always 5e-2 in typical mixed precision workflows.
  // When D is demoted to BF16, the error of O/dV/dK/dQ is still 5e-2.
  //
  // Benchmarks suggest that keeping D in BF16, measurably improves ALU
  // utilization in the backward dK/dV pass. Samples were taken at every
  // whole number head dimension from 32 to 96 (e.g. 32, 33, 34, ...) and a
  // constant sequence length. The improvement was ~1% on both architectures.
  //
  // M1 Max, Sequence Dimension = 8192
  //
  // |         | BWD    | dQ     | dK/dV  |
  // | ------- | ------ | ------ | ------ |
  // | Average |  0.0%  | +0.1%  | +1.1%  |
  // | Minimum | -0.2%  | -1.2%  | -1.9%  |
  // | Median  |  0.0%  |  0.0%  | +1.4%  |
  // | Maximum | +0.2%  | +4.4%  | +5.6%  |
  //
  // M4, Sequence Dimension = 4096
  //
  // |         | BWD    | dQ     | dK/dV  |
  // | ------- | ------ | ------ | ------ |
  // | Average |  0.0%  |  0.0%  | +0.8%  |
  // | Minimum | -0.4%  | -0.2%  | -0.1%  |
  // | Median  |  0.0%  |  0.0%  | +0.8%  |
  // | Maximum |  0.3%  | +0.2%  | +3.0%  |
  //
  // To confirm this conclusion, a second study was performed on M1 Max at
  // large head dimensions (95 to 160). In addition, examining only the
  // subset of head dimensions that divide evenly by 8.
  //
  // M1 Max, dK/dV
  //
  // |         | 32 to 96 | 96 to 160 | 32 to 160 (div. 8) |
  // | ------- | -------- | --------- | ------------------ |
  // | Average | +1.1%    | +0.3%     | +0.6%              |
  // | Minimum | -1.9%    | -1.5%     | -1.5%              |
  // | Median  | +1.4%    | +0.2%     | +0.0%              |
  // | Maximum | +5.6%    | +2.5%     | +5.6%              |
  //
  // The improvement diminishes to ~0.3% at larger head dimensions. This
  // makes sense, as the overhead of one elementwise operation is amortized
  // over a larger dot product. The head dimension increased 2x and the
  // improvement shrunk 2-3x. For heads divisible by 8 (the target use case),
  // the improvement shrunk from major at small heads, to zero at large
  // ones. The cutoff aligns with the point where the GEMM loops cannot be
  // unrolled (head dimension vastly exceeds head block dimension).
  if (lowPrecisionIntermediates) {
    memoryPrecisions[AttentionOperand::L] = isBF16 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16;
    memoryPrecisions[AttentionOperand::D] = GEMMOperandPrecision::BF16;
  } else {
    memoryPrecisions[AttentionOperand::L] = GEMMOperandPrecision::FP32;
    memoryPrecisions[AttentionOperand::D] = GEMMOperandPrecision::FP32;
  }
  
  // Data for low precision outputs.
  //
  // Traversal block = 64, sequence length = 256, head size = 32
  // FP16 (O)          | cached: 3e-4 | paged: 5e-4   | 2x
  // BF16 (dV, dK, dQ) | cached: 4e-3 | paged: 1.3e-2 | 3x
  //
  // Traversal block = 64, sequence length = 1024, head size = 32
  // FP16 (O)          | cached: 2e-4 | paged: 5e-4   | 3x
  // BF16 (dV, dK, dQ) | cached: 4e-3 | paged: 1.5e-2 | 4x
  //
  // Traversal block = 64, sequence length = 4096, head size = 32
  // FP16 (O)          | cached: 1e-4 | paged: 5e-4   | 5x
  // BF16 (dV, dK, dQ) | cached: 1e-3 | paged: 4e-2   | 40x
  //
  // Traversal block = 64, sequence length = 8192, head size = 32
  // FP16 (O)          | cached: 4e-5 | paged: 5e-4   | 13x
  // BF16 (dV, dK, dQ) | cached: 1e-3 | paged: 4e-2   | 40x
  //
  // The benchmarks were taken in the case where O/dV/dK/dQ are spilled to
  // memory. Hence, the impact of writing them to memory scales with N^2.
  // M1 was slower when packing/unpacking BF16, while M4 was faster. This
  // was without utilizing the native hardware instructions for BF16 to
  // FP32 conversion on M4.
  //
  // M4 is faster when the accumulators are stored in registers, up to at
  // least head dimension 256. The cost of storing scales with N on that
  // architecture. BF16 would only bring harm on M1 and no change on M3 with
  // proper heuristics. I am forcing dV/dK/dQ to be stored in RAM as FP32,
  // based on performance alone (although it does help the rounding error).
  //
  // Clients can issue a subsequent kernel that casts the FP32 scalars to
  // BF16, within a smaller memory allocation. Then, deallocate the FP32
  // allocation. The overall training process will not be any slower than
  // if MFA outputted BF16 into the final buffer.
  //
  // ## Update
  //
  // Paging O as FP16 was found to be slower on M1. Like with BF16, the M3
  // generation was faster. Writing O directly to FP16 is a very
  // important use case: attention inference. Small head dimensions fit
  // inside the registers and don't convert FP16 -> FP32 -> FP16 every loop
  // iteration. They only convert once at the end of the kernel. It is the
  // supermassive head dimensions that require register spilling, and
  // therefore an FP32 memory allocation for O.
  //
  // I don't know the best way to resolve this. It seems like something the
  // client should deal with. Therefore, the MFA reference implementation
  // will always write O as FP32 in memory. This choice simplifies
  // everything, just like the choice to always store log-sum-exp during the
  // forward pass. It also removes the concern of rounding error from
  if (type.value != AttentionKernelType::forward && lowPrecisionInputs) {
    memoryPrecisions[AttentionOperand::O] = isBF16 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16;
  } else {
    memoryPrecisions[AttentionOperand::O] = GEMMOperandPrecision::FP32;
  }
  memoryPrecisions[AttentionOperand::dV] = GEMMOperandPrecision::FP32;
  memoryPrecisions[AttentionOperand::dK] = GEMMOperandPrecision::FP32;
  memoryPrecisions[AttentionOperand::dQ] = GEMMOperandPrecision::FP32;
  
  return memoryPrecisions;
}

AttentionOperands<GEMMOperandPrecision> createRegisterPrecisions(AttentionKernelType type, bool lowPrecisionInputs, bool lowPrecisionIntermediates, bool isBF16, bool family1009) noexcept {
  AttentionOperands<GEMMOperandPrecision> registerPrecisions;
  
  // Query whether the hardware fuses the promotion of BF16 to FP32 with
  // the FMA assembly instruction.
  const bool hasNativeBF16Casting = family1009;
  
  // Inputs have the same register precision across kernels.
  if (lowPrecisionInputs) {
    if (isBF16) {
      registerPrecisions[AttentionOperand::Q] = GEMMOperandPrecision::BF16;
      registerPrecisions[AttentionOperand::K] = GEMMOperandPrecision::BF16;
      registerPrecisions[AttentionOperand::V] = GEMMOperandPrecision::BF16;
      registerPrecisions[AttentionOperand::dO] = GEMMOperandPrecision::BF16;
    } else {
      registerPrecisions[AttentionOperand::Q] = GEMMOperandPrecision::FP16;
      registerPrecisions[AttentionOperand::K] = GEMMOperandPrecision::FP16;
      registerPrecisions[AttentionOperand::V] = GEMMOperandPrecision::FP16;
      registerPrecisions[AttentionOperand::dO] = GEMMOperandPrecision::FP16;
    }
  } else {
    registerPrecisions[AttentionOperand::Q] = GEMMOperandPrecision::FP32;
    registerPrecisions[AttentionOperand::K] = GEMMOperandPrecision::FP32;
    registerPrecisions[AttentionOperand::V] = GEMMOperandPrecision::FP32;
    registerPrecisions[AttentionOperand::dO] = GEMMOperandPrecision::FP32;
  }
  
  // The register precision of L/D only counts for backward key-value.
  if (lowPrecisionIntermediates) {
    registerPrecisions[AttentionOperand::L] = isBF16 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16;
    registerPrecisions[AttentionOperand::D] = hasNativeBF16Casting ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP32;
  } else {
    registerPrecisions[AttentionOperand::L] = GEMMOperandPrecision::FP32;
    registerPrecisions[AttentionOperand::D] = GEMMOperandPrecision::FP32;
  }
  
  // The register precision for the attention matrix.
  if (lowPrecisionIntermediates) {
    // There is a special FP16xFP16->FP16 instruction that reaches peak ALU
    // throughput. S = Q * K is the only place where it can be employed
    // in attention kernels.
    //
    // S = Q * K is the most often recomputed intermediate (3 out of 9 GEMMs,
    // 2 out of 3 unnecessary GEMMs). If we optimize this, the impact on
    // performance will be greater than for any other multiplication.
    //
    // Accumulating S in FP16 increased the rounding error tenfold in one
    // experiment (5e-3 to 5e-2). For reference, the average magnitude of any
    // scalar was 1.0 to 10.0.
    //
    // FP16 (Q, K)    | 5e-3
    // FP16 (Q, K, S) | 5e-2
    // FP16 (P)       | 2.7e-3
    // BF16 (dS)      | 8e-3
    registerPrecisions[AttentionOperand::S] = lowPrecisionInputs ? GEMMOperandPrecision::FP16 : GEMMOperandPrecision::FP32;
    registerPrecisions[AttentionOperand::P] = GEMMOperandPrecision::FP16;
    registerPrecisions[AttentionOperand::dP] = GEMMOperandPrecision::FP32;
    registerPrecisions[AttentionOperand::dS] = hasNativeBF16Casting ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP32;
  } else {
    registerPrecisions[AttentionOperand::S] = GEMMOperandPrecision::FP32;
    registerPrecisions[AttentionOperand::P] = GEMMOperandPrecision::FP32;
    registerPrecisions[AttentionOperand::dP] = GEMMOperandPrecision::FP32;
    registerPrecisions[AttentionOperand::dS] = GEMMOperandPrecision::FP32;
  }
  
  // All of the outputs are accumulated in FP32.
  if (type.value != AttentionKernelType::forward && lowPrecisionInputs) {
    registerPrecisions[AttentionOperand::O] = isBF16 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16;
  } else {
    registerPrecisions[AttentionOperand::O] = GEMMOperandPrecision::FP32;
  }
  registerPrecisions[AttentionOperand::dV] = GEMMOperandPrecision::FP32;
  registerPrecisions[AttentionOperand::dK] = GEMMOperandPrecision::FP32;
  registerPrecisions[AttentionOperand::dQ] = GEMMOperandPrecision::FP32;
  
  return registerPrecisions;

}

// MARK: - AttentionDescriptor+Parameters

static AttentionParameterRow fetchRow(const std::vector<AttentionParameterRow>& table, unsigned short headDimension) noexcept {
  int matchedRowID = table.size() - 1;
  for (int i = 0; i < table.size(); i++) {
    if (headDimension <= table[i].maximumHeadDimension) {
      matchedRowID = i;
      break;
    }
  }
  return table[matchedRowID];
}

static std::vector<AttentionParameterRow> defaultParameters(bool family1009) noexcept {
  if (family1009) {
    return { AttentionParameterRow(0, 16, 128, 16, {}) };
  } else {
    return { AttentionParameterRow(0, 32, 80, 16, {}) };
  }
}

std::vector<AttentionParameterRow> forwardMixedParameters(bool family1009) noexcept {
  if (family1009) {
    return {
      AttentionParameterRow(32, 16, 128, 16, { AttentionOperand::Q, AttentionOperand::O }),
      AttentionParameterRow(96, 16, 128, 32, { AttentionOperand::Q, AttentionOperand::O }),
      AttentionParameterRow(160, 16, 128, 32, { AttentionOperand::O }),
      AttentionParameterRow(224, 16, 128, 32, { AttentionOperand::Q }),
      AttentionParameterRow(384, 16, 128, 32, {})
    };
  } else {
    return {
      AttentionParameterRow(96, 32, 128, 32, { AttentionOperand::Q, AttentionOperand::O }),
      AttentionParameterRow(128, 32, 128, 32, { AttentionOperand::Q }),
      AttentionParameterRow(384, 32, 128, 32, {})
    };
  }
}

std::vector<AttentionParameterRow> forwardParameters(bool family1009) noexcept {
  if (family1009) {
    return {
      AttentionParameterRow(8, 16, 128, 16, { AttentionOperand::Q, AttentionOperand::O }),
      AttentionParameterRow(16, 16, 64, 16, { AttentionOperand::Q, AttentionOperand::O }),
      AttentionParameterRow(48, 16, 32, 8, { AttentionOperand::Q, AttentionOperand::O }),
      AttentionParameterRow(192, 16, 64, 16, { AttentionOperand::O }),
      AttentionParameterRow(384, 16, 128, 16, {})
    };
  } else {
    return {
      AttentionParameterRow(24, 32, 64, 24, { AttentionOperand::Q, AttentionOperand::O }),
      AttentionParameterRow(32, 32, 64, 32, { AttentionOperand::O }),
      AttentionParameterRow(56, 32, 32, 56, { AttentionOperand::Q }),
      AttentionParameterRow(384, 32, 80, 16, {})
    };
  }
}

std::vector<AttentionParameterRow> backwardQueryMixedParameters(bool family1009) noexcept {
  if (family1009) {
    return {
      AttentionParameterRow(80, 16, 64, 8, { AttentionOperand::Q, AttentionOperand::dQ }),
      AttentionParameterRow(192, 16, 64, 32, { AttentionOperand::Q, AttentionOperand::dQ }),
      AttentionParameterRow(384, 16, 128, 32, {})
    };
  } else {
    return {
      AttentionParameterRow(32, 32, 64, 32, { AttentionOperand::Q, AttentionOperand::dQ }),
      AttentionParameterRow(96, 32, 64, 32, { AttentionOperand::dQ }),
      AttentionParameterRow(384, 32, 64, 32, {})
    };
  }
}

std::vector<AttentionParameterRow> backwardQueryParameters(bool family1009) noexcept {
  if (family1009) {
    return {
      AttentionParameterRow(16, 16, 64, 8, { AttentionOperand::Q, AttentionOperand::dO, AttentionOperand::dQ }),
      AttentionParameterRow(32, 16, 64, 16, { AttentionOperand::Q, AttentionOperand::dQ }),
      AttentionParameterRow(192, 16, 64, 32, { AttentionOperand::Q, AttentionOperand::dQ }),
      AttentionParameterRow(384, 16, 128, 16, {})
    };
  } else {
    return {
      AttentionParameterRow(16, 32, 64, 16, { AttentionOperand::Q, AttentionOperand::dQ }),
      AttentionParameterRow(32, 32, 64, 32, { AttentionOperand::dQ }),
      AttentionParameterRow(56, 32, 64, 24, { AttentionOperand::dQ }),
      AttentionParameterRow(384, 32, 80, 16, {})
    };
  }
}

std::vector<AttentionParameterRow> backwardKeyValueMixedParameters(bool family1009) noexcept {
  if (family1009) {
    return {
      AttentionParameterRow(56, 16, 64, 8, { AttentionOperand::K, AttentionOperand::V, AttentionOperand::dV, AttentionOperand::dK }),
      AttentionParameterRow(80, 16, 32, 16, { AttentionOperand::V, AttentionOperand::dV, AttentionOperand::dK }),
      AttentionParameterRow(144, 16, 128, 16, { AttentionOperand::dV, AttentionOperand::dK }),
      AttentionParameterRow(224, 16, 128, 16, { AttentionOperand::dV }),
      AttentionParameterRow(384, 16, 128, 32, {})
    };
  } else {
    return {
      AttentionParameterRow(16, 32, 64, 16, { AttentionOperand::V, AttentionOperand::dV, AttentionOperand::dK }),
      AttentionParameterRow(32, 32, 64, 32, { AttentionOperand::dV, AttentionOperand::dK }),
      AttentionParameterRow(56, 32, 80, 32, { AttentionOperand::dV }),
      AttentionParameterRow(96, 32, 64, 32, { AttentionOperand::dV }),
      AttentionParameterRow(384, 32, 64, 32, {})
    };
  }
}

std::vector<AttentionParameterRow> backwardKeyValueParameters(bool family1009) noexcept {
  if (family1009) {
    return {
      AttentionParameterRow(16, 16, 64, 8, { AttentionOperand::K, AttentionOperand::V, AttentionOperand::dV, AttentionOperand::dK }),
      AttentionParameterRow(32, 16, 32, 16, { AttentionOperand::K, AttentionOperand::V, AttentionOperand::dV, AttentionOperand::dK }),
      AttentionParameterRow(64, 16, 32, 16, { AttentionOperand::V, AttentionOperand::dV, AttentionOperand::dK }),
      AttentionParameterRow(128, 16, 128, 16, { AttentionOperand::dV, AttentionOperand::dK }),
      AttentionParameterRow(160, 16, 128, 16, { AttentionOperand::dV }),
      AttentionParameterRow(384, 16, 128, 16, {})
    };
  } else {
    return {
      AttentionParameterRow(16, 32, 32, 16, { AttentionOperand::V, AttentionOperand::dV, AttentionOperand::dK }),
      AttentionParameterRow(24, 32, 64, 24, { AttentionOperand::dV, AttentionOperand::dK }),
      AttentionParameterRow(56, 32, 80, 16, { AttentionOperand::dV }),
      AttentionParameterRow(384, 32, 80, 16, {})
    };
  }
}

static std::vector<AttentionParameterRow> parameterFile(AttentionKernelType type, bool lowPrecisionInputs, bool lowPrecisionIntermediates, bool family1009) noexcept {
  if (lowPrecisionInputs && lowPrecisionIntermediates) {
    switch (type.value) {
    case AttentionKernelType::forward: 
      return forwardMixedParameters(family1009);
    case AttentionKernelType::backwardQuery:
      return backwardQueryMixedParameters(family1009);
    case AttentionKernelType::backwardKeyValue:
      return backwardKeyValueMixedParameters(family1009);
    }
  } else {
    switch (type.value) {
    case AttentionKernelType::forward: 
      return forwardParameters(family1009);
    case AttentionKernelType::backwardQuery:
      return backwardQueryParameters(family1009);
    case AttentionKernelType::backwardKeyValue:
      return backwardKeyValueParameters(family1009);
    }
  }
  return defaultParameters(family1009);
}

static AttentionKernelDescriptor kernelDescriptor(AttentionKernelType type, bool lowPrecisionInputs, bool lowPrecisionIntermediates, bool isBF16, bool family1009, unsigned short headDimension, bool isCausal, bool masked, bool isVarlen, bool attentionSinks = false, bool slidingWindow = false) noexcept {
  std::vector table = parameterFile(type, lowPrecisionInputs, lowPrecisionIntermediates, family1009);
  auto row = fetchRow(table, headDimension);
  auto createBlockDimensions =
  [=]() -> simd::ushort3 {
    unsigned short parallelization = row.parallelization;
    unsigned short traversal = row.traversal;
    unsigned short originalHead = row.head;
    // Enforce the rule that head block dimension <= head dimension.
    unsigned short paddedHeadDimension = (headDimension + 7) / 8 * 8;
    unsigned short revisedHead = std::min(originalHead, paddedHeadDimension);
 
    return simd::ushort3 { parallelization, traversal, revisedHead };
  };
  
  auto createCacheState =
  [=]() -> AttentionOperands<bool> {
    AttentionOperands<bool> output;
    switch (type.value) {
    case AttentionKernelType::forward:
      output[AttentionOperand::Q] = false;
      output[AttentionOperand::O] = false;
      break;
    case AttentionKernelType::backwardQuery:
      output[AttentionOperand::Q] = false;
      output[AttentionOperand::dO] = false;
      output[AttentionOperand::dQ] = false;
      break;
    case AttentionKernelType::backwardKeyValue:
      output[AttentionOperand::K] = false;
      output[AttentionOperand::V] = false;
      output[AttentionOperand::dV] = false;
      output[AttentionOperand::dK] = false;
      break;
    }
    auto cachedOperands = row.cachedOperands;
    for (const auto& operand : cachedOperands) {
      output[operand] = true;
    }
    return output;
  };
  
  auto createTransposeState =
  [=]() -> AttentionOperands<bool> {
    AttentionOperands<bool> output;
    output[AttentionOperand::Q] = false;
    output[AttentionOperand::K] = false;
    output[AttentionOperand::V] = false;
    output[AttentionOperand::O] = false;
 
    output[AttentionOperand::dO] = false;
    output[AttentionOperand::dV] = false;
    output[AttentionOperand::dK] = false;
    output[AttentionOperand::dQ] = false;
    return output;
  };

  auto createLeadingDimensions =
  [=]() -> AttentionOperands<bool> {
    AttentionOperands<bool> output;
    output[AttentionOperand::Q] = true;
    output[AttentionOperand::K] = true;
    output[AttentionOperand::V] = true;
    output[AttentionOperand::O] = true;
 
    output[AttentionOperand::dO] = true;
    output[AttentionOperand::dV] = true;
    output[AttentionOperand::dK] = true;
    output[AttentionOperand::dQ] = true;
    return output;
  };

  if (family1009) {
    return AttentionKernelDescriptor(createBlockDimensions(), createCacheState(), headDimension, createMemoryPrecisions(type, lowPrecisionInputs, lowPrecisionIntermediates, isBF16), true, false, createRegisterPrecisions(type, lowPrecisionInputs, lowPrecisionIntermediates, isBF16, family1009), createTransposeState(), createLeadingDimensions(), type, isCausal, masked, isVarlen, attentionSinks, slidingWindow);
  } else {
    return AttentionKernelDescriptor(createBlockDimensions(), createCacheState(), headDimension, createMemoryPrecisions(type, lowPrecisionInputs, lowPrecisionIntermediates, isBF16), false, true, createRegisterPrecisions(type, lowPrecisionInputs, lowPrecisionIntermediates, isBF16, family1009), createTransposeState(), createLeadingDimensions(), type, isCausal, masked, isVarlen, attentionSinks, slidingWindow);
  }
}

int main(int argc, char** argv)
{
	bool emitMetal = false;
	std::string metalDirectory;
	for (int i = 1; i < argc; i++) {
		std::string arg(argv[i]);
		if (arg == "--emit-metal") {
			if (++i >= argc) {
				std::cerr << "--emit-metal requires an output directory" << std::endl;
				return 1;
			}
			emitMetal = true;
			metalDirectory = argv[i];
		} else if (arg == "--emit-selector") {
			emitMetal = false;
		} else {
			std::cerr << "usage: " << argv[0] << " [--emit-selector] [--emit-metal <directory>]" << std::endl;
			return 1;
		}
	}
	if (emitMetal)
		std::cout.setstate(std::ios_base::failbit);
	ccv_nnc_init();
	{
		AttentionOperands<bool> transposeState;
		transposeState[AttentionOperand::Q] = false;
		transposeState[AttentionOperand::K] = false;
		transposeState[AttentionOperand::V] = false;
		transposeState[AttentionOperand::O] = false;
		transposeState[AttentionOperand::dQ] = false;
		transposeState[AttentionOperand::dK] = false;
		transposeState[AttentionOperand::dV] = false;
		transposeState[AttentionOperand::dO] = false;
		AttentionOperands<bool> leadingDimensions;
		leadingDimensions[AttentionOperand::Q] = true;
		leadingDimensions[AttentionOperand::K] = true;
		leadingDimensions[AttentionOperand::V] = true;
		leadingDimensions[AttentionOperand::O] = true;
		leadingDimensions[AttentionOperand::dQ] = true;
		leadingDimensions[AttentionOperand::dK] = true;
		leadingDimensions[AttentionOperand::dV] = true;
		leadingDimensions[AttentionOperand::dO] = true;
		int j, k, l, m;
		unsigned short headDimensions[] = { 40, 64, 80, 128, 160, 256 };
		bool lowPrecisionInputs = true;
		for (j = 0; j < 2; j++)
		{
			bool lowPrecisionIntermediates = j == 0 ? true : false;
			for (k = 0; k < 2; k++)
			{
				bool family1009 = k == 0 ? false : true;
				for (l = 0; l < 2; l++)
				{
					bool isBF16 = l == 0 ? false : true;
					if (isBF16 && lowPrecisionIntermediates) // These two are not compatible.
						continue;
					for (m = 0; m < 6; m++)
					{
						unsigned short headDimension = headDimensions[m];
						bool forwardIsCausalVariants[] = { false, true, false, true, false, true, true, true };
						bool forwardMaskedVariants[] = { false, false, true, true, false, false, false, true };
						bool forwardIsVarlenVariants[] = { false, false, false, false, true, true, false, false };
						bool forwardSlidingWindowVariants[] = { false, false, false, false, false, false, true, true };
						for (int n = 0; n < 8; n++)
						{
							bool isCausal = forwardIsCausalVariants[n];
							bool masked = forwardMaskedVariants[n];
							bool isVarlen = forwardIsVarlenVariants[n];
							bool slidingWindow = forwardSlidingWindowVariants[n];
							for (int s = 0; s < 2; s++)
							{
								bool attentionSinks = s == 1;
								AttentionKernelDescriptor kernelDesc = kernelDescriptor(AttentionKernelType::forward, lowPrecisionInputs, lowPrecisionIntermediates, isBF16, family1009, headDimension, isCausal, masked, isVarlen, attentionSinks, slidingWindow);
								std::string file = std::string("f_b") + std::to_string(kernelDesc.blockDimensions[0]) + "x" + std::to_string(kernelDesc.blockDimensions[1]) + "x" + std::to_string(kernelDesc.blockDimensions[2]) + "_h" + std::to_string(headDimension) + "_i" + std::to_string(lowPrecisionInputs) + "_t" + std::to_string(lowPrecisionIntermediates) + "_c" + cacheState(kernelDesc.cacheState) + "_b" + std::to_string(isBF16) + "_c" + std::to_string(kernelDesc.preferAsyncCache) + "_l" + std::to_string(kernelDesc.preferAsyncLoad) + forwardVariantSuffix(isCausal, masked, isVarlen, attentionSinks, slidingWindow);
								if (emitMetal) {
									AttentionKernel kernel(kernelDesc, nullptr);
									if (!writeMetalFile(metalDirectory, file, kernel.source))
										return 1;
								}
								/*
								std::cout << "///filename: " << file << std::endl;
								std::cout << "#include <metal_stdlib>" << std::endl;
								auto kernel = new AttentionKernel(kernelDesc, device.get());
								delete kernel;
								*/
								file = toLower(file);
								std::cout << R"(
  } else if (type.value == AttentionKernelType::forward &&
)";
								std::cout << "    blockDimensions[0] == " << kernelDesc.blockDimensions[0] << " && blockDimensions[1] == " << kernelDesc.blockDimensions[1] << " && blockDimensions[2] == " << kernelDesc.blockDimensions[2] << " &&" << std::endl;
								std::cout << "    headDimension == " << headDimension << " &&" << std::endl;
								std::cout << "    lowPrecisionIntermediates == " << lowPrecisionIntermediates << " && isBF16 == " << isBF16 << " &&" << std::endl;
								std::cout << "    isCausal == " << isCausal << " && masked == " << masked << " && isVarlen == " << isVarlen << " &&" << std::endl;
								std::cout << "    attentionSinks == " << attentionSinks << " && slidingWindow " << (slidingWindow ? "> 0" : "== 0") << " &&" << std::endl;
								std::cout << "    preferAsyncCache == " << kernelDesc.preferAsyncCache << " && preferAsyncLoad == " << kernelDesc.preferAsyncLoad << ") {" << std::endl;
								std::cout << "#if TARGET_OS_IPHONE" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_iphoneos_metallib, sizeof(" << file << "_iphoneos_metallib), NULL, 0);" << std::endl;
								std::cout << "#else" << std::endl;
								std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_macosx_metallib, sizeof(" << file << "_macosx_metallib), NULL, 0);" << std::endl;
								std::cout << "#endif";
								std::cout << R"(
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;
)";
							}
						}
						{
							AttentionKernelDescriptor kernelDesc = kernelDescriptor(AttentionKernelType::backwardQuery, lowPrecisionInputs, lowPrecisionIntermediates, isBF16, family1009, headDimension, false, false, false);
							std::string file = std::string("bq_b") + std::to_string(kernelDesc.blockDimensions[0]) + "x" + std::to_string(kernelDesc.blockDimensions[1]) + "x" + std::to_string(kernelDesc.blockDimensions[2]) + "_h" + std::to_string(headDimension) + "_i" + std::to_string(lowPrecisionInputs) + "_t" + std::to_string(lowPrecisionIntermediates) + "_c" + cacheState(kernelDesc.cacheState) + "_b" + std::to_string(isBF16) + "_c" + std::to_string(kernelDesc.preferAsyncCache) + "_l" + std::to_string(kernelDesc.preferAsyncLoad);
							if (emitMetal) {
								AttentionKernel kernel(kernelDesc, nullptr);
								if (!writeMetalFile(metalDirectory, file, kernel.source))
									return 1;
							}
							/*
							std::cout << "///filename: " << file << std::endl;
							std::cout << "#include <metal_stdlib>" << std::endl;
							auto kernel = new AttentionKernel(kernelDesc, device.get());
							delete kernel;
							*/
							file = toLower(file);
							std::cout << R"(
  } else if (type.value == AttentionKernelType::backwardQuery &&
)";
							std::cout << "    blockDimensions[0] == " << kernelDesc.blockDimensions[0] << " && blockDimensions[1] == " << kernelDesc.blockDimensions[1] << " && blockDimensions[2] == " << kernelDesc.blockDimensions[2] << " &&" << std::endl;
							std::cout << "    headDimension == " << headDimension << " &&" << std::endl;
							std::cout << "    lowPrecisionIntermediates == " << lowPrecisionIntermediates << " && isBF16 == " << isBF16 << " &&" << std::endl;
							std::cout << "    isCausal == 0 && masked == 0 && isVarlen == 0 &&" << std::endl;
							std::cout << "    attentionSinks == 0 && slidingWindow == 0 &&" << std::endl;
							std::cout << "    preferAsyncCache == " << kernelDesc.preferAsyncCache << " && preferAsyncLoad == " << kernelDesc.preferAsyncLoad << ") {" << std::endl;
							std::cout << "#if TARGET_OS_IPHONE" << std::endl;
							std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_iphoneos_metallib, sizeof(" << file << "_iphoneos_metallib), NULL, 0);" << std::endl;
							std::cout << "#else" << std::endl;
							std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_macosx_metallib, sizeof(" << file << "_macosx_metallib), NULL, 0);" << std::endl;
							std::cout << "#endif";
							std::cout << R"(
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;
)";
						}
						{
							AttentionKernelDescriptor kernelDesc = kernelDescriptor(AttentionKernelType::backwardKeyValue, lowPrecisionInputs, lowPrecisionIntermediates, isBF16, family1009, headDimension, false, false, false);
							std::string file = std::string("bkv_b") + std::to_string(kernelDesc.blockDimensions[0]) + "x" + std::to_string(kernelDesc.blockDimensions[1]) + "x" + std::to_string(kernelDesc.blockDimensions[2]) + "_h" + std::to_string(headDimension) + "_i" + std::to_string(lowPrecisionInputs) + "_t" + std::to_string(lowPrecisionIntermediates) + "_c" + cacheState(kernelDesc.cacheState) + "_b" + std::to_string(isBF16) + "_c" + std::to_string(kernelDesc.preferAsyncCache) + "_l" + std::to_string(kernelDesc.preferAsyncLoad);
							if (emitMetal) {
								AttentionKernel kernel(kernelDesc, nullptr);
								if (!writeMetalFile(metalDirectory, file, kernel.source))
									return 1;
							}
							/*
							std::cout << "///filename: " << file << std::endl;
							std::cout << "#include <metal_stdlib>" << std::endl;
							auto kernel = new AttentionKernel(kernelDesc, device.get());
							delete kernel;
							*/
							file = toLower(file);
							std::cout << R"(
  } else if (type.value == AttentionKernelType::backwardKeyValue &&
)";
							std::cout << "    blockDimensions[0] == " << kernelDesc.blockDimensions[0] << " && blockDimensions[1] == " << kernelDesc.blockDimensions[1] << " && blockDimensions[2] == " << kernelDesc.blockDimensions[2] << " &&" << std::endl;
							std::cout << "    headDimension == " << headDimension << " &&" << std::endl;
							std::cout << "    lowPrecisionIntermediates == " << lowPrecisionIntermediates << " && isBF16 == " << isBF16 << " &&" << std::endl;
							std::cout << "    isCausal == 0 && masked == 0 && isVarlen == 0 &&" << std::endl;
							std::cout << "    attentionSinks == 0 && slidingWindow == 0 &&" << std::endl;
							std::cout << "    preferAsyncCache == " << kernelDesc.preferAsyncCache << " && preferAsyncLoad == " << kernelDesc.preferAsyncLoad << ") {" << std::endl;
							std::cout << "#if TARGET_OS_IPHONE" << std::endl;
							std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_iphoneos_metallib, sizeof(" << file << "_iphoneos_metallib), NULL, 0);" << std::endl;
							std::cout << "#else" << std::endl;
							std::cout << "    dispatch_data_t data = dispatch_data_create(" << file << "_macosx_metallib, sizeof(" << file << "_macosx_metallib), NULL, 0);" << std::endl;
							std::cout << "#endif";
							std::cout << R"(
    auto library = device->newLibrary(data, error);
    dispatch_release(data);
    return library;
)";
						}
					}
				}
			}
		}
	}
	return 0;
}
