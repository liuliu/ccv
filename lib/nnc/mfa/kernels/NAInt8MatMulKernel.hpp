#ifndef NAInt8MatMulKernel_hpp
#define NAInt8MatMulKernel_hpp

#include "NAInt8MatMulKernelDescriptor.hpp"
#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

class CodeWriter;

struct NAInt8MatMulKernel {
  NS::SharedPtr<MTL::Library> library;
  NS::SharedPtr<MTL::Library> registerLibrary;
  std::string source;

  simd::ushort3 blockDimensions;
  uint16_t executionSIMDGroups;
  GEMMOperandPrecision ioPrecision;
  bool useRegisterOperands;
  bool useBias;
  bool loadM;
  bool useLeadingDimensions;
  uint16_t activationQuantizeThreads;
  bool activationHadamard256;
  uint32_t groupM;
  uint32_t groupN;

  NAInt8MatMulKernel(NAInt8MatMulKernelDescriptor descriptor, MTL::Device *const device);

  uint16_t threadgroupSize(MTL::ComputePipelineState *const pipelineState) const noexcept;
  MTL::Size threadgroupsPerGrid(uint32_t M, uint32_t N, uint32_t batchDimension) const noexcept;

private:
  std::string createSource() const noexcept;
  std::string createRegisterSource() const noexcept;
};

struct NAInt8MatMulRegisterParams {
  int32_t M, N, K, lda, ldb, ldd, tiles_n, tiles_m;
  int64_t batch_stride_a, batch_stride_b, batch_stride_d;
  int32_t swizzle_log, gemm_k_iterations_aligned, batch_ndim;
};
static_assert(sizeof(NAInt8MatMulRegisterParams) == 72, "Metal GEMM parameter ABI");

#endif
