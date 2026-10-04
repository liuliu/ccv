#ifndef NARegisterMatMulDescriptor_hpp
#define NARegisterMatMulDescriptor_hpp

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include "DeviceProperties.hpp"
#include "PipelineValue.hpp"

// Register fragments: FP16 inputs / FP32 accumulation or INT8 / INT32.
// FP16 supports both 64x128 and 128x64 output tiles; INT8 uses 64x128.
// Matrix lengths and strides are runtime arguments, never kernel cache state.
struct NARegisterMatMulKernelDescriptor {
  bool quantized = false;
  bool wideM = false;
  bool splitK = false;
  // Widen the rounded half epilogue, preserving GEMM -> cast semantics.
  bool castOutputToFloat = false;
  bool operator==(const NARegisterMatMulKernelDescriptor& rhs) const {
    return quantized == rhs.quantized && wideM == rhs.wideM && splitK == rhs.splitK &&
        castOutputToFloat == rhs.castOutputToFloat;
  }
};

template<> struct std::hash<NARegisterMatMulKernelDescriptor> {
  size_t operator()(const NARegisterMatMulKernelDescriptor& value) const noexcept {
    return (value.castOutputToFloat << 3) | (value.splitK << 2) | (value.wideM << 1) | value.quantized;
  }
};

struct NARegisterMatMulKernel;

struct NARegisterMatMulDescriptor {
  bool alignedM;
  bool alignedN;
  bool alignedK;
  bool useBias;
  bool quantized = false;
  bool wideM = false;
  bool splitK = false;
  bool castOutputToFloat = false;

  bool operator==(const NARegisterMatMulDescriptor& rhs) const {
    return alignedM == rhs.alignedM && alignedN == rhs.alignedN &&
        alignedK == rhs.alignedK && useBias == rhs.useBias && quantized == rhs.quantized &&
        wideM == rhs.wideM && splitK == rhs.splitK && castOutputToFloat == rhs.castOutputToFloat;
  }

  std::pair<NARegisterMatMulKernelDescriptor, PipelineValue<NARegisterMatMulKernel>*> findKernel(
      MTL::Device* device, const DeviceProperties& dprops,
      NS::Array* binaryArchivesToRead, MTL::BinaryArchive* binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<NARegisterMatMulKernelDescriptor, std::unique_ptr<NARegisterMatMulKernel>>* libraryCache) const noexcept;
};

template<> struct std::hash<NARegisterMatMulDescriptor> {
  size_t operator()(const NARegisterMatMulDescriptor& value) const noexcept {
    return (value.castOutputToFloat << 7) | (value.splitK << 6) | (value.wideM << 5) | (value.quantized << 4) | (value.alignedM << 3) | (value.alignedN << 2) |
        (value.alignedK << 1) | value.useBias;
  }
};

struct NARegisterMatMulParams {
  int32_t M, N, K, lda, ldb, ldd, tiles_n, tiles_m;
  int64_t batch_stride_a, batch_stride_b, batch_stride_d;
  int32_t swizzle_log, gemm_k_iterations_aligned, batch_ndim;
};
static_assert(sizeof(NARegisterMatMulParams) == 72, "Metal GEMM parameter ABI");

struct NARegisterMatMulBiasParams {
  int32_t ldc, fdc;
  int64_t batch_stride_c;
  float alpha, beta;
};
static_assert(sizeof(NARegisterMatMulBiasParams) == 24, "Metal GEMM bias parameter ABI");

struct NARegisterMatMulSplitKParams {
  int32_t M, N, K, lda, ldb, ldc, tiles_n, tiles_m;
  int32_t partitions, partition_stride, partition_size, swizzle_log, iterations;
};
static_assert(sizeof(NARegisterMatMulSplitKParams) == 52, "Metal split-K parameter ABI");

#endif
