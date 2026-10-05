#ifndef NAInt8MatMulKernelDescriptor_hpp
#define NAInt8MatMulKernelDescriptor_hpp

#include "GEMMOperandPrecision.hpp"
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
#include <simd/simd.h>

struct NAInt8MatMulDescriptor;
struct DeviceProperties;

struct NAInt8MatMulKernelDescriptor {
  simd::ushort3 blockDimensions;
  uint16_t executionSIMDGroups;
  GEMMOperandPrecision ioPrecision;
  // Selects register fragment loading and MPP assembly adapted from MLX Steel's
  // gemm_nax.h (Copyright 2023-2025 Apple Inc., MIT). See ../3rdparty/mlx/README
  // for the pinned upstream source and ../3rdparty/mlx/LICENSE for its license.
  bool useRegisterOperands;
  bool useBias;
  bool loadM;
  bool useLeadingDimensions;
  uint16_t activationQuantizeThreads;
  bool activationHadamard256;
  uint32_t groupM;
  uint32_t groupN;

  NAInt8MatMulKernelDescriptor() = delete;
  // Choose the source configuration from execution requirements and device
  // properties. The shader generator consumes these choices without tuning.
  NAInt8MatMulKernelDescriptor(
      const NAInt8MatMulDescriptor& descriptor,
      const DeviceProperties& dprops) noexcept;
  NAInt8MatMulKernelDescriptor(
      simd::ushort3 blockDimensions,
      uint16_t executionSIMDGroups,
      GEMMOperandPrecision ioPrecision,
      bool useBias,
      bool loadM,
      uint16_t activationQuantizeThreads,
      uint32_t groupM,
      uint32_t groupN,
      bool useLeadingDimensions = false,
      bool activationHadamard256 = false,
      bool useRegisterOperands = false) noexcept;

  bool operator==(const NAInt8MatMulKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NAInt8MatMulKernelDescriptor>
{
  std::size_t operator()(const NAInt8MatMulKernelDescriptor& hash) const noexcept;
};

#endif
