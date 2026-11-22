#ifndef MFA_MESSAGEGATEDESCRIPTOR_HPP_
#define MFA_MESSAGEGATEDESCRIPTOR_HPP_

#include <functional>
#include <memory>
#include <simd/simd.h>
#include <string>
#include <unordered_map>
#include <utility>
#include "DeviceProperties.hpp"
#include "GEMMOperandPrecision.hpp"
#include "PipelineValue.hpp"

// The library holds the kernels of the few-rows mix (the normalizations and the down projection), of
// the normalization the down GEMM of many rows reads, and of the gate.
struct MessageGateKernelDescriptor {
  GEMMOperandPrecision memoryPrecision;
  GEMMOperandPrecision downPrecision;
  GEMMOperandPrecision weightPrecision;
  bool downQuantized;
  bool quantized;
  // The mix width sizes the gate's mix rows and unrolls the projections.
  uint32_t rank;
  // The rows a threadgroup of the few-rows mix takes, 1, 2, 4 or 8.
  uint32_t mixRows;
  bool operator==(const MessageGateKernelDescriptor& rhs) const {
    return memoryPrecision == rhs.memoryPrecision && downPrecision == rhs.downPrecision &&
      weightPrecision == rhs.weightPrecision && downQuantized == rhs.downQuantized &&
      quantized == rhs.quantized && rank == rhs.rank && mixRows == rhs.mixRows;
  }
};

template<>
struct std::hash<MessageGateKernelDescriptor> {
  std::size_t operator()(const MessageGateKernelDescriptor& descriptor) const noexcept {
    return std::hash<uint64_t>()((uint64_t)descriptor.memoryPrecision.value |
      ((uint64_t)descriptor.downPrecision.value << 4) | ((uint64_t)descriptor.weightPrecision.value << 8) |
      ((uint64_t)descriptor.downQuantized << 12) | ((uint64_t)descriptor.quantized << 13) |
      ((uint64_t)descriptor.mixRows << 16) | ((uint64_t)descriptor.rank << 32));
  }
};

struct MessageGateKernel;

struct MessageGateDescriptor {
  GEMMOperandPrecision memoryPrecision;
  GEMMOperandPrecision downPrecision;
  GEMMOperandPrecision weightPrecision;
  bool downQuantized;
  bool quantized;
  uint32_t rank;
  uint32_t mixRows;
  // Function constants of the pipeline. The rows are bound at encode time.
  uint32_t channels;
  float epsilon;
  // The pipeline's kernel.
  enum Kind : uint8_t { gate, mix, normalize } kind;

  bool operator==(const MessageGateDescriptor& rhs) const;
  std::pair<MessageGateKernelDescriptor, PipelineValue<MessageGateKernel>*> findKernel(MTL::Device* const device, const DeviceProperties& dprops, NS::Array* const binaryArchivesToRead, MTL::BinaryArchive* const binaryArchiveToWrite, const std::string& pathToWrite, std::unordered_map<MessageGateKernelDescriptor, std::unique_ptr<MessageGateKernel>>* const libraryCache) const noexcept;
};

template<>
struct std::hash<MessageGateDescriptor> {
  std::size_t operator()(const MessageGateDescriptor& descriptor) const noexcept;
};

#endif
