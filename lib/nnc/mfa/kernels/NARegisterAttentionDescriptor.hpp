#ifndef NARegisterAttentionDescriptor_hpp
#define NARegisterAttentionDescriptor_hpp

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include "DeviceProperties.hpp"
#include "PipelineValue.hpp"

// Only the head width changes generated source. Sequence lengths, strides and
// scale are runtime arguments, and alignment, causal masking and saved statistics are constants.
struct NARegisterAttentionKernelDescriptor {
  uint32_t D;

  bool operator==(const NARegisterAttentionKernelDescriptor& rhs) const {
    return D == rhs.D;
  }
};

template<> struct std::hash<NARegisterAttentionKernelDescriptor> {
  size_t operator()(const NARegisterAttentionKernelDescriptor& value) const noexcept {
    return value.D;
  }
};

struct NARegisterAttentionKernel;

struct NARegisterAttentionDescriptor {
  uint32_t D;
  bool alignedQ;
  bool alignedK;
  bool saveL;
  bool causal;

  bool operator==(const NARegisterAttentionDescriptor& rhs) const {
    return D == rhs.D && alignedQ == rhs.alignedQ &&
        alignedK == rhs.alignedK && saveL == rhs.saveL && causal == rhs.causal;
  }

  std::pair<NARegisterAttentionKernelDescriptor, PipelineValue<NARegisterAttentionKernel>*> findKernel(
      MTL::Device* device, const DeviceProperties& dprops,
      NS::Array* binaryArchivesToRead, MTL::BinaryArchive* binaryArchiveToWrite,
      const std::string& pathToWrite,
      std::unordered_map<NARegisterAttentionKernelDescriptor, std::unique_ptr<NARegisterAttentionKernel>>* libraryCache) const noexcept;
};

template<> struct std::hash<NARegisterAttentionDescriptor> {
  size_t operator()(const NARegisterAttentionDescriptor& value) const noexcept {
    return (size_t(value.D) << 4) | (value.alignedQ << 3) |
        (value.alignedK << 2) | (value.saveL << 1) | value.causal;
  }
};

// Matches the embedded Metal AttnParams ABI; strides count elements in SHD.
struct NARegisterAttentionParams {
  int32_t B, H, D, qL, kL, gqa_factor;
  float scale;
  int32_t NQ, NK, NQ_aligned, NK_aligned, qL_rem, kL_rem, qL_off;
  int64_t Q_strides[3], K_strides[3], V_strides[3], O_strides[3];
};
static_assert(sizeof(NARegisterAttentionParams) == 152, "Metal attention parameter ABI");

#endif
