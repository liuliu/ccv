#include "GEMMKernelDescriptor.hpp"
#include "../ccv_nnc_mfa_error.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

// MARK: - Hash Conformance

bool GEMMKernelDescriptor::operator==(const GEMMKernelDescriptor& rhs) const {
  return
  simd_all(blockDimensions == rhs.blockDimensions) &&
  memoryPrecisions == rhs.memoryPrecisions &&
  leadingBlockDimensions.has_value() == rhs.leadingBlockDimensions.has_value() &&
  simd_all(leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX)) == rhs.leadingBlockDimensions.value_or(simd::ushort3(UINT16_MAX))) &&
  (preferAsyncLoad == rhs.preferAsyncLoad) &&
  (preferAsyncStore == rhs.preferAsyncStore) &&
  registerPrecisions == rhs.registerPrecisions &&
  simd_all(splits == rhs.splits) &&
  simd_all(transposeState == rhs.transposeState) &&
  (useBias == rhs.useBias) &&
  (loadM == rhs.loadM);
}

std::size_t std::hash<GEMMKernelDescriptor>::operator()(const GEMMKernelDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, pack_64(simd_make_ushort4(hash.blockDimensions, 0)));
  combine_64(seed, pack_64(simd::ushort4 { hash.memoryPrecisions.A.value, hash.memoryPrecisions.B.value, hash.memoryPrecisions.C.value, hash.memoryPrecisions.bias.value }));
  if (hash.leadingBlockDimensions.has_value()) {
    combine_64(seed, pack_64(simd_make_ushort4(hash.leadingBlockDimensions.value())));
  }
  combine_32(seed, pack_32(simd::uchar4 { hash.preferAsyncLoad, hash.preferAsyncStore, 0, 0 }));
  combine_64(seed, pack_64(simd::ushort4 { hash.registerPrecisions.A.value, hash.registerPrecisions.B.value, hash.registerPrecisions.C.value, hash.registerPrecisions.bias.value }));
  combine_32(seed, pack_32(hash.splits));
  combine_32(seed, pack_32(simd::uchar4 { hash.transposeState[0], hash.transposeState[1], hash.transposeState[2], hash.useBias }));
  combine_32(seed, pack_32(simd::uchar4 { hash.loadM, 0, 0, 0 }));
  return seed;
}

// MARK: - Initializer

GEMMKernelDescriptor::GEMMKernelDescriptor(simd::ushort3 blockDimensions, GEMMOperandPrecisions memoryPrecisions, std::optional<simd::ushort3> leadingBlockDimensions, bool preferAsyncLoad, bool preferAsyncStore, GEMMOperandPrecisions registerPrecisions, simd::ushort2 splits, simd::uchar3 transposeState, bool useBias, bool loadM) noexcept {
  this->blockDimensions = blockDimensions;
  this->memoryPrecisions = memoryPrecisions;
  this->leadingBlockDimensions = leadingBlockDimensions;
  this->preferAsyncLoad = preferAsyncLoad;
  this->preferAsyncStore = preferAsyncStore;
  this->registerPrecisions = registerPrecisions;
  this->splits = splits;
  this->transposeState = transposeState;
  this->useBias = useBias;
  this->loadM = loadM;
}

std::pair<simd::ushort3, std::optional<simd::ushort3>> GEMMKernelDescriptor::getBlockDimensions(MTL::Device* const mtlDevice, const uint32_t coreCount, const simd::uint3 matrixDimensions, const int64_t batchDimension, const GEMMOperandPrecisions memoryPrecisions, const GEMMOperandPrecision registerPrecisionC, const simd::uchar3 transposeState) noexcept {
  if (mtlDevice->supportsFamily(MTL::GPUFamily(1009))) {
    if (!transposeState[0] && transposeState[1] && registerPrecisionC == GEMMOperandPrecision::FP32)
    {
      unsigned short paddedAK = (memoryPrecisions.A == GEMMOperandPrecision::FP32) ? 8 : 32;
      unsigned short paddedBK = (memoryPrecisions.B == GEMMOperandPrecision::FP32) ? 8 : 32;
      return std::make_pair(simd::ushort3 { 32, 32, 8 }, simd::ushort3 { paddedAK, paddedBK, 32 });
    } else {
      return std::make_pair(simd::ushort3 { 32, 32, 8 }, std::nullopt);
    }
  }
  
  // Find the actual number of threadgroups, with a large block size.
  auto ceilDivide =
  [=](uint32_t target, uint16_t granularity) -> uint32_t {
    return (target + uint32_t(granularity) - 1) / uint32_t(granularity);
  };
  int64_t actualGroups = 1;
  actualGroups *= ceilDivide(matrixDimensions[0], 48);
  actualGroups *= ceilDivide(matrixDimensions[1], 48);
  actualGroups *= batchDimension;
  
  // Does the kernel use 48x48x24xFP32 (9 KB) or 48x48x32xFP16/BF16 (6 KB)?
  bool useLargeAllocation = false;
  if (memoryPrecisions.A == GEMMOperandPrecision::FP32 ||
      memoryPrecisions.B == GEMMOperandPrecision::FP32 ||
      memoryPrecisions.C == GEMMOperandPrecision::FP32) {
    useLargeAllocation = true;
  }
  
  // Branch on whether the allocation is large / target occupancy is low.
  if (useLargeAllocation) {
    // Remove CoreCount based block size logic, per https://github.com/philipturner/ccv/commit/e8b0682b4344410eb43cdafb9a9c721ba7fdb726
    auto blockDimensions = simd::ushort3 { 48, 48, 24 };
      
    // This is verified to be optimal for:
    // - (memA, memB, memC) = (FP32, FP32, FP32)
    // - (memA, memB, memC) = (FP16, FP16, FP32)
    // - (memA, memB, memC) = (FP16, FP32, FP32)
    // - (memA, memB, memC) = (FP16, FP32, FP16)
    if (!transposeState[0] && !transposeState[1]) {
      return std::make_pair(blockDimensions, simd::ushort3 { 24, 48, 48 });
    } else if (!transposeState[0] && transposeState[1]) {
      if (memoryPrecisions.B == GEMMOperandPrecision::FP32) {
        return std::make_pair(blockDimensions, simd::ushort3 { 24, 28, 48 });
      } else {
        return std::make_pair(blockDimensions, simd::ushort3 { 24, 24, 48 });
      }
    } else if (transposeState[0] && !transposeState[1]) {
      if (memoryPrecisions.A == GEMMOperandPrecision::FP32) {
        return std::make_pair(blockDimensions, simd::ushort3 { 52, 48, 48 });
      } else {
        return std::make_pair(blockDimensions, simd::ushort3 { 56, 48, 48 });
      }
    } else {
      if (memoryPrecisions.A == GEMMOperandPrecision::FP32) {
        return std::make_pair(blockDimensions, simd::ushort3 { 52, 24, 48 });
      } else {
        return std::make_pair(blockDimensions, simd::ushort3 { 56, 24, 48 });
      }
    }
  } else {
    return std::make_pair(simd::ushort3 { 48, 48, 32 }, std::nullopt);
  }
}
