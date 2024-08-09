#include "GEMMDescriptor.hpp"
#include "../ccv_nnc_mfa_hash.hpp"

// MARK: - Hash Conformance

GEMMKey::GEMMKey(GEMMDescriptor descriptor) {
  batchDimension = descriptor.batchDimension;
  matrixDimensions = descriptor.matrixDimensions.value_or
  (simd::uint3(UINT32_MAX));
  
  if (descriptor.memoryPrecisions.has_value()) {
    auto precisions = descriptor.memoryPrecisions.value();
    memoryPrecisions = simd::ushort3 {
      precisions.A.value,
      precisions.B.value,
      precisions.C.value,
    };
  } else {
    memoryPrecisions = simd::ushort3(UINT16_MAX);
  }
  transposeState = descriptor.transposeState.value_or
  (simd::uchar3(UINT8_MAX));
  useBias = descriptor.useBias.value_or(UINT8_MAX);
}

bool GEMMKey::operator==(const GEMMKey& rhs) const {
  return
  (batchDimension == rhs.batchDimension) &&
  simd_all(matrixDimensions == rhs.matrixDimensions) &&
  simd_all(memoryPrecisions == rhs.memoryPrecisions) &&
  simd_all(transposeState == rhs.transposeState) &&
  (useBias == rhs.useBias);
}

std::size_t std::hash<GEMMKey>::operator()(const GEMMKey& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_64(seed, hash.batchDimension);
  combine_32(seed, hash.matrixDimensions[0]);
  combine_32(seed, hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  combine_64(seed, pack_64(simd_make_ushort4(hash.memoryPrecisions, 0)));
  combine_32(seed, pack_32(simd::uchar4 { hash.transposeState[0], hash.transposeState[1], hash.transposeState[2], hash.useBias }));
  return seed;
}
