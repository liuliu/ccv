#include "AttentionDescriptor.hpp"
#include "AttentionKernelDescriptor.hpp"
// #include "AttentionKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool AttentionDescriptor::operator==(const AttentionDescriptor& rhs) const {
  return
  (lowPrecisionInputs == rhs.lowPrecisionInputs) &&
  (lowPrecisionIntermediates == rhs.lowPrecisionIntermediates) &&
  simd_all(matrixDimensions == rhs.matrixDimensions) &&
  simd_all(transposeState == rhs.transposeState);
}

std::size_t std::hash<AttentionDescriptor>::operator()(const AttentionDescriptor& hash) const noexcept {
  std::size_t seed = 0;
  using namespace ccv::nnc::mfa::hash;
  combine_32(seed, hash.matrixDimensions[0]);
  combine_32(seed, hash.matrixDimensions[1]);
  combine_32(seed, hash.matrixDimensions[2]);
  combine_32(seed, pack_32(simd::uchar4 { hash.transposeState[0], hash.transposeState[1], hash.transposeState[2], hash.transposeState[3] }));
  combine_32(seed, pack_32(simd::uchar4 { hash.lowPrecisionInputs, hash.lowPrecisionIntermediates, 0, 0 }));
  return seed;
}

/*
std::pair<AttentionKernelDescriptor, PipelineValue<AttentionKernel> *> AttentionDescriptor::findKernel(MTL::Device *const device, const DeviceProperties &dprops, std::unordered_map<AttentionKernelDescriptor, std::unique_ptr<AttentionKernel>> *const libraryCache) const noexcept {
}
*/
