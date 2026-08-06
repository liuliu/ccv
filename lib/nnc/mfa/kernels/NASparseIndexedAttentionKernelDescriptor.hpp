#ifndef MFA_NASPARSEINDEXEDATTENTIONKERNELDESCRIPTOR_HPP_
#define MFA_NASPARSEINDEXEDATTENTIONKERNELDESCRIPTOR_HPP_

#include "GEMMOperandPrecision.hpp"

enum class NASparseIndexedAttentionVariant : uint32_t {
  Threadgroup16 = 0,
  Threadgroup24 = 1,
  Threadgroup64 = 3,
  Threadgroup64D128 = 4,
};

struct NASparseIndexedAttentionKernelDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP16;
  bool attentionSinks = false;
  bool denseOnly = false;
  bool loadRows = false;
  NASparseIndexedAttentionVariant variant = NASparseIndexedAttentionVariant::Threadgroup16;

  bool operator==(const NASparseIndexedAttentionKernelDescriptor& rhs) const;
};

template<>
struct std::hash<NASparseIndexedAttentionKernelDescriptor>
{
  std::size_t operator()(const NASparseIndexedAttentionKernelDescriptor& hash) const noexcept;
};

#endif
