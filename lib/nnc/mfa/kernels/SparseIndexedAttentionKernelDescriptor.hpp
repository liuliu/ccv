#ifndef MFA_SPARSEINDEXEDATTENTIONKERNELDESCRIPTOR_HPP_
#define MFA_SPARSEINDEXEDATTENTIONKERNELDESCRIPTOR_HPP_

#include "GEMMOperandPrecision.hpp"

struct SparseIndexedAttentionKernelDescriptor {
  GEMMOperandPrecision memoryPrecision = GEMMOperandPrecision::FP16;
  bool attentionSinks = false;
  bool loadRows = false;

  bool operator==(const SparseIndexedAttentionKernelDescriptor& rhs) const;
};

template<>
struct std::hash<SparseIndexedAttentionKernelDescriptor>
{
  std::size_t operator()(const SparseIndexedAttentionKernelDescriptor& hash) const noexcept;
};

#endif
