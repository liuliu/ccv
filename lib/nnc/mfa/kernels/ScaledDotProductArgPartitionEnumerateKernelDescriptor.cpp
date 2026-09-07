#include "ScaledDotProductArgPartitionEnumerateKernelDescriptor.hpp"

bool ScaledDotProductArgPartitionEnumerateKernelDescriptor::operator==(const ScaledDotProductArgPartitionEnumerateKernelDescriptor& rhs) const {
  return loadM == rhs.loadM;
}

std::size_t std::hash<ScaledDotProductArgPartitionEnumerateKernelDescriptor>::operator()(const ScaledDotProductArgPartitionEnumerateKernelDescriptor& hash) const noexcept {
  return hash.loadM ? 1 : 0;
}
