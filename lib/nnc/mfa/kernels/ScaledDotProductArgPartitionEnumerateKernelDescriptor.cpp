#include "ScaledDotProductArgPartitionEnumerateKernelDescriptor.hpp"

bool ScaledDotProductArgPartitionEnumerateKernelDescriptor::operator==(const ScaledDotProductArgPartitionEnumerateKernelDescriptor& rhs) const {
  (void)rhs;
  return true;
}

std::size_t std::hash<ScaledDotProductArgPartitionEnumerateKernelDescriptor>::operator()(const ScaledDotProductArgPartitionEnumerateKernelDescriptor& hash) const noexcept {
  (void)hash;
  return 0;
}
