#include "GEMMShaderCache.hpp"

std::unordered_map<int, int> GEMMShaderCache::map = {};

int GEMMShaderCache::fetchKernel(int descriptor) {
  map[descriptor] = descriptor + 1;
  return map[descriptor];
}
