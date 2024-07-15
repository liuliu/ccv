#ifndef GEMMShaderCache_hpp
#define GEMMShaderCache_hpp

#include <unordered_map>

struct GEMMShaderCache {
  static std::unordered_map<int, int> map;
  
  static int fetchKernel(int descriptor);
};
#endif /* GEMMShaderCache_hpp */
