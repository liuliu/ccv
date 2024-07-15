#ifndef GUARD_ccv_nnc_mfa_hash_hpp
#define GUARD_ccv_nnc_mfa_hash_hpp

#include <simd/simd.h>

// Source:
// https://stackoverflow.com/a/50978188

namespace {
template<typename T>
T xorshift(const T& n,int i){
  return n^(n>>i);
}

// a hash function with another name as to not confuse with std::hash
uint32_t distribute_32(const uint32_t& n){
  uint32_t p = 0x55555555ul; // pattern of alternating 0 and 1
  uint32_t c = 3423571495ul; // random uneven integer constant;
  return c*xorshift(p*xorshift(n,16),16);
}

// a hash function with another name as to not confuse with std::hash
uint64_t distribute_64(const uint64_t& n){
  uint64_t p = 0x5555555555555555ull; // pattern of alternating 0 and 1
  uint64_t c = 17316035218449499591ull;// random uneven integer constant;
  return c*xorshift(p*xorshift(n,32),32);
}

// if c++20 rotl is not available:
template <typename T,typename S>
typename std::enable_if<std::is_unsigned<T>::value,T>::type
constexpr rotl(const T n, const S i){
  const T m = (std::numeric_limits<T>::digits-1);
  const T c = i&m;
  return (n<<c)|(n>>((T(0)-c)&m)); // this is usually recognized by the compiler to mean rotation, also c++20 now gives us rotl directly
}
}

namespace ccv {
namespace nnc {
namespace mfa {
namespace hash {

// call this function with the old seed and the new key to be hashed and combined into the new seed value, respectively the final hash
inline size_t combine_32(std::size_t& seed, const uint32_t& v) {
    return rotl(seed, std::numeric_limits<size_t>::digits/3) ^ distribute_32(v);
}

inline uint32_t pack_32(const simd::uchar4& v) {
  return reinterpret_cast<const uint32_t&>(v);
}

inline uint32_t pack_32(const simd::ushort2& v) {
  return reinterpret_cast<const uint32_t&>(v);
}

inline size_t combine_64(std::size_t& seed, const uint64_t& v) {
    return rotl(seed, std::numeric_limits<size_t>::digits/3) ^ distribute_64(v);
}

inline uint64_t pack_64(const simd::ushort4& v) {
  return reinterpret_cast<const uint64_t&>(v);
}

inline uint64_t pack_64(const simd::uint2& v) {
  return reinterpret_cast<const uint64_t&>(v);
}

inline simd::ulong2 pack_128(const simd::ushort8& v) {
  return reinterpret_cast<const simd::ulong2&>(v);
}

} // namespace hash
} // namespace mfa
} // namespace nnc
} // namespace ccv

#endif
