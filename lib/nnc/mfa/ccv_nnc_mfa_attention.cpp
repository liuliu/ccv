#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

// MARK: - C++

mfa::attention::hash::hash(ccv_nnc_mfa_attention_params_t params) {
  data_type = params.data_type;
  R = params.R;
  C = params.C;
  H = params.H;
  D = params.D;
  Q_trans = params.Q_trans;
  K_trans = params.K_trans;
  V_trans = params.V_trans;
  O_trans = params.O_trans;
  alpha = params.alpha;
  batched = params.batched;
  masked = params.masked;
}

bool mfa::attention::hash::operator==(const mfa::attention::hash& hash) const {
  return
  (data_type == hash.data_type) &&
  (R == hash.R) &&
  (C == hash.C) &&
  (H == hash.H) &&
  (D == hash.D) &&
  (Q_trans == hash.Q_trans) &&
  (K_trans == hash.K_trans) &&
  (V_trans == hash.V_trans) &&
  (O_trans == hash.O_trans) &&
  (alpha == hash.alpha) &&
  (batched == hash.batched) &&
  (masked == hash.masked);
}

std::ostream& operator<<(std::ostream& os, const mfa::attention::hash& hash) {
  os << "mfa::attention::hash {";
  os << " .data_type = " << hash.data_type << ',';
  os << " .R = " << hash.R << ',';
  os << " .C = " << hash.C << ',';
  os << " .H = " << hash.H << ',';
  os << " .D = " << hash.D << ',';
  os << " .Q_trans = " << bool(hash.Q_trans) << ',';
  os << " .K_trans = " << bool(hash.K_trans) << ',';
  os << " .V_trans = " << bool(hash.V_trans) << ',';
  os << " .O_trans = " << bool(hash.O_trans) << ',';
  os << " .alpha = " << double(hash.alpha) << ',';
  os << " .batched = " << bool(hash.batched) << ',';
  os << " .masked = " << bool(hash.masked) << ',';
  os << "}";
  return os;
}

std::size_t std::hash<mfa::attention::hash>::operator()(const mfa::attention::hash& hash) const noexcept {
  std::size_t seed = 0;
  using namespace mfa::hash;
  combine_64(seed, hash.data_type);
  combine_64(seed, pack_64(simd::uint2 { hash.R, hash.C }));
  combine_64(seed, pack_64(simd::uint2 { hash.H, hash.D }));
  combine_64(seed, pack_64(simd::uint2 { pack_32(simd::uchar4 { hash.Q_trans, hash.K_trans, hash.V_trans, hash.O_trans }), *reinterpret_cast<const uint32_t*>(&hash.alpha) }));
  combine_32(seed, pack_32(simd::uchar4 { hash.batched, hash.masked, 0, 0 }));
  return seed;
}
