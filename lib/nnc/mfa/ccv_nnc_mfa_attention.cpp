#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

void ccv_nnc_mfa_async_prepare_attention(mfa::context* context, ccv_nnc_mfa_attention_params_t params)
{
  context->attention_cache.prepare(context, mfa::attention::hash(params), true);
}

void ccv_nnc_mfa_sync_prepare_attention(mfa::context* context, ccv_nnc_mfa_attention_params_t params)
{
  context->attention_cache.prepare(context, mfa::attention::hash(params), false);
}

void ccv_nnc_mfa_encode_attention(mfa::context* context, ccv_nnc_mfa_attention_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  mfa::attention::hash hash(params);
  auto iterator = context->attention_cache.map.find(hash);
  if (iterator == context->attention_cache.map.end()) {
    mfa::precondition_failure("Attention hash not cached.", __LINE__, __FILE__, __FUNCTION__);
  }
}

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
  os << " .masked = " << bool(hash.masked) << " ";
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

mfa::attention::pipeline::pipeline(mfa::context* context, mfa::attention::hash hash, bool async) {
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf))
  
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  
  
  pool->drain();
}

mfa::attention::pipeline::~pipeline() {
  if (semaphore) {
    delete semaphore;
  }
  attention_pso->release();
  generate_mask_pso->release();
}

void mfa::attention::pipeline::wait() {
  if (!flags[0]) {
    semaphore->wait();
    flags[0] = true;
  }
}

MTL::ComputePipelineState* mfa::attention::pipeline::get_attention_pso() const {
  if (flags[0]) {
    return attention_pso;
  } else {
    return nullptr;
  }
}

MTL::ComputePipelineState* mfa::attention::pipeline::get_generate_mask_pso() const {
  if (flags[0]) {
    return generate_mask_pso;
  } else {
    return nullptr;
  }
}

simd::uchar4 mfa::attention::pipeline::get_flags() const {
  if (flags[0]) {
    return flags;
  } else {
    return false;
  }
}

uint16_t mfa::attention::pipeline::get_threadgroup_memory_length() const {
  if (flags[0]) {
    return threadgroup_memory_length;
  } else {
    return UINT16_MAX;
  }
}

MTL::Size mfa::attention::pipeline::get_grid_size() const {
  if (flags[0]) {
    return grid_size;
  } else {
    return MTL::Size(0, UINT64_MAX, UINT64_MAX);
  }
}

MTL::Size mfa::attention::pipeline::get_group_size() const {
  if (flags[0]) {
    return group_size;
  } else {
    return MTL::Size(0, UINT64_MAX, UINT64_MAX);
  }
}
