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
  
  if (async) {
    flags[0] = false;
    semaphore = new Dispatch::Semaphore(0);
  } else {
    flags[0] = true;
    semaphore = nullptr;
  }
  this->flags[1] = hash.batched;
  this->flags[2] = hash.masked;
  
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  constants->setConstantValue(&hash.R, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&hash.C, MTL::DataTypeUInt, 1);
  constants->setConstantValue(&hash.H, MTL::DataTypeUInt, 2);
  constants->setConstantValue(&hash.D, MTL::DataTypeUInt, 3);
  constants->setConstantValue(&hash.Q_trans, MTL::DataTypeBool, 10);
  constants->setConstantValue(&hash.K_trans, MTL::DataTypeBool, 11);
  constants->setConstantValue(&hash.V_trans, MTL::DataTypeBool, 12);
  constants->setConstantValue(&hash.O_trans, MTL::DataTypeBool, 13);
  constants->setConstantValue(&hash.alpha, MTL::DataTypeFloat, 20);
  constants->setConstantValue(&hash.data_type, MTL::DataTypeUInt, 30);
  constants->setConstantValue(&hash.batched, MTL::DataTypeBool, 100);
  constants->setConstantValue(&hash.masked, MTL::DataTypeBool, 50000);
  
  {
    bool block_sparse = hash.masked;
    bool triangular = false;
    bool forward = true;
    bool backward = false;
    bool generate_block_mask = false;
    bool grouped_query = false;
    constants->setConstantValue(&block_sparse, MTL::DataTypeBool, 102);
    constants->setConstantValue(&triangular, MTL::DataTypeBool, 103);
    constants->setConstantValue(&forward, MTL::DataTypeBool, 110);
    constants->setConstantValue(&backward, MTL::DataTypeBool, 111);
    constants->setConstantValue(&generate_block_mask, MTL::DataTypeBool, 112);
    constants->setConstantValue(&grouped_query, MTL::DataTypeBool, 113);
  }
  
  uint16_t R_simd;
  uint16_t C_simd;
  uint16_t R_splits;
  bool fuse_async_loads = false;
  if (hash.data_type == MTL::DataTypeFloat) {
    R_simd = 8;
    C_simd = 32;
    R_splits = 4;
  } else {
    uint32_t D = hash.D;
    if (hash.masked) {
      if (D <= 16) {
        R_simd = 16;
        C_simd = 64;
        R_splits = 4;
      } else if (D <= 24) {
        R_simd = 8;
        C_simd = 64;
        R_splits = 8;
      } else if (D <= 80) {
        R_simd = 8;
        C_simd = 64;
        R_splits = 4;
      } else {
        R_simd = 8;
        C_simd = 32;
        R_splits = 4;
      }
    } else {
      R_simd = 8;
      R_splits = 8;
      
      if (D <= 8) {
        R_simd = 16;
        C_simd = 64;
      } else if (D <= 16) {
        C_simd = 72;
        fuse_async_loads = true;
      } else if (D <= 24) {
        C_simd = 56;
        fuse_async_loads = true;
      } else if (D <= 56) {
        C_simd = 64;
      } else if (D <= 64) {
        C_simd = 40;
        fuse_async_loads = true;
      } else if (D <= 96) {
        C_simd = 64;
      } else if (D <= 304) {
        C_simd = 32;
        R_splits = 4;
      } else {
        C_simd = 40;
        R_splits = 8;
      }
    }
  }
  
  constants->setConstantValue(&R_simd, MTL::DataTypeUShort, 200);
  constants->setConstantValue(&C_simd, MTL::DataTypeUShort, 201);
  constants->setConstantValue(&R_splits, MTL::DataTypeUShort, 210);
  constants->setConstantValue(&fuse_async_loads, MTL::DataTypeBool, 213);
  
  uint16_t data_type_size = UINT16_MAX;
  switch (hash.data_type) {
    case MTL::DataTypeHalf: {
      data_type_size = 2;
      break;
    }
    case MTL::DataTypeFloat: {
      data_type_size = 4;
      break;
    }
    default: {
      CCV_NNC_MFA_PRECONDITION(false)
      break;
    }
  }
  
  uint16_t D_simd = (hash.D + 7) / 8 * 8;
  uint16_t R_group = R_simd * R_splits;
  uint16_t Q_block_length;
  uint16_t K_block_length;
  uint16_t V_block_length;
  uint16_t O_block_length;
  
  uint16_t R_block_dim = R_group;
  uint16_t C_block_dim = C_simd;
  uint16_t D_block_dim = D_simd;
  std::function<void(uint16_t*, NS::UInteger)> set_bank_offset = [=](uint16_t* dim, NS::UInteger index) {
    CCV_NNC_MFA_PRECONDITION(*dim % 8 == 0);
    
    uint16_t dim_bytes = *dim * data_type_size;
    uint16_t dim_bytes_modulo = dim_bytes % 64;
    if (dim_bytes_modulo == 16 || dim_bytes_modulo == 48) {
      return;
    } else if (dim_bytes_modulo == 0 || dim_bytes_modulo == 32) {
      constexpr uint16_t bank_offset_bytes = 16;
      uint16_t bank_offset = bank_offset_bytes / data_type_size;
      constants->setConstantValue(&bank_offset, MTL::DataTypeUShort, index);
      *dim += bank_offset;
    } else {
      CCV_NNC_MFA_PRECONDITION(false)
    }
  };
  set_bank_offset(&R_block_dim, 220);
  set_bank_offset(&C_block_dim, 221);
  set_bank_offset(&D_block_dim, 222);
  
  // TODO: Find amount of threadgroup memory and grid sizes.
  
  auto swift_name = NS::String::string("attention", NS::UTF8StringEncoding);
  for (int i = 0; i < 2; ++i) {
    MTL::ComputePipelineState** pso;
    if (i == 0) {
      pso = &attention_pso;
    } else {
      pso = &generate_mask_pso;
      if (hash.masked) {
        bool generate_block_mask = true;
        constants->setConstantValue(&generate_block_mask, MTL::DataTypeBool, 112);
      } else {
        continue;
      }
    }
    
    NS::Error *error;
    auto function = NS::TransferPtr(context->library->newFunction(swift_name, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error)
    
    if (async) {
      context->device->newComputePipelineState(function.get(), [=](MTL::ComputePipelineState* pipeline, NS::Error* error) {
        CCV_NNC_MFA_CHECK_ERROR(error)
        
        pipeline->retain();
        *pso = pipeline;
        semaphore->signal();
      });
    } else {
      *pso = context->device->newComputePipelineState(function.get(), &error);
      CCV_NNC_MFA_CHECK_ERROR(error)
    }
  }
  
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
    if (flags[2]) {
      semaphore->wait();
    }
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
