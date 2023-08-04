#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_attention(mfa::context* context, ccv_nnc_mfa_attention_params_t params)
{
  context->attention_cache.prepare(context, mfa::attention::hash(params));
}

void ccv_nnc_mfa_encode_attention(mfa::context* context, ccv_nnc_mfa_attention_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  mfa::attention::hash hash(params);
  auto iterator = context->attention_cache.map.find(hash);
  if (iterator == context->attention_cache.map.end()) {
    mfa::precondition_failure("Attention hash not cached.", __LINE__, __FILE__, __FUNCTION__);
  }
  
  auto* pipeline = iterator->second;
  auto encoder = command_batch->startCommand();
  
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == (hash.masked ? 5 : 4));
  
  uint16_t data_type_size = 0;
  switch (params.data_type) {
    case MTL::DataTypeHalf: {
      data_type_size = 2;
      break;
    }
    case MTL::DataTypeFloat: {
      data_type_size = 4;
      break;
    }
    default:
      CCV_NNC_MFA_PRECONDITION(false);
      break;
  }
  
  // Simple broadcasting rules; not yet support for NumPy broadcasting rules.
  simd::ushort2 num_batch_dims(0);
  simd::ulong2 batch_sizes(1);
  if (params.batched) {
    for (uint16_t operand = 0; operand < 2; ++operand) {
      uint32_t* batch_dims;
      if (operand == 0) {
        batch_dims = params.batch_dims_q;
      } else if (operand == 1) {
        batch_dims = params.batch_dims_mask;
      }
      
      for (int i = 0; i < CCV_NNC_MAX_DIM_ALLOC; ++i) {
        if (batch_dims[i] == 0) {
          break;
        }
        num_batch_dims[operand] += 1;
        batch_sizes[operand] *= batch_dims[i];
      }
      
      bool dims_match_q = true;
      if (num_batch_dims[0] != num_batch_dims[operand]) {
        dims_match_q = false;
      } else if (batch_sizes[0] != batch_sizes[operand]) {
        dims_match_q = false;
      } else {
        for (int i = 0; i < CCV_NNC_MAX_DIM_ALLOC; ++i) {
          if (params.batch_dims_q[i] != batch_dims[i]) {
            dims_match_q = false;
          }
        }
      }
      
      if (!dims_match_q) {
        CCV_NNC_MFA_PRECONDITION(batch_sizes[operand] == 1);
      }
    }
    
    uint64_t byte_stride_mask = 0;
    uint64_t byte_stride_block_mask = 0;
    if (batch_sizes[0] > 1) {
      byte_stride_mask = hash.R * hash.C * data_type_size;
    }
    if (batch_sizes[1] > 1) {
      auto grid_size = pipeline->grid_size;
      byte_stride_block_mask = grid_size.width * grid_size.height * 1;
    }
    
    simd::ulong4 matrix_offsets[batch_sizes[0]];
    for (int i = 0; i < batch_sizes[0]; ++i) {
      matrix_offsets[i] = simd::ulong4 {
        i * byte_stride_mask,
        i * byte_stride_block_mask,
        0,
        0,
      };
    }
    encoder->setBytes(matrix_offsets, batch_sizes[0] * 32, 10);
  }
  
  if (params.masked) {
    encoder->setComputePipelineState(pipeline->generate_block_mask_pso.get());
    encoder->setThreadgroupMemoryLength(48, 0);
    encoder->useResource(tensors[4], MTL::ResourceUsageRead);
    encoder->setBuffer(tensors[4], tensor_offsets[4], 12);
    
    auto grid_size = pipeline->grid_size;
    auto scratch_size = grid_size.width * grid_size.height * 1;
    grid_size.depth = batch_sizes[1];
    scratch_size *= batch_sizes[1];
    
    auto scratch = context->request_scratch(scratch_size);
    encoder->useResource(scratch, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->setBuffer(scratch, 0, 13);
    encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
    command_batch->finishCommand(encoder);
  }
  
  encoder->setComputePipelineState(pipeline->attention_pso.get());
  encoder->setThreadgroupMemoryLength(pipeline->threadgroup_memory_length, 0);
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[2], MTL::ResourceUsageRead);
  encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
  for (int i = 0; i < 4; ++i) {
    encoder->setBuffer(tensors[i], tensor_offsets[i], i);
  }
  
  auto grid_size = pipeline->grid_size;
  grid_size.height = params.H;
  grid_size.depth = batch_sizes[0];
  encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
  command_batch->finishCommand(encoder);
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

mfa::attention::pipeline::pipeline(mfa::context* context, mfa::attention::hash hash) {
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf))
  
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
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

  simd::ulong4 garbage(0);
  constants->setConstantValue(&garbage, MTL::DataTypeBool, 101);
  
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
  
  uint16_t Q_block_length;
  uint16_t K_block_length;
  uint16_t V_block_length;
  uint16_t O_block_length;
  if (hash.Q_trans) {
    Q_block_length = D_simd * R_block_dim;
  } else {
    Q_block_length = R_group * D_block_dim;
  }
  if (hash.K_trans) {
    K_block_length = C_simd * D_block_dim;
  } else {
    K_block_length = D_simd * C_block_dim;
  }
  if (hash.V_trans) {
    V_block_length = D_simd * C_block_dim;
  } else {
    V_block_length = C_simd * D_block_dim;
  }
  if (hash.O_trans) {
    O_block_length = D_simd * R_block_dim;
  } else {
    O_block_length = R_group * D_block_dim;
  }
  
  uint16_t block_elements;
  if (fuse_async_loads) {
    block_elements = K_block_length + V_block_length;
  } else {
    block_elements = std::max(K_block_length, V_block_length);
  }
  block_elements = std::max(block_elements, Q_block_length);
  block_elements = std::max(block_elements, O_block_length);
  this->threadgroup_memory_length = block_elements * data_type_size;
  
  std::function<size_t(size_t, uint16_t)> ceil_divide = [](size_t original, uint16_t granularity) {
    return (original + size_t(granularity) - 1) / size_t(granularity);
  };
  this->grid_size = MTL::Size(ceil_divide(hash.R, R_group), ceil_divide(hash.C, C_simd), 1);
  this->group_size = MTL::Size(32 * R_splits, 1, 1);
  
  auto swift_name = NS::String::string("attention", NS::UTF8StringEncoding);
  for (int i = 0; i < 2; ++i) {
    NS::SharedPtr<MTL::ComputePipelineState>* pso;
    if (i == 0) {
      pso = &attention_pso;
    } else {
      pso = &generate_block_mask_pso;
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
    
    *pso = NS::TransferPtr(context->device->newComputePipelineState(function.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error)
  }
  
  pool->drain();
}
