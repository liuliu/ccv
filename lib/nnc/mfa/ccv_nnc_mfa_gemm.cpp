#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_gemm(mfa::context* context, ccv_nnc_mfa_gemm_params_t params)
{
  context->gemm_cache.prepare(context, mfa::gemm::hash(params));
}

void ccv_nnc_mfa_encode_gemm(mfa::context* context, ccv_nnc_mfa_gemm_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  mfa::gemm::hash hash(params);
  auto iterator = context->gemm_cache.map.find(hash);
  if (iterator == context->gemm_cache.map.end()) {
    mfa::precondition_failure("GEMM hash not cached.", __LINE__, __FILE__, __FUNCTION__);
  }
  
  auto* pipeline = iterator->second;
  auto encoder = command_batch->startCommand();
  encoder->setComputePipelineState(pipeline->pso.get());
  encoder->setThreadgroupMemoryLength(pipeline->threadgroup_memory_length, 0);
  
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION((num_tensors == 3) || (num_tensors == 4))
  
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
  if (num_tensors >= 4) {
    encoder->useResource(tensors[3], MTL::ResourceUsageRead);
  }
  for (int i = 0; i < num_tensors; ++i) {
    encoder->setBuffer(tensors[i], tensor_offsets[i], i);
  }
  
  // Simple broadcasting rules; not yet support for NumPy broadcasting rules.
  simd::ushort4 num_batch_dims(0);
  simd::ulong4 batch_sizes(1);
  if (params.batched) {
    for (uint16_t operand = 0; operand < 4; ++operand) {
      uint32_t* batch_dims;
      if (operand == 0) {
        batch_dims = params.batch_dims_a;
      } else if (operand == 1) {
        batch_dims = params.batch_dims_b;
      } else if (operand == 2) {
        // Skip the C operand.
        continue;
      } else if (operand == 3) {
        // Skip the D operand if unavailable.
        if (!(params.fused_activation_function || params.fused_bias)) {
          continue;
        }
        batch_dims = params.batch_dims_d;
      }
      
      for (int i = 0; i < CCV_NNC_MAX_DIM_ALLOC; ++i) {
        if (batch_dims[i] == 0) {
          break;
        }
        num_batch_dims[operand] += 1;
        batch_sizes[operand] *= batch_dims[i];
      }
      
      bool dims_match_a = true;
      if (num_batch_dims[0] != num_batch_dims[operand]) {
        dims_match_a = false;
      } else if (batch_sizes[0] != batch_sizes[operand]) {
        dims_match_a = false;
      } else {
        for (int i = 0; i < CCV_NNC_MAX_DIM_ALLOC; ++i) {
          if (params.batch_dims_a[i] != batch_dims[i]) {
            dims_match_a = false;
          }
        }
      }
      
      if (!dims_match_a) {
        CCV_NNC_MFA_PRECONDITION(batch_sizes[operand] == 1);
      }
    }
    
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
    uint64_t byte_stride_a = hash.M * hash.K * data_type_size;
    uint64_t byte_stride_b = hash.K * hash.N * data_type_size;
    uint64_t byte_stride_c = hash.M * hash.N * data_type_size;
    uint64_t byte_stride_d = (hash.D_trans ? hash.M : hash.N) * data_type_size;
    if (batch_sizes[1] == 1) {
      byte_stride_b = 0;
    }
    if (batch_sizes[3] == 1) {
      byte_stride_d = 0;
    }
    
    simd::ulong4 matrix_offsets[batch_sizes[0]];
    for (int i = 0; i < batch_sizes[0]; ++i) {
      matrix_offsets[i] = simd::ulong4 {
        i * byte_stride_a,
        i * byte_stride_b,
        i * byte_stride_c,
        i * byte_stride_d,
      };
    }
    encoder->setBytes(matrix_offsets, batch_sizes[0] * 32, 10);
  }
  
  auto grid_size = pipeline->grid_size;
  grid_size.depth = batch_sizes[0];
  encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
  command_batch->finishCommand(encoder);
}

// MARK: - C++

mfa::gemm::hash::hash(ccv_nnc_mfa_gemm_params_t params) {
  data_type = params.data_type;
  M = params.M;
  N = params.N;
  K = params.K;
  A_trans = params.A_trans;
  B_trans = params.B_trans;
  D_trans = params.D_trans;
  alpha = params.alpha;
  beta = params.beta;
  batched = params.batched;
  fused_activation_function = params.fused_activation_function;
  fused_bias = params.fused_bias;
}

bool mfa::gemm::hash::operator==(const mfa::gemm::hash& hash) const {
  return
  (data_type == hash.data_type) &&
  (M == hash.M) &&
  (N == hash.N) &&
  (K == hash.K) &&
  (A_trans == hash.A_trans) &&
  (B_trans == hash.B_trans) &&
  (D_trans == hash.D_trans) &&
  (alpha == hash.alpha) &&
  (beta == hash.beta) &&
  (batched == hash.batched) &&
  (fused_activation_function == hash.fused_activation_function) &&
  (fused_bias == hash.fused_bias);
}

std::ostream& operator<<(std::ostream& os, const mfa::gemm::hash& hash) {
  os << "mfa::gemm::hash {";
  os << " .data_type = " << hash.data_type << ',';
  os << " .M = " << hash.M << ',';
  os << " .N = " << hash.N << ',';
  os << " .K = " << hash.K << ',';
  os << " .A_trans = " << bool(hash.A_trans) << ',';
  os << " .B_trans = " << bool(hash.B_trans) << ',';
  os << " .D_trans = " << bool(hash.D_trans) << ',';
  os << " .alpha = " << double(hash.alpha) << ',';
  os << " .beta = " << double(hash.beta) << ',';
  os << " .batched = " << bool(hash.batched) << ',';
  os << " .fused_activation_function = " << bool(hash.fused_activation_function) << ',';
  os << " .fused_bias = " << bool(hash.fused_bias) << " ";
  os << "}";
  return os;
}

std::size_t std::hash<mfa::gemm::hash>::operator()(const mfa::gemm::hash& hash) const noexcept {
  std::size_t seed = 0;
  using namespace mfa::hash;
  combine_64(seed, hash.data_type);
  combine_64(seed, pack_64(simd::uint2 { hash.M, hash.N }));
  combine_64(seed, pack_64(simd::uint2 { hash.K, pack_32(simd::uchar4 { hash.A_trans, hash.B_trans, hash.D_trans, 0 }) }));
  combine_64(seed, pack_64(simd::uint2 { *reinterpret_cast<const uint32_t*>(&hash.alpha), *reinterpret_cast<const uint32_t*>(&hash.beta) }));
  combine_32(seed, pack_32(simd::uchar4 { hash.batched, hash.fused_activation_function, hash.fused_bias, 0 }));
  return seed;
}

mfa::gemm::pipeline::pipeline(mfa::context* context, mfa::gemm::hash hash) {
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf))
  CCV_NNC_MFA_PRECONDITION(hash.alpha == 1.0)
  CCV_NNC_MFA_PRECONDITION(hash.beta == 0.0)
  CCV_NNC_MFA_PRECONDITION(hash.fused_activation_function == false)
  
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  constants->setConstantValue(&hash.M, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&hash.N, MTL::DataTypeUInt, 1);
  constants->setConstantValue(&hash.K, MTL::DataTypeUInt, 2);
  constants->setConstantValue(&hash.A_trans, MTL::DataTypeBool, 10);
  constants->setConstantValue(&hash.B_trans, MTL::DataTypeBool, 11);
  constants->setConstantValue(&hash.D_trans, MTL::DataTypeBool, 13);
  constants->setConstantValue(&hash.alpha, MTL::DataTypeFloat, 20);
  constants->setConstantValue(&hash.beta, MTL::DataTypeFloat, 21);
  constants->setConstantValue(&hash.batched, MTL::DataTypeBool, 100);
  constants->setConstantValue(&hash.fused_activation_function, MTL::DataTypeBool, 101);
  constants->setConstantValue(&hash.fused_bias, MTL::DataTypeBool, 50001);
  simd::ulong4 garbage(0);
  constants->setConstantValue(&garbage, MTL::DataTypeBool, 102);
  constants->setConstantValue(&garbage, MTL::DataTypeBool, 103);
  constants->setConstantValue(&garbage, MTL::DataTypeBool, 113);
  constants->setConstantValue(&garbage, MTL::DataTypeBool, 50000);
  
  // Eventually, this may incorporate the batch size.
  // BxMxN > 1,000,000 -> 48x48, only if M >= 88 and N >= 88
  // BxMxN > 4,000,000 -> 64x64, only if M >= 120 and N >= 120
  uint64_t C_elements = uint64_t(hash.M) * uint64_t(hash.N);
  if (hash.batched) {
    C_elements *= 2;
  }
  int is_half = (hash.data_type == MTL::DataTypeHalf); // SD v1 attention
  int is_float = (hash.data_type == MTL::DataTypeFloat); // SD v2 attention
  
  uint16_t M_group = 32;
  uint16_t N_group = 32;
  uint16_t K_simd = 32;
  if (C_elements > 1000 * 1000) {
    M_group = 48;
    N_group = 48;
  }
  
  // If K_simd is perfectly equal to matrix K, the compiler can elide a large
  // amount of logic in the kernel.
  if (hash.K >= 33 && hash.K <= 40) {
    K_simd = 40; // 1 * 40
  } else if (is_half && hash.K >= 73 && hash.K <= 80) {
    K_simd = 40; // 2 * 40
  } else if (C_elements > 1000 * 1000) {
    if (hash.K <= 16) {
      K_simd = 16; // 1 * 16
    } else if (hash.K <= 24) {
      K_simd = 24; // 1 * 24
    } else if (hash.K <= 32) {
      K_simd = 32; // 1 * 32
    } else if (hash.K <= 48) {
      K_simd = 24;
    } else if (hash.K <= 64) {
      K_simd = 32;
    } else if (is_float) {
      K_simd = 24;
    }
  }
  
  uint16_t M_splits = 2;
  uint16_t N_splits = 2;
  uint16_t M_simd = M_group / M_splits;
  uint16_t N_simd = N_group / N_splits;
  
  constants->setConstantValue(&M_simd, MTL::DataTypeUShort, 200);
  constants->setConstantValue(&N_simd, MTL::DataTypeUShort, 201);
  constants->setConstantValue(&K_simd, MTL::DataTypeUShort, 202);
  constants->setConstantValue(&M_splits, MTL::DataTypeUShort, 210);
  constants->setConstantValue(&N_splits, MTL::DataTypeUShort, 211);
  
  std::string cpp_name;
  uint16_t data_type_size = UINT16_MAX;
  switch (hash.data_type) {
    case MTL::DataTypeHalf: {
      cpp_name = "hgemm";
      data_type_size = 2;
      break;
    }
    case MTL::DataTypeFloat: {
      cpp_name = "sgemm";
      data_type_size = 4;
      break;
    }
    default: {
      CCV_NNC_MFA_PRECONDITION(false)
      break;
    }
  }
  auto* swift_name = NS::String::string(cpp_name.c_str(), NS::UTF8StringEncoding);
  
  uint16_t A_block_bytes = M_group * K_simd * data_type_size;
  uint16_t B_block_bytes = K_simd * N_group * data_type_size;
  uint16_t C_block_bytes = M_group * N_group * data_type_size;
  threadgroup_memory_length = A_block_bytes + B_block_bytes;
  
  if ((hash.M % 8 > 0) && (hash.N % 8 > 0)) {
    if (C_block_bytes > threadgroup_memory_length) {
      threadgroup_memory_length = C_block_bytes;
    }
  }
  if (hash.fused_bias) {
    uint16_t D_block_bytes = (hash.D_trans ? M_group : N_group) * data_type_size;
    if (D_block_bytes > threadgroup_memory_length) {
      threadgroup_memory_length = D_block_bytes;
    }
  }
  
  std::function<size_t(size_t, uint16_t)> ceil_divide = [](size_t original, uint16_t granularity) {
    return (original + size_t(granularity) - 1) / size_t(granularity);
  };
  grid_size = MTL::Size(ceil_divide(hash.N, N_group), ceil_divide(hash.M, M_group), 1);
  group_size = MTL::Size(32 * M_splits * N_splits, 1, 1);
  
  NS::Error* error = NULL;
  auto function = NS::TransferPtr(context->library->newFunction(swift_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error)
  
  pso = NS::TransferPtr(context->device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error)
  
  pool->drain();
}
