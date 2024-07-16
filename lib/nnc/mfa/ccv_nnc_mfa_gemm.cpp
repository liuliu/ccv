#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

// MARK: - Testing the New GEMM Kernel

#include "ccv_nnc_mfa_error.hpp"
#include "GEMM/CoreCount.hpp"
#include "GEMM/GEMMDescriptor.hpp"
#include "GEMM/GEMMKernel.hpp"
#include "GEMM/GEMMShaderCache.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

// MARK: - Imports that Existed Previously

#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_gemm(mfa::context* context, ccv_nnc_mfa_gemm_params_t params)
{
  context->gemm_cache.prepare(context, mfa::gemm::hash(params));
}

void ccv_nnc_mfa_encode_gemm(mfa::context* context, ccv_nnc_mfa_gemm_params_t params, MTL::CommandBatch* command_batch, MTL::Buffer** tensors, size_t* tensor_offsets)
{
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION((num_tensors == 3) || (num_tensors == 4))
  
  
  
  // Count the number of GEMMs at all.
  //
  // MFA       | 39 |  60% |
  // MPSMatrix |  5 |   8% |
  // MPSGraph  | 21 |  32% |
  // Total     | 65 | 100% |
  ccv_nnc_mfa_log_message("\n");
  ccv_nnc_mfa_log_message("MFA\n");
  
  // Count the percentage of GEMMs that match a specific set of criteria.
  // - data_type = any
  // - M = any
  // - N = any
  // - K = any
  // - A_trans = any
  // - B_trans = any
  // - D_trans = any
  // - alpha = 1.0
  // - beta = 0.0
  // - batched = false
  // - fused_activation_function = false
  // - fused_bias = false
  //
  // - batch_dims_a = any
  // - batch_dims_b = any
  // - batch_dims_d = any
  //
  // - num_tensors = 3
  //
  // YES   | 17 |  44% |
  // NO    | 22 |  56% |
  // Total | 39 | 100% |
  bool canEncodeNewGEMM = false;
  if ((params.alpha == 1.0) &&
      (params.beta == 0.0) &&
      (params.batched == false) &&
      (params.fused_activation_function == false) &&
      (params.fused_bias == false) &&
      (num_tensors == 3))
  {
    ccv_nnc_mfa_log_message("\n");
    ccv_nnc_mfa_log_message("YES\n");
    canEncodeNewGEMM = true;
  }
  else
  {
    ccv_nnc_mfa_log_message("\n");
    ccv_nnc_mfa_log_message("NO\n");
  }
  
  // Branch on whether to use the new kernel.
  if (canEncodeNewGEMM) {
    // Instantiate the descriptor.
    GEMMDescriptor gemmDesc;
    gemmDesc.matrixDimensions = simd::uint3 {
      params.M,
      params.N,
      params.K,
    };
    switch (params.data_type) {
      case MTL::DataTypeHalf: {
        gemmDesc.memoryPrecisions = {
          .A = GEMMOperandPrecision::FP16,
          .B = GEMMOperandPrecision::FP16,
          .C = GEMMOperandPrecision::FP16,
        };
        break;
      }
      case MTL::DataTypeFloat: {
        gemmDesc.memoryPrecisions = {
          .A = GEMMOperandPrecision::FP32,
          .B = GEMMOperandPrecision::FP32,
          .C = GEMMOperandPrecision::FP32,
        };
        break;
      }
      default:
        CCV_NNC_MFA_PRECONDITION(false);
        break;
    }
    gemmDesc.transposeState = simd::uchar2 { params.A_trans, params.B_trans };
    
    // Instantiate the kernel.
    //
    // TODO: Remove the autoreleasepool, once you confirm the caller always
    // makes one. Or find a different solution, like spawning a pool when a new
    // kernel variant is compiled.
    auto pool = NS::AutoreleasePool::alloc()->init();
    GEMMShaderCache::fetchKernel(gemmDesc);
    auto pipelineValue = GEMMShaderCache::fetchKernel(gemmDesc);
    pool->drain();
    auto kernel = pipelineValue->kernel;
    auto pipeline = pipelineValue->pipeline;
    
    // Allocate a new command.
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipeline.get());
    encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation, 0);
    
    // Bind the function arguments.
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
    for (int i = 0; i < 3; ++i) {
      encoder->setBuffer(tensors[i], tensor_offsets[i], i);
    }
    
    // Calculate the grid size.
    auto ceilDivide =
    [=](int64_t target, uint16_t granularity) -> int64_t {
      return (target + int64_t(granularity) - 1) / int64_t(granularity);
    };
    MTL::Size gridSize
    (ceilDivide(int64_t(params.N), kernel->blockDimensions[1]),
     ceilDivide(int64_t(params.M), kernel->blockDimensions[0]),
     1);
    MTL::Size groupSize
    (int64_t(kernel->threadgroupSize), 1, 1);
    
    // Dispatch the required number of threads.
    encoder->dispatchThreadgroups(gridSize, groupSize);
    
    // Finish the command.
    command_batch->finishCommand(encoder);
  } else {
    mfa::gemm::hash hash(params);
    auto iterator = context->gemm_cache.map.find(hash);
    if (iterator == context->gemm_cache.map.end()) {
      mfa::precondition_failure("GEMM hash not cached.", __LINE__, __FILE__, __FUNCTION__);
    }
    
    auto* pipeline = iterator->second;
    auto encoder = command_batch->startCommand();
    encoder->setComputePipelineState(pipeline->pso.get());
    encoder->setThreadgroupMemoryLength(pipeline->threadgroup_memory_length, 0);
    
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
      if (batch_sizes[0] == 1) {
        byte_stride_a = 0;
      }
      if (batch_sizes[1] == 1) {
        byte_stride_b = 0;
      }
      if (batch_sizes[3] == 1) {
        byte_stride_d = 0;
      }
      
      const unsigned long batch_size = std::max(batch_sizes[0], batch_sizes[1]);
      simd::ulong4 matrix_offsets[batch_size];
      for (int i = 0; i < batch_size; ++i) {
        matrix_offsets[i] = simd::ulong4 {
          i * byte_stride_a,
          i * byte_stride_b,
          i * byte_stride_c,
          i * byte_stride_d,
        };
      }
      if (batch_size * 32 > 4096) {
        auto buffer = context->device->newBuffer(matrix_offsets, batch_size * 32, MTL::ResourceStorageModeShared);
        encoder->useResource(buffer, MTL::ResourceUsageRead);
        encoder->setBuffer(buffer, 0, 10);
        buffer->release();
      } else {
        encoder->setBytes(matrix_offsets, batch_size * 32, 10);
      }
    }
    
    auto grid_size = pipeline->grid_size;
    grid_size.depth = batch_sizes[0];
    encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
    command_batch->finishCommand(encoder);
  }
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
    if (hash.K <= 24) {
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
  
  NS::Error* error = nullptr;
  auto function = NS::TransferPtr(context->library->newFunction(swift_name, constants.get(), &error));
  if (!function) {
    CCV_NNC_MFA_CHECK_ERROR(error)
  }
  
  pso = NS::TransferPtr(context->device->newComputePipelineState(function.get(), &error));
  if (!pso) {
    CCV_NNC_MFA_CHECK_ERROR(error)
  }
  
  pool->drain();
}
