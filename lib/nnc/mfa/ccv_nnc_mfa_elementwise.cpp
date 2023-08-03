#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

mfa::elementwise::hash::hash(ccv_nnc_mfa_elementwise_params_t params) {
  data_type = params.data_type;
  operation_id = params.operation_id;
  reduction_dim = params.reduction_dim;
}

bool mfa::elementwise::hash::operator==(const mfa::elementwise::hash& hash) const {
  return
  (data_type == hash.data_type) &&
  (operation_id == hash.operation_id) &&
  (reduction_dim == hash.reduction_dim);
}

std::ostream& operator<<(std::ostream& os, const mfa::elementwise::hash& hash) {
  os << "mfa::elementwise::hash {";
  os << " .data_type = " << hash.data_type << ',';
  os << " .operation_id = " << hash.operation_id << ',';
  os << " .reduction_dim = " << hash.reduction_dim << " ";
  os << "}";
  return os;
}

std::size_t std::hash<mfa::elementwise::hash>::operator()(const mfa::elementwise::hash& hash) const noexcept {
  std::size_t seed = 0;
  using namespace mfa::hash;
  combine_64(seed, hash.data_type);
  combine_64(seed, pack_64(simd::uint2 { hash.operation_id, hash.reduction_dim }));
  return seed;
}

mfa::elementwise::pipeline::pipeline(mfa::context* context, mfa::elementwise::hash hash) {
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf))
  CCV_NNC_MFA_PRECONDITION(hash.operation_id == 0)
  
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  std::string shader = "";
  if (hash.data_type == MTL::DataTypeFloat) {
    shader += std::string("typedef float real;");
    shader += "\n";
  } else {
    shader += std::string("typedef half real;");
    shader += "\n";
  }
  shader += "constant uint row_size = " + std::to_string(hash.reduction_dim) + ";";
  shader += "\n";
  
  uint32_t group_size;
  if (hash.reduction_dim <= 384) {
    group_size = 128;
  } else {
    group_size = 256;
  }
  shader += "constant ushort group_size = " + std::to_string(group_size) + ";";
  shader += "\n";
  this->group_size = MTL::Size(group_size, 1, 1);
  
  shader += R"(
  constant uint bulk_size = row_size / group_size * group_size;
  constant uint padding_size = row_size - bulk_size;
  constant float scale_factor = 1 / float(row_size);
  
  kernel void normalization(device real *source [[buffer(0)]],
                            device real *destination [[buffer(1)]],
                            
                            uint tgid [[threadgroup_position_in_grid]],
                            ushort sidx [[simdgroup_index_in_threadgroup]],
                            ushort lid [[thread_position_in_threadgroup]])
  {
    source += tgid * row_size + lid;
    destination += tgid * row_size + lid;
    real cache_bulk[bulk_size / group_size];
    real cache_padding;
    threadgroup float partials[group_size / 32];
    
    float sum = 0;
    #pragma clang loop unroll(full)
    for (uint i = 0; i < bulk_size; i += group_size) {
      cache_bulk[i / group_size] = source[i];
      sum += cache_bulk[i / group_size];
    }
    if (padding_size > 0 && lid < padding_size) {
      cache_padding = source[bulk_size];
      sum += cache_padding;
    }
    partials[sidx] = simd_sum(sum);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < (group_size / 32)) {
      float sum = quad_sum(partials[lid]);
      if (group_size == 256) {
         sum += simd_shuffle_xor(sum, 4);
      }
      partials[lid] = sum * scale_factor;
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = partials[sidx];
    float variance = 0;
    for (uint i = 0; i < bulk_size; i += group_size) {
      cache_bulk[i / group_size] -= mean;
      real deviation = cache_bulk[i / group_size];
      variance = fma(deviation, deviation, variance);
    }
    if (padding_size > 0 && lid < padding_size) {
      cache_padding -= mean;
      variance += fma(cache_padding, cache_padding, variance);
    }
    partials[sidx] = simd_sum(variance);
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < (group_size / 32)) {
      float variance = quad_sum(partials[lid]);
      if (group_size == 256) {
        variance += simd_shuffle_xor(variance, 4);
      }
      partials[lid] = rsqrt(variance);
    }
  
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float standard_deviation_reciprocal = partials[sidx];
    for (uint i = 0; i < bulk_size; i += group_size) {
      real deviation = cache_bulk[i / group_size];
      destination[i] = deviation * standard_deviation_reciprocal;
    }
    if (padding_size > 0 && lid < padding_size) {
      destination[bulk_size] = cache_padding * standard_deviation_reciprocal;
    }
  }
  )";
  
  NS::Error *error;
  auto swift_string = NS::String::string(shader.c_str(),
   NS::UTF8StringEncoding);
  auto library = NS::TransferPtr(context->device->newLibrary(swift_string, nullptr, &error));
  CCV_NNC_MFA_CHECK_ERROR(error)
  
  auto swift_name = NS::String::string("normalization", NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(context->library->newFunction(swift_name));
  CCV_NNC_MFA_PRECONDITION(function.get() != nullptr)
  
  pso = NS::TransferPtr(context->device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error)
  
  pool->drain();
}
