#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

mfa::normalization::hash::hash(ccv_nnc_mfa_normalization_params_t params) {
  data_type = params.data_type;
  channel_count = params.channel_count;
  channel_groups = params.channel_groups;
  sequence_count = params.sequence_count;
  is_layer_normalization = params.is_layer_normalization;
}

bool mfa::normalization::hash::operator==(const mfa::normalization::hash& hash) const {
  return
  (data_type == hash.data_type) &&
  (channel_count == hash.channel_count) &&
  (channel_groups == hash.channel_groups) &&
  (sequence_count == hash.sequence_count) &&
  (is_layer_normalization == hash.is_layer_normalization);
}

std::ostream& operator<<(std::ostream& os, const mfa::normalization::hash& hash) {
  os << "mfa::normalization::hash {";
  os << " .data_type = " << hash.data_type << ',';
  os << " .channel_count = " << hash.channel_count << ',';
  os << " .channel_groups = " << hash.channel_groups << ',';
  os << " .sequence_count = " << hash.sequence_count << ',';
  os << " .is_layer_normalization = " << hash.is_layer_normalization << " ";
  os << "}";
  return os;
}

std::size_t std::hash<mfa::normalization::hash>::operator()(const mfa::normalization::hash& hash) const noexcept {
  std::size_t seed = 0;
  using namespace mfa::hash;
  combine_64(seed, hash.data_type);
  combine_64(seed, pack_64(simd::uint2 { hash.channel_count, hash.channel_groups }));
  combine_64(seed, pack_64(simd::uint2 { hash.sequence_count, uint32_t(hash.is_layer_normalization) }));
  return seed;
}

mfa::normalization::pipeline::pipeline(mfa::context* context, mfa::normalization::hash hash) {
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf))
  
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  std::string shader = R"(
  constant uint bulk_size = sample_count / threadgroup_size * threadgroup_size;
  constant uint padding_size = sample_count - bulk_size;
  
  constant uint channel_group_size = channel_count / channel_groups;
  constant uint population_count = channel_group_size * sequence_count;
  
  // Partially sourced from:
  // https://github.com/nvpro-samples/gl_vk_raytrace_interop/blob/master/shaders/sampling.h
  uint tea(uint val0, uint val1) {
    uint v0 = val0;
    uint v1 = val1;
    uint s0 = 0;
    
    for (uint n = 0; n < 9; n++) {
      s0 += 0x9e3779b9;
      v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
      v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }
    return v0;
  }
  
  // Compute radical inverse of n to the base 2, then convert into an integer.
  uint radinv2(uint n) {
    float random = as_type<float>(0x3F800000 | ((reverse_bits(n)) >> 9)) - 1.0f;
    uint index = uint(progress * float(population_count));
    return min(index, population_count - 1);
  }
  
  // Generate 2D coordinates within the population.
  uint population_coords(uint index) {
    uint x = index % channel_group_size;
    uint y = index - x * channel_group_size;
    return uint2(x, y);
  }
  
  kernel void normalization(
    device real *source [[buffer(0)]],
    device real *destination [[buffer(1)]],
    device real *saved_mean [[buffer(2)]],
    device real *saved_standard_deviation_reciprocal [[buffer(3)],
    device real *channel_scales [[buffer(4)]],
    device real *channel_translations [[buffer(5)]],
  #if SAMPLE_POPULATION
    constant uint *batch_seeds [[buffer(6)]],
  #endif
  
  #if LAYER_NORMALIZATION
    uint tgid [[threadgroup_position_in_grid]],
  #else
    uint3 tgid [[threadgroup_position_in_grid]],
  #endif
    ushort sidx [[simdgroup_index_in_threadgroup]],
    ushort lid [[thread_position_in_threadgroup]]
  ) {
  #if LAYER_NORMALIZATION
    uint io_offset = tgid * row_size + lid;
  #else
    uint io_offset = tgid.z * sequence_size * channel_size;
    io_offset += tgid.y * channel_group_size;
    io_offset += tgid.x * sequence_group_size;
  
    uint channel_offset = tgid.y * channel_group_size;
    channel_scale += channel_offset;
    channel_translation += channel_offset;
  #endif
    source += io_offset
    destination += io_offset
  
  #if !LAYER_NORMALIZATION
    threadgroup real scales[channel_group_size];
    threadgroup real translations[channel_group_size];
    for (ushort i = lid; i < channel_group_size; i += threadgroup_size) {
      uint address = tgid.y * channel_group_size + i;
      scales[i] = channel_scale[address];
      translations[i] = channel_translations[address];
    }
    uint saved_address = tgid.y + tgid.z * channel_groups;
  #endif
  
  #if (GROUP_NORMALIZATION && !SAMPLE_POPULATION)
    {
      float mean = saved_mean[saved_address];
      float standard_deviation_reciprocal = saved_standard_deviation_reciprocal[saved_address];
      
    #pragma clang loop unroll(full)
      for (uint i = 0; i < sample_size; i += threadgroup_size) {
        uint2 coords = population_coords(i);
        real scale = scales[coords.x];
        real translation = translations[coords.x];
        
        uint io_offset = coords.y * channel_count + coords.x;
        real source_value = source[io_offset];
        source_value = source_value * scale + translation;
        source_value -= mean;
        source_value *= standard_deviation_reciprocal;
        destination[io_offset] = source_value;
      }
    }
  #else
    {
      real cache_bulk[bulk_size / group_size];
      real cache_padding;
      threadgroup float partials[threadgroup_size / 32];
    #if SAMPLE_POPULATION
      threadgroup uint group_seed[1];
      if (lid == 0) {
        group_seed[0] = tea(batch_seeds[tgid.z], tgid.y);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      
      uint seed = group_seed[0] + lid;
    #endif
      
      float sum = 0;
    #pragma clang loop unroll(full)
      for (uint i = 0; i < bulk_size; i += threadgroup_size) {
    #if SAMPLE_POPULATION
        uint2 coords = population_coords(radinv2(seed + i));
        real scale = scales[coords.x];
        real translation = translations[coords.x];
        real source_value = source[coords.y * channel_count + coords.x];
    #else
        real scale = channel_scales[lid + i];
        real translation = channel_translations[lid + i];
        real source_value = source[i];
    #endif
        source_value = source_value * scale + translation;
        cache_bulk[i / threadgroup_size] = source_value;
        sum += cache_bulk[i / threadgroup_size];
      }
    #if !SAMPLE_POPULATION
      if (padding_size > 0 && lid < padding_size) {
        real scale = channel_scales[lid + bulk_size];
        real translation = channel_translations[lid + bulk_size];
        real source_value = source[bulk_size];
        
        source_value = source_value * scale + translation;
        cache_padding = source[bulk_size];
        sum += cache_padding;
      }
    #endif
      partials[sidx] = simd_sum(sum);
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lid < (threadgroup_size / 32)) {
        float sum = quad_sum(partials[lid]);
        if (threadgroup_size >= 256) {
          sum += simd_shuffle_xor(sum, 4);
        }
        if (threadgroup_size >= 512) {
          sum += simd_shuffle_xor(sum, 8);
        }
        partials[lid] = sum * (1.0 / float(sample_count));
      }
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float mean = partials[sidx];
      float variance = 0;
    #pragma clang loop unroll(full)
      for (uint i = 0; i < bulk_size; i += threadgroup_size) {
        cache_bulk[i / threadgroup_size] -= mean;
        real deviation = cache_bulk[i / threadgroup_size];
        variance = fma(deviation, deviation, variance);
      }
    #if !SAMPLE_POPULATION
      if (padding_size > 0 && lid < padding_size) {
        cache_padding -= mean;
        variance += fma(cache_padding, cache_padding, variance);
      }
    #endif
      partials[sidx] = simd_sum(variance);
      
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (lid < (threadgroup_size / 32)) {
        float variance = quad_sum(partials[lid]);
        if (threadgroup_size >= 256) {
          variance += simd_shuffle_xor(variance, 4);
        }
        if (threadgroup_size >= 512) {
          variance += simd_shuffle_xor(variance, 8);
        }
        
        variance *= 1.0 / float(sample_count);
      #if !SAMPLE_POPULATION
        float standard_deviation_reciprocal = rsqrt(variance);
        partials[lid] = standard_deviation_reciprocal;
        uint saved_address = tgid * sequence_count;
      #endif
        saved_mean[saved_address] = mean;
        saved_standard_deviation_reciprocal[saved_address] = standard_deviation_reciprocal;
      }
  
    #if !SAMPLE_POPULATION
      threadgroup_barrier(mem_flags::mem_threadgroup);
      float standard_deviation_reciprocal = partials[sidx];
    #pragma clang loop unroll(full)
      for (uint i = 0; i < bulk_size; i += threadgroup_size) {
        real deviation = cache_bulk[i / threadgroup_size];
        destination[i] = deviation * standard_deviation_reciprocal;
      }
      if (padding_size > 0 && lid < padding_size) {
        destination[bulk_size] = cache_padding * standard_deviation_reciprocal;
      }
    #endif
    }
  #endif
  }
  )";
  
  std::string defines = "";
  if (hash.data_type == MTL::DataTypeFloat) {
    defines += std::string("typedef float real;");
    defines += "\n";
  } else {
    defines += std::string("typedef half real;");
    defines += "\n";
  }
  
  defines += "constant uint channel_count = ";
  defines += std::to_string(hash.channel_count) + ";";
  defines += "\n";
  
  defines += "constant uint channel_groups = ";
  defines += std::to_string(hash.channel_groups) + ";";
  defines += "\n";
  
  defines += "constant uint sequence_count = ";
  defines += std::to_string(hash.sequence_count) + ";";
  defines += "\n";
  
  uint16_t threadgroup_size;
  if (hash.is_layer_normalization) {
    CCV_NNC_MFA_PRECONDITION(hash.channel_groups == 1);
    defines += "constant ushort sample_count = ";
    defines += std::to_string(hash.channel_count) + ";";
    defines += "\n";
    
    if (hash.channel_count <= 384) {
      threadgroup_size = 128;
    } else {
      threadgroup_size = 256;
    }
  } else {
    CCV_NNC_MFA_PRECONDITION(hash.channel_count % hash.channel_groups == 0);
    threadgroup_size = 512;
    
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
    
    // Allocate 64 KB of registers per threadgroup.
    uint16_t thread_bytes = uint32_t(64 * 1024) / uint32_t(threadgroup_size);
    uint16_t thread_elements = thread_bytes / data_type_size;
    uint16_t sample_count = thread_elements * threadgroup_size;
    
    uint32_t channel_group_size = hash.channel_count / hash.channel_groups;
    uint32_t population_count = hash.sequence_count * channel_group_size;
    
    // Aim to sample no more than 25% of the population.
    if (sample_count >= 2 * population_count) {
      sample_count /= 8;
    } else if (sample_count >= population_count) {
      sample_count /= 4;
    } else if (sample_count * 2 >= population_count) {
      sample_count /= 2;
    }
    
    defines += "constant ushort sample_count = ";
    defines += std::to_string(sample_count) + ";";
    defines += "\n";
  }
  
  defines += "constant ushort threadgroup_size = ";
  defines += std::to_string(threadgroup_size) + ";";
  defines += "\n";
  this->group_size = MTL::Size(threadgroup_size, 1, 1);
  
  for (int i = 0; i < 2; ++i) {
    NS::SharedPtr<MTL::ComputePipelineState>* pso;
    std::string macro;
    if (i == 0) {
      if (hash.is_layer_normalization) {
        continue;
      }
      macro = "SAMPLE_POPULATION";
      pso = &sampling_pso;
    } else {
      if (hash.is_layer_normalization) {
        macro = "LAYER_NORMALIZATION";
      } else {
        macro = "GROUP_NORMALIZATION";
      }
      pso = &normalization_pso;
    }
    
    std::string source = defines;
    source += "#define " + macro + "1";
    source += "\n";
    source += shader;
    
    NS::Error *error;
    auto swift_source = NS::String::string(source.c_str(),
     NS::UTF8StringEncoding);
    auto library = NS::TransferPtr(context->device->newLibrary(swift_source, nullptr, &error));
    CCV_NNC_MFA_CHECK_ERROR(error)
    
    auto swift_name = NS::String::string("normalization", NS::UTF8StringEncoding);
    auto function = NS::TransferPtr(context->library->newFunction(swift_name));
    CCV_NNC_MFA_PRECONDITION(function.get() != nullptr)
    
    *pso = NS::TransferPtr(context->device->newComputePipelineState(function.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error)
  }
  
  pool->drain();
}
