#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>
#include <sstream>
#include <iomanip>

// MARK: - C

void ccv_nnc_mfa_prepare_normalization(mfa::context* context, ccv_nnc_mfa_normalization_params_t params)
{
  context->normalization_cache.prepare(context, mfa::normalization::hash(params));
}

void ccv_nnc_mfa_encode_normalization(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_normalization_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  mfa::normalization::hash hash(params);
  auto iterator = context->normalization_cache.map.find(hash);
  if (iterator == context->normalization_cache.map.end()) {
    mfa::precondition_failure("Normalization hash not cached.", __LINE__, __FILE__, __FUNCTION__);
  }
  
  auto* pipeline = iterator->second;
  auto encoder = command_batch->startCommand();
  
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 6 || num_tensors == 4 || num_tensors == 3);
  
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
  for (uint16_t operand = 0; operand < 2; ++operand) {
    uint32_t* batch_dims;
    if (operand == 0) {
      batch_dims = params.batch_dims_data;
    } else if (operand == 1) {
      if (params.scale_translation_batched) {
        batch_dims = params.batch_dims_scale_translation;
      } else {
        continue;
      }
    }
    
    for (int i = 0; i < CCV_NNC_MAX_DIM_ALLOC; ++i) {
      if (batch_dims[i] == 0) {
        break;
      }
      num_batch_dims[operand] += 1;
      batch_sizes[operand] *= batch_dims[i];
    }
    
    bool dims_match_data = true;
    if (num_batch_dims[0] != num_batch_dims[operand]) {
      dims_match_data = false;
    } else if (batch_sizes[0] != batch_sizes[operand]) {
      dims_match_data = false;
    } else {
      for (int i = 0; i < CCV_NNC_MAX_DIM_ALLOC; ++i) {
        if (params.batch_dims_data[i] != batch_dims[i]) {
          dims_match_data = false;
        }
      }
    }
    
    if (!dims_match_data) {
      CCV_NNC_MFA_PRECONDITION(batch_sizes[operand] == 1);
    }
  }
  
  if (params.scale_translation_batched) {
    uint64_t byte_stride_scale_translation = 0;
    if (batch_sizes[1] > 1) {
      byte_stride_scale_translation = params.channel_count * data_type_size;
    }
    
    simd::ulong4 scale_translation_offsets[batch_sizes[0]];
    for (int i = 0; i < batch_sizes[0]; ++i) {
      scale_translation_offsets[i] = simd::ulong4 {
        i * byte_stride_scale_translation,
        i * byte_stride_scale_translation,
        0,
        0,
      };
    }
    encoder->setBytes(scale_translation_offsets, batch_sizes[0] * 32, 10);
  }
  
  encoder->setComputePipelineState(pipeline->normalization_pso.get());
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
  if (num_tensors == 6) { // This is for layer norm.
    if (params.reuse_saved_statistics) {
      encoder->useResource(tensors[2], MTL::ResourceUsageRead);
      encoder->useResource(tensors[3], MTL::ResourceUsageRead);
    } else {
      encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
      encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
    }
    encoder->useResource(tensors[4], MTL::ResourceUsageRead);
    encoder->useResource(tensors[5], MTL::ResourceUsageRead);
  } else if (num_tensors == 4) {
    if (params.elementwise_affine) { // This is for RMSNorm.
      encoder->useResource(tensors[3], MTL::ResourceUsageRead);
      if (params.reuse_saved_statistics) {
        encoder->useResource(tensors[2], MTL::ResourceUsageRead);
      } else {
        encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
      }
    } else { // This is for layer norm without elementwise affine.
      if (params.reuse_saved_statistics) {
        encoder->useResource(tensors[2], MTL::ResourceUsageRead);
        encoder->useResource(tensors[3], MTL::ResourceUsageRead);
      } else {
        encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
        encoder->useResource(tensors[3], MTL::ResourceUsageWrite);
      }
    }
  } else { // This is for RMSNorm.
    if (params.reuse_saved_statistics) {
      encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    } else {
      encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
    }
    encoder->useResource(tensors[2], MTL::ResourceUsageRead);
  }
  
  auto grid_size = pipeline->grid_size;
  grid_size.depth = batch_sizes[0];
  CCV_NNC_MFA_PRECONDITION(grid_size.depth > 0);
  encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
  command_batch->finishCommand(encoder);
}

// MARK: - C++

mfa::normalization::hash::hash(ccv_nnc_mfa_normalization_params_t params) {
  data_type = params.data_type;
  channel_count = params.channel_count;
  channel_groups = params.channel_groups;
  sequence_count = params.sequence_count;
  epsilon = params.epsilon;
  elementwise_affine = params.elementwise_affine;
  scale_translation_batched = params.scale_translation_batched;
  normalization_type = (Type)params.normalization_type;
  reuse_saved_statistics = params.reuse_saved_statistics;
  src_batch_stride = params.src_batch_stride;
  dst_batch_stride = params.dst_batch_stride;
}

bool mfa::normalization::hash::operator==(const mfa::normalization::hash& hash) const {
  return
  (data_type == hash.data_type) &&
  (channel_count == hash.channel_count) &&
  (channel_groups == hash.channel_groups) &&
  (sequence_count == hash.sequence_count) &&
  (epsilon == hash.epsilon) &&
  (elementwise_affine == hash.elementwise_affine) &&
  (scale_translation_batched == hash.scale_translation_batched) &&
  (normalization_type == hash.normalization_type) &&
  (reuse_saved_statistics == hash.reuse_saved_statistics) &&
  (src_batch_stride == hash.src_batch_stride) &&
  (dst_batch_stride == hash.dst_batch_stride);
}

std::ostream& operator<<(std::ostream& os, const mfa::normalization::hash& hash) {
  os << "mfa::normalization::hash {";
  os << " .data_type = " << hash.data_type << ',';
  os << " .channel_count = " << hash.channel_count << ',';
  os << " .channel_groups = " << hash.channel_groups << ',';
  os << " .sequence_count = " << hash.sequence_count << ',';
  os << " .epsilon = " << double(hash.epsilon) << ',';
  os << " .elementwise_affine = " << bool(hash.elementwise_affine) << ',';
  os << " .scale_translation_batched = " << bool(hash.scale_translation_batched) << ',';
  os << " .normalization_type = " << hash.normalization_type << ',';
  os << " .reuse_saved_statistics = " << bool(hash.reuse_saved_statistics) << ",";
  os << " .src_batch_stride = " << hash.src_batch_stride << ',';
  os << " .dst_batch_stride = " << hash.dst_batch_stride << ' ';
  os << "}";
  return os;
}

static std::string high_precision_to_string(float value) {
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return oss.str();
}

std::size_t std::hash<mfa::normalization::hash>::operator()(const mfa::normalization::hash& hash) const noexcept {
  std::size_t seed = 0;
  using namespace mfa::hash;
  combine_64(seed, hash.data_type);
  combine_64(seed, pack_64(simd::uint2 { hash.channel_count, hash.channel_groups }));
  combine_64(seed, pack_64(simd::uint2 { hash.sequence_count, *reinterpret_cast<const uint32_t*>(&hash.epsilon) }));
  combine_32(seed, pack_32(simd::uchar4 { hash.elementwise_affine, hash.scale_translation_batched, hash.normalization_type, hash.reuse_saved_statistics }));
  combine_64(seed, pack_64(simd::uint2 { hash.src_batch_stride, hash.dst_batch_stride }));
  return seed;
}

mfa::normalization::pipeline::pipeline(mfa::context* context, mfa::normalization::hash hash) {
  // FlashNorm not supported for group normalization yet.
  CCV_NNC_MFA_PRECONDITION(hash.normalization_type == Type::layer_norm || hash.normalization_type == Type::rmsnorm);
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf))
  
  auto* pool = NS::AutoreleasePool::alloc()->init();

  std::string rmsnorm_shader = R"(
constant uint bulk_size = sample_count / threadgroup_size * threadgroup_size;
constant uint padding_size = sample_count - bulk_size;

#include <metal_stdlib>
using namespace metal;

kernel void normalization(
  device real *source [[buffer(0)]],
  device real *destination [[buffer(1)]],
  device real *saved_standard_deviation_reciprocal [[buffer(2)]],
#if ELEMENTWISE_AFFINE
  device real *channel_scales [[buffer(3)]],
  
#if SCALE_TRANSLATION_BATCHED
  constant ulong4 *scale_translation_offsets [[buffer(10)]],
#endif
#endif
  
  uint3 tgid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  uint lid [[thread_index_in_threadgroup]]
) {
  uint threadgroup_index = tgid.z * sequence_count + tgid.x;
  {
    uint io_offset = tgid.x * channel_count + lid;
    source += tgid.z * src_batch_stride + io_offset;
    destination += tgid.z * dst_batch_stride + io_offset;
  }
#if ELEMENTWISE_AFFINE
  channel_scales += lid;

#if SCALE_TRANSLATION_BATCHED
  {
    ulong2 offsets = scale_translation_offsets[tgid.z].xy;
    channel_scale = (device real*)((device uchar*)channel_scale + offsets[0]);
    channel_translation = (device real*)((device uchar*)channel_translation + offsets[1]);
  }
#endif
#endif
  
  const uint cache_bulk_size = bulk_size / threadgroup_size;
  real cache_bulk[cache_bulk_size > 0 ? cache_bulk_size : 1];
  real cache_padding;
  threadgroup float partials[threadgroup_size / 32];
  
#pragma clang loop unroll(full)
  for (uint i = 0; i < bulk_size; i += threadgroup_size) {
    cache_bulk[i / threadgroup_size] = source[i];
  }
  if (padding_size > 0 && lid < padding_size) {
    cache_padding = source[bulk_size];
  }
#if REUSE_SAVED_STATISTICS
  float standard_deviation_reciprocal = saved_standard_deviation_reciprocal[threadgroup_index];
#else
  float variance = 0;
#pragma clang loop unroll(full)
  for (ushort slot = 0; slot < cache_bulk_size; ++slot) {
    float centered = float(cache_bulk[slot]);
    variance += centered * centered;
  }
  if (padding_size > 0 && lid < padding_size) {
    variance += cache_padding * cache_padding;
  }
  partials[sidx] = simd_sum(variance);
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < (threadgroup_size / 32)) {
    float variance = quad_sum(partials[lid]);
    if (threadgroup_size >= 256) {
      variance += simd_shuffle_xor(variance, 4);
    }
    variance = variance / float(sample_count) + epsilon;
    
    float standard_deviation_reciprocal = rsqrt(variance);
    partials[lid] = standard_deviation_reciprocal;
    
    saved_standard_deviation_reciprocal[threadgroup_index] =  standard_deviation_reciprocal;
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float standard_deviation_reciprocal = partials[sidx];
#endif

#pragma clang loop unroll(full)
  for (uint i = 0; i < bulk_size; i += threadgroup_size) {
    real deviation = cache_bulk[i / threadgroup_size];
    deviation *= standard_deviation_reciprocal;

#if ELEMENTWISE_AFFINE
    real scale = channel_scales[i];
    destination[i] = scale * deviation;
#else
    destination[i] = deviation;
#endif
  }
  if (padding_size > 0 && lid < padding_size) {
    real deviation = cache_padding;
    deviation *= standard_deviation_reciprocal;

#if ELEMENTWISE_AFFINE
    real scale = channel_scales[bulk_size];
    destination[bulk_size] = scale * deviation;
#else
    destination[bulk_size] = deviation;
#endif
  }
}
)";
  
  std::string norm_shader = R"(
constant uint bulk_size = sample_count / threadgroup_size * threadgroup_size;
constant uint padding_size = sample_count - bulk_size;

#include <metal_stdlib>
using namespace metal;

kernel void normalization(
  device real *source [[buffer(0)]],
  device real *destination [[buffer(1)]],
  device real *saved_mean [[buffer(2)]],
  device real *saved_standard_deviation_reciprocal [[buffer(3)]],
#if ELEMENTWISE_AFFINE
  device real *channel_scales [[buffer(4)]],
  device real *channel_translations [[buffer(5)]],
  
#if SCALE_TRANSLATION_BATCHED
  constant ulong4 *scale_translation_offsets [[buffer(10)]],
#endif
#endif
  
  uint3 tgid [[threadgroup_position_in_grid]],
  ushort sidx [[simdgroup_index_in_threadgroup]],
  uint lid [[thread_index_in_threadgroup]]
) {
  uint threadgroup_index = tgid.z * sequence_count + tgid.x;
  {
    uint io_offset = tgid.x * channel_count + lid;
    source += tgid.z * src_batch_stride + io_offset;
    destination += tgid.z * dst_batch_stride + io_offset;
  }
#if ELEMENTWISE_AFFINE
  channel_scales += lid;
  channel_translations += lid;

#if SCALE_TRANSLATION_BATCHED
  {
    ulong2 offsets = scale_translation_offsets[tgid.z].xy;
    channel_scale = (device real*)((device uchar*)channel_scale + offsets[0]);
    channel_translation = (device real*)((device uchar*)channel_translation + offsets[1]);
  }
#endif
#endif
  
  const uint cache_bulk_size = bulk_size / threadgroup_size;
  real cache_bulk[cache_bulk_size > 0 ? cache_bulk_size : 1];
  real cache_padding;
  threadgroup float partials[threadgroup_size / 32];
  
  float sum = 0;
#pragma clang loop unroll(full)
  for (uint i = 0; i < bulk_size; i += threadgroup_size) {
    cache_bulk[i / threadgroup_size] = source[i];
    sum += cache_bulk[i / threadgroup_size];
  }
  if (padding_size > 0 && lid < padding_size) {
    cache_padding = source[bulk_size];
    sum += cache_padding;
  }
#if REUSE_SAVED_STATISTICS
  float mean = saved_mean[threadgroup_index];
  float standard_deviation_reciprocal = saved_standard_deviation_reciprocal[threadgroup_index];
#else
  partials[sidx] = simd_sum(sum);
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < (threadgroup_size / 32)) {
    float sum = quad_sum(partials[lid]);
    if (threadgroup_size >= 256) {
      sum += simd_shuffle_xor(sum, 4);
    }
    partials[lid] = sum / float(sample_count);
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mean = partials[sidx];
  float variance = 0;
#pragma clang loop unroll(full)
  for (ushort slot = 0; slot < cache_bulk_size; ++slot) {
    float centered = float(cache_bulk[slot]) - mean;
    variance += centered * centered;
    cache_bulk[slot] = real(centered);
  }
  if (padding_size > 0 && lid < padding_size) {
    cache_padding -= mean;
    variance += cache_padding * cache_padding;
  }
  partials[sidx] = simd_sum(variance);
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < (threadgroup_size / 32)) {
    float variance = quad_sum(partials[lid]);
    if (threadgroup_size >= 256) {
      variance += simd_shuffle_xor(variance, 4);
    }
    variance = variance / float(sample_count) + epsilon;
    
    float standard_deviation_reciprocal = rsqrt(variance);
    partials[lid] = standard_deviation_reciprocal;
    
    saved_mean[threadgroup_index] = mean;
    saved_standard_deviation_reciprocal[threadgroup_index] =  standard_deviation_reciprocal;
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float standard_deviation_reciprocal = partials[sidx];
#endif

#pragma clang loop unroll(full)
  for (uint i = 0; i < bulk_size; i += threadgroup_size) {
    real deviation = cache_bulk[i / threadgroup_size];
    deviation *= standard_deviation_reciprocal;

#if ELEMENTWISE_AFFINE
    real scale = channel_scales[i];
    real translation = channel_translations[i];
    destination[i] = scale * deviation + translation;
#else
    destination[i] = deviation;
#endif
  }
  if (padding_size > 0 && lid < padding_size) {
    real deviation = cache_padding;
    deviation *= standard_deviation_reciprocal;

#if ELEMENTWISE_AFFINE
    real scale = channel_scales[bulk_size];
    real translation = channel_translations[bulk_size];
    destination[bulk_size] = scale * deviation + translation;
#else
    destination[bulk_size] = deviation;
#endif
  }
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
  
  defines += "constant uint sequence_count = ";
  defines += std::to_string(hash.sequence_count) + ";";
  defines += "\n";
  
  defines += "constant float epsilon = ";
  defines += high_precision_to_string(hash.epsilon) + ";";
  defines += "\n";
  
  uint16_t threadgroup_size;
  if (hash.normalization_type == Type::layer_norm || hash.normalization_type == Type::rmsnorm) {
    CCV_NNC_MFA_PRECONDITION(hash.channel_groups == 1);
    defines += "constant ushort sample_count = ";
    defines += std::to_string(hash.channel_count) + ";";
    defines += "\n";
    
    if (hash.channel_count <= 384) {
      threadgroup_size = 128;
    } else {
      threadgroup_size = 256;
    }
    
    this->grid_size = MTL::Size(hash.sequence_count, 1, 1);
  } else {
    CCV_NNC_MFA_PRECONDITION(hash.channel_count % hash.channel_groups == 0);

    threadgroup_size = 512;
    
    uint16_t sample_count = 16384;
    uint32_t channel_group_size = hash.channel_count / hash.channel_groups;
    uint32_t population_count = hash.sequence_count * channel_group_size;
    
    // Aim to sample no more than 50% of the population, a tradeoff between
    // sampling cost and accuracy. This only kicks in for asymptotically small
    // population sizes, such as (16 * 16) / (1280 / 32).
    //
    // Such sizes could probably fit inside SRAM, but the cost of maintaining
    // a separate code path for these cases outweighs the benefits.
    if (sample_count >= 4 * population_count) {
      sample_count /= 8;
    } else if (sample_count >= 2 * population_count) {
      sample_count /= 4;
    } else if (sample_count >= population_count) {
      sample_count /= 2;
    }

    defines += "constant uint channel_groups = ";
    defines += std::to_string(hash.channel_groups) + ";";
    defines += "\n";
    
    defines += "constant ushort sample_count = ";
    defines += std::to_string(sample_count) + ";";
    defines += "\n";
    
    uint x_dim = (hash.sequence_count + sample_count - 1) / sample_count * sample_count;
    this->grid_size = MTL::Size(x_dim, hash.channel_groups, 1);
  }

  defines += "constant uint src_batch_stride = ";
  defines += std::to_string(hash.src_batch_stride) + ";";
  defines += "\n";

  defines += "constant uint dst_batch_stride = ";
  defines += std::to_string(hash.dst_batch_stride) + ";";
  defines += "\n";
  
  defines += "constant ushort threadgroup_size = ";
  defines += std::to_string(threadgroup_size) + ";";
  defines += "\n";
  this->group_size = MTL::Size(threadgroup_size, 1, 1);
  
  if (hash.elementwise_affine) {
    defines += "#define ELEMENTWISE_AFFINE 1";
    defines += "\n";
  } else {
    defines += "#define ELEMENTWISE_AFFINE 0";
    defines += "\n";
  }
  
  if (hash.scale_translation_batched) {
    defines += "#define SCALE_TRANSLATION_BATCHED 1";
    defines += "\n";
  }
  
  if (hash.reuse_saved_statistics) {
    defines += "#define REUSE_SAVED_STATISTICS 1";
    defines += "\n";
  }
  
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  std::string shader = norm_shader;
  for (int i = 0; i < 2; ++i) {
    NS::SharedPtr<MTL::ComputePipelineState>* pso;
    std::string macro;
    if (i == 0) {
      if (hash.normalization_type != Type::group_norm) {
        continue;
      }
      macro = "SAMPLE_POPULATION";
      pso = &sampling_pso;
    } else {
      if (hash.normalization_type == Type::layer_norm) {
        macro = "LAYER_NORMALIZATION";
      } else if (hash.normalization_type == Type::group_norm) {
        macro = "GROUP_NORMALIZATION";
	  } else if (hash.normalization_type == Type::rmsnorm) {
        macro = "RMSNORM";
		shader = rmsnorm_shader;
      }
      pso = &normalization_pso;
    }
    
    std::string source = defines;
    source += "#define " + macro + " 1";
    source += "\n";
    if (METAL_LOG_LEVEL(context) >= 4) {
      std::cerr << source << std::endl;
    }
    source += shader;
    
    NS::Error *error = nullptr;
    auto swift_source = NS::String::string(source.c_str(),
     NS::UTF8StringEncoding);
    auto library = NS::TransferPtr(context->device->newLibrary(swift_source, nullptr, &error));
    if (!library) {
      CCV_NNC_MFA_CHECK_ERROR(error)
    }
    
    auto swift_name = NS::String::string("normalization", NS::UTF8StringEncoding);
    auto function = NS::TransferPtr(library->newFunction(swift_name, constants.get(), &error));
    if (!function) {
      CCV_NNC_MFA_CHECK_ERROR(error)
    }
    
    *pso = NS::TransferPtr(context->device->newComputePipelineState(function.get(), &error));
    if (!*pso) {
      CCV_NNC_MFA_CHECK_ERROR(error)
    }
  }
  
  pool->drain();
}
