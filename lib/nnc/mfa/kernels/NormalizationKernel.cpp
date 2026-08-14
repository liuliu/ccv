#include "NormalizationKernel.hpp"
#include "../ccv_nnc_mfa.hpp"

#include <iomanip>
#include <limits>
#include <sstream>
#include <string>

namespace {

std::string high_precision_to_string(const float value)
{
  std::ostringstream oss;
  oss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
  return oss.str();
}

}

NormalizationKernel::NormalizationKernel(NormalizationKernelDescriptor descriptor, MTL::Device* const device) {
  dataType = descriptor.dataType;
  channelCount = descriptor.channelCount;
  channelGroups = descriptor.channelGroups;
  sequenceCount = descriptor.sequenceCount;
  epsilon = descriptor.epsilon;
  scale = descriptor.scale;
  elementwiseAffine = descriptor.elementwiseAffine;
  scaleTranslationBatched = descriptor.scaleTranslationBatched;
  normalizationType = descriptor.normalizationType;
  reuseSavedStatistics = descriptor.reuseSavedStatistics;
  loadM = descriptor.loadM;
  srcBatchStride = descriptor.srcBatchStride;
  dstBatchStride = descriptor.dstBatchStride;

  // FlashNorm not supported for group normalization yet.
  CCV_NNC_MFA_PRECONDITION(normalizationType == 0 || normalizationType == 2);
  CCV_NNC_MFA_PRECONDITION(dataType == MTL::DataTypeFloat || dataType == MTL::DataTypeHalf || dataType == MTL::DataTypeBFloat);
  CCV_NNC_MFA_PRECONDITION(channelGroups == 1);

  const uint16_t threadgroup_size = (channelCount <= 384) ? 128 : 256;
  gridSize = MTL::Size(sequenceCount, 1, 1);
  groupSize = MTL::Size(threadgroup_size, 1, 1);

  source = createSource();

  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
}

std::string NormalizationKernel::createSource() const noexcept {
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
  ushort simd_lane [[thread_index_in_simdgroup]],
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
  accumulator cache_bulk[cache_bulk_size > 0 ? cache_bulk_size : 1];
  accumulator cache_padding;
  threadgroup float partials[threadgroup_size / 32];
  
#pragma clang loop unroll(full)
  for (uint i = 0; i < bulk_size; i += threadgroup_size) {
    cache_bulk[i / threadgroup_size] = accumulator(source[i]);
  }
  if (padding_size > 0 && lid < padding_size) {
    cache_padding = accumulator(source[bulk_size]);
  }
#if REUSE_SAVED_STATISTICS
  float standard_deviation_reciprocal = accumulator(saved_standard_deviation_reciprocal[threadgroup_index]);
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
  const float simd_variance = simd_sum(variance);
  if (simd_lane == 0) {
    partials[sidx] = simd_variance;
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sidx == 0) {
    variance = simd_lane < (threadgroup_size / 32) ? partials[simd_lane] : 0;
    variance = quad_sum(variance);
    if (threadgroup_size >= 256) {
      variance += simd_shuffle_xor(variance, 4);
    }
    if (simd_lane == 0) {
      variance = variance / float(sample_count) + epsilon;
      const float standard_deviation_reciprocal = rsqrt(variance);
      partials[0] = standard_deviation_reciprocal;
      saved_standard_deviation_reciprocal[threadgroup_index] = real(standard_deviation_reciprocal);
    }
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float standard_deviation_reciprocal = partials[0];
#endif

#pragma clang loop unroll(full)
  for (uint i = 0; i < bulk_size; i += threadgroup_size) {
    accumulator deviation = cache_bulk[i / threadgroup_size];
    deviation *= standard_deviation_reciprocal;

#if ELEMENTWISE_AFFINE
    accumulator scale = accumulator(channel_scales[i]);
    destination[i] = real(FINAL_SCALE(scale * deviation));
#else
    destination[i] = real(FINAL_SCALE(deviation));
#endif
  }
  if (padding_size > 0 && lid < padding_size) {
    accumulator deviation = cache_padding;
    deviation *= standard_deviation_reciprocal;

#if ELEMENTWISE_AFFINE
    accumulator scale = accumulator(channel_scales[bulk_size]);
    destination[bulk_size] = real(FINAL_SCALE(scale * deviation));
#else
    destination[bulk_size] = real(FINAL_SCALE(deviation));
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
  ushort simd_lane [[thread_index_in_simdgroup]],
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
  accumulator cache_bulk[cache_bulk_size > 0 ? cache_bulk_size : 1];
  accumulator cache_padding;
  threadgroup float partials[threadgroup_size / 32];
  
  float sum = 0;
#pragma clang loop unroll(full)
  for (uint i = 0; i < bulk_size; i += threadgroup_size) {
    cache_bulk[i / threadgroup_size] = accumulator(source[i]);
    sum += cache_bulk[i / threadgroup_size];
  }
  if (padding_size > 0 && lid < padding_size) {
    cache_padding = accumulator(source[bulk_size]);
    sum += cache_padding;
  }
#if REUSE_SAVED_STATISTICS
  float mean = accumulator(saved_mean[threadgroup_index]);
  float standard_deviation_reciprocal = accumulator(saved_standard_deviation_reciprocal[threadgroup_index]);
#else
  const float simd_sum_value = simd_sum(sum);
  if (simd_lane == 0) {
    partials[sidx] = simd_sum_value;
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sidx == 0) {
    sum = simd_lane < (threadgroup_size / 32) ? partials[simd_lane] : 0;
    sum = quad_sum(sum);
    if (threadgroup_size >= 256) {
      sum += simd_shuffle_xor(sum, 4);
    }
    if (simd_lane == 0) {
      partials[0] = sum / float(sample_count);
    }
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mean = partials[0];
  float variance = 0;
#pragma clang loop unroll(full)
  for (ushort slot = 0; slot < cache_bulk_size; ++slot) {
    float centered = float(cache_bulk[slot]) - mean;
    variance += centered * centered;
    cache_bulk[slot] = accumulator(centered);
  }
  if (padding_size > 0 && lid < padding_size) {
    cache_padding -= mean;
    variance += cache_padding * cache_padding;
  }
  const float simd_variance = simd_sum(variance);
  if (simd_lane == 0) {
    partials[sidx] = simd_variance;
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sidx == 0) {
    variance = simd_lane < (threadgroup_size / 32) ? partials[simd_lane] : 0;
    variance = quad_sum(variance);
    if (threadgroup_size >= 256) {
      variance += simd_shuffle_xor(variance, 4);
    }
    if (simd_lane == 0) {
      variance = variance / float(sample_count) + epsilon;
      const float standard_deviation_reciprocal = rsqrt(variance);
      partials[0] = standard_deviation_reciprocal;
      saved_mean[threadgroup_index] = real(mean);
      saved_standard_deviation_reciprocal[threadgroup_index] = real(standard_deviation_reciprocal);
    }
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float standard_deviation_reciprocal = partials[0];
#endif

#pragma clang loop unroll(full)
  for (uint i = 0; i < bulk_size; i += threadgroup_size) {
    accumulator deviation = cache_bulk[i / threadgroup_size];
    deviation *= standard_deviation_reciprocal;

#if ELEMENTWISE_AFFINE
    accumulator scale = accumulator(channel_scales[i]);
    accumulator translation = accumulator(channel_translations[i]);
    destination[i] = real(FINAL_SCALE(scale * deviation + translation));
#else
    destination[i] = real(FINAL_SCALE(deviation));
#endif
  }
  if (padding_size > 0 && lid < padding_size) {
    accumulator deviation = cache_padding;
    deviation *= standard_deviation_reciprocal;

#if ELEMENTWISE_AFFINE
    accumulator scale = accumulator(channel_scales[bulk_size]);
    accumulator translation = accumulator(channel_translations[bulk_size]);
    destination[bulk_size] = real(FINAL_SCALE(scale * deviation + translation));
#else
    destination[bulk_size] = real(FINAL_SCALE(deviation));
#endif
  }
}
)";

  std::string defines = "";
  if (dataType == MTL::DataTypeFloat) {
    defines += std::string("typedef float real;");
    defines += "\n";
    defines += std::string("typedef real accumulator;");
    defines += "\n";
  } else if (dataType == MTL::DataTypeHalf) {
    defines += std::string("typedef half real;");
    defines += "\n";
    defines += std::string("typedef real accumulator;");
    defines += "\n";
  } else {
    defines += std::string("typedef bfloat real;");
    defines += "\n";
    defines += std::string("typedef float accumulator;");
    defines += "\n";
  }

  defines += "constant uint channel_count = ";
  defines += std::to_string(channelCount) + ";";
  defines += "\n";

  if (!loadM) {
    defines += "constant uint sequence_count = ";
    defines += std::to_string(sequenceCount) + ";";
    defines += "\n";
  }

  defines += "constant float epsilon = ";
  defines += high_precision_to_string(epsilon) + ";";
  defines += "\n";

  if (scale != 1) {
    defines += "constant float final_scale = ";
    defines += high_precision_to_string(scale) + ";";
    defines += "\n";
    defines += "#define FINAL_SCALE(value) (final_scale * (value))";
    defines += "\n";
  } else {
    defines += "#define FINAL_SCALE(value) (value)";
    defines += "\n";
  }

  defines += "constant ushort sample_count = ";
  defines += std::to_string(channelCount) + ";";
  defines += "\n";

  defines += "constant uint src_batch_stride = ";
  defines += std::to_string(srcBatchStride) + ";";
  defines += "\n";

  defines += "constant uint dst_batch_stride = ";
  defines += std::to_string(dstBatchStride) + ";";
  defines += "\n";

  defines += "constant ushort threadgroup_size = ";
  defines += std::to_string(groupSize.width) + ";";
  defines += "\n";

  if (elementwiseAffine) {
    defines += "#define ELEMENTWISE_AFFINE 1";
    defines += "\n";
  } else {
    defines += "#define ELEMENTWISE_AFFINE 0";
    defines += "\n";
  }

  if (scaleTranslationBatched) {
    defines += "#define SCALE_TRANSLATION_BATCHED 1";
    defines += "\n";
  }

  if (reuseSavedStatistics) {
    defines += "#define REUSE_SAVED_STATISTICS 1";
    defines += "\n";
  }

  std::string shader;
  if (normalizationType == 0) {
    defines += "#define LAYER_NORMALIZATION 1\n";
    shader = defines + norm_shader;
  } else {
    defines += "#define RMSNORM 1\n";
    shader = defines + rmsnorm_shader;
  }
  if (loadM) {
    const std::string::size_type argumentPosition = shader.find("  uint3 tgid [[threadgroup_position_in_grid]]");
    CCV_NNC_MFA_PRECONDITION(argumentPosition != std::string::npos);
    shader.insert(argumentPosition, "  const device uint *loadM [[buffer(11)]],\n");
    const std::string::size_type sequenceCountPosition = shader.find("  uint threadgroup_index = tgid.z * sequence_count + tgid.x;");
    CCV_NNC_MFA_PRECONDITION(sequenceCountPosition != std::string::npos);
    shader.insert(sequenceCountPosition, "  const uniform<uint> sequence_count = make_uniform(loadM[0]);\n");
  }
  return shader;
}
