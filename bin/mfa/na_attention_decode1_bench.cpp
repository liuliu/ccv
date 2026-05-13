#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"

namespace {

using half_float = _Float16;

constexpr MTL::ResourceOptions kPrivateResourceOptions =
    MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked;
constexpr MTL::ResourceOptions kSharedResourceOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;

struct AttentionCase {
  uint32_t batch = 1;
  uint32_t C = 4096;
  uint32_t Hq = 32;
  uint32_t Hk = 8;
  uint32_t D = 128;
};

struct BenchmarkConfig {
  int warmup_iterations = 5;
  int timed_iterations = 40;
  uint32_t simdgroups = 4;
  uint32_t workgroups = 1;
};

struct Stats {
  double average_seconds = 0;
  double median_seconds = 0;
  double best3_seconds = 0;
  double min_seconds = 0;
  double max_seconds = 0;
};

struct DecodeArgs {
  float scale_log2e;
};

struct DecodePipeline {
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> partial_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> reduce_pipeline;
  size_t threadgroup_memory_bytes = 0;
  uint32_t threadgroup_size = 128;
  uint32_t workgroups = 1;
};

size_t q_index(const AttentionCase& attention, uint32_t batch, uint32_t head, uint32_t dim)
{
  return ((size_t)batch * attention.Hq + head) * attention.D + dim;
}

size_t kv_index(const AttentionCase& attention, uint32_t batch, uint32_t column, uint32_t head, uint32_t dim)
{
  return (((size_t)batch * attention.C + column) * attention.Hk + head) * attention.D + dim;
}

size_t o_index(const AttentionCase& attention, uint32_t batch, uint32_t head, uint32_t dim)
{
  return ((size_t)batch * attention.Hq + head) * attention.D + dim;
}

float create_scale(const AttentionCase& attention)
{
  return 1.0f / std::sqrt((float)attention.D);
}

std::vector<float> make_data(size_t count, float scale, uint32_t salt)
{
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i) {
    const float x = (float)((i * 1103515245u + salt * 12345u) & 0xffffu) / 65535.0f;
    values[i] = std::sin((float)i * 0.013f + (float)salt) * scale + (x - 0.5f) * scale;
  }
  return values;
}

std::vector<half_float> encode_fp16(const std::vector<float>& values)
{
  std::vector<half_float> encoded(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    encoded[i] = (half_float)values[i];
  return encoded;
}

std::vector<float> decode_fp16(const void* data, size_t count)
{
  const auto* values = static_cast<const half_float*>(data);
  std::vector<float> decoded(count);
  for (size_t i = 0; i < count; ++i)
    decoded[i] = (float)values[i];
  return decoded;
}

std::vector<float> decode_fp16_vector(const std::vector<half_float>& data)
{
  std::vector<float> decoded(data.size());
  for (size_t i = 0; i < data.size(); ++i)
    decoded[i] = (float)data[i];
  return decoded;
}

void copy_buffer(MTL::CommandQueue* command_queue, MTL::Buffer* source, MTL::Buffer* destination, size_t size)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto blit = NS::TransferPtr(command_buffer->blitCommandEncoder());
  blit->copyFromBuffer(source, 0, destination, 0, size);
  blit->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
}

std::string create_decode_source()
{
  return R"(
#include <metal_stdlib>
using namespace metal;

constant uint C_LEN [[function_constant(0)]];
constant uint Hq [[function_constant(1)]];
constant uint Hk [[function_constant(2)]];
constant uint D_LEN [[function_constant(3)]];
constant uint NSG [[function_constant(4)]];
constant uint NWG [[function_constant(5)]];

struct DecodeArgs {
  float scale_log2e;
};

kernel void decode1_no_scratch(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device half* O [[buffer(3)]],
    constant DecodeArgs& args [[buffer(4)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
  const uint hq = tgid.x;
  const uint batch = tgid.y;
  const uint hk = hq / (Hq / Hk);
  threadgroup float* q_shared = scratch;
  threadgroup float* partial_o = q_shared + D_LEN;
  threadgroup float* partial_s = partial_o + NSG * D_LEN;
  threadgroup float* partial_m = partial_s + NSG;

  if (sgid == 0) {
    device const half* q_row = Q + (batch * Hq + hq) * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      q_shared[d] = (float)q_row[d];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float acc[8];
  for (uint i = 0; i < 8; ++i) {
    acc[i] = 0;
  }
  float row_m = -INFINITY;
  float row_s = 0;

  for (uint c = sgid; c < C_LEN; c += NSG) {
    float dot_acc = 0;
    device const half* k_row = K + ((batch * C_LEN + c) * Hk + hk) * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      dot_acc += q_shared[d] * (float)k_row[d];
    }
    const float score = simd_sum(dot_acc) * args.scale_log2e;
    const float new_m = max(row_m, score);
    const float old_scale = fast::exp2(row_m - new_m);
    const float p = fast::exp2(score - new_m);
    device const half* v_row = V + ((batch * C_LEN + c) * Hk + hk) * D_LEN;
    for (uint i = 0; i < 8; ++i) {
      const uint d = lane_id + i * 32;
      if (d < D_LEN) {
        acc[i] = acc[i] * old_scale + p * (float)v_row[d];
      }
    }
    row_s = row_s * old_scale + p;
    row_m = new_m;
  }

  for (uint i = 0; i < 8; ++i) {
    const uint d = lane_id + i * 32;
    if (d < D_LEN) {
      partial_o[sgid * D_LEN + d] = acc[i];
    }
  }
  if (lane_id == 0) {
    partial_s[sgid] = row_s;
    partial_m[sgid] = row_m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgid == 0) {
    float global_m = -INFINITY;
    for (uint i = 0; i < NSG; ++i) {
      global_m = max(global_m, partial_m[i]);
    }
    float global_s = 0;
    for (uint i = 0; i < NSG; ++i) {
      global_s += partial_s[i] * fast::exp2(partial_m[i] - global_m);
    }
    device half* o_row = O + (batch * Hq + hq) * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      float numerator = 0;
      for (uint i = 0; i < NSG; ++i) {
        numerator += partial_o[i * D_LEN + d] * fast::exp2(partial_m[i] - global_m);
      }
      o_row[d] = (half)(numerator / global_s);
    }
  }
}

kernel void decode1_split_partials(
    device const half* Q [[buffer(0)]],
    device const half* K [[buffer(1)]],
    device const half* V [[buffer(2)]],
    device float* partial [[buffer(3)]],
    constant DecodeArgs& args [[buffer(4)]],
    threadgroup float* scratch [[threadgroup(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]])
{
  const uint hq = tgid.x;
  const uint batch = tgid.y;
  const uint iwg = tgid.z;
  const uint hk = hq / (Hq / Hk);
  threadgroup float* q_shared = scratch;
  threadgroup float* partial_o = q_shared + D_LEN;
  threadgroup float* partial_s = partial_o + NSG * D_LEN;
  threadgroup float* partial_m = partial_s + NSG;

  if (sgid == 0) {
    device const half* q_row = Q + (batch * Hq + hq) * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      q_shared[d] = (float)q_row[d];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float acc[8];
  for (uint i = 0; i < 8; ++i) {
    acc[i] = 0;
  }
  float row_m = -INFINITY;
  float row_s = 0;

  for (uint c = iwg * NSG + sgid; c < C_LEN; c += NWG * NSG) {
    float dot_acc = 0;
    device const half* k_row = K + ((batch * C_LEN + c) * Hk + hk) * D_LEN;
    for (uint d = lane_id; d < D_LEN; d += 32) {
      dot_acc += q_shared[d] * (float)k_row[d];
    }
    const float score = simd_sum(dot_acc) * args.scale_log2e;
    const float new_m = max(row_m, score);
    const float old_scale = fast::exp2(row_m - new_m);
    const float p = fast::exp2(score - new_m);
    device const half* v_row = V + ((batch * C_LEN + c) * Hk + hk) * D_LEN;
    for (uint i = 0; i < 8; ++i) {
      const uint d = lane_id + i * 32;
      if (d < D_LEN) {
        acc[i] = acc[i] * old_scale + p * (float)v_row[d];
      }
    }
    row_s = row_s * old_scale + p;
    row_m = new_m;
  }

  for (uint i = 0; i < 8; ++i) {
    const uint d = lane_id + i * 32;
    if (d < D_LEN) {
      partial_o[sgid * D_LEN + d] = acc[i];
    }
  }
  if (lane_id == 0) {
    partial_s[sgid] = row_s;
    partial_m[sgid] = row_m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sgid == 0) {
    float global_m = -INFINITY;
    for (uint i = 0; i < NSG; ++i) {
      global_m = max(global_m, partial_m[i]);
    }
    float global_s = 0;
    for (uint i = 0; i < NSG; ++i) {
      global_s += partial_s[i] * fast::exp2(partial_m[i] - global_m);
    }
    device float* row = partial + (((batch * Hq + hq) * NWG + iwg) * (D_LEN + 2));
    for (uint d = lane_id; d < D_LEN; d += 32) {
      float numerator = 0;
      for (uint i = 0; i < NSG; ++i) {
        numerator += partial_o[i * D_LEN + d] * fast::exp2(partial_m[i] - global_m);
      }
      row[d] = numerator;
    }
    if (lane_id == 0) {
      row[D_LEN] = global_s;
      row[D_LEN + 1] = global_m;
    }
  }
}

kernel void decode1_split_reduce(
    device const float* partial [[buffer(0)]],
    device half* O [[buffer(1)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    ushort lane_id [[thread_index_in_simdgroup]])
{
  const uint hq = tgid.x;
  const uint batch = tgid.y;
  device const float* base = partial + ((batch * Hq + hq) * NWG * (D_LEN + 2));
  float global_m = -INFINITY;
  float global_s = 0;
  if (NWG <= 32) {
    float local_m = -INFINITY;
    if (lane_id < NWG) {
      local_m = base[lane_id * (D_LEN + 2) + D_LEN + 1];
    }
    global_m = simd_max(local_m);
    float local_s = 0;
    if (lane_id < NWG) {
      local_s = base[lane_id * (D_LEN + 2) + D_LEN] *
          fast::exp2(base[lane_id * (D_LEN + 2) + D_LEN + 1] - global_m);
    }
    global_s = simd_sum(local_s);
  } else {
    for (uint i = 0; i < NWG; ++i) {
      global_m = max(global_m, base[i * (D_LEN + 2) + D_LEN + 1]);
    }
    for (uint i = 0; i < NWG; ++i) {
      global_s += base[i * (D_LEN + 2) + D_LEN] *
          fast::exp2(base[i * (D_LEN + 2) + D_LEN + 1] - global_m);
    }
  }
  device half* o_row = O + (batch * Hq + hq) * D_LEN;
  for (uint d = lane_id; d < D_LEN; d += 32) {
    float numerator = 0;
    for (uint i = 0; i < NWG; ++i) {
      device const float* row = base + i * (D_LEN + 2);
      numerator += row[d] * fast::exp2(row[D_LEN + 1] - global_m);
    }
    o_row[d] = (half)(numerator / global_s);
  }
}
)";
}

DecodePipeline create_decode_pipeline(
    MTL::Device* device,
    const AttentionCase& attention,
    uint32_t simdgroups,
    uint32_t workgroups)
{
  auto source = NS::TransferPtr(NS::String::string(create_decode_source().c_str(), NS::UTF8StringEncoding));
  NS::Error* error = nil;
  auto library = NS::TransferPtr(device->newLibrary(source.get(), nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  constants->setConstantValue(&attention.C, MTL::DataTypeUInt, NS::UInteger(0));
  constants->setConstantValue(&attention.Hq, MTL::DataTypeUInt, NS::UInteger(1));
  constants->setConstantValue(&attention.Hk, MTL::DataTypeUInt, NS::UInteger(2));
  constants->setConstantValue(&attention.D, MTL::DataTypeUInt, NS::UInteger(3));
  constants->setConstantValue(&simdgroups, MTL::DataTypeUInt, NS::UInteger(4));
  constants->setConstantValue(&workgroups, MTL::DataTypeUInt, NS::UInteger(5));
  auto function_name = NS::TransferPtr(NS::String::string("decode1_no_scratch", NS::UTF8StringEncoding));
  auto function = NS::TransferPtr(library->newFunction(function_name.get(), constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);

  DecodePipeline result;
  result.pipeline = std::move(pipeline);
  result.threadgroup_size = 32 * simdgroups;
  result.threadgroup_memory_bytes = ((size_t)attention.D + (size_t)simdgroups * attention.D + (size_t)simdgroups * 2) * sizeof(float);
  result.workgroups = workgroups;
  if (workgroups > 1) {
    auto partial_name = NS::TransferPtr(NS::String::string("decode1_split_partials", NS::UTF8StringEncoding));
    auto partial_function = NS::TransferPtr(library->newFunction(partial_name.get(), constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    result.partial_pipeline = NS::TransferPtr(device->newComputePipelineState(partial_function.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto reduce_name = NS::TransferPtr(NS::String::string("decode1_split_reduce", NS::UTF8StringEncoding));
    auto reduce_function = NS::TransferPtr(library->newFunction(reduce_name.get(), constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    result.reduce_pipeline = NS::TransferPtr(device->newComputePipelineState(reduce_function.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
  }
  return result;
}

double run_decode_once(
    MTL::CommandQueue* command_queue,
    const DecodePipeline& pipeline,
    const DecodeArgs& args,
    const AttentionCase& attention,
    MTL::Buffer* q_buffer,
    MTL::Buffer* k_buffer,
    MTL::Buffer* v_buffer,
    MTL::Buffer* o_buffer,
    MTL::Buffer* partial_buffer)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  if (pipeline.workgroups == 1) {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(pipeline.pipeline.get());
    encoder->setBuffer(q_buffer, 0, 0);
    encoder->setBuffer(k_buffer, 0, 1);
    encoder->setBuffer(v_buffer, 0, 2);
    encoder->setBuffer(o_buffer, 0, 3);
    encoder->setBytes(&args, sizeof(args), 4);
    encoder->setThreadgroupMemoryLength(pipeline.threadgroup_memory_bytes, 0);
    encoder->dispatchThreadgroups(
        MTL::Size(attention.Hq, attention.batch, 1),
        MTL::Size(pipeline.threadgroup_size, 1, 1));
    encoder->endEncoding();
  } else {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(pipeline.partial_pipeline.get());
    encoder->setBuffer(q_buffer, 0, 0);
    encoder->setBuffer(k_buffer, 0, 1);
    encoder->setBuffer(v_buffer, 0, 2);
    encoder->setBuffer(partial_buffer, 0, 3);
    encoder->setBytes(&args, sizeof(args), 4);
    encoder->setThreadgroupMemoryLength(pipeline.threadgroup_memory_bytes, 0);
    encoder->dispatchThreadgroups(
        MTL::Size(attention.Hq, attention.batch, pipeline.workgroups),
        MTL::Size(pipeline.threadgroup_size, 1, 1));
    encoder->endEncoding();

    auto reduce_encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    reduce_encoder->setComputePipelineState(pipeline.reduce_pipeline.get());
    reduce_encoder->setBuffer(partial_buffer, 0, 0);
    reduce_encoder->setBuffer(o_buffer, 0, 1);
    reduce_encoder->dispatchThreadgroups(
        MTL::Size(attention.Hq, attention.batch, 1),
        MTL::Size(32, 1, 1));
    reduce_encoder->endEncoding();
  }
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

void compute_reference(
    const AttentionCase& attention,
    const std::vector<float>& q_values,
    const std::vector<float>& k_values,
    const std::vector<float>& v_values,
    std::vector<float>* output)
{
  output->assign((size_t)attention.batch * attention.Hq * attention.D, 0);
  const float scale = create_scale(attention);
  for (uint32_t batch = 0; batch < attention.batch; ++batch) {
    for (uint32_t hq = 0; hq < attention.Hq; ++hq) {
      const uint32_t hk = hq / (attention.Hq / attention.Hk);
      std::vector<float> scores(attention.C);
      float max_score = -std::numeric_limits<float>::infinity();
      for (uint32_t c = 0; c < attention.C; ++c) {
        float dot = 0;
        for (uint32_t d = 0; d < attention.D; ++d)
          dot += q_values[q_index(attention, batch, hq, d)] * k_values[kv_index(attention, batch, c, hk, d)];
        scores[c] = dot * scale;
        max_score = std::max(max_score, scores[c]);
      }
      double sum = 0;
      for (uint32_t c = 0; c < attention.C; ++c) {
        scores[c] = std::exp(scores[c] - max_score);
        sum += scores[c];
      }
      const double inv_sum = 1.0 / sum;
      for (uint32_t c = 0; c < attention.C; ++c) {
        const float weight = (float)(scores[c] * inv_sum);
        for (uint32_t d = 0; d < attention.D; ++d)
          (*output)[o_index(attention, batch, hq, d)] += weight * v_values[kv_index(attention, batch, c, hk, d)];
      }
    }
  }
}

bool validate_output(const std::vector<float>& reference, const std::vector<float>& actual)
{
  double max_abs = 0;
  double max_rel = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    const double abs = std::fabs((double)reference[i] - actual[i]);
    const double rel = abs / std::max<double>(std::max(std::fabs(reference[i]), std::fabs(actual[i])), 1.0);
    max_abs = std::max(max_abs, abs);
    max_rel = std::max(max_rel, rel);
  }
  std::cout << "validation max_abs_o=" << max_abs << " max_rel_o=" << max_rel << '\n';
  return max_abs <= 2e-2 || max_rel <= 2e-2;
}

template <typename RunOnce>
bool benchmark(const BenchmarkConfig& config, RunOnce&& run_once, Stats* gpu_stats, Stats* wall_stats)
{
  std::vector<double> gpu_samples;
  std::vector<double> wall_samples;
  gpu_samples.reserve(config.timed_iterations);
  wall_samples.reserve(config.timed_iterations);
  for (int i = 0; i < config.warmup_iterations + config.timed_iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    const double gpu_seconds = run_once();
    const auto end = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(end - start).count();
    if (!(gpu_seconds > 0) || std::isnan(gpu_seconds))
      return false;
    if (i >= config.warmup_iterations) {
      gpu_samples.push_back(gpu_seconds);
      wall_samples.push_back(wall_seconds);
    }
  }
  auto compute_stats = [](std::vector<double>* samples, Stats* stats) {
    stats->average_seconds = std::accumulate(samples->begin(), samples->end(), 0.0) / samples->size();
    std::sort(samples->begin(), samples->end());
    stats->median_seconds = (*samples)[samples->size() / 2];
    const size_t best3_count = std::min<size_t>(samples->size(), 3);
    stats->best3_seconds = std::accumulate(samples->begin(), samples->begin() + best3_count, 0.0) / (double)best3_count;
    stats->min_seconds = samples->front();
    stats->max_seconds = samples->back();
  };
  compute_stats(&gpu_samples, gpu_stats);
  compute_stats(&wall_samples, wall_stats);
  return true;
}

void print_stats(const char* label, const Stats& stats)
{
  std::cout << std::fixed
            << label
            << " avg_ms=" << std::setprecision(4) << stats.average_seconds * 1e3
            << " median_ms=" << stats.median_seconds * 1e3
            << " best3_ms=" << stats.best3_seconds * 1e3
            << " min_ms=" << stats.min_seconds * 1e3
            << " max_ms=" << stats.max_seconds * 1e3
            << '\n';
}

} // namespace

int main(int argc, char** argv)
{
  std::cout.setf(std::ios::unitbuf);
  AttentionCase attention;
  BenchmarkConfig config;
  if (argc >= 6) {
    attention.C = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    attention.D = (uint32_t)std::strtoul(argv[2], nullptr, 10);
    attention.batch = (uint32_t)std::strtoul(argv[3], nullptr, 10);
    attention.Hq = (uint32_t)std::strtoul(argv[4], nullptr, 10);
    attention.Hk = (uint32_t)std::strtoul(argv[5], nullptr, 10);
  }
  if (argc >= 8) {
    config.warmup_iterations = std::atoi(argv[6]);
    config.timed_iterations = std::atoi(argv[7]);
  }
  if (argc >= 9)
    config.simdgroups = (uint32_t)std::strtoul(argv[8], nullptr, 10);
  if (argc >= 10)
    config.workgroups = (uint32_t)std::strtoul(argv[9], nullptr, 10);

  if (attention.D == 0 || attention.D > 256 || attention.D % 32 != 0 ||
      attention.Hq == 0 || attention.Hk == 0 || attention.Hq % attention.Hk != 0 ||
      config.simdgroups == 0 || config.simdgroups > 32 ||
      config.workgroups == 0 || config.workgroups > 128) {
    std::cerr << "usage: na_attention_decode1_bench C D B Hq Hk [warmup timed nsg nwg]\n";
    std::cerr << "constraints: D in {32,64,...,256}, Hq divisible by Hk, 1 <= nsg <= 32, 1 <= nwg <= 128\n";
    return 1;
  }

  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    std::cerr << "Metal device unavailable\n";
    pool->drain();
    return 1;
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue) {
    std::cerr << "Metal command queue unavailable\n";
    pool->drain();
    return 1;
  }

  const auto q_values_raw = make_data((size_t)attention.batch * attention.Hq * attention.D, 0.25f, 1);
  const auto k_values_raw = make_data((size_t)attention.batch * attention.C * attention.Hk * attention.D, 0.25f, 2);
  const auto v_values_raw = make_data((size_t)attention.batch * attention.C * attention.Hk * attention.D, 0.25f, 3);
  const auto q_encoded = encode_fp16(q_values_raw);
  const auto k_encoded = encode_fp16(k_values_raw);
  const auto v_encoded = encode_fp16(v_values_raw);
  const auto q_values = decode_fp16_vector(q_encoded);
  const auto k_values = decode_fp16_vector(k_encoded);
  const auto v_values = decode_fp16_vector(v_encoded);

  const size_t q_bytes = q_encoded.size() * sizeof(half_float);
  const size_t k_bytes = k_encoded.size() * sizeof(half_float);
  const size_t v_bytes = v_encoded.size() * sizeof(half_float);
  const size_t o_count = (size_t)attention.batch * attention.Hq * attention.D;
  const size_t o_bytes = o_count * sizeof(half_float);

  auto q_stage = NS::TransferPtr(device->newBuffer(q_encoded.data(), q_bytes, kSharedResourceOptions));
  auto k_stage = NS::TransferPtr(device->newBuffer(k_encoded.data(), k_bytes, kSharedResourceOptions));
  auto v_stage = NS::TransferPtr(device->newBuffer(v_encoded.data(), v_bytes, kSharedResourceOptions));
  auto o_stage = NS::TransferPtr(device->newBuffer(o_bytes, kSharedResourceOptions));
  auto q_buffer = NS::TransferPtr(device->newBuffer(q_bytes, kPrivateResourceOptions));
  auto k_buffer = NS::TransferPtr(device->newBuffer(k_bytes, kPrivateResourceOptions));
  auto v_buffer = NS::TransferPtr(device->newBuffer(v_bytes, kPrivateResourceOptions));
  auto o_buffer = NS::TransferPtr(device->newBuffer(o_bytes, kPrivateResourceOptions));
  const size_t partial_bytes = (config.workgroups > 1) ?
      (size_t)attention.batch * attention.Hq * config.workgroups * (attention.D + 2) * sizeof(float) : 4;
  auto partial_buffer = NS::TransferPtr(device->newBuffer(partial_bytes, kPrivateResourceOptions));
  copy_buffer(command_queue.get(), q_stage.get(), q_buffer.get(), q_bytes);
  copy_buffer(command_queue.get(), k_stage.get(), k_buffer.get(), k_bytes);
  copy_buffer(command_queue.get(), v_stage.get(), v_buffer.get(), v_bytes);

  auto pipeline = create_decode_pipeline(device.get(), attention, config.simdgroups, config.workgroups);
  DecodeArgs args = { create_scale(attention) * 1.442695041f };

  const double validation_seconds =
      run_decode_once(command_queue.get(), pipeline, args, attention, q_buffer.get(), k_buffer.get(), v_buffer.get(), o_buffer.get(), partial_buffer.get());
  if (!(validation_seconds > 0)) {
    std::cerr << "validation dispatch failed\n";
    pool->drain();
    return 1;
  }
  copy_buffer(command_queue.get(), o_buffer.get(), o_stage.get(), o_bytes);
  const auto actual = decode_fp16(o_stage->contents(), o_count);
  std::vector<float> reference;
  compute_reference(attention, q_values, k_values, v_values, &reference);
  if (!validate_output(reference, actual)) {
    std::cerr << "validation failed\n";
    pool->drain();
    return 1;
  }

  Stats gpu_stats;
  Stats wall_stats;
  if (!benchmark(config, [&]() {
        return run_decode_once(command_queue.get(), pipeline, args, attention, q_buffer.get(), k_buffer.get(), v_buffer.get(), o_buffer.get(), partial_buffer.get());
      }, &gpu_stats, &wall_stats)) {
    std::cerr << "benchmark failed\n";
    pool->drain();
    return 1;
  }

  std::cout << "shape"
            << " B=" << attention.batch
            << " R=1"
            << " C=" << attention.C
            << " Hq=" << attention.Hq
            << " Hk=" << attention.Hk
            << " D=" << attention.D
            << " warmup=" << config.warmup_iterations
            << " timed=" << config.timed_iterations
            << " nsg=" << config.simdgroups
            << " nwg=" << config.workgroups
            << " tgmem_bytes=" << pipeline.threadgroup_memory_bytes
            << " partial_bytes=" << partial_bytes
            << '\n';
  const char* const label = config.workgroups == 1 ? "decode1_no_scratch" : "decode1_split_reduce";
  print_stats((std::string(label) + "_gpu").c_str(), gpu_stats);
  print_stats((std::string(label) + "_wall").c_str(), wall_stats);
  std::cout.flush();
  std::cerr.flush();
  std::_Exit(0);
}
