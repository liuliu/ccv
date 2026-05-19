#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/GEMMOperandPrecision.hpp"
#include "nnc/mfa/kernels/ShaderCache.hpp"
#include "nnc/mfa/kernels/WalshHadamardTransformDescriptor.hpp"
#include "nnc/mfa/kernels/WalshHadamardTransformKernel.hpp"

namespace {

struct BenchmarkConfig {
  int warmup_iterations = 5;
  int timed_iterations = 20;
  int dispatch_repeats = 1;
};

struct Stats {
  double average_seconds = 0;
  double best3_average_seconds = 0;
  double median_seconds = 0;
  double min_seconds = 0;
  double max_seconds = 0;
};

struct CopyPipeline {
  NS::SharedPtr<MTL::Library> library;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;
};

constexpr MTL::ResourceOptions kSharedResourceOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;

bool is_power_of_two(uint32_t x)
{
  return x > 0 && (x & (x - 1)) == 0;
}

std::vector<float> make_data(size_t size)
{
  std::vector<float> values(size);
  for (size_t i = 0; i < size; ++i) {
    const int centered = (int)((i * 17 + 13) % 31) - 15;
    values[i] = centered * 0.0625f;
  }
  return values;
}

uint16_t float_to_half_bits(float value)
{
  __fp16 half_value = (__fp16)value;
  uint16_t bits;
  std::memcpy(&bits, &half_value, sizeof(bits));
  return bits;
}

float half_bits_to_float(uint16_t bits)
{
  __fp16 half_value;
  std::memcpy(&half_value, &bits, sizeof(bits));
  return (float)half_value;
}

std::vector<uint16_t> make_data_fp16(const std::vector<float>& values)
{
  std::vector<uint16_t> output(values.size());
  for (size_t i = 0; i < values.size(); i++)
    output[i] = float_to_half_bits(values[i]);
  return output;
}

void walsh_hadamard_transform_cpu(float* row, uint32_t dim)
{
  for (uint32_t stride = 1; stride < dim; stride <<= 1) {
    for (uint32_t base = 0; base < dim; base += stride * 2) {
      for (uint32_t i = 0; i < stride; i++) {
        const float a = row[base + i];
        const float b = row[base + i + stride];
        row[base + i] = a + b;
        row[base + i + stride] = a - b;
      }
    }
  }
}

std::vector<float> walsh_hadamard_transform_reference(const std::vector<float>& input, uint32_t rows, uint32_t dim, float scale)
{
  std::vector<float> output(input.size());
  std::vector<float> row(dim);
  for (uint32_t r = 0; r < rows; r++) {
    std::memcpy(row.data(), input.data() + (size_t)r * dim, sizeof(float) * dim);
    walsh_hadamard_transform_cpu(row.data(), dim);
    for (uint32_t i = 0; i < dim; i++)
      output[(size_t)r * dim + i] = row[i] * scale;
  }
  return output;
}

bool benchmark(const BenchmarkConfig& config, const std::function<double()>& run_once, Stats* stats)
{
  std::vector<double> samples;
  samples.reserve(config.timed_iterations);
  for (int i = 0; i < config.warmup_iterations + config.timed_iterations; ++i) {
    const double elapsed = run_once();
    if (!(elapsed >= 0))
      return false;
    if (i >= config.warmup_iterations)
      samples.push_back(elapsed);
  }
  if (samples.empty())
    return false;
  stats->average_seconds = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  const size_t best_count = std::min<size_t>(3, samples.size());
  stats->best3_average_seconds = std::accumulate(samples.begin(), samples.begin() + best_count, 0.0) / best_count;
  stats->median_seconds = samples[samples.size() / 2];
  stats->min_seconds = samples.front();
  stats->max_seconds = samples.back();
  return true;
}

void print_stats(const char* label, const Stats& stats, size_t bytes)
{
  const double gib = (double)bytes / (1024.0 * 1024.0 * 1024.0);
  const double average_bandwidth = gib / stats.average_seconds;
  const double best3_bandwidth = gib / stats.best3_average_seconds;
  const double median_bandwidth = gib / stats.median_seconds;
  std::cout << label
            << " avg_ms=" << std::fixed << std::setprecision(4) << stats.average_seconds * 1e3
            << " best3_avg_ms=" << stats.best3_average_seconds * 1e3
            << " median_ms=" << stats.median_seconds * 1e3
            << " min_ms=" << stats.min_seconds * 1e3
            << " max_ms=" << stats.max_seconds * 1e3
            << " avg_GiBps=" << average_bandwidth
            << " best3_GiBps=" << best3_bandwidth
            << " median_GiBps=" << median_bandwidth
            << '\n';
}

CopyPipeline create_copy_pipeline(MTL::Device* device, uint32_t count, GEMMOperandPrecision precision)
{
  std::string source = R"(
#include <metal_stdlib>
using namespace metal;

typedef )" + precision.name() + R"( real;
constant uint count [[function_constant(0)]];

kernel void copy_kernel(
  device const real* source [[buffer(0)]],
  device real* destination [[buffer(1)]],
  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = source[idx];
}
  )";
  CopyPipeline output;
  auto string = NS::String::string(source.c_str(), NS::UTF8StringEncoding);
  NS::Error* error = nil;
  output.library = NS::TransferPtr(device->newLibrary(string, nil, &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  constants->setConstantValue(&count, MTL::DataTypeUInt, NS::UInteger(0));
  NS::String* functionName = NS::String::string("copy_kernel", NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(output.library->newFunction(functionName, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  output.pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  return output;
}

double run_wht_once(
    MTL::CommandQueue* command_queue,
    PipelineValue<WalshHadamardTransformKernel>* pipeline_value,
    MTL::Buffer* src,
    MTL::Buffer* dst,
    uint32_t rows,
    uint32_t dim,
    int dispatch_repeats)
{
  const uint32_t max_radix = (dim > 1) ? std::min<uint32_t>(dim, 16) : 1;
  const uint32_t strategy = (dim >= 32 && dim <= 128) ? 2 : ((dim <= 128) ? 1 : 0);
  const uint32_t rows_per_threadgroup = (strategy == 2) ? std::max<uint32_t>(1, std::min<uint32_t>(8, (uint32_t)pipeline_value->pipeline->maxTotalThreadsPerThreadgroup() / dim)) : 1;
  const uint32_t num_threads = (strategy == 2) ? dim * rows_per_threadgroup : ((strategy == 1) ? dim : std::max<uint32_t>(dim / max_radix, 1));
  const auto start = std::chrono::steady_clock::now();
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline_value->pipeline.get());
  MTL::Buffer* current_src = src;
  MTL::Buffer* current_dst = dst;
  for (int i = 0; i < dispatch_repeats; i++) {
    encoder->setBuffer(current_src, 0, 0);
    encoder->setBuffer(current_dst, 0, 1);
    encoder->setThreadgroupMemoryLength(NS::UInteger(dim * rows_per_threadgroup * sizeof(float)), NS::UInteger(0));
    encoder->dispatchThreadgroups(MTL::Size((rows + rows_per_threadgroup - 1) / rows_per_threadgroup, 1, 1), MTL::Size(num_threads, 1, 1));
    std::swap(current_src, current_dst);
  }
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

double run_copy_once(
    MTL::CommandQueue* command_queue,
    MTL::ComputePipelineState* pipeline,
    MTL::Buffer* src,
    MTL::Buffer* dst,
    uint32_t count,
    int dispatch_repeats)
{
  const auto start = std::chrono::steady_clock::now();
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline);
  MTL::Buffer* current_src = src;
  MTL::Buffer* current_dst = dst;
  for (int i = 0; i < dispatch_repeats; i++) {
    encoder->setBuffer(current_src, 0, 0);
    encoder->setBuffer(current_dst, 0, 1);
    encoder->dispatchThreadgroups(MTL::Size((count + 255) / 256, 1, 1), MTL::Size(256, 1, 1));
    std::swap(current_src, current_dst);
  }
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

bool check_correctness(MTL::Device* device, MTL::CommandQueue* command_queue, PipelineValue<WalshHadamardTransformKernel>* pipeline_value, GEMMOperandPrecision precision, uint32_t dim, float scale)
{
  const uint32_t rows = 19;
  const size_t count = (size_t)rows * dim;
  const auto input = make_data(count);
  const auto expected = walsh_hadamard_transform_reference(input, rows, dim, scale);
  const size_t scalar_bytes = precision.size();
  const size_t bytes = count * scalar_bytes;
  const auto input_fp16 = make_data_fp16(input);
  const void* input_data = (precision == GEMMOperandPrecision::FP16) ? (const void*)input_fp16.data() : (const void*)input.data();
  auto src = NS::TransferPtr(device->newBuffer(input_data, bytes, kSharedResourceOptions));
  auto dst = NS::TransferPtr(device->newBuffer(bytes, kSharedResourceOptions));
  if (!src || !dst) {
    std::cerr << "Correctness buffer allocation failed.\n";
    return false;
  }
  run_wht_once(command_queue, pipeline_value, src.get(), dst.get(), rows, dim, 1);
  const float* actual_fp32 = (const float*)dst->contents();
  const uint16_t* actual_fp16 = (const uint16_t*)dst->contents();
  double max_abs_diff = 0;
  double max_rel_diff = 0;
  size_t max_diff_index = 0;
  for (size_t i = 0; i < count; i++) {
    const double actual = (precision == GEMMOperandPrecision::FP16) ? half_bits_to_float(actual_fp16[i]) : actual_fp32[i];
    const double diff = std::abs(actual - expected[i]);
    const double rel = diff / std::max(std::abs((double)expected[i]), 1.0);
    if (diff > max_abs_diff) {
      max_abs_diff = diff;
      max_diff_index = i;
    }
    max_rel_diff = std::max(max_rel_diff, rel);
  }
  std::cout << "correctness rows=" << rows << " dim=" << dim
            << " max_abs_diff=" << std::setprecision(9) << max_abs_diff
            << " max_rel_diff=" << max_rel_diff
            << " index=" << max_diff_index << '\n';
  return (precision == GEMMOperandPrecision::FP16) ? (max_abs_diff <= 2e-2) : (max_abs_diff <= 2e-5);
}

} // namespace

int main(int argc, char** argv)
{
  BenchmarkConfig config;
  uint32_t rows = 262144;
  uint32_t dim = 128;
  if (argc >= 3) {
    rows = (uint32_t)std::strtoul(argv[1], nullptr, 10);
    dim = (uint32_t)std::strtoul(argv[2], nullptr, 10);
  }
  if (argc >= 5) {
    config.warmup_iterations = std::atoi(argv[3]);
    config.timed_iterations = std::atoi(argv[4]);
  }
  if (rows == 0 || !is_power_of_two(dim) || dim > 8192) {
    std::cerr << "usage: " << argv[0] << " [rows dim warmup timed dispatch_repeats precision], with rows > 0, dim power-of-two, dim <= 8192, precision fp32 or fp16.\n";
    return 1;
  }
  if (argc >= 6)
    config.dispatch_repeats = std::max(1, std::atoi(argv[5]));
  GEMMOperandPrecision precision = GEMMOperandPrecision::FP32;
  if (argc >= 7) {
    const std::string precision_name = argv[6];
    if (precision_name == "fp16")
      precision = GEMMOperandPrecision::FP16;
    else if (precision_name != "fp32") {
      std::cerr << "precision must be fp32 or fp16.\n";
      return 1;
    }
  }
  const size_t count = (size_t)rows * dim;
  if (count > UINT32_MAX) {
    std::cerr << "rows * dim must fit in uint32_t for the copy baseline.\n";
    return 1;
  }

  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    std::cerr << "Metal device unavailable.\n";
    (void)pool;
    return 1;
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue) {
    std::cerr << "Metal command queue unavailable.\n";
    (void)pool;
    return 1;
  }

  const float scale = 1.0f / std::sqrt((float)dim);
  WalshHadamardTransformDescriptor descriptor;
  descriptor.memoryPrecision = precision;
  descriptor.rowCount = rows;
  descriptor.dim = dim;
  descriptor.scale = scale;
  ShaderCache shader_cache;
  DeviceProperties dprops = DeviceProperties();
  auto pipeline_value = shader_cache.findKernel<WalshHadamardTransformKernel, WalshHadamardTransformDescriptor, WalshHadamardTransformKernelDescriptor>(descriptor, device.get(), dprops);
  if (!check_correctness(device.get(), command_queue.get(), pipeline_value, precision, dim, scale)) {
    std::cerr << "Correctness check failed.\n";
    (void)pool;
    return 1;
  }

  const size_t bytes = count * precision.size();
  const auto input = make_data(count);
  const auto input_fp16 = make_data_fp16(input);
  const void* input_data = (precision == GEMMOperandPrecision::FP16) ? (const void*)input_fp16.data() : (const void*)input.data();
  auto src = NS::TransferPtr(device->newBuffer(input_data, bytes, kSharedResourceOptions));
  auto wht_dst = NS::TransferPtr(device->newBuffer(bytes, kSharedResourceOptions));
  auto copy_dst = NS::TransferPtr(device->newBuffer(bytes, kSharedResourceOptions));
  if (!src || !wht_dst || !copy_dst) {
    std::cerr << "Benchmark buffer allocation failed.\n";
    (void)pool;
    return 1;
  }
  auto copy_pipeline = create_copy_pipeline(device.get(), (uint32_t)count, precision);

  Stats wht_stats;
  Stats copy_stats;
  const bool wht_ok = benchmark(config, [&]() {
    return run_wht_once(command_queue.get(), pipeline_value, src.get(), wht_dst.get(), rows, dim, config.dispatch_repeats);
  }, &wht_stats);
  const bool copy_ok = benchmark(config, [&]() {
    return run_copy_once(command_queue.get(), copy_pipeline.pipeline.get(), src.get(), copy_dst.get(), (uint32_t)count, config.dispatch_repeats);
  }, &copy_stats);

  std::cout << precision.name() << " rows=" << rows << " dim=" << dim << " elements=" << count << " repeats=" << config.dispatch_repeats << " traffic_bytes=" << (bytes * 2 * (size_t)config.dispatch_repeats) << '\n';
  if (wht_ok)
    print_stats("  wht       ", wht_stats, bytes * 2 * (size_t)config.dispatch_repeats);
  if (copy_ok)
    print_stats("  copy      ", copy_stats, bytes * 2 * (size_t)config.dispatch_repeats);
  if (wht_ok && copy_ok)
    std::cout << "  wht_vs_copy_median=" << std::fixed << std::setprecision(4) << (wht_stats.median_seconds / copy_stats.median_seconds) << "x\n";

  std::cout << std::flush;
  (void)pool;
  return 0;
}
