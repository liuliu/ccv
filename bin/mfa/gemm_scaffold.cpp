#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <QuartzCore/QuartzCore.h>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/GEMMDescriptor.hpp"
#include "nnc/mfa/kernels/GEMMKernel.hpp"
#include "nnc/mfa/kernels/GEMMKernelDescriptor.hpp"
#include "nnc/mfa/kernels/ShaderCache.hpp"

namespace {

struct GemmCase {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  bool transpose_a;
  bool transpose_b;
};

struct BenchmarkConfig {
  int warmup_iterations;
  int timed_iterations;
  int duplicated_dispatches;
};

using half_float = _Float16;

std::vector<float> make_matrix(size_t size, float offset)
{
  std::vector<float> values(size);
  for (size_t i = 0; i < size; ++i)
    values[i] = offset + static_cast<float>(static_cast<int>(i % 13) - 6) * 0.125f;
  return values;
}

float read_a(const std::vector<float>& a, const GemmCase& gemm, uint32_t m, uint32_t k)
{
  if (gemm.transpose_a)
    return a[k * gemm.M + m];
  return a[m * gemm.K + k];
}

float read_b(const std::vector<float>& b, const GemmCase& gemm, uint32_t k, uint32_t n)
{
  if (gemm.transpose_b)
    return b[n * gemm.K + k];
  return b[k * gemm.N + n];
}

std::vector<float> cpu_reference(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const GemmCase& gemm)
{
  std::vector<float> c(static_cast<size_t>(gemm.M) * gemm.N, 0.0f);
  for (uint32_t m = 0; m < gemm.M; ++m)
    for (uint32_t n = 0; n < gemm.N; ++n)
    {
      float sum = 0.0f;
      for (uint32_t k = 0; k < gemm.K; ++k)
        sum += read_a(a, gemm, m, k) * read_b(b, gemm, k, n);
      c[m * gemm.N + n] = sum;
    }
  return c;
}

void print_matrix(
    const std::string& name,
    const std::vector<float>& matrix,
    uint32_t rows,
    uint32_t cols)
{
  std::cout << name << " (" << rows << "x" << cols << ")\n";
  for (uint32_t row = 0; row < rows; ++row)
  {
    for (uint32_t col = 0; col < cols; ++col)
      std::cout << std::setw(9) << std::fixed << std::setprecision(4)
                << matrix[row * cols + col] << ' ';
    std::cout << '\n';
  }
}

GEMMDescriptor make_descriptor(const GemmCase& gemm, GEMMOperandPrecision precision)
{
  GEMMDescriptor descriptor;
  descriptor.matrixDimensions = simd::uint3{gemm.M, gemm.N, gemm.K};
  descriptor.memoryPrecisions = {
      .A = precision,
      .B = precision,
      .C = precision,
      .bias = precision,
  };
  descriptor.registerPrecisionC = std::nullopt;
  descriptor.leadingDimensions = std::nullopt;
  descriptor.batchStrides = std::nullopt;
  descriptor.transposeState =
      simd::uchar3{static_cast<unsigned char>(gemm.transpose_a),
                   static_cast<unsigned char>(gemm.transpose_b),
                   0};
  descriptor.loadPreviousC = false;
  descriptor.useBias = false;
  descriptor.loadM = false;
  descriptor.supportIndirectCommandBuffers = false;
  return descriptor;
}

template <typename Scalar>
std::vector<Scalar> cast_values(const std::vector<float>& values)
{
  std::vector<Scalar> output(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    output[i] = static_cast<Scalar>(values[i]);
  return output;
}

template <typename Scalar>
std::vector<float> to_float_vector(const std::vector<Scalar>& values)
{
  std::vector<float> output(values.size());
  for (size_t i = 0; i < values.size(); ++i)
    output[i] = static_cast<float>(values[i]);
  return output;
}

template <typename Scalar>
double benchmark_case(
    MTL::Device* device,
    MTL::CommandQueue* command_queue,
    ShaderCache& shader_cache,
    const GemmCase& gemm,
    const BenchmarkConfig& config,
    GEMMOperandPrecision precision,
    std::vector<float>* output)
{
  DeviceProperties dprops{};
  const auto descriptor = make_descriptor(gemm, precision);
  auto pipeline_value =
      shader_cache.findKernel<GEMMKernel, GEMMDescriptor, GEMMKernelDescriptor>(
          descriptor, device, dprops);
  std::cerr << "[benchmark] pipeline ready for M=" << gemm.M
            << " N=" << gemm.N
            << " K=" << gemm.K
            << " precision=" << static_cast<int>(precision.value)
            << '\n';

  const auto a = cast_values<Scalar>(make_matrix(static_cast<size_t>(gemm.M) * gemm.K, 0.5f));
  const auto b = cast_values<Scalar>(make_matrix(static_cast<size_t>(gemm.N) * gemm.K, -0.25f));
  std::vector<Scalar> c(static_cast<size_t>(gemm.M) * gemm.N, Scalar(0));

  auto buffer_a = NS::TransferPtr(device->newBuffer(
      a.data(),
      a.size() * sizeof(Scalar),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto buffer_b = NS::TransferPtr(device->newBuffer(
      b.data(),
      b.size() * sizeof(Scalar),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  auto buffer_c = NS::TransferPtr(device->newBuffer(
      c.data(),
      c.size() * sizeof(Scalar),
      MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked));
  std::cerr << "[benchmark] buffers allocated\n";

  double wall_seconds = 0.0;
  const auto ceil_divide =
      [](uint32_t target, uint16_t granularity) -> uint32_t {
    return (target + granularity - 1) / granularity;
  };
  const auto grid_size = MTL::Size(
      ceil_divide(gemm.N, pipeline_value->kernel->blockDimensions[1]),
      ceil_divide(gemm.M, pipeline_value->kernel->blockDimensions[0]),
      1);
  const auto group_size =
      MTL::Size(pipeline_value->kernel->threadgroupSize, 1, 1);
  for (int iteration = 0;
       iteration < config.warmup_iterations + config.timed_iterations;
       ++iteration)
  {
    auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(pipeline_value->pipeline.get());
    encoder->setThreadgroupMemoryLength(
        pipeline_value->kernel->threadgroupMemoryAllocation, 0);
    encoder->setBuffer(buffer_a.get(), 0, 0);
    encoder->setBuffer(buffer_b.get(), 0, 1);
    encoder->setBuffer(buffer_c.get(), 0, 2);
    if (descriptor.loadM)
      encoder->setBytes(&gemm.M, sizeof(gemm.M), 3);
    encoder->useResource(buffer_a.get(), MTL::ResourceUsageRead);
    encoder->useResource(buffer_b.get(), MTL::ResourceUsageRead);
    encoder->useResource(buffer_c.get(), MTL::ResourceUsageWrite);

    for (int duplicate = 0; duplicate < config.duplicated_dispatches; ++duplicate)
      encoder->dispatchThreadgroups(grid_size, group_size);

    encoder->endEncoding();
    const double start_time = CACurrentMediaTime();
    command_buffer->commit();
    command_buffer->waitUntilCompleted();
    const double end_time = CACurrentMediaTime();

    if (iteration >= config.warmup_iterations)
      wall_seconds += end_time - start_time;
  }

  if (output)
  {
    const auto* raw = static_cast<const Scalar*>(buffer_c->contents());
    std::vector<Scalar> gpu_values(raw, raw + c.size());
    *output = to_float_vector(gpu_values);
  }

  return wall_seconds / config.timed_iterations;
}

} // namespace

int main(int argc, char** argv)
{
  auto* pool = NS::AutoreleasePool::alloc()->init();
  std::cerr << "[scaffold] creating device\n";
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device)
  {
    std::cerr << "Metal device unavailable.\n";
    pool->drain();
    return 1;
  }
  std::cerr << "[scaffold] creating command queue\n";

  const GemmCase gemm{
      .M = 4,
      .N = 8,
      .K = 16,
      .transpose_a = false,
      .transpose_b = true,
  };

  ShaderCache shader_cache;
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue)
  {
    std::cerr << "Metal command queue unavailable.\n";
    pool->drain();
    return 1;
  }
  std::cerr << "[scaffold] running correctness case\n";
  const auto a = make_matrix(static_cast<size_t>(gemm.M) * gemm.K, 0.5f);
  const auto b = make_matrix(static_cast<size_t>(gemm.N) * gemm.K, -0.25f);
  const auto expected = cpu_reference(a, b, gemm);
  std::vector<float> c;
  benchmark_case<float>(
      device.get(),
      command_queue.get(),
      shader_cache,
      gemm,
      BenchmarkConfig{0, 1, 1},
      GEMMOperandPrecision::FP32,
      &c);

  float max_abs_error = 0.0f;
  size_t max_error_index = 0;
  for (size_t i = 0; i < c.size(); ++i)
  {
    const float abs_error = std::fabs(c[i] - expected[i]);
    if (abs_error > max_abs_error)
    {
      max_abs_error = abs_error;
      max_error_index = i;
    }
  }

  std::cout << "device: " << device->name()->utf8String() << '\n';
  std::cout << "supports apple10 / neural accelerators: "
            << (device->supportsFamily(MTL::GPUFamily(1010)) ? "yes" : "no")
            << '\n';
  std::cout << "gemm: M=" << gemm.M << " N=" << gemm.N << " K=" << gemm.K
            << " transposeA=" << gemm.transpose_a
            << " transposeB=" << gemm.transpose_b << '\n';
  std::cout << "max abs error: " << max_abs_error << '\n';

  if (!(max_abs_error <= 1e-4f))
  {
    print_matrix("A", a, gemm.M, gemm.K);
    print_matrix("B storage (N x K, transposeB=true)", b, gemm.N, gemm.K);
    print_matrix("GPU C", c, gemm.M, gemm.N);
    print_matrix("CPU C", expected, gemm.M, gemm.N);
    std::cerr << "largest mismatch at linear index " << max_error_index << '\n';
    pool->drain();
    return 2;
  }

  print_matrix("GPU C", c, gemm.M, gemm.N);

  uint32_t benchmark_m = 3072;
  uint32_t benchmark_n = 3072;
  uint32_t benchmark_k = 3072;
  if (argc >= 4)
  {
    const long parsed_m = std::strtol(argv[1], nullptr, 10);
    const long parsed_n = std::strtol(argv[2], nullptr, 10);
    const long parsed_k = std::strtol(argv[3], nullptr, 10);
    if (parsed_m <= 0 || parsed_n <= 0 || parsed_k <= 0)
    {
      std::cerr << "Invalid GEMM shape arguments.\n";
      pool->drain();
      return 2;
    }
    benchmark_m = static_cast<uint32_t>(parsed_m);
    benchmark_n = static_cast<uint32_t>(parsed_n);
    benchmark_k = static_cast<uint32_t>(parsed_k);
  }

  const GemmCase benchmark_gemm{
      .M = benchmark_m,
      .N = benchmark_n,
      .K = benchmark_k,
      .transpose_a = false,
      .transpose_b = true,
  };
  const BenchmarkConfig benchmark_config{
      .warmup_iterations = 2,
      .timed_iterations = 10,
      .duplicated_dispatches = 1,
  };
  DeviceProperties dprops{};
  const auto benchmark_descriptor =
      make_descriptor(benchmark_gemm, GEMMOperandPrecision::FP16);
  auto benchmark_pipeline =
      shader_cache.findKernel<GEMMKernel, GEMMDescriptor, GEMMKernelDescriptor>(
          benchmark_descriptor, device.get(), dprops);
  const auto benchmark_block = benchmark_pipeline->kernel->blockDimensions;
  const auto benchmark_threadgroup_size = benchmark_pipeline->kernel->threadgroupSize;
  const auto benchmark_tgm_bytes = benchmark_pipeline->kernel->threadgroupMemoryAllocation;
  const bool benchmark_async_load = benchmark_pipeline->kernel->preferAsyncLoad;
  const bool benchmark_async_store = benchmark_pipeline->kernel->preferAsyncStore;
  std::cerr << "[scaffold] running fp16 benchmark case\n";
  const double average_gpu_seconds = benchmark_case<half_float>(
      device.get(),
      command_queue.get(),
      shader_cache,
      benchmark_gemm,
      benchmark_config,
      GEMMOperandPrecision::FP16,
      nullptr);
  std::cerr << "[scaffold] benchmark completed\n";
  const double total_flops = 2.0 * static_cast<double>(benchmark_gemm.M) *
      static_cast<double>(benchmark_gemm.N) * static_cast<double>(benchmark_gemm.K) *
      static_cast<double>(benchmark_config.duplicated_dispatches);
  std::cerr << "[scaffold] computed flop count\n";
  const double tflops = total_flops / average_gpu_seconds / 1e12;
  std::cerr << "[scaffold] computed tflops\n";
  const auto ceil_divide =
      [](uint32_t target, uint16_t granularity) -> uint32_t {
    return (target + granularity - 1) / granularity;
  };
  const auto benchmark_grid = MTL::Size(
      ceil_divide(benchmark_gemm.N, benchmark_block[1]),
      ceil_divide(benchmark_gemm.M, benchmark_block[0]),
      1);
  std::cerr << "[scaffold] computed grid\n";
  fprintf(
      stderr,
      "fp16 benchmark: M=%u N=%u K=%u transposeB=%d block=%u x %u x %u "
      "threadgroup_size=%u tgm_bytes=%u async_load=%d async_store=%d "
      "grid=%llu x %llu duplicate_dispatches=%d avg_ms=%.3f tflops=%.3f\n",
      benchmark_gemm.M,
      benchmark_gemm.N,
      benchmark_gemm.K,
      static_cast<int>(benchmark_gemm.transpose_b),
      static_cast<unsigned>(benchmark_block[0]),
      static_cast<unsigned>(benchmark_block[1]),
      static_cast<unsigned>(benchmark_block[2]),
      static_cast<unsigned>(benchmark_threadgroup_size),
      static_cast<unsigned>(benchmark_tgm_bytes),
      static_cast<int>(benchmark_async_load),
      static_cast<int>(benchmark_async_store),
      static_cast<unsigned long long>(benchmark_grid.width),
      static_cast<unsigned long long>(benchmark_grid.height),
      benchmark_config.duplicated_dispatches,
      average_gpu_seconds * 1e3,
      tflops);
  fflush(stdout);
  fflush(stderr);
  std::_Exit(0);
}
