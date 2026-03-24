#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/NAMatMulDescriptor.hpp"
#include "nnc/mfa/kernels/NAMatMulKernel.hpp"
#include "nnc/mfa/kernels/NAMatMulKernelDescriptor.hpp"

namespace {

using half_float = _Float16;

struct BenchmarkConfig {
  int warmup_iterations = 3;
  int timed_iterations = 10;
};

struct BenchmarkCase {
  uint32_t M = 32768;
  uint32_t N = 4096;
  uint32_t K = 4096;
};

struct PipelineBundle {
  NAMatMulDescriptor descriptor;
  std::unique_ptr<NAMatMulKernel> kernel;
  NS::SharedPtr<MTL::ComputePipelineState> main_pipeline;
  NS::SharedPtr<MTL::ComputePipelineState> reduction_pipeline;
};

struct Stats {
  double average_seconds = 0;
  double median_seconds = 0;
  double min_seconds = 0;
  double max_seconds = 0;
};

struct VariantResult {
  uint16_t split_k = 1;
  uint32_t group_n = 0;
  bool thread_barrier_over_k = false;
  Stats stats;
  bool valid = true;
};

constexpr MTL::ResourceOptions kPrivateResourceOptions =
    MTL::ResourceStorageModePrivate | MTL::ResourceHazardTrackingModeTracked;

PipelineBundle create_pipeline_bundle(
    MTL::Device* device,
    const BenchmarkCase& bench,
    uint16_t forced_split_k,
    uint32_t group_m,
    uint32_t group_n)
{
  PipelineBundle bundle;
  bundle.descriptor.batchDimension = 1;
  bundle.descriptor.matrixDimensions =
      simd::uint3{bench.M, bench.N, bench.K};
  bundle.descriptor.memoryPrecisions = {
      .A = GEMMOperandPrecision::FP16,
      .B = GEMMOperandPrecision::FP16,
      .C = GEMMOperandPrecision::FP16,
      .bias = GEMMOperandPrecision::FP16,
  };
  bundle.descriptor.registerPrecisionC = std::nullopt;
  bundle.descriptor.batchStrides = std::nullopt;
  bundle.descriptor.transposeState = simd::uchar3{0, 1, 0};
  bundle.descriptor.useBias = false;
  bundle.descriptor.loadM = true;
  bundle.descriptor.supportIndirectCommandBuffers = false;

  const GEMMOperandPrecisions register_precisions = {
      .A = GEMMOperandPrecision::FP16,
      .B = GEMMOperandPrecision::FP16,
      .C = GEMMOperandPrecision::FP16,
      .bias = GEMMOperandPrecision::FP16,
  };
  const bool thread_barrier_over_k =
      NAMatMulDescriptor::threadBarrierOverK(bench.K, forced_split_k);
  const NAMatMulKernelDescriptor kernel_descriptor(
      simd::ushort3{128, 64, 64},
      bundle.descriptor.memoryPrecisions,
      register_precisions,
      forced_split_k,
      4,
      thread_barrier_over_k,
      bundle.descriptor.transposeState,
      bundle.descriptor.useBias,
      bundle.descriptor.loadM,
      group_m,
      group_n);
  bundle.kernel = std::make_unique<NAMatMulKernel>(kernel_descriptor, device);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  uint32_t N = bench.N;
  uint32_t K = bench.K;
  constants->setConstantValue(&N, MTL::DataTypeUInt, 1);
  constants->setConstantValue(&K, MTL::DataTypeUInt, 2);
  bool batched = false;
  uint32_t zero = 0;
  constants->setConstantValue(&batched, MTL::DataTypeBool, 11);
  constants->setConstantValue(&zero, MTL::DataTypeUInt, 15);
  constants->setConstantValue(&zero, MTL::DataTypeUInt, 16);
  constants->setConstantValue(&zero, MTL::DataTypeUInt, 17);
  constants->setConstantValue(&zero, MTL::DataTypeUInt, 18);

  NS::Error* error = nil;
  auto matmul_name = NS::String::string("matmul", NS::UTF8StringEncoding);
  auto matmul_function = NS::TransferPtr(
      bundle.kernel->library->newFunction(matmul_name, constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto main_descriptor =
      NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
  main_descriptor->setComputeFunction(matmul_function.get());
  auto main_pipeline = device->newComputePipelineState(
      main_descriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
  CCV_NNC_MFA_CHECK_ERROR(error);
  bundle.main_pipeline = NS::TransferPtr(main_pipeline);

  if (forced_split_k > 1) {
    auto reduce_name = NS::String::string(
        (bench.N % 2) == 0 ? "reduce_sum_2" : "reduce_sum",
        NS::UTF8StringEncoding);
    auto reduce_function = NS::TransferPtr(
        bundle.kernel->library->newFunction(reduce_name, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);
    auto reduce_descriptor =
        NS::TransferPtr(MTL::ComputePipelineDescriptor::alloc()->init());
    reduce_descriptor->setComputeFunction(reduce_function.get());
    auto reduce_pipeline = device->newComputePipelineState(
        reduce_descriptor.get(), MTL::PipelineOptionNone, nullptr, &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    bundle.reduction_pipeline = NS::TransferPtr(reduce_pipeline);
  }

  return bundle;
}

double run_once(
    MTL::CommandQueue* command_queue,
    const PipelineBundle& bundle,
    const BenchmarkCase& bench,
    MTL::Buffer* buffer_a,
    MTL::Buffer* buffer_b,
    MTL::Buffer* buffer_c,
    MTL::Buffer* scratch)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());

  {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(bundle.main_pipeline.get());
    encoder->useResource(buffer_a, MTL::ResourceUsageRead);
    encoder->useResource(buffer_b, MTL::ResourceUsageRead);
    if (bundle.kernel->splitK > 1) {
      encoder->useResource(scratch, MTL::ResourceUsageWrite);
    } else {
      encoder->useResource(buffer_c, MTL::ResourceUsageWrite);
    }
    encoder->setBuffer(buffer_a, 0, 0);
    encoder->setBuffer(buffer_b, 0, 1);
    encoder->setBuffer(bundle.kernel->splitK > 1 ? scratch : buffer_c, 0, 2);
    encoder->setBytes(&bench.M, sizeof(bench.M), 3);
    const auto grid_size = bundle.kernel->threadgroupsPerGrid(bundle.descriptor);
    const auto group_size = MTL::Size(
        int64_t(bundle.kernel->threadgroupSize(
            bundle.main_pipeline.get(), bundle.descriptor)),
        1,
        1);
    encoder->dispatchThreadgroups(grid_size, group_size);
    encoder->endEncoding();
  }

  if (bundle.kernel->splitK > 1) {
    auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
    encoder->setComputePipelineState(bundle.reduction_pipeline.get());
    encoder->setBuffer(scratch, 0, 0);
    encoder->setBuffer(buffer_c, 0, 1);
    encoder->setBytes(&bench.M, sizeof(bench.M), 2);
    encoder->useResource(scratch, MTL::ResourceUsageRead);
    encoder->useResource(buffer_c, MTL::ResourceUsageWrite);
    if ((bench.N % 2) == 0) {
      encoder->dispatchThreadgroups(
          MTL::Size((bench.M * bench.N / 2 + 255) / 256, 1, 1),
          MTL::Size(256, 1, 1));
    } else {
      encoder->dispatchThreadgroups(
          MTL::Size((bench.M * bench.N + 255) / 256, 1, 1),
          MTL::Size(256, 1, 1));
    }
    encoder->endEncoding();
  }

  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  if (command_buffer->status() != MTL::CommandBufferStatusCompleted) {
    std::cerr << "command buffer failed with status="
              << static_cast<int>(command_buffer->status());
    if (auto* error = command_buffer->error()) {
      auto* description = error->localizedDescription();
      if (description) {
        std::cerr << " error=" << description->utf8String();
      }
    }
    std::cerr << std::endl;
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double gpu_start = command_buffer->GPUStartTime();
  const double gpu_end = command_buffer->GPUEndTime();
  if (!(gpu_end > gpu_start)) {
    std::cerr << "invalid gpu timestamps start=" << gpu_start
              << " end=" << gpu_end
              << " splitK=" << bundle.kernel->splitK
              << std::endl;
    return std::numeric_limits<double>::quiet_NaN();
  }
  return gpu_end - gpu_start;
}

bool benchmark_variant(
    MTL::CommandQueue* command_queue,
    const PipelineBundle& bundle,
    const BenchmarkCase& bench,
    const BenchmarkConfig& config,
    MTL::Buffer* buffer_a,
    MTL::Buffer* buffer_b,
    MTL::Buffer* buffer_c,
    MTL::Buffer* scratch,
    Stats* const stats)
{
  std::vector<double> samples;
  samples.reserve(config.timed_iterations);
  for (int i = 0; i < config.warmup_iterations + config.timed_iterations; ++i) {
    const double seconds =
        run_once(command_queue, bundle, bench, buffer_a, buffer_b, buffer_c, scratch);
    if (std::isnan(seconds)) {
      return false;
    }
    if (i >= config.warmup_iterations) {
      samples.push_back(seconds);
    }
  }

  stats->average_seconds =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  stats->median_seconds = samples[samples.size() / 2];
  stats->min_seconds = samples.front();
  stats->max_seconds = samples.back();
  return true;
}

void print_stats(
    const char* label,
    const BenchmarkCase& bench,
    const Stats& stats)
{
  const double flops = 2.0 * static_cast<double>(bench.M) *
      static_cast<double>(bench.N) * static_cast<double>(bench.K);
  const double gflops = flops / stats.average_seconds / 1e9;
  std::cout << label
            << " avg_ms=" << std::fixed << std::setprecision(3)
            << stats.average_seconds * 1e3
            << " median_ms=" << stats.median_seconds * 1e3
            << " min_ms=" << stats.min_seconds * 1e3
            << " max_ms=" << stats.max_seconds * 1e3
            << " avg_gflops=" << gflops
            << '\n';
}

} // namespace

int main(int argc, char** argv)
{
  BenchmarkCase bench;
  BenchmarkConfig config;
  int forced_split_k = 0;
  if (argc >= 4) {
    bench.M = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
    bench.N = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
    bench.K = static_cast<uint32_t>(std::strtoul(argv[3], nullptr, 10));
  }
  if (argc >= 6) {
    config.warmup_iterations = std::atoi(argv[4]);
    config.timed_iterations = std::atoi(argv[5]);
  }
  if (argc >= 7) {
    forced_split_k = std::atoi(argv[6]);
  }

  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    std::cerr << "Metal device unavailable.\n";
    pool->drain();
    return 1;
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue) {
    std::cerr << "Metal command queue unavailable.\n";
    pool->drain();
    return 1;
  }

  std::cout << "shape M=" << bench.M
            << " N=" << bench.N
            << " K=" << bench.K
            << " warmup=" << config.warmup_iterations
            << " timed=" << config.timed_iterations
            << '\n';

  const size_t a_count = static_cast<size_t>(bench.M) * bench.K;
  const size_t b_count = static_cast<size_t>(bench.N) * bench.K;
  const size_t c_count = static_cast<size_t>(bench.M) * bench.N;
  auto buffer_a = NS::TransferPtr(
      device->newBuffer(a_count * sizeof(half_float), kPrivateResourceOptions));
  auto buffer_b = NS::TransferPtr(
      device->newBuffer(b_count * sizeof(half_float), kPrivateResourceOptions));
  auto buffer_c = NS::TransferPtr(
      device->newBuffer(c_count * sizeof(half_float), kPrivateResourceOptions));
  auto scratch = NS::TransferPtr(
      device->newBuffer(c_count * 8 * sizeof(half_float), kPrivateResourceOptions));
  if (!buffer_a || !buffer_b || !buffer_c || !scratch) {
    std::cerr << "Failed to allocate benchmark buffers.\n";
    pool->drain();
    return 1;
  }

  const uint32_t group_m = (bench.M >= 4096) ? 4096 : 0;
  std::vector<uint32_t> group_ns = {0};
  if (bench.N >= 4096) {
    group_ns.push_back(4096);
  }
  std::vector<VariantResult> results;
  const std::vector<uint16_t> split_ks = {1, 2, 4, 8};
  results.reserve(group_ns.size() * split_ks.size());
  for (const auto group_n : group_ns) {
    for (const auto split_k : split_ks) {
      if (forced_split_k > 0 &&
          split_k != static_cast<uint16_t>(forced_split_k)) {
        continue;
      }
      if (split_k > 1 && bench.K / split_k < 128) {
        continue;
      }
      std::cerr << "running splitK=" << split_k
                << " groupN=" << group_n << std::endl;
      auto bundle = create_pipeline_bundle(
          device.get(), bench, split_k, group_m, group_n);
      Stats stats;
      const bool valid = benchmark_variant(
          command_queue.get(),
          bundle,
          bench,
          config,
          buffer_a.get(),
          buffer_b.get(),
          buffer_c.get(),
          split_k > 1 ? scratch.get() : nullptr,
          &stats);
      results.push_back(VariantResult{
          .split_k = split_k,
          .group_n = group_n,
          .thread_barrier_over_k =
              NAMatMulDescriptor::threadBarrierOverK(bench.K, split_k),
          .stats = stats,
          .valid = valid,
      });
    }
  }

  for (const auto& result : results) {
    std::ostringstream label;
    label << "groupM=" << group_m
          << " groupN=" << result.group_n
          << " splitK=" << result.split_k
          << " threadBarrierOverK="
          << (result.thread_barrier_over_k ? 1 : 0);
    if (result.valid) {
      print_stats(label.str().c_str(), bench, result.stats);
    } else {
      std::cout << label.str() << " invalid\n";
    }
  }

  pool->drain();
  return 0;
}
