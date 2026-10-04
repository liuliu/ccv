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
  uint32_t batch = 1;
  bool transpose_a = false;
  bool transpose_b = true;
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
    bool load_m,
    uint32_t group_m,
    uint32_t group_n,
    int forced_thread_barrier,
    simd::ushort3 block_dimensions)
{
  PipelineBundle bundle;
  bundle.descriptor.batchDimension = bench.batch;
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
  bundle.descriptor.transposeState = simd::uchar3{(uint8_t)bench.transpose_a, (uint8_t)bench.transpose_b, 0};
  bundle.descriptor.useBias = false;
  bundle.descriptor.loadM = load_m;
  bundle.descriptor.supportIndirectCommandBuffers = false;

  const GEMMOperandPrecisions register_precisions = {
      .A = GEMMOperandPrecision::FP16,
      .B = GEMMOperandPrecision::FP16,
      .C = GEMMOperandPrecision::FP16,
      .bias = GEMMOperandPrecision::FP16,
  };
  const bool thread_barrier_over_k = forced_thread_barrier >= 0 ?
      (forced_thread_barrier != 0) :
      NAMatMulDescriptor::threadBarrierOverK(bench.K, forced_split_k);
  const NAMatMulKernelDescriptor kernel_descriptor(
      block_dimensions,
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
  if (!load_m) {
    uint32_t M = bench.M;
    constants->setConstantValue(&M, MTL::DataTypeUInt, NS::UInteger(0));
  }
  uint32_t N = bench.N;
  uint32_t K = bench.K;
  constants->setConstantValue(&N, MTL::DataTypeUInt, 1);
  constants->setConstantValue(&K, MTL::DataTypeUInt, 2);
  bool batched = bench.batch > 1;
  uint32_t zero = 0;
  uint32_t a_stride = bench.M * bench.K, b_stride = bench.N * bench.K, c_stride = bench.M * bench.N;
  constants->setConstantValue(&batched, MTL::DataTypeBool, 11);
  constants->setConstantValue(&a_stride, MTL::DataTypeUInt, 15);
  constants->setConstantValue(&b_stride, MTL::DataTypeUInt, 16);
  constants->setConstantValue(&c_stride, MTL::DataTypeUInt, 17);
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
  uint32_t params[] = {bench.M, bench.M * bench.K, bench.N * bench.K, bench.M * bench.N, 0};

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
    encoder->setBytes(params, sizeof(params), 3);
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
    encoder->setBytes(params, sizeof(params), 2);
    encoder->useResource(scratch, MTL::ResourceUsageRead);
    encoder->useResource(buffer_c, MTL::ResourceUsageWrite);
    if ((bench.N % 2) == 0) {
      encoder->dispatchThreadgroups(
          MTL::Size((bench.M * bench.N / 2 + 255) / 256, bench.batch, 1),
          MTL::Size(256, 1, 1));
    } else {
      encoder->dispatchThreadgroups(
          MTL::Size((bench.M * bench.N + 255) / 256, bench.batch, 1),
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
  // A fixed iteration count under-warms small dispatches after shader compilation.
  const double minimum_warmup = std::getenv("CCV_NA_WARMUP_SECONDS") ?
      std::atof(std::getenv("CCV_NA_WARMUP_SECONDS")) : 0;
  double warmup_seconds = 0;
  for (int i = 0; samples.size() < (size_t)config.timed_iterations; ++i) {
    const double seconds =
        run_once(command_queue, bundle, bench, buffer_a, buffer_b, buffer_c, scratch);
    if (std::isnan(seconds)) {
      return false;
    }
    warmup_seconds += seconds;
    if (i >= config.warmup_iterations && warmup_seconds >= minimum_warmup) {
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
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  BenchmarkCase bench;
  BenchmarkConfig config;
  int forced_split_k = 0;
  int forced_thread_barrier = -1;
  bool load_m = true;
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
  if (argc >= 8) {
    load_m = std::strtoul(argv[7], nullptr, 10) != 0;
  }
  if (argc >= 9) {
    forced_thread_barrier = std::atoi(argv[8]);
  }
  simd::ushort3 block_dimensions {128, 64, 64};
  if (argc >= 12) {
    block_dimensions = simd::ushort3 {
      (uint16_t)std::strtoul(argv[9], nullptr, 10),
      (uint16_t)std::strtoul(argv[10], nullptr, 10),
      (uint16_t)std::strtoul(argv[11], nullptr, 10)};
  }

  if (argc >= 15) {
    bench.batch = std::strtoul(argv[12], nullptr, 10);
    bench.transpose_a = std::atoi(argv[13]) != 0;
    bench.transpose_b = std::atoi(argv[14]) != 0;
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
            << " loadM=" << (load_m ? 1 : 0)
            << '\n';

  const size_t a_count = static_cast<size_t>(bench.M) * bench.K * bench.batch;
  const size_t b_count = static_cast<size_t>(bench.N) * bench.K * bench.batch;
  const size_t c_count = static_cast<size_t>(bench.M) * bench.N * bench.batch;
  auto buffer_a = NS::TransferPtr(
      device->newBuffer(a_count * sizeof(half_float), MTL::ResourceStorageModeShared));
  auto buffer_b = NS::TransferPtr(
      device->newBuffer(b_count * sizeof(half_float), MTL::ResourceStorageModeShared));
  auto buffer_c = NS::TransferPtr(
      device->newBuffer(c_count * sizeof(half_float), MTL::ResourceStorageModeShared));
  auto scratch = NS::TransferPtr(
      device->newBuffer(c_count * 8 * sizeof(half_float), MTL::ResourceStorageModeShared));
  if (!buffer_a || !buffer_b || !buffer_c || !scratch) {
    std::cerr << "Failed to allocate benchmark buffers.\n";
    pool->drain();
    return 1;
  }

  for (size_t i = 0; i < a_count; ++i)
    ((half_float*)buffer_a->contents())[i] = half_float(float(int((i * 17 + 31) % 127) - 63) / 128);
  for (size_t i = 0; i < b_count; ++i)
    ((half_float*)buffer_b->contents())[i] = half_float(float(int((i * 29 + 13) % 127) - 63) / 128);
  const uint32_t group_m = (bench.M >= 4096) ? 4096 : 0;
  std::vector<uint32_t> group_ns = {0};
  if (bench.N >= 4096 && bench.transpose_b) {
    group_ns.push_back(4096);
  }
  std::vector<VariantResult> results;
  std::vector<uint16_t> split_ks = {1, 2, 4, 8};
  if (forced_split_k > 0 &&
      std::find(split_ks.begin(), split_ks.end(),
          static_cast<uint16_t>(forced_split_k)) == split_ks.end()) {
    split_ks.push_back(static_cast<uint16_t>(forced_split_k));
  }
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
                << " loadM=" << (load_m ? 1 : 0)
                << " groupN=" << group_n << std::endl;
      auto bundle = create_pipeline_bundle(
          device.get(), bench, split_k, load_m, group_m, group_n,
          forced_thread_barrier, block_dimensions);
      Stats stats;
      bool valid = benchmark_variant(
          command_queue.get(),
          bundle,
          bench,
          config,
          buffer_a.get(),
          buffer_b.get(),
          buffer_c.get(),
          split_k > 1 ? scratch.get() : nullptr,
          &stats);
      // Check batch offsets, transpose layouts and split reduction against CPU FP32.
      double max_error = 0;
      for (uint32_t z = 0; z < bench.batch; ++z) {
        for (uint32_t sample = 0; sample < 32; ++sample) {
          const uint32_t row = sample == 0 ? bench.M - 1 : (sample * 997) % bench.M;
          const uint32_t col = sample == 0 ? bench.N - 1 : (sample * 293) % bench.N;
          float expected = 0;
          for (uint32_t k = 0; k < bench.K; ++k) {
            const size_t ai = size_t(z) * bench.M * bench.K + (bench.transpose_a ? size_t(k) * bench.M + row : size_t(row) * bench.K + k);
            const size_t bi = size_t(z) * bench.N * bench.K + (bench.transpose_b ? size_t(col) * bench.K + k : size_t(k) * bench.N + col);
            expected += float(((half_float*)buffer_a->contents())[ai]) * float(((half_float*)buffer_b->contents())[bi]);
          }
          const float actual = float(((half_float*)buffer_c->contents())[(size_t(z) * bench.M + row) * bench.N + col]);
          if (!std::isfinite(actual)) valid = false;
          max_error = std::max(max_error, double(std::abs(actual - expected) / std::max(1.0f, std::abs(expected))));
        }
      }
      valid = valid && max_error < 0.03;
      std::cout << "validation max_relative_error=" << max_error << " passed=" << valid << '\n';
      results.push_back(VariantResult{
          .split_k = split_k,
          .group_n = group_n,
          .thread_barrier_over_k =
              forced_thread_barrier >= 0 ?
                  (forced_thread_barrier != 0) :
                  NAMatMulDescriptor::threadBarrierOverK(bench.K, split_k),
          .stats = stats,
          .valid = valid,
      });
    }
  }

  for (const auto& result : results) {
    std::ostringstream label;
    label << "loadM=" << (load_m ? 1 : 0)
          << " groupM=" << group_m
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

  fflush(stdout);
  fflush(stderr);
  std::_Exit(std::all_of(results.begin(), results.end(), [](const auto& r) { return r.valid; }) ? 0 : 2);
}
