#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/HyperConnectionDescriptor.hpp"
#include "nnc/mfa/kernels/HyperConnectionKernel.hpp"
#include "nnc/mfa/kernels/ShaderCache.hpp"

namespace {

constexpr MTL::ResourceOptions kSharedResourceOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;

struct BenchmarkConfig {
  int warmup_iterations = 5;
  int timed_iterations = 20;
  int duplicated_dispatches = 64;
};

struct HyperConnectionCase {
  const char* label;
  uint32_t row_count;
  uint32_t count;
  uint32_t hidden;
  uint32_t operation;
};

struct Stats {
  double average_seconds;
  double median_seconds;
  double min_seconds;
};

struct Buffers {
  NS::SharedPtr<MTL::Buffer> values[8];
};

NS::SharedPtr<MTL::Buffer> make_buffer(MTL::Device* const device, const size_t count)
{
  const size_t length = std::max<size_t>(count, 1) * sizeof(float);
  auto buffer = NS::TransferPtr(device->newBuffer(length, kSharedResourceOptions));
  if (buffer)
    std::memset(buffer->contents(), 0, length);
  return buffer;
}

Buffers create_buffers(MTL::Device* const device, const HyperConnectionCase& bench)
{
  const size_t rows = bench.row_count;
  const size_t hc = bench.count;
  const size_t hidden = bench.hidden;
  const size_t mix_dim = 2 * hc + hc * hc;
  Buffers buffers;
  if (bench.operation == 2)
  {
    buffers.values[0] = make_buffer(device, rows * hidden);
    buffers.values[1] = make_buffer(device, rows * hc * hidden);
    buffers.values[2] = make_buffer(device, rows * hc);
    buffers.values[3] = make_buffer(device, rows * hc * hc);
    buffers.values[4] = make_buffer(device, rows * hc * hidden);
    buffers.values[5] = make_buffer(device, 1);
    buffers.values[6] = make_buffer(device, 1);
    buffers.values[7] = make_buffer(device, 1);
  } else {
    buffers.values[0] = make_buffer(device, rows * mix_dim);
    buffers.values[1] = make_buffer(device, 3);
    buffers.values[2] = make_buffer(device, mix_dim);
    buffers.values[3] = make_buffer(device, bench.operation == 1 ? rows * hc * hidden : 1);
    buffers.values[4] = make_buffer(device, bench.operation == 0 ? rows * hc : 1);
    buffers.values[5] = make_buffer(device, rows * hc);
    buffers.values[6] = make_buffer(device, rows * hc * hc);
    buffers.values[7] = make_buffer(device, bench.operation == 1 ? rows * hidden : 1);
  }
  return buffers;
}

bool benchmark(const BenchmarkConfig& config, const std::function<double()>& run_once, Stats* const stats)
{
  std::vector<double> samples;
  for (int i = 0; i < config.warmup_iterations + config.timed_iterations; ++i)
  {
    const double elapsed = run_once();
    if (!(elapsed >= 0))
      return false;
    if (i >= config.warmup_iterations)
      samples.push_back(elapsed / config.duplicated_dispatches);
  }
  std::sort(samples.begin(), samples.end());
  stats->average_seconds = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  stats->median_seconds = samples[samples.size() / 2];
  stats->min_seconds = samples.front();
  return true;
}

double run_once(MTL::CommandQueue* const command_queue, PipelineValue<HyperConnectionKernel>* const pipeline_value, const HyperConnectionCase& bench, const BenchmarkConfig& config, const Buffers& buffers)
{
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline_value->pipeline.get());
  for (uint32_t i = 0; i < 8; ++i)
    encoder->setBuffer(buffers.values[i].get(), 0, i);
  const MTL::Size grid_size(bench.row_count, 1, 1);
  const MTL::Size group_size(bench.operation == 0 ? 32 : 256, 1, 1);
  for (int i = 0; i < config.duplicated_dispatches; ++i)
    encoder->dispatchThreadgroups(grid_size, group_size);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  return command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
}

} // namespace

int main()
{
  std::cout << std::unitbuf;
  std::cerr << std::unitbuf;
  const BenchmarkConfig config;
  const std::vector<HyperConnectionCase> cases = {
    { "split_decode", 1, 4, 0, 0 },
    { "split_16", 16, 4, 0, 0 },
    { "split_256", 256, 4, 0, 0 },
    { "weighted_decode", 1, 4, 4096, 1 },
    { "weighted_16", 16, 4, 4096, 1 },
    { "weighted_256", 256, 4, 4096, 1 },
    { "split_hc3_256", 256, 3, 0, 0 },
  };
  const HyperConnectionDescriptor hash_a { 1, 4, 0, 20, 1e-6f, 0 };
  const HyperConnectionDescriptor hash_b { 256, 4, 4096, 20, 1e-6f, 1 };
  if (std::hash<HyperConnectionDescriptor> {}(hash_a) == std::hash<HyperConnectionDescriptor> {}(hash_b))
  {
    std::cerr << "HyperConnection descriptor hash smoke check failed.\n";
    return 1;
  }
  auto* const pool = NS::AutoreleasePool::alloc()->init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device)
  {
    std::cerr << "Metal device unavailable.\n";
    return 1;
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  ShaderCache shader_cache;
  const DeviceProperties dprops {};
  std::cout << "device=" << device->name()->utf8String()
            << " warmup=" << config.warmup_iterations
            << " iters=" << config.timed_iterations
            << " duplicated_dispatches=" << config.duplicated_dispatches << '\n';
  for (const auto& bench : cases)
  {
    std::cout << "prepare=" << bench.label << '\n';
    const HyperConnectionDescriptor descriptor {
      bench.row_count, bench.count, bench.hidden, 20, 1e-6f, bench.operation
    };
    auto pipeline_value = shader_cache.findKernel<HyperConnectionKernel, HyperConnectionDescriptor, HyperConnectionKernelDescriptor>(descriptor, device.get(), dprops);
    const Buffers buffers = create_buffers(device.get(), bench);
    Stats stats;
    if (!benchmark(config, [&]() {
      return run_once(command_queue.get(), pipeline_value, bench, config, buffers);
    }, &stats))
      return 1;
    std::cout << std::left << std::setw(20) << bench.label
              << " rows=" << std::setw(4) << bench.row_count
              << " hc=" << bench.count
              << " hidden=" << std::setw(4) << bench.hidden
              << " median_us=" << std::fixed << std::setprecision(3) << stats.median_seconds * 1e6
              << " average_us=" << stats.average_seconds * 1e6
              << " min_us=" << stats.min_seconds * 1e6 << '\n';
  }
  (void)pool;
  return 0;
}
