#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "nnc/mfa/ccv_nnc_mfa_error.hpp"
#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/GatedDeltaDescriptor.hpp"
#include "nnc/mfa/kernels/GatedDeltaKernel.hpp"
#include "nnc/mfa/kernels/ShaderCache.hpp"

namespace {

constexpr MTL::ResourceOptions kSharedResourceOptions =
    MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeTracked;

struct BenchmarkConfig {
  int warmup_iterations = 3;
  int timed_iterations = 10;
  int duplicated_dispatches = 8;
  bool log_decay_input = true;
};

struct GatedDeltaCase {
  std::string label = "qwen_decode";
  uint32_t B = 1;
  uint32_t T = 1;
  uint32_t Hk = 16;
  uint32_t Hv = 64;
  uint32_t Dk = 192;
  uint32_t Dv = 128;
};

struct Stats {
  double average_seconds = 0;
  double best3_average_seconds = 0;
  double median_seconds = 0;
  double min_seconds = 0;
  double max_seconds = 0;
};

struct Buffers {
  NS::SharedPtr<MTL::Buffer> q;
  NS::SharedPtr<MTL::Buffer> k;
  NS::SharedPtr<MTL::Buffer> v;
  NS::SharedPtr<MTL::Buffer> log_decay;
  NS::SharedPtr<MTL::Buffer> beta;
  NS::SharedPtr<MTL::Buffer> state_in;
  NS::SharedPtr<MTL::Buffer> y;
  NS::SharedPtr<MTL::Buffer> state_out;
};

size_t qk_count(const GatedDeltaCase& bench)
{
  return (size_t)bench.B * bench.T * bench.Hk * bench.Dk;
}

size_t v_count(const GatedDeltaCase& bench)
{
  return (size_t)bench.B * bench.T * bench.Hv * bench.Dv;
}

size_t gate_count(const GatedDeltaCase& bench)
{
  return (size_t)bench.B * bench.T * bench.Hv;
}

size_t state_count(const GatedDeltaCase& bench)
{
  return (size_t)bench.B * bench.Hv * bench.Dv * bench.Dk;
}

template <typename T>
std::vector<T> make_data(size_t count, float scale, uint32_t salt)
{
  std::vector<T> values(count);
  for (size_t i = 0; i < count; ++i) {
    const int centered = (int)((i * 1103515245u + salt * 12345u) & 0x3fu) - 32;
    values[i] = (T)(centered * scale);
  }
  return values;
}

std::vector<float> make_decay(size_t count, bool log_decay_input)
{
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i)
  {
    const float log_decay = -0.01f * (float)((i % 7) + 1);
    values[i] = log_decay_input ? log_decay : std::exp(log_decay);
  }
  return values;
}

std::vector<float> make_beta(size_t count)
{
  std::vector<float> values(count);
  for (size_t i = 0; i < count; ++i)
    values[i] = 0.1f + 0.03f * (float)(i % 5);
  return values;
}

NS::SharedPtr<MTL::Buffer> make_buffer(MTL::Device* device, const std::vector<float>& values)
{
  return NS::TransferPtr(device->newBuffer(
      values.data(), values.size() * sizeof(float), kSharedResourceOptions));
}

Buffers create_buffers(MTL::Device* device, const GatedDeltaCase& bench, bool log_decay_input)
{
  Buffers buffers;
  buffers.q = make_buffer(device, make_data<float>(qk_count(bench), 0.0015f, 1));
  buffers.k = make_buffer(device, make_data<float>(qk_count(bench), 0.0013f, 2));
  buffers.v = make_buffer(device, make_data<float>(v_count(bench), 0.0011f, 3));
  buffers.log_decay = make_buffer(device, make_decay(gate_count(bench), log_decay_input));
  buffers.beta = make_buffer(device, make_beta(gate_count(bench)));
  buffers.state_in = make_buffer(device, make_data<float>(state_count(bench), 0.0007f, 4));
  buffers.y = NS::TransferPtr(device->newBuffer(v_count(bench) * sizeof(float), kSharedResourceOptions));
  buffers.state_out = NS::TransferPtr(device->newBuffer(state_count(bench) * sizeof(float), kSharedResourceOptions));
  return buffers;
}

GatedDeltaDescriptor create_descriptor(const GatedDeltaCase& bench, bool log_decay_input)
{
  GatedDeltaDescriptor descriptor;
  descriptor.batchSize = bench.B;
  descriptor.sequenceLength = bench.T;
  descriptor.keyHeadCount = bench.Hk;
  descriptor.valueHeadCount = bench.Hv;
  descriptor.keyDim = bench.Dk;
  descriptor.valueDim = bench.Dv;
  descriptor.inputMemoryPrecision = GEMMOperandPrecision::FP32;
  descriptor.logDecay = log_decay_input;
  return descriptor;
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
      samples.push_back(elapsed / (double)config.duplicated_dispatches);
  }
  if (samples.empty())
    return false;
  stats->average_seconds =
      std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  std::sort(samples.begin(), samples.end());
  const size_t best_count = std::min<size_t>(3, samples.size());
  stats->best3_average_seconds =
      std::accumulate(samples.begin(), samples.begin() + best_count, 0.0) / best_count;
  stats->median_seconds = samples[samples.size() / 2];
  stats->min_seconds = samples.front();
  stats->max_seconds = samples.back();
  return true;
}

double run_once(
    MTL::CommandQueue* command_queue,
    PipelineValue<GatedDeltaKernel>* pipeline_value,
    const GatedDeltaCase& bench,
    const BenchmarkConfig& config,
    const Buffers& buffers)
{
  const auto start = std::chrono::steady_clock::now();
  auto command_buffer = NS::TransferPtr(command_queue->commandBuffer());
  auto encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
  encoder->setComputePipelineState(pipeline_value->pipeline.get());
  encoder->setThreadgroupMemoryLength(pipeline_value->kernel->threadgroupMemoryAllocation, 0);
  encoder->setBuffer(buffers.q.get(), 0, 0);
  encoder->setBuffer(buffers.k.get(), 0, 1);
  encoder->setBuffer(buffers.v.get(), 0, 2);
  encoder->setBuffer(buffers.log_decay.get(), 0, 3);
  encoder->setBuffer(buffers.beta.get(), 0, 4);
  encoder->setBuffer(buffers.state_in.get(), 0, 5);
  encoder->setBuffer(buffers.y.get(), 0, 6);
  encoder->setBuffer(buffers.state_out.get(), 0, 7);
  const MTL::Size grid_size(1, (bench.Dv + 3) / 4, bench.B * bench.Hv);
  for (int i = 0; i < config.duplicated_dispatches; ++i)
    encoder->dispatchThreadgroups(grid_size, pipeline_value->kernel->threadgroupSize);
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

double modeled_flops(const GatedDeltaCase& bench)
{
  const double rows = (double)bench.B * bench.T * bench.Hv * bench.Dv;
  return rows * (7.0 * (double)bench.Dk + 2.0);
}

double modeled_legacy_state_bytes(const GatedDeltaCase& bench)
{
  const double state_bytes = (double)state_count(bench) * sizeof(float);
  return 2.0 * state_bytes + (double)bench.T * 4.0 * state_bytes;
}

double modeled_register_state_bytes(const GatedDeltaCase& bench)
{
  const double state_bytes = (double)state_count(bench) * sizeof(float);
  return 2.0 * state_bytes;
}

void print_stats(const GatedDeltaCase& bench, const BenchmarkConfig& config, const Stats& stats)
{
  const double flops = modeled_flops(bench);
  const double legacy_state_gib =
      modeled_legacy_state_bytes(bench) / (1024.0 * 1024.0 * 1024.0);
  const double register_state_gib =
      modeled_register_state_bytes(bench) / (1024.0 * 1024.0 * 1024.0);
  std::cout << bench.label
            << " B=" << bench.B
            << " T=" << bench.T
            << " Hk=" << bench.Hk
            << " Hv=" << bench.Hv
            << " Dk=" << bench.Dk
            << " Dv=" << bench.Dv
            << " log_decay_input=" << (config.log_decay_input ? 1 : 0)
            << " avg_ms=" << std::fixed << std::setprecision(4) << stats.average_seconds * 1e3
            << " best3_avg_ms=" << stats.best3_average_seconds * 1e3
            << " median_ms=" << stats.median_seconds * 1e3
            << " min_ms=" << stats.min_seconds * 1e3
            << " max_ms=" << stats.max_seconds * 1e3
            << " modeled_tflops_median=" << flops / stats.median_seconds / 1e12
            << " legacy_state_GiBps_median=" << legacy_state_gib / stats.median_seconds
            << " register_state_GiBps_median=" << register_state_gib / stats.median_seconds
            << " legacy_to_register_state_ratio=" << (legacy_state_gib / register_state_gib)
            << '\n';
}

void print_usage(const char* argv0)
{
  std::cerr
      << "usage: " << argv0
      << " [--single] [--label NAME] [--B N] [--T N] [--Hk N] [--Hv N] [--Dk N] [--Dv N]"
      << " [--warmup N] [--iters N] [--dups N] [--log-decay-input|--decay-input]\n";
}

bool parse_u32(const char* text, uint32_t* value)
{
  char* end = nullptr;
  const unsigned long parsed = std::strtoul(text, &end, 10);
  if (!end || *end != 0)
    return false;
  *value = (uint32_t)parsed;
  return true;
}

bool parse_int(const char* text, int* value)
{
  char* end = nullptr;
  const long parsed = std::strtol(text, &end, 10);
  if (!end || *end != 0)
    return false;
  *value = (int)parsed;
  return true;
}

} // namespace

int main(int argc, char** argv)
{
  BenchmarkConfig config;
  GatedDeltaCase single_case;
  bool single = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto require_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << name << " requires a value.\n";
        return nullptr;
      }
      return argv[++i];
    };
    if (arg == "--single") {
      single = true;
    } else if (arg == "--label") {
      const char* value = require_value("--label");
      if (!value)
        return 1;
      single_case.label = value;
    } else if (arg == "--B" || arg == "--T" || arg == "--Hk" || arg == "--Hv" || arg == "--Dk" || arg == "--Dv") {
      const char* value = require_value(arg.c_str());
      if (!value)
        return 1;
      uint32_t parsed = 0;
      if (!parse_u32(value, &parsed)) {
        std::cerr << "invalid integer for " << arg << ".\n";
        return 1;
      }
      if (arg == "--B")
        single_case.B = parsed;
      else if (arg == "--T")
        single_case.T = parsed;
      else if (arg == "--Hk")
        single_case.Hk = parsed;
      else if (arg == "--Hv")
        single_case.Hv = parsed;
      else if (arg == "--Dk")
        single_case.Dk = parsed;
      else
        single_case.Dv = parsed;
      single = true;
    } else if (arg == "--warmup" || arg == "--iters" || arg == "--dups") {
      const char* value = require_value(arg.c_str());
      if (!value)
        return 1;
      int parsed = 0;
      if (!parse_int(value, &parsed)) {
        std::cerr << "invalid integer for " << arg << ".\n";
        return 1;
      }
      if (arg == "--warmup")
        config.warmup_iterations = parsed;
      else if (arg == "--iters")
        config.timed_iterations = parsed;
      else
        config.duplicated_dispatches = parsed;
    } else if (arg == "--log-decay-input") {
      config.log_decay_input = true;
    } else if (arg == "--decay-input") {
      config.log_decay_input = false;
    } else if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << '\n';
      print_usage(argv[0]);
      return 1;
    }
  }
  if (config.warmup_iterations < 0 || config.timed_iterations <= 0 || config.duplicated_dispatches <= 0) {
    std::cerr << "invalid benchmark iteration counts.\n";
    return 1;
  }

  std::vector<GatedDeltaCase> cases;
  if (single) {
    cases.push_back(single_case);
  } else {
    cases = {
      { "qwen_decode", 1, 1, 16, 64, 192, 128 },
      { "qwen_prefill_16", 1, 16, 16, 64, 192, 128 },
      { "qwen_prefill_64", 1, 64, 16, 64, 192, 128 },
      { "qwen_prefill_256", 1, 256, 16, 64, 192, 128 },
    };
  }

  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) {
    std::cerr << "Metal device unavailable.\n";
    return 1;
  }
  auto command_queue = NS::TransferPtr(device->newCommandQueue());
  if (!command_queue) {
    std::cerr << "Metal command queue unavailable.\n";
    return 1;
  }
  std::cout << "device=" << device->name()->utf8String()
            << " warmup=" << config.warmup_iterations
            << " iters=" << config.timed_iterations
            << " duplicated_dispatches=" << config.duplicated_dispatches
            << " log_decay_input=" << (config.log_decay_input ? 1 : 0)
            << '\n';

  ShaderCache shader_cache;
  DeviceProperties dprops{};
  for (const auto& bench : cases) {
    if (bench.B == 0 || bench.T == 0 || bench.Hk == 0 || bench.Hv == 0 ||
        bench.Dk == 0 || bench.Dv == 0 || (bench.Hv % bench.Hk) != 0) {
      std::cerr << "invalid shape for " << bench.label << ".\n";
      return 1;
    }
    const auto descriptor = create_descriptor(bench, config.log_decay_input);
    auto pipeline_value =
        shader_cache.findKernel<GatedDeltaKernel, GatedDeltaDescriptor, GatedDeltaKernelDescriptor>(
            descriptor, device.get(), dprops);
    Buffers buffers = create_buffers(device.get(), bench, config.log_decay_input);
    if (!buffers.q || !buffers.k || !buffers.v || !buffers.log_decay || !buffers.beta ||
        !buffers.state_in || !buffers.y || !buffers.state_out) {
      std::cerr << "buffer allocation failed for " << bench.label << ".\n";
      return 1;
    }
    Stats stats;
    const bool ok = benchmark(config, [&]() {
      return run_once(command_queue.get(), pipeline_value, bench, config, buffers);
    }, &stats);
    if (!ok) {
      std::cerr << "benchmark failed for " << bench.label << ".\n";
      return 1;
    }
    print_stats(bench, config, stats);
  }
  (void)pool;
  return 0;
}
