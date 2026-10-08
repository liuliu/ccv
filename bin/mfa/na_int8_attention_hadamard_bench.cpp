// Paired timings using production descriptors/pipelines. Hadamard is opt-in and
// forward-only. Includes CPU quantizer and sampled original-basis SDPA checks.
#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionDescriptor.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionKernel.hpp"
extern "C" {
#include "nnc/ccv_nnc_easy.h"
#ifdef CCV_NA_HADAMARD_EMBEDDED
// Reuse the reference row helper without linking the entire command registry.
#include "nnc/cmd/walsh_hadamard_transform/ccv_nnc_walsh_hadamard_transform_cpu_ref.c"
#endif
}

static double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return (values[(values.size() - 1) / 2] + values[values.size() / 2]) * 0.5;
}

#ifdef CCV_NA_HADAMARD_EMBEDDED
int na_int8_tuning_run(int argc, char** argv)
#else
int main(int argc, char** argv)
#endif
{
  if (argc < 9 || argc > 12) {
    fprintf(stderr, "Usage: %s R C B Hq Hk precision causal dynamic_flags [distribution=0] [varlen=0] [D=128]\n"
        "precision: 0=FP16 1=BF16 2=FP32; distribution: 0=normal 1=channel-outliers 2=uniform 3=zeros\n", argv[0]);
    return 2;
  }
  std::array<uint32_t, 11> arguments{};
  for (int i = 1; i < argc; ++i) {
    char* end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(argv[i], &end, 10);
    if (argv[i][0] == '-' || end == argv[i] || *end || errno || value > UINT32_MAX) return 2;
    arguments[i - 1] = uint32_t(value);
  }
  const uint32_t R = arguments[0], C = arguments[1], B = arguments[2];
  const uint32_t Hq = arguments[3], Hk = arguments[4];
  const uint32_t precision = arguments[5], causal = arguments[6], dynamic = arguments[7];
  const uint32_t distribution = arguments[8];
  const bool varlen = arguments[9];
  const uint32_t D = argc > 11 ? arguments[10] : 128;
  const char* quant_only_arg = getenv("CCV_NA_QUANT_ONLY");
  if (quant_only_arg && strcmp(quant_only_arg, "0") && strcmp(quant_only_arg, "1")) return 2;
  const bool quant_only = quant_only_arg && !strcmp(quant_only_arg, "1");
  if (!R || !C || !B || !Hq || !Hk || Hq % Hk || precision > 2 || causal > 1 ||
      (dynamic & ~3u) || distribution > 3 || arguments[9] > 1 || B > 65535 ||
      Hq > 65535 || R > 1048576 || C > 1048576 || D < 8 || D > 256 || D % 8) return 2;
  const uint32_t hadamard_block = std::min<uint32_t>(D & -D, 256);
  const size_t q_count = size_t(B) * R * Hq * D, kv_count = size_t(B) * C * Hk * D;
  // Precision conversion helpers take int counts; shader indices are uint.
  if (q_count > INT_MAX || kv_count > INT_MAX) return 2;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
#ifndef CCV_NA_HADAMARD_EMBEDDED
  ccv_nnc_init();
#endif
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) { fprintf(stderr, "No Metal device available.\n"); return 2; }
  auto context = ccv_nnc_init_mfa_context(device.get());
  if (!ccv_nnc_mfa_has_neural_accelerators(context)) {
    fprintf(stderr, "Neural accelerators are required for this benchmark.\n");
    ccv_nnc_deinit_mfa_context(context);
    return 2;
  }
  auto queue = NS::TransferPtr(device->newCommandQueue());
  const size_t element_size = precision == 2 ? 4 : 2;
  const uint32_t q_tiles = (R + 15) / 16, k_tiles = (C + 63) / 64;
  const size_t counts[] = {q_count, kv_count, kv_count};
  std::array<NS::SharedPtr<MTL::Buffer>, 3> inputs;
  std::array<std::vector<float>, 3> data;
  std::mt19937 random(42);
  std::normal_distribution<float> normal(0, 0.5f);
  std::uniform_real_distribution<float> uniform(-0.5f, 0.5f);
  for (int operand = 0; operand < 3; ++operand) {
    data[operand].resize(counts[operand]);
    for (size_t i = 0; i < counts[operand]; ++i) {
      float x = distribution == 3 ? 0 : (distribution == 2 ? uniform(random) : normal(random));
      if (distribution == 1 && operand < 2 && i % D == 7) x *= 10;
      data[operand][i] = x;
    }
    inputs[operand] = NS::TransferPtr(device->newBuffer(counts[operand] * element_size, MTL::ResourceStorageModeShared));
    if (!inputs[operand]) return 2;
    if (precision == 0) {
      ccv_float_to_half_precision(data[operand].data(), static_cast<uint16_t*>(inputs[operand]->contents()), counts[operand]);
      ccv_half_precision_to_float(static_cast<uint16_t*>(inputs[operand]->contents()), data[operand].data(), counts[operand]);
    } else if (precision == 1) {
      ccv_float_to_bfloat(data[operand].data(), static_cast<uint16_t*>(inputs[operand]->contents()), counts[operand]);
      ccv_bfloat_to_float(static_cast<uint16_t*>(inputs[operand]->contents()), data[operand].data(), counts[operand]);
    } else memcpy(inputs[operand]->contents(), data[operand].data(), counts[operand] * 4);
  }
  std::array<std::vector<int>, 2> seq;
  std::array<NS::SharedPtr<MTL::Buffer>, 2> seq_buffers;
  for (int operand = 0; operand < 2; ++operand) {
    const uint32_t length = operand ? C : R;
    seq[operand].push_back(0);
    for (uint32_t batch = 0; batch < B; ++batch)
      seq[operand].push_back(seq[operand].back() + (varlen ? length - batch * 13 % length : length));
    seq_buffers[operand] = NS::TransferPtr(device->newBuffer(seq[operand].data(), seq[operand].size() * sizeof(int), MTL::ResourceStorageModeShared));
  }
  NAInt8AttentionDescriptor descriptor;
  descriptor.matrixDimensions = simd::uint3{R, C, D};
  descriptor.batchDimension = B; descriptor.Hq = Hq; descriptor.Hk = Hk;
  descriptor.ioPrecision = precision == 0 ? GEMMOperandPrecision::FP16 :
      (precision == 1 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP32);
  descriptor.scale = 1.0f / std::sqrt(float(D));
  descriptor.isCausal = causal; descriptor.isVarlen = varlen;
  descriptor.loadR = dynamic & 1; descriptor.loadC = dynamic & 2;
  descriptor.lowPrecisionIntermediates = precision == 0;
  descriptor.batchStrides[AttentionOperand::Q] = B > 1 ? R * Hq * D : 0;
  descriptor.batchStrides[AttentionOperand::K] = B > 1 ? C * Hk * D : 0;
  descriptor.batchStrides[AttentionOperand::V] = B > 1 ? C * Hk * D : 0;
  descriptor.batchStrides[AttentionOperand::O] = B > 1 ? R * Hq * D : 0;
  std::array<PipelineValue<NAInt8AttentionKernel>*, 2> pipelines;
  std::array<std::array<NS::SharedPtr<MTL::Buffer>, 8>, 2> scratch;
  const size_t bytes[] = {q_count, kv_count, kv_count, size_t(B) * Hq * q_tiles * 4,
      size_t(B) * Hk * k_tiles * 4, size_t(B) * Hk * k_tiles * 4,
      size_t(B) * Hk * D * 4, q_count * element_size};
  std::array<NS::SharedPtr<MTL::Buffer>, 2> l;
  for (int variant = 0; variant < 2; ++variant) {
    descriptor.qkHadamard = variant;
    pipelines[variant] = context->kernel_cache.findKernel<NAInt8AttentionKernel, NAInt8AttentionDescriptor,
        NAInt8AttentionKernelDescriptor>(descriptor, device.get(), context->device_properties);
    for (int i = 0; i < 8; ++i) {
      scratch[variant][i] = NS::TransferPtr(device->newBuffer(bytes[i], MTL::ResourceStorageModeShared));
      if (!scratch[variant][i]) return 2;
      memset(scratch[variant][i]->contents(), 0, bytes[i]);
    }
    l[variant] = NS::TransferPtr(device->newBuffer(size_t(B) * Hq * R * 4, MTL::ResourceStorageModeShared));
    if (!l[variant]) return 2;
    printf("resources variant=%d q_tg_bytes=%lu k_tg_bytes=%lu q_threads=%lu k_threads=%lu\n", variant,
        pipelines[variant]->second->staticThreadgroupMemoryLength(), pipelines[variant]->third->staticThreadgroupMemoryLength(),
        pipelines[variant]->second->maxTotalThreadsPerThreadgroup(), pipelines[variant]->third->maxTotalThreadsPerThreadgroup());
  }
  const auto& baseline_source = pipelines[0]->kernel->source;
  const auto& hadamard_source = pipelines[1]->kernel->source;
  const size_t baseline_attention = baseline_source.find("kernel void int8_attention(");
  const size_t hadamard_attention = hadamard_source.find("kernel void int8_attention(");
  const bool attention_source_identical = baseline_attention != std::string::npos &&
      hadamard_attention != std::string::npos &&
      baseline_source.substr(baseline_attention) == hadamard_source.substr(hadamard_attention);
  printf("attention_source_identical=%d\n", attention_source_identical);
  if (!attention_source_identical) return 4;
  const uint32_t dimensions[] = {R, C, B > 1 ? R * Hq * D : 0, B > 1 ? C * Hk * D : 0,
      B > 1 ? C * Hk * D : 0, B > 1 ? R * Hq * D : 0, 0, 0, 0, 0,
      B > 1 ? Hq * q_tiles : 0, B > 1 ? Hk * k_tiles : 0, B > 1 ? Hk * k_tiles : 0, 0, 0, 0, 0};
  auto run = [&](int variant, bool full, int repeats) {
    auto iteration_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto cb = queue->commandBuffer();
    auto p = pipelines[variant]; auto& buffers = scratch[variant];
    auto dispatch = [&](MTL::ComputePipelineState* pipeline, MTL::Size grid, MTL::Size threads,
                        const std::vector<std::pair<int, MTL::Buffer*>>& bindings, size_t tg_bytes = 0) {
      auto encoder = cb->computeCommandEncoder();
      encoder->setComputePipelineState(pipeline);
      if (dynamic) encoder->setBytes(dimensions, sizeof(dimensions), 21);
      if (varlen) {
        encoder->setBuffer(seq_buffers[0].get(), 0, 17);
        encoder->setBuffer(seq_buffers[1].get(), 0, 18);
      }
      for (const auto& binding : bindings) encoder->setBuffer(binding.second, 0, binding.first);
      if (tg_bytes) encoder->setThreadgroupMemoryLength(tg_bytes, 0);
      encoder->dispatchThreadgroups(grid, threads);
      encoder->endEncoding();
    };
    for (int repeat = 0; repeat < repeats; ++repeat) {
      dispatch(p->second.get(), MTL::Size(q_tiles, Hq, B), MTL::Size(128, 1, 1),
          {{0, inputs[0].get()}, {1, buffers[0].get()}, {2, buffers[3].get()}});
      dispatch(p->third.get(), MTL::Size(k_tiles, Hk, B), MTL::Size(256, 1, 1),
          {{0, inputs[1].get()}, {1, buffers[1].get()}, {2, buffers[4].get()}, {17, seq_buffers[1].get()}});
      if (full) {
        dispatch(pipelines[0]->fifth.get(), p->kernel->vMeanThreadgroupsPerGrid(B),
            MTL::Size(p->kernel->vMeanThreadgroupSize(), 1, 1), {{0, inputs[2].get()}, {1, buffers[6].get()}, {17, seq_buffers[1].get()}});
        dispatch(pipelines[0]->fourth.get(), MTL::Size(k_tiles, Hk, B), MTL::Size(256, 1, 1),
            {{0, inputs[2].get()}, {1, buffers[2].get()}, {2, buffers[5].get()}, {3, buffers[6].get()}, {17, seq_buffers[1].get()}});
        dispatch(pipelines[0]->pipeline.get(), p->kernel->threadgroupsPerGrid(B, R),
            MTL::Size(p->kernel->threadgroupSize(pipelines[0]->pipeline.get()), 1, 1),
            {{0, buffers[0].get()}, {1, buffers[1].get()}, {2, buffers[2].get()}, {3, buffers[7].get()},
             {4, l[variant].get()}, {10, buffers[3].get()}, {11, buffers[4].get()},
             {12, buffers[5].get()}, {14, buffers[6].get()}}, p->kernel->threadgroupMemoryAllocation());
      }
    }
    cb->commit(); cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted) {
      fprintf(stderr, "GPU failed: %s\n", cb->error() ? cb->error()->localizedDescription()->utf8String() : "unknown");
      std::exit(3);
    }
    return (cb->GPUEndTime() - cb->GPUStartTime()) * 1000 / repeats;
  };
  for (int variant = 0; variant < 2; ++variant) run(variant, !quant_only, 1);
  bool correct = true;
  for (int variant = 0; variant < 2; ++variant) for (int operand = 0; operand < 2; ++operand) {
    const uint32_t tile_size = operand ? 64 : 16, tiles = operand ? k_tiles : q_tiles;
    const uint32_t heads = operand ? Hk : Hq, length = operand ? C : R;
    const auto gpu = static_cast<const int8_t*>(scratch[variant][operand]->contents());
    const auto scales = static_cast<const float*>(scratch[variant][3 + operand]->contents());
    double max_scale_diff = 0; int max_quant_diff = 0;
    for (uint32_t batch = 0; batch < B; ++batch) for (uint32_t head = 0; head < heads; ++head)
      for (uint32_t tile : {0u, tiles / 2, tiles - 1}) {
        const uint32_t actual_length = varlen ? seq[operand][batch + 1] - seq[operand][batch] : length;
        const uint32_t start = std::min(actual_length, tile * tile_size), extent = std::min(tile_size, actual_length - start);
        const size_t input_offset = size_t(varlen ? seq[operand][batch] : batch * length) * heads * D;
        std::vector<float> transformed(size_t(extent) * D); float maximum = 0;
        for (uint32_t row = 0; row < extent; ++row) {
          memcpy(transformed.data() + row * D, data[operand].data() + input_offset + ((start + row) * heads + head) * D, D * sizeof(float));
        }
        if (variant && extent) {
#ifdef CCV_NA_HADAMARD_EMBEDDED
          for (size_t block = 0; block < transformed.size(); block += hadamard_block)
            _ccv_nnc_walsh_hadamard_transform_row(transformed.data() + block, hadamard_block);
#else
          auto tensor = ccv_nnc_tensor(transformed.data(), CPU_TENSOR_NHWC(32F, int(extent * D / hadamard_block), int(hadamard_block)), 0);
          if (ccv_nnc_cmd_exec(CMD_WALSH_HADAMARD_TRANSFORM_FORWARD(1), ccv_nnc_no_hint, 0,
              TENSOR_LIST(&tensor), TENSOR_LIST(&tensor), 0) != CCV_NNC_EXEC_SUCCESS) return 4;
#endif
        }
        for (float x : transformed) maximum = std::max(maximum, std::abs(x));
        const float normalization = variant ? 1.0f / std::sqrt(float(hadamard_block)) : 1;
        const float scale = maximum > 0 ? maximum / 127 * normalization : 1.0f / 127;
        const float gpu_scale = scales[(batch * heads + head) * tiles + tile];
        max_scale_diff = std::max(max_scale_diff, double(std::abs(gpu_scale - scale) / scale));
        for (uint32_t row = 0; row < extent; ++row) for (uint32_t dim = 0; dim < D; ++dim) {
          const int expected = std::clamp(int(std::rint(transformed[row * D + dim] * (maximum > 0 ? 127 / maximum : 127))), -127, 127);
          max_quant_diff = std::max(max_quant_diff, std::abs(expected - int(gpu[input_offset + ((start + row) * heads + head) * D + dim])));
        }
      }
    printf("quant_check variant=%d operand=%d max_int8_diff=%d scale_rel_diff=%.9g\n", variant, operand, max_quant_diff, max_scale_diff);
    correct &= max_quant_diff <= 1 && max_scale_diff < 1e-5;
  }
  for (int variant = 0; variant < (quant_only ? 0 : 2); ++variant) {
    double squared_error = 0, norm = 0, maximum_error = 0;
    auto read_output = [&](size_t index) {
      const auto output = scratch[variant][7]->contents();
      if (precision == 2) return static_cast<float*>(output)[index];
      if (precision == 0) return float(static_cast<_Float16*>(output)[index]);
      const uint32_t bits = uint32_t(static_cast<uint16_t*>(output)[index]) << 16;
      float value; memcpy(&value, &bits, 4); return value;
    };
    for (uint32_t batch = 0; batch < B; ++batch) for (uint32_t head : {0u, Hq - 1}) {
      const uint32_t rows = varlen ? seq[0][batch + 1] - seq[0][batch] : R;
      const uint32_t cols = varlen ? seq[1][batch + 1] - seq[1][batch] : C;
      const size_t q_base = size_t(varlen ? seq[0][batch] : batch * R) * Hq * D;
      const size_t kv_base = size_t(varlen ? seq[1][batch] : batch * C) * Hk * D;
      for (uint32_t row : {0u, rows / 2, rows - 1}) {
        const size_t qi = q_base + (row * Hq + head) * D;
        const uint32_t kh = head / (Hq / Hk);
        const int end = causal ? std::clamp(int(row) + int(cols) - int(rows) + 1, 0, int(cols)) : cols;
        std::vector<double> scores(end); double maximum = -INFINITY, sum = 0;
        for (int col = 0; col < end; ++col) {
          double dot = 0; const size_t ki = kv_base + (col * Hk + kh) * D;
          for (uint32_t dim = 0; dim < D; ++dim) dot += double(data[0][qi + dim]) * data[1][ki + dim];
          scores[col] = dot * descriptor.scale; maximum = std::max(maximum, scores[col]);
        }
        for (auto& score : scores) { score = std::exp(score - maximum); sum += score; }
        for (uint32_t dim = 0; dim < D; ++dim) {
          double expected = 0;
          for (int col = 0; col < end; ++col) expected += scores[col] * data[2][kv_base + (col * Hk + kh) * D + dim];
          if (sum > 0) expected /= sum;
          const float actual = read_output(qi + dim);
          correct &= std::isfinite(actual);
          const double diff = actual - expected;
          squared_error += diff * diff; norm += expected * expected; maximum_error = std::max(maximum_error, std::abs(diff));
        }
      }
    }
    printf("attention_check variant=%d sampled_rel_l2=%.9g max_abs=%.9g\n", variant,
        std::sqrt(squared_error / std::max(norm, 1e-30)), maximum_error);
  }
  int samples = 20;
  if (const char* value = getenv("CCV_NA_SAMPLES")) {
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (end == value || *end || errno || parsed < 1 || parsed > 10000) return 2;
    samples = int(parsed);
  }
  printf("case device=%s R=%u C=%u D=%u B=%u Hq=%u Hk=%u precision=%u causal=%u dynamic=%u distribution=%u varlen=%d quant_only=%d hadamard_block=%u\n",
      device->name()->utf8String(), R, C, D, B, Hq, Hk, precision, causal, dynamic, distribution, varlen, quant_only, hadamard_block);
  // Both variants now use identical addresses and the same V/attention pipelines.
  // Only Q/K quantization changes in timed pairs; correctness above kept outputs separate.
  scratch[1] = scratch[0];
  l[1] = l[0];
  for (int full = 0; full < (quant_only ? 1 : 2); ++full) {
    const int repeats = full ? 1 : 8;
    std::array<std::vector<double>, 2> times;
    std::vector<double> ratios;
    double warm_ms = 0;
    for (int warmup = 0; warmup < 5 || warm_ms < 150; ++warmup)
      for (int variant = 0; variant < 2; ++variant) warm_ms += run(variant, full, repeats) * repeats;
    for (int round = 0; round < samples; ++round) {
      double pair[2];
      for (int j = 0; j < 2; ++j) {
        const int variant = (round + j) % 2;
        pair[variant] = run(variant, full, repeats); times[variant].push_back(pair[variant]);
      }
      ratios.push_back(pair[1] / pair[0]);
      printf("pair full=%d round=%d baseline_ms=%.9g hadamard_ms=%.9g ratio=%.9g\n", full, round, pair[0], pair[1], pair[1] / pair[0]);
    }
    printf("timing full=%d baseline_ms=%.9g hadamard_ms=%.9g paired_overhead_pct=%.6g samples=%d\n",
        full, median(times[0]), median(times[1]), (median(ratios) - 1) * 100, samples);
  }
  ccv_nnc_deinit_mfa_context(context);
  printf("correct=%d\n", correct);
  return correct ? 0 : 4;
}
