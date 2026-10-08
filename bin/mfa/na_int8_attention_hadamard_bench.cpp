// Paired timings using production descriptors/pipelines. Hadamard is opt-in and
// forward-only. Includes CPU quantizer and sampled original-basis SDPA checks.
#include <CommonCrypto/CommonDigest.h>
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
  if (argc < 9 || argc > 15) {
    fprintf(stderr, "Usage: %s R C B Hq Hk precision causal dynamic_flags [distribution=0] [varlen=0] [D=128] [quant_only=0] [samples=20] [center_pair=0]\n"
        "precision: 0=FP16 1=BF16 2=FP32; distribution: 0=normal 1=channel-outliers 2=uniform 3=zeros 4=dispatcher-bench\n", argv[0]);
    return 2;
  }
  std::array<uint32_t, 14> arguments{};
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
  const bool quant_only = arguments[11];
  const int samples = argc > 13 ? arguments[12] : 20;
  const bool center_pair = arguments[13];
  if (!R || !C || !B || !Hq || !Hk || Hq % Hk || precision > 2 || causal > 1 ||
      (dynamic & ~3u) || distribution > 4 || arguments[9] > 1 || B > 65535 ||
      Hq > 65535 || R > 1048576 || C > 1048576 || D < 8 || D > 256 || D % 8 ||
      arguments[11] > 1 || arguments[13] > 1 || samples < 1 || samples > 10000) return 2;
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
      if (distribution == 4) {
        uint32_t value = uint32_t(i) * 747796405u + uint32_t(operand + 1) * 2891336453u;
        value = ((value >> ((value >> 28) + 4)) ^ value) * 277803737u;
        value = (value >> 22) ^ value;
        x = float(int(value % 2047) - 1023) / 1024;
        if (operand == 2) x = x * 0.3f + float((i / D) % Hk) * 0.01f;
      }
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
  std::array<std::array<NS::SharedPtr<MTL::Buffer>, 10>, 2> scratch;
  const size_t bytes[] = {q_count, kv_count, kv_count, size_t(B) * Hq * q_tiles * 4,
      size_t(B) * Hk * k_tiles * 4, size_t(B) * Hk * k_tiles * 4,
      size_t(B) * Hk * D * 4, q_count * element_size, size_t(B) * Hk * D * 4};
  std::array<NS::SharedPtr<MTL::Buffer>, 2> l;
  for (int variant = 0; variant < 2; ++variant) {
    descriptor.qkHadamard = variant;
    pipelines[variant] = context->kernel_cache.findKernel<NAInt8AttentionKernel, NAInt8AttentionDescriptor,
        NAInt8AttentionKernelDescriptor>(descriptor, device.get(), context->device_properties);
    for (int i = 0; i < 9; ++i) {
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
  // Per-chunk mean sums for long sequences. Timed pairs share scratch, so both
  // variants get the larger (two-operand Hadamard) requirement.
  const size_t partial_bytes = std::max<size_t>({16, pipelines[0]->kernel->vMeanPartialBytes(B, C),
      pipelines[1]->kernel->vMeanPartialBytes(B, C)});
  for (int variant = 0; variant < 2; ++variant) {
    scratch[variant][9] = NS::TransferPtr(device->newBuffer(partial_bytes, MTL::ResourceStorageModeShared));
    if (!scratch[variant][9]) return 2;
  }
  PipelineValue<NAInt8AttentionKernel> uncentered;
  if (center_pair) {
    // Rebuild only the old K specialization; Q, V, mean-V, and attention are unchanged.
    std::string source = pipelines[1]->kernel->source;
    const std::string centered = ", true, false>";
    const size_t position = source.find(centered, source.find("kernel void quantize_k("));
    if (position == std::string::npos) return 4;
    source.replace(position, centered.size(), ", false, false>");
    auto options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
    options->setLanguageVersion(MTL::LanguageVersion(0x40000));
    NS::Error* error = nullptr;
    auto library = NS::TransferPtr(device->newLibrary(NS::String::string(source.c_str(), NS::UTF8StringEncoding), options.get(), &error));
    if (!library) { fprintf(stderr, "%s\n", error->localizedDescription()->utf8String()); return 3; }
    auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
    const uint32_t values[] = {R, C, Hq, Hk, 16, 64, q_tiles, k_tiles,
        B > 1 ? R * Hq * D : 0, B > 1 ? C * Hk * D : 0, B > 1 ? C * Hk * D : 0,
        B > 1 ? Hq * q_tiles : 0, B > 1 ? Hk * k_tiles : 0};
    for (int i = 0; i < 13; ++i) {
      if ((i == 0 && (dynamic & 1)) || (i == 1 && (dynamic & 2)) || (i >= 6 && dynamic)) continue;
      constants->setConstantValue(values + i, MTL::DataTypeUInt, NS::UInteger(900 + i));
    }
    auto function = NS::TransferPtr(library->newFunction(NS::String::string("quantize_k", NS::UTF8StringEncoding), constants.get(), &error));
    if (!function) { fprintf(stderr, "%s\n", error->localizedDescription()->utf8String()); return 3; }
    uncentered = *pipelines[0];
    uncentered.second = pipelines[1]->second;
    uncentered.third = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
    if (!uncentered.third) { fprintf(stderr, "%s\n", error->localizedDescription()->utf8String()); return 3; }
    pipelines[0] = &uncentered;
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
    // Production CommandBatch keeps all dispatches in one serial compute encoder.
    auto encoder = cb->computeCommandEncoder();
    auto dispatch = [&](MTL::ComputePipelineState* pipeline, MTL::Size grid, MTL::Size threads,
                        const std::vector<std::pair<int, MTL::Buffer*>>& bindings, size_t tg_bytes = 0) {
      encoder->setComputePipelineState(pipeline);
      if (dynamic) encoder->setBytes(dimensions, sizeof(dimensions), 21);
      if (varlen) {
        encoder->setBuffer(seq_buffers[0].get(), 0, 17);
        encoder->setBuffer(seq_buffers[1].get(), 0, 18);
      }
      for (const auto& binding : bindings) encoder->setBuffer(binding.second, 0, binding.first);
      if (tg_bytes) encoder->setThreadgroupMemoryLength(tg_bytes, 0);
      encoder->dispatchThreadgroups(grid, threads);
    };
    // The production helper chooses the mean grid and adds its finalize pass.
    auto encode_mean = [&](const std::vector<std::pair<int, MTL::Buffer*>>& bindings) {
      if (dynamic) encoder->setBytes(dimensions, sizeof(dimensions), 21);
      for (const auto& binding : bindings) encoder->setBuffer(binding.second, 0, binding.first);
      p->kernel->encodeVMean(encoder, p->fifth.get(), p->seventh.get(), buffers[9].get(), 0, B, C);
    };
    for (int repeat = 0; repeat < repeats; ++repeat) {
      dispatch(p->second.get(), MTL::Size(q_tiles, Hq, B), MTL::Size(128, 1, 1),
          {{0, inputs[0].get()}, {1, buffers[0].get()}, {2, buffers[3].get()}, {3, buffers[8].get()}});
      if (variant)
        encode_mean({{0, inputs[2].get()}, {1, buffers[6].get()}, {2, inputs[1].get()}, {3, buffers[8].get()}, {17, seq_buffers[1].get()}});
      dispatch(p->third.get(), MTL::Size(k_tiles, Hk, B), MTL::Size(256, 1, 1),
          {{0, inputs[1].get()}, {1, buffers[1].get()}, {2, buffers[4].get()}, {3, buffers[8].get()}, {17, seq_buffers[1].get()}});
      if (full) {
        if (!variant)
          encode_mean({{0, inputs[2].get()}, {1, buffers[6].get()}, {17, seq_buffers[1].get()}});
        dispatch(pipelines[0]->fourth.get(), MTL::Size(k_tiles, Hk, B), MTL::Size(256, 1, 1),
            {{0, inputs[2].get()}, {1, buffers[2].get()}, {2, buffers[5].get()}, {3, buffers[6].get()}, {17, seq_buffers[1].get()}});
        dispatch(pipelines[0]->pipeline.get(), p->kernel->threadgroupsPerGrid(B, R),
            MTL::Size(p->kernel->threadgroupSize(pipelines[0]->pipeline.get()), 1, 1),
            {{0, buffers[0].get()}, {1, buffers[1].get()}, {2, buffers[2].get()}, {3, buffers[7].get()},
             {4, l[variant].get()}, {10, buffers[3].get()}, {11, buffers[4].get()},
             {12, buffers[5].get()}, {14, buffers[6].get()}}, p->kernel->threadgroupMemoryAllocation());
      }
    }
    encoder->endEncoding();
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
          if (variant && operand)
            for (uint32_t dim = 0; dim < D; ++dim)
              transformed[row * D + dim] -= static_cast<const float*>(scratch[variant][8]->contents())[(batch * Hk + head) * D + dim];
        }
        if ((variant || center_pair) && extent) {
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
        const float normalization = variant || center_pair ? 1.0f / std::sqrt(float(hadamard_block)) : 1;
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
    if (center_pair && distribution == 4) {
      CC_SHA256_CTX hash;
      CC_SHA256_Init(&hash);
      std::vector<float> block(1 << 18);
      for (size_t offset = 0; offset < q_count; offset += block.size()) {
        const size_t count = std::min(block.size(), q_count - offset);
        for (size_t i = 0; i < count; ++i) block[i] = read_output(offset + i);
        CC_SHA256_Update(&hash, block.data(), CC_LONG(count * sizeof(float)));
      }
      unsigned char digest[CC_SHA256_DIGEST_LENGTH];
      CC_SHA256_Final(digest, &hash);
      printf("output_sha256 variant=%d value=", variant);
      for (auto byte : digest) printf("%02x", byte);
      printf("\n");
    }
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
  printf("case device=%s R=%u C=%u D=%u B=%u Hq=%u Hk=%u precision=%u causal=%u dynamic=%u distribution=%u varlen=%d quant_only=%d hadamard_block=%u\n",
      device->name()->utf8String(), R, C, D, B, Hq, Hk, precision, causal, dynamic, distribution, varlen, quant_only, hadamard_block);
  printf("center_pair=%d\n", center_pair);
  // Timed pairs reuse addresses; correctness above kept outputs separate.
  // Hadamard also centers K in the combined K/V mean reduction.
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
