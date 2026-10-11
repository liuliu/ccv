// Three-way forward prototype: INT8 PV, floating PV with INT8 QK, NAAttention.
// Rotating GPU timings, including preparation, and sampled CPU SDPA reference.
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
extern "C" {
#include "ccv.h"
}
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionDescriptor.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionKernel.hpp"
#include "nnc/mfa/kernels/NAAttentionDescriptor.hpp"
#include "nnc/mfa/kernels/NAAttentionKernel.hpp"
static double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return (values[(values.size() - 1) / 2] + values[values.size() / 2]) * 0.5;
}

#ifdef CCV_NA_PV_EMBEDDED
int na_int8_tuning_run(int argc, char** argv)
#else
int main(int argc, char** argv)
#endif
{
  setbuf(stdout, nullptr);
  if (argc < 9 || argc > 18) {
    fprintf(stderr, "Usage: %s R C B Hq Hk precision causal dynamic_flags [distribution=0] [varlen=0] [D=128] [hadamard=0] [samples=20] [hybrid_block_d=0] [hybrid_simdgroups=0] [hybrid_output_tile=0] [long_profile=0]\n"
        "precision: 0=FP16 1=BF16 2=FP32; distribution: 0=normal 1=channel-outliers 2=uniform 3=zeros 4=dispatcher-bench\n", argv[0]);
    return 2;
  }
  std::array<uint32_t, 17> arguments{};
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
  const bool hadamard = arguments[11];
  const bool long_profile = arguments[16];
  const bool validate_quantizers = std::getenv("CCV_PV_VALIDATE_QUANTIZERS") != nullptr;
  // Compare two floating-PV tile configurations without mixing in V quantization.
  const bool compare_configs = std::getenv("CCV_PV_COMPARE_CONFIGS") != nullptr;
  const bool pair_only = compare_configs || std::getenv("CCV_PV_PAIR_ONLY") != nullptr;
  const int variant_count = pair_only ? 2 : 3;
  const int check_variant = std::getenv("CCV_PV_CHECK_VARIANT") ? std::atoi(std::getenv("CCV_PV_CHECK_VARIANT")) : -1;
  // Feature checks: masks 1=shared, 2=batched, 3=fully masked; sinks
  // 1=per head, 2=shared, 3=per head with stride 2. Zero disables either.
  const int mask_mode = std::getenv("CCV_PV_MASK") ? std::atoi(std::getenv("CCV_PV_MASK")) : 0;
  const int sink_mode = std::getenv("CCV_PV_SINKS") ? std::atoi(std::getenv("CCV_PV_SINKS")) : 0;
  const int requested_samples = argc > 13 ? arguments[12] : 20;
  const int samples = long_profile && !compare_configs ? std::min(requested_samples, 3) : requested_samples;
  if (!R || !C || !B || !Hq || !Hk || Hq % Hk || precision > 2 || causal > 1 ||
      (dynamic & ~3u) || distribution > 4 || arguments[9] > 1 || B > 65535 ||
      Hq > 65535 || R > 1048576 || C > 1048576 || D < 8 || D > 512 || D % 8 ||
      arguments[11] > 1 || arguments[16] > 1 || samples < 1 || samples > 10000 ||
      mask_mode < 0 || mask_mode > 3 || sink_mode < 0 || sink_mode > 3 ||
      check_variant < -1 || check_variant >= variant_count ||
      (arguments[13] && arguments[13] != 32 && arguments[13] != 64 && arguments[13] != 128) ||
      (arguments[14] && arguments[14] != 1 && arguments[14] != 2 && arguments[14] != 4 && arguments[14] != 8 && arguments[14] != 16) ||
      (arguments[15] && (arguments[15] > D || arguments[15] % 32))) return 2;

  const size_t q_count = size_t(B) * R * Hq * D, kv_count = size_t(B) * C * Hk * D;
  // Keep host counts and shader indexing in range.
  if (q_count > INT_MAX || kv_count > INT_MAX) return 2;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
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
      float x = (distribution == 3 || distribution == 4) ? 0 : (distribution == 2 ? uniform(random) : normal(random));
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
      for (size_t i = 0; i < counts[operand]; ++i) {
        auto out = static_cast<_Float16*>(inputs[operand]->contents());
        out[i] = data[operand][i]; data[operand][i] = out[i];
      }
    } else if (precision == 1) {
      for (size_t i = 0; i < counts[operand]; ++i) {
        uint32_t bits; memcpy(&bits, &data[operand][i], 4);
        bits += 0x7fff + ((bits >> 16) & 1);
        static_cast<uint16_t*>(inputs[operand]->contents())[i] = bits >> 16;
        bits &= 0xffff0000; memcpy(&data[operand][i], &bits, 4);
      }
    } else memcpy(inputs[operand]->contents(), data[operand].data(), counts[operand] * 4);
    std::vector<float>().swap(data[operand]);
  }
  auto read_input = [&](int operand, size_t index) {
    const auto input = inputs[operand]->contents();
    if (precision == 2) return static_cast<float*>(input)[index];
    if (precision == 0) return float(static_cast<_Float16*>(input)[index]);
    const uint32_t bits = uint32_t(static_cast<uint16_t*>(input)[index]) << 16;
    float value; memcpy(&value, &bits, 4); return value;
  };
  std::array<std::vector<int>, 2> seq;
  std::array<NS::SharedPtr<MTL::Buffer>, 2> seq_buffers;
  for (int operand = 0; operand < 2; ++operand) {
    const uint32_t length = operand ? C : R;
    seq[operand].push_back(0);
    for (uint32_t batch = 0; batch < B; ++batch)
      seq[operand].push_back(seq[operand].back() + (varlen ? length - batch * 13 % length : length));
    seq_buffers[operand] = NS::TransferPtr(device->newBuffer(seq[operand].data(), seq[operand].size() * sizeof(int), MTL::ResourceStorageModeShared));
  }
  const uint32_t mask_batch_stride = mask_mode >= 2 ? R * C : 0;
  const uint32_t sink_head_stride = sink_mode == 2 ? 0 : (sink_mode == 3 ? 2 : 1);
  std::array<std::vector<float>, 2> feature_data;
  feature_data[0].resize(mask_mode ? size_t(mask_batch_stride ? B : 1) * R * C : 1);
  feature_data[1].resize(Hq * 2);
  if (mask_mode) {
    for (uint32_t batch = 0; batch < (mask_batch_stride ? B : 1); ++batch)
      for (uint32_t row = 0; row < R; ++row) for (uint32_t col = 0; col < C; ++col) {
        // Exercise skipped, zero and mixed tiles, plus entirely masked rows.
        const float value = mask_mode == 3 || (R > 16 && row % 64 < 16) || (col >= 64 && col < 128) ? -INFINITY :
            (row % 64 < 32 && col < 64 ? 0.0f : ((row + col + batch) % 7 == 0 ? -INFINITY : -0.25f * ((row + col + batch) % 3)));
        feature_data[0][size_t(batch) * R * C + row * C + col] = value;
      }
  }
  for (uint32_t head = 0; head < Hq; ++head)
    feature_data[1][head * sink_head_stride] = 4.0f + 2.0f * (head % 3);
  std::array<NS::SharedPtr<MTL::Buffer>, 2> features;
  for (int i = 0; i < 2; ++i) {
    features[i] = NS::TransferPtr(device->newBuffer(feature_data[i].size() * element_size, MTL::ResourceStorageModeShared));
    if (!features[i]) return 2;
    if (precision == 0) ccv_float_to_half_precision(feature_data[i].data(), static_cast<uint16_t*>(features[i]->contents()), feature_data[i].size());
    else if (precision == 1) ccv_float_to_bfloat(feature_data[i].data(), static_cast<uint16_t*>(features[i]->contents()), feature_data[i].size());
    else memcpy(features[i]->contents(), feature_data[i].data(), feature_data[i].size() * sizeof(float));
  }
  NAInt8AttentionDescriptor descriptor;
  descriptor.matrixDimensions = simd::uint3{R, C, D};
  descriptor.batchDimension = B; descriptor.Hq = Hq; descriptor.Hk = Hk;
  descriptor.ioPrecision = precision == 0 ? GEMMOperandPrecision::FP16 :
      (precision == 1 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP32);
  descriptor.scale = 1.0f / std::sqrt(float(D));
  descriptor.isCausal = causal; descriptor.isVarlen = varlen;
  descriptor.masked = mask_mode != 0; descriptor.maskBatchStride = mask_batch_stride;
  descriptor.attentionSinks = sink_mode != 0;
  descriptor.qkMeanCorrection = hadamard && sink_mode;
  descriptor.loadR = dynamic & 1; descriptor.loadC = dynamic & 2;
  descriptor.lowPrecisionIntermediates = precision == 0;
  descriptor.batchStrides[AttentionOperand::Q] = B > 1 ? R * Hq * D : 0;
  descriptor.batchStrides[AttentionOperand::K] = B > 1 ? C * Hk * D : 0;
  descriptor.batchStrides[AttentionOperand::V] = B > 1 ? C * Hk * D : 0;
  descriptor.batchStrides[AttentionOperand::O] = B > 1 ? R * Hq * D : 0;
  std::array<PipelineValue<NAInt8AttentionKernel>*, 2> pipelines;
  std::unordered_map<NAInt8AttentionKernelDescriptor, std::unique_ptr<NAInt8AttentionKernel>> hybrid_cache;
  std::array<std::unique_ptr<PipelineValue<NAInt8AttentionKernel>>, 2> overridden_pipelines;
  std::array<std::array<NS::SharedPtr<MTL::Buffer>, 10>, 3> scratch;
  const size_t bytes[] = {q_count, kv_count, kv_count, size_t(B) * Hq * q_tiles * 4,
      size_t(B) * Hk * k_tiles * 4, size_t(B) * Hk * k_tiles * 4,
      size_t(B) * Hk * D * 4, q_count * element_size, size_t(B) * Hk * D * 4};
  std::array<NS::SharedPtr<MTL::Buffer>, 3> l;
  const char* block_r_override = std::getenv("CCV_PV_BLOCK_R");
  const char* block_c_override = std::getenv("CCV_PV_BLOCK_C");
  const char* barrier_override = std::getenv("CCV_PV_BARRIER");
  for (int variant = 0; variant < 2; ++variant) {
    descriptor.qkHadamard = hadamard;
    descriptor.quantizePV = compare_configs ? false : !variant;
    if (compare_configs || (variant && (arguments[13] || arguments[14] || arguments[15] || block_r_override || block_c_override || barrier_override))) {
      NAInt8AttentionKernelDescriptor kd(descriptor, context->device_properties);
      if (block_r_override) {
        kd.blockDimensions[0] = std::atoi(block_r_override);
        kd.qScaleTileSize = std::max<uint16_t>(16, kd.blockDimensions[0]);
      }
      if (block_c_override) {
        kd.blockDimensions[1] = std::atoi(block_c_override);
        kd.kvScaleTileSize = std::max<uint16_t>(64, kd.blockDimensions[1]);
      }
      if (barrier_override) kd.threadBarrierEveryC = std::atoi(barrier_override);
      if (arguments[13]) kd.blockDimensions[2] = arguments[13];
      if (arguments[14]) kd.executionSIMDGroups = arguments[14];
      if (arguments[15]) kd.outputTileSize = arguments[15];
      if (compare_configs) {
        // Explicit C tile, head tile, SIMD groups, output tile for each variant.
        // All four are required, so selector changes cannot alter the experiment.
        const char* config = std::getenv(variant ? "CCV_PV_CONFIG_1" : "CCV_PV_CONFIG_0");
        unsigned bc, bd, groups, output;
        char trailing;
        if (!config || sscanf(config, "%u,%u,%u,%u%c", &bc, &bd, &groups, &output, &trailing) != 4 ||
            (bc != 16 && bc != 32 && bc != 64) || (bd != 32 && bd != 64 && bd != 128) ||
            (groups != 1 && groups != 2 && groups != 4 && groups != 8) || !output || output > D)
          return 2;
        kd.blockDimensions = simd::ushort3{16, uint16_t(bc), uint16_t(bd)};
        kd.qScaleTileSize = 16; kd.kvScaleTileSize = 64;
        kd.executionSIMDGroups = groups; kd.outputTileSize = output;
      }
      // Benchmark overrides must update the properties derived from the tiles.
      if (!kd.blockDimensions[0] || !kd.blockDimensions[1]) return 2;
      kd.hasCRemainder = varlen || C % kd.blockDimensions[1] != 0;
      kd.hasRRemainder = !descriptor.loadR || varlen || R % kd.blockDimensions[0] != 0;
      kd.mortonTraversal = kd.outputTileSize == D && causal && dynamic;
      overridden_pipelines[variant].reset(descriptor.findKernel(device.get(), kd, nullptr, nullptr, "", &hybrid_cache).second);
      pipelines[variant] = overridden_pipelines[variant].get();
    } else {
      pipelines[variant] = context->kernel_cache.findKernel<NAInt8AttentionKernel, NAInt8AttentionDescriptor,
          NAInt8AttentionKernelDescriptor>(descriptor, device.get(), context->device_properties);
    }
    printf("kernel variant=%d block_r=%u block_c=%u block_d=%u simdgroups=%u output_tile=%u\n", variant,
        pipelines[variant]->kernel->blockDimensions[0], pipelines[variant]->kernel->blockDimensions[1],
        pipelines[variant]->kernel->blockDimensions[2], pipelines[variant]->kernel->executionSIMDGroups,
        pipelines[variant]->kernel->outputTileSize);
    for (int i = 0; i < 9; ++i) {
      if (variant) { scratch[variant][i] = scratch[0][i]; continue; }
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
  NAAttentionDescriptor baseline;
  baseline.matrixDimensions = descriptor.matrixDimensions;
  baseline.batchDimension = B; baseline.Hq = Hq; baseline.Hk = Hk;
  baseline.lowPrecisionInputs = precision != 2; baseline.isBF16 = precision == 1;
  baseline.lowPrecisionIntermediates = precision != 2;
  baseline.type = AttentionKernelType::forward;
  baseline.scale = descriptor.scale; baseline.isCausal = causal;
  baseline.isVarlen = varlen; baseline.loadR = dynamic & 1; baseline.loadC = dynamic & 2;
  baseline.masked = mask_mode != 0; baseline.maskBatchStride = mask_batch_stride;
  baseline.attentionSinks = sink_mode != 0;
  baseline.batchStrides = descriptor.batchStrides;
  auto fp = pair_only ? nullptr : context->kernel_cache.findKernel<NAAttentionKernel, NAAttentionDescriptor,
      NAAttentionKernelDescriptor>(baseline, device.get(), context->device_properties);
  const uint32_t split_kv = fp ? fp->kernel->splitKV : 1;
  NS::SharedPtr<MTL::Buffer> partial_o, partial_l;
  if (split_kv > 1) {
    partial_o = NS::TransferPtr(device->newBuffer(size_t(B) * Hq * split_kv * R * D * sizeof(float), MTL::ResourceStorageModeShared));
    partial_l = NS::TransferPtr(device->newBuffer(size_t(B) * Hq * split_kv * R * sizeof(float), MTL::ResourceStorageModeShared));
    if (!partial_o || !partial_l) return 2;
  }
  printf("baseline split_kv=%u\n", split_kv);
  const uint32_t na_q_tiles = fp ? (R + fp->kernel->blockDimensions[0] - 1) / fp->kernel->blockDimensions[0] : q_tiles;
  const uint32_t na_k_tiles = fp ? (C + fp->kernel->blockDimensions[1] - 1) / fp->kernel->blockDimensions[1] : k_tiles;
  std::array<NS::SharedPtr<MTL::Buffer>, 3> block_mask;
  for (int variant = 0; variant < 3; ++variant) {
    const uint32_t q_mask_tiles = variant == 2 ? na_q_tiles :
        (R + pipelines[variant]->kernel->blockDimensions[0] - 1) / pipelines[variant]->kernel->blockDimensions[0];
    const uint32_t k_mask_tiles = variant == 2 ? na_k_tiles :
        (C + pipelines[variant]->kernel->blockDimensions[1] - 1) / pipelines[variant]->kernel->blockDimensions[1];
    block_mask[variant] = NS::TransferPtr(device->newBuffer(std::max<size_t>(16, size_t(B) * q_mask_tiles * k_mask_tiles), MTL::ResourceStorageModeShared));
  }
  auto qk_correction = NS::TransferPtr(device->newBuffer(descriptor.qkMeanCorrection ? size_t(B) * Hq * R * sizeof(float) : 16, MTL::ResourceStorageModeShared));
  scratch[2] = scratch[0];
  scratch[2][7] = scratch[0][7];
  l[2] = NS::TransferPtr(device->newBuffer(size_t(B) * Hq * R * 4, MTL::ResourceStorageModeShared));
  auto padded_v_tail = NS::TransferPtr(device->newBuffer(std::max<size_t>({16,
      pipelines[0]->kernel->paddedVTailBytes(B), pipelines[1]->kernel->paddedVTailBytes(B)}), MTL::ResourceStorageModeShared));
  const uint32_t dimensions[] = {R, C, B > 1 ? R * Hq * D : 0, B > 1 ? C * Hk * D : 0,
      B > 1 ? C * Hk * D : 0, B > 1 ? R * Hq * D : 0, 0, 0, 0, 0,
      B > 1 ? Hq * q_tiles : 0, B > 1 ? Hk * k_tiles : 0, B > 1 ? Hk * k_tiles : 0, 0, 0,
      mask_batch_stride, mask_batch_stride ? q_tiles * k_tiles : 0};
  auto run = [&](int variant, bool full, int repeats) {
    auto iteration_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
    auto cb = queue->commandBuffer();
    auto p = pipelines[std::min(variant, 1)]; auto& buffers = scratch[variant];
    uint32_t variant_dimensions[17];
    memcpy(variant_dimensions, dimensions, sizeof(dimensions));
    const uint32_t variant_q_tiles = (R + p->kernel->qScaleTileSize - 1) / p->kernel->qScaleTileSize;
    const uint32_t variant_k_tiles = (C + p->kernel->kvScaleTileSize - 1) / p->kernel->kvScaleTileSize;
    const uint32_t variant_q_mask_tiles = (R + p->kernel->blockDimensions[0] - 1) / p->kernel->blockDimensions[0];
    const uint32_t variant_k_mask_tiles = (C + p->kernel->blockDimensions[1] - 1) / p->kernel->blockDimensions[1];
    variant_dimensions[10] = B > 1 ? Hq * variant_q_tiles : 0;
    variant_dimensions[11] = variant_dimensions[12] = B > 1 ? Hk * variant_k_tiles : 0;
    variant_dimensions[16] = mask_batch_stride ? variant_q_mask_tiles * variant_k_mask_tiles : 0;
    // Production CommandBatch keeps all dispatches in one serial compute encoder.
    auto encoder = cb->computeCommandEncoder();
    auto dispatch = [&](MTL::ComputePipelineState* pipeline, MTL::Size grid, MTL::Size threads,
                        const std::vector<std::pair<int, MTL::Buffer*>>& bindings, size_t tg_bytes = 0) {
      encoder->setComputePipelineState(pipeline);
      if (variant == 2) {
        if (dynamic & 2) encoder->setBytes(&C, sizeof(C), 21);
        const uint32_t fp_dimensions[] = {R, dimensions[2], dimensions[3], dimensions[4], dimensions[5], mask_batch_stride, mask_batch_stride ? na_q_tiles * na_k_tiles : 0};
        if (dynamic & 1) encoder->setBytes(fp_dimensions, sizeof(fp_dimensions), 22);
      } else if (dynamic) encoder->setBytes(variant_dimensions, sizeof(variant_dimensions), 21);
      if (varlen) {
        encoder->setBuffer(seq_buffers[0].get(), 0, 17);
        encoder->setBuffer(seq_buffers[1].get(), 0, 18);
      }
      for (const auto& binding : bindings) encoder->setBuffer(binding.second, 0, binding.first);
      if (mask_mode) {
        encoder->setBuffer(features[0].get(), 0, 15);
        encoder->setBuffer(block_mask[variant].get(), 0, 16);
      }
      if (sink_mode) {
        encoder->setBuffer(features[1].get(), 0, 19);
        encoder->setBytes(&sink_head_stride, sizeof(sink_head_stride), 20);
      }
      if (variant != 2 && descriptor.qkMeanCorrection) encoder->setBuffer(qk_correction.get(), 0, 22);
      if (tg_bytes) encoder->setThreadgroupMemoryLength(tg_bytes, 0);
      encoder->dispatchThreadgroups(grid, threads);
    };
    // The production helper chooses the mean grid and adds its finalize pass.
    auto encode_mean = [&](const std::vector<std::pair<int, MTL::Buffer*>>& bindings) {
      if (dynamic) encoder->setBytes(variant_dimensions, sizeof(variant_dimensions), 21);
      for (const auto& binding : bindings) encoder->setBuffer(binding.second, 0, binding.first);
      p->kernel->encodeVMean(encoder, p->fifth.get(), p->seventh.get(), buffers[9].get(), 0, B, C);
    };
    for (int repeat = 0; repeat < repeats; ++repeat) {
      if (variant == 2) {
        if (mask_mode && full)
          dispatch(fp->second.get(), MTL::Size(na_q_tiles, na_k_tiles, mask_batch_stride ? B : 1),
              MTL::Size(NAAttentionKernel::blockMaskThreads, 1, 1), {}, NAAttentionKernel::blockMaskThreads * sizeof(uint32_t) * 2);
        if (split_kv > 1) {
          encoder->setBuffer(partial_o.get(), 0, 5);
          encoder->setBuffer(partial_l.get(), 0, 6);
        }
        dispatch(fp->pipeline.get(), fp->kernel->threadgroupsPerGrid(baseline),
            MTL::Size(fp->kernel->threadgroupSize(fp->pipeline.get(), baseline), 1, 1),
            {{0, inputs[0].get()}, {1, inputs[1].get()}, {2, inputs[2].get()},
             {3, buffers[7].get()}, {4, l[variant].get()}}, fp->kernel->threadgroupMemoryAllocation(fp->pipeline.get(), baseline));
        if (split_kv > 1)
          dispatch(fp->second.get(), MTL::Size((q_count + NAAttentionKernel::splitKVCombineThreads - 1) / NAAttentionKernel::splitKVCombineThreads, 1, 1),
              MTL::Size(NAAttentionKernel::splitKVCombineThreads, 1, 1),
              {{3, buffers[7].get()}, {4, l[variant].get()}, {5, partial_o.get()}, {6, partial_l.get()}});
        continue;
      }
      if (full) {
      if (hadamard)
        encode_mean({{0, inputs[2].get()}, {1, buffers[6].get()}, {2, inputs[1].get()}, {3, buffers[8].get()}, {17, seq_buffers[1].get()}});
      dispatch(p->second.get(), MTL::Size(variant_q_tiles, Hq, B), MTL::Size(128, 1, 1),
          {{0, inputs[0].get()}, {1, buffers[0].get()}, {2, buffers[3].get()}, {3, buffers[8].get()}, {4, qk_correction.get()}});
      dispatch(p->third.get(), MTL::Size(variant_k_tiles, Hk, B), MTL::Size(256, 1, 1),
          {{0, inputs[1].get()}, {1, buffers[1].get()}, {2, buffers[4].get()}, {3, buffers[8].get()}, {4, inputs[2].get()}, {5, padded_v_tail.get()}, {17, seq_buffers[1].get()}});
        if (!hadamard && p->kernel->quantizePV)
          encode_mean({{0, inputs[2].get()}, {1, buffers[6].get()}, {17, seq_buffers[1].get()}});
        if (p->kernel->quantizePV) dispatch(p->fourth.get(), MTL::Size(k_tiles, Hk, B), MTL::Size(256, 1, 1),
            {{0, inputs[2].get()}, {1, buffers[2].get()}, {2, buffers[5].get()}, {3, buffers[6].get()}, {17, seq_buffers[1].get()}});
        if (mask_mode)
          dispatch(p->sixth.get(), MTL::Size(variant_q_mask_tiles, variant_k_mask_tiles, mask_batch_stride ? B : 1),
              MTL::Size(NAInt8AttentionKernel::blockMaskThreads, 1, 1), {}, NAInt8AttentionKernel::blockMaskThreads * sizeof(uint32_t) * 2);
      }
        if (validate_quantizers) continue;
        if (p->kernel->usesPaddedVTail()) encoder->setBuffer(padded_v_tail.get(), 0, 5);
        dispatch(p->pipeline.get(), p->kernel->threadgroupsPerGrid(B, R),
            MTL::Size(p->kernel->threadgroupSize(p->pipeline.get()), 1, 1),
            {{0, buffers[0].get()}, {1, buffers[1].get()}, {2, p->kernel->quantizePV ? buffers[2].get() : inputs[2].get()}, {3, buffers[7].get()},
             {4, l[variant].get()}, {10, buffers[3].get()}, {11, buffers[4].get()},
             {12, buffers[5].get()}, {14, buffers[6].get()}}, p->kernel->threadgroupMemoryAllocation());
    }
    encoder->endEncoding();
    cb->commit(); cb->waitUntilCompleted();
    if (cb->status() != MTL::CommandBufferStatusCompleted) {
      fprintf(stderr, "GPU failed: %s\n", cb->error() ? cb->error()->localizedDescription()->utf8String() : "unknown");
      std::exit(3);
    }
    return (cb->GPUEndTime() - cb->GPUStartTime()) * 1000 / repeats;
  };
  // Isolate preparation for shader validation: the shipped attention kernel's
  // bounded tensor slices themselves trigger validation on partial query tiles.
  if (validate_quantizers) {
    run(0, true, 1);
    run(1, true, 1);
    ccv_nnc_deinit_mfa_context(context);
    printf("quantizer_validation=1\n");
    return 0;
  }
  bool correct = true;
  for (int variant = 0; variant < variant_count; ++variant) {
    if (check_variant >= 0 && variant != check_variant) continue;
    printf("checking_variant=%d\n", variant);
    run(variant, true, 1);
    bool variant_correct = true;
    double squared_error = 0, norm = 0, maximum_error = 0;
    auto read_output = [&](size_t index) {
      const auto output = scratch[variant][7]->contents();
      if (precision == 2) return static_cast<float*>(output)[index];
      if (precision == 0) return float(static_cast<_Float16*>(output)[index]);
      const uint32_t bits = uint32_t(static_cast<uint16_t*>(output)[index]) << 16;
      float value; memcpy(&value, &bits, 4); return value;
    };
    if (distribution == 4) {
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
    const bool full_reference = uint64_t(R) * C * B * Hq <= 1048576;
    std::vector<uint32_t> heads;
    if (full_reference) for (uint32_t h = 0; h < Hq; ++h) heads.push_back(h);
    else heads = {0u, Hq - 1};
    for (uint32_t batch = 0; batch < B; ++batch) for (uint32_t head : heads) {
      const uint32_t rows = varlen ? seq[0][batch + 1] - seq[0][batch] : R;
      const uint32_t cols = varlen ? seq[1][batch + 1] - seq[1][batch] : C;
      const size_t q_base = size_t(varlen ? seq[0][batch] : batch * R) * Hq * D;
      const size_t kv_base = size_t(varlen ? seq[1][batch] : batch * C) * Hk * D;
      std::vector<uint32_t> checked_rows;
      if (full_reference) for (uint32_t row = 0; row < rows; ++row) checked_rows.push_back(row);
      else checked_rows = {0u, std::min(15u, rows - 1), std::min(16u, rows - 1), rows / 2, rows - 1};
      for (uint32_t row : checked_rows) {
        const size_t qi = q_base + (row * Hq + head) * D;
        const uint32_t kh = head / (Hq / Hk);
        const int end = causal ? std::clamp(int(row) + int(cols) - int(rows) + 1, 0, int(cols)) : cols;
        std::vector<double> scores(end);
        const double sink = sink_mode ? feature_data[1][head * sink_head_stride] : -INFINITY;
        double maximum = sink, sum = 0;
        for (int col = 0; col < end; ++col) {
          double dot = 0; const size_t ki = kv_base + (col * Hk + kh) * D;
          for (uint32_t dim = 0; dim < D; ++dim) dot += double(read_input(0, qi + dim)) * read_input(1, ki + dim);
          scores[col] = dot * descriptor.scale + (mask_mode ? feature_data[0][size_t(mask_batch_stride) * batch + row * C + col] : 0);
          maximum = std::max(maximum, scores[col]);
        }
        if (sink_mode) sum = std::exp(sink - maximum);
        for (auto& score : scores) { score = std::isfinite(score) ? std::exp(score - maximum) : 0; sum += score; }
        for (uint32_t dim = 0; dim < D; ++dim) {
          double expected = 0;
          for (int col = 0; col < end; ++col) expected += scores[col] * read_input(2, kv_base + (col * Hk + kh) * D + dim);
          if (sum > 0) expected /= sum;
          const float actual = read_output(qi + dim);
          if (!std::isfinite(actual)) fprintf(stderr, "nonfinite variant=%d batch=%u head=%u row=%u dim=%u\n", variant, batch, head, row, dim);
          variant_correct &= std::isfinite(actual);
          const double diff = actual - expected;
          squared_error += diff * diff; norm += expected * expected; maximum_error = std::max(maximum_error, std::abs(diff));
        }
      }
    }
    const double relative_l2 = std::sqrt(squared_error / std::max(norm, 1e-30));
    variant_correct &= relative_l2 < (distribution == 1 ? 0.15 : 0.05);
    correct &= variant_correct;
    printf("attention_check variant=%d full_reference=%d rel_l2=%.9g max_abs=%.9g correct=%d\n", variant, full_reference,
        std::sqrt(squared_error / std::max(norm, 1e-30)), maximum_error, variant_correct);
  }
  printf("case device=%s R=%u C=%u D=%u B=%u Hq=%u Hk=%u precision=%u causal=%u dynamic=%u distribution=%u varlen=%d hadamard=%d\n",
      device->name()->utf8String(), R, C, D, B, Hq, Hk, precision, causal, dynamic, distribution, varlen, hadamard);
  printf("features mask=%d sinks=%d sink_head_stride=%u\n", mask_mode, sink_mode, sink_head_stride);
  if (check_variant >= 0) {
    ccv_nnc_deinit_mfa_context(context);
    printf("correct=%d\n", correct);
    return correct ? 0 : 4;
  }
  // Same scratch/output addresses for each timed run; rotating order mitigates drift.
  scratch[1] = scratch[0]; scratch[2] = scratch[0]; l[1] = l[0]; l[2] = l[0];
  for (int full = long_profile || compare_configs ? 1 : 0; full < 2; ++full) {
    // One dispatch per sample also keeps large iPad runs within the app launch budget.
    const int repeats = compare_configs && uint64_t(R) * C * Hq * B < (1ull << 27) ? 10 : 1;
    std::array<std::vector<double>, 3> times;
    double warm_ms = 0;
    for (int warmup = 0; warmup < (long_profile ? 1 : 5) || warm_ms < 300; ++warmup)
      for (int variant = 0; variant < variant_count; ++variant) warm_ms += run(variant, true, repeats) * repeats;
    for (int round = 0; round < samples; ++round) {
      double triplet[3] = {};
      for (int j = 0; j < variant_count; ++j) {
        const int variant = (round + j) % variant_count;
        triplet[variant] = run(variant, full, repeats); times[variant].push_back(triplet[variant]);
      }
      if (compare_configs) printf("config_sample round=%d baseline_ms=%.9g candidate_ms=%.9g\n", round, triplet[0], triplet[1]);
      else if (pair_only) printf("pair_sample full=%d round=%d int8_ms=%.9g hybrid_ms=%.9g\n", full, round, triplet[0], triplet[1]);
      else printf("sample full=%d round=%d int8_ms=%.9g hybrid_ms=%.9g na_ms=%.9g\n", full, round, triplet[0], triplet[1], triplet[2]);
    }
    if (compare_configs) printf("config_timing baseline_ms=%.9g candidate_ms=%.9g samples=%d\n", median(times[0]), median(times[1]), samples);
    else if (pair_only) printf("pair_timing full=%d int8_ms=%.9g hybrid_ms=%.9g samples=%d\n", full, median(times[0]), median(times[1]), samples);
    else printf("timing full=%d int8_ms=%.9g hybrid_ms=%.9g na_ms=%.9g samples=%d\n",
        full, median(times[0]), median(times[1]), median(times[2]), samples);
  }
  ccv_nnc_deinit_mfa_context(context);
  printf("correct=%d\n", correct);
  return correct ? 0 : 4;
}
