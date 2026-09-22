// Compare the production V-mean pipeline with the previous single-pass reduction.
// Usage: na_int8_v_mean_bench [T=32768] [H=56] [N=1] [rounds=20] [D=128] [precision=16F] [full=0] [R=T]
// Precision: 16F, 16BF, or 32F. BF16 and FP32 use FP32 intermediates.
// Build from repo root (after building libccv.a):
// clang++ -std=c++17 -O3 -fblocks -Ilib bin/mfa/na_int8_v_mean_bench.cpp lib/libccv.a -framework Accelerate -framework Metal -framework Foundation -framework QuartzCore -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework CoreML -framework CoreVideo -framework IOSurface -o /tmp/na_int8_v_mean_bench
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionDescriptor.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionKernel.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// Preserve the old shader, including the exact accumulation and shuffle order.
static const char* referenceMean = R"(
kernel void compute_v_mean(
    device const half *src [[buffer(0)]],
    device float *mean [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
  threadgroup float4 scratch[QUANTIZE_V_MEAN_SIMDGROUPS];
  device const io_vec4 *src4 = reinterpret_cast<device const io_vec4 *>(src);
  device v_mean_vec4 *mean4 = reinterpret_cast<device v_mean_vec4 *>(mean);
  const uint mean_tiles = 128 / 4;
  const uint vec_bits = ceil_log2_u32(mean_tiles);
  const uint head_bits = ceil_log2_u32(QUANTIZE_KV_HEADS);
  const uint2 morton = morton_decode_rectangular_2d(tgid.x, vec_bits, head_bits);
  const uint vec_dim = morton.x;
  const uint head = morton.y;
  const uint batch = tgid.z;
  if (vec_dim >= mean_tiles || head >= QUANTIZE_KV_HEADS)
    return;
  float4 local_sum = float4(0.0f);
  for (uint column = tid; column < QUANTIZE_KV_SEQUENCE; column += QUANTIZE_V_MEAN_THREADS) {
    const uint index = batch * QUANTIZE_V_BATCH_STRIDE + ((column * QUANTIZE_KV_HEADS + head) * 128 + vec_dim * 4);
    local_sum += float4(src4[index / 4]);
  }
  local_sum[0] += simd_shuffle_xor(local_sum[0], 16);
  local_sum[1] += simd_shuffle_xor(local_sum[1], 16);
  local_sum[2] += simd_shuffle_xor(local_sum[2], 16);
  local_sum[3] += simd_shuffle_xor(local_sum[3], 16);
  local_sum[0] += simd_shuffle_xor(local_sum[0], 8);
  local_sum[1] += simd_shuffle_xor(local_sum[1], 8);
  local_sum[2] += simd_shuffle_xor(local_sum[2], 8);
  local_sum[3] += simd_shuffle_xor(local_sum[3], 8);
  local_sum[0] += simd_shuffle_xor(local_sum[0], 4);
  local_sum[1] += simd_shuffle_xor(local_sum[1], 4);
  local_sum[2] += simd_shuffle_xor(local_sum[2], 4);
  local_sum[3] += simd_shuffle_xor(local_sum[3], 4);
  local_sum[0] += simd_shuffle_xor(local_sum[0], 2);
  local_sum[1] += simd_shuffle_xor(local_sum[1], 2);
  local_sum[2] += simd_shuffle_xor(local_sum[2], 2);
  local_sum[3] += simd_shuffle_xor(local_sum[3], 2);
  local_sum[0] += simd_shuffle_xor(local_sum[0], 1);
  local_sum[1] += simd_shuffle_xor(local_sum[1], 1);
  local_sum[2] += simd_shuffle_xor(local_sum[2], 1);
  local_sum[3] += simd_shuffle_xor(local_sum[3], 1);
  if (lane_id == 0)
    scratch[sgid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid == 0) {
    float4 reduced = lane_id < QUANTIZE_V_MEAN_SIMDGROUPS ? scratch[lane_id] : float4(0.0f);
    reduced[0] += simd_shuffle_xor(reduced[0], 16);
    reduced[1] += simd_shuffle_xor(reduced[1], 16);
    reduced[2] += simd_shuffle_xor(reduced[2], 16);
    reduced[3] += simd_shuffle_xor(reduced[3], 16);
    reduced[0] += simd_shuffle_xor(reduced[0], 8);
    reduced[1] += simd_shuffle_xor(reduced[1], 8);
    reduced[2] += simd_shuffle_xor(reduced[2], 8);
    reduced[3] += simd_shuffle_xor(reduced[3], 8);
    reduced[0] += simd_shuffle_xor(reduced[0], 4);
    reduced[1] += simd_shuffle_xor(reduced[1], 4);
    reduced[2] += simd_shuffle_xor(reduced[2], 4);
    reduced[3] += simd_shuffle_xor(reduced[3], 4);
    reduced[0] += simd_shuffle_xor(reduced[0], 2);
    reduced[1] += simd_shuffle_xor(reduced[1], 2);
    reduced[2] += simd_shuffle_xor(reduced[2], 2);
    reduced[3] += simd_shuffle_xor(reduced[3], 2);
    reduced[0] += simd_shuffle_xor(reduced[0], 1);
    reduced[1] += simd_shuffle_xor(reduced[1], 1);
    reduced[2] += simd_shuffle_xor(reduced[2], 1);
    reduced[3] += simd_shuffle_xor(reduced[3], 1);
    if (lane_id == 0) {
      mean4[((batch * QUANTIZE_KV_HEADS + head) * 128 + vec_dim * 4) / 4] =
          reduced * (1.0f / float(QUANTIZE_KV_SEQUENCE));
    }
  }
}
)";

static const char* referenceMeanScalar = R"(
kernel void compute_v_mean(
    device const half *src [[buffer(0)]],
    device float *mean [[buffer(1)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort sgid [[simdgroup_index_in_threadgroup]],
    ushort lane_id [[thread_index_in_simdgroup]],
    uint3 tgid [[threadgroup_position_in_grid]]
  ) {
  threadgroup float scratch[QUANTIZE_V_MEAN_SIMDGROUPS];
  const uint dim_bits = ceil_log2_u32(128);
  const uint head_bits = ceil_log2_u32(QUANTIZE_KV_HEADS);
  const uint2 morton = morton_decode_rectangular_2d(tgid.x, dim_bits, head_bits);
  const uint dim = morton.x;
  const uint head = morton.y;
  const uint batch = tgid.z;
  if (dim >= 128 || head >= QUANTIZE_KV_HEADS)
    return;
  float local_sum = 0.0f;
  for (uint column = tid; column < QUANTIZE_KV_SEQUENCE; column += QUANTIZE_V_MEAN_THREADS) {
    const uint index = batch * QUANTIZE_V_BATCH_STRIDE + ((column * QUANTIZE_KV_HEADS + head) * 128 + dim);
    local_sum += (float)src[index];
  }
  local_sum += simd_shuffle_xor(local_sum, 16);
  local_sum += simd_shuffle_xor(local_sum, 8);
  local_sum += simd_shuffle_xor(local_sum, 4);
  local_sum += simd_shuffle_xor(local_sum, 2);
  local_sum += simd_shuffle_xor(local_sum, 1);
  if (lane_id == 0)
    scratch[sgid] = local_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sgid == 0) {
    float reduced = lane_id < QUANTIZE_V_MEAN_SIMDGROUPS ? scratch[lane_id] : 0.0f;
    reduced += simd_shuffle_xor(reduced, 16);
    reduced += simd_shuffle_xor(reduced, 8);
    reduced += simd_shuffle_xor(reduced, 4);
    reduced += simd_shuffle_xor(reduced, 2);
    reduced += simd_shuffle_xor(reduced, 1);
    if (lane_id == 0)
      mean[(batch * QUANTIZE_KV_HEADS + head) * 128 + dim] =
          reduced * (1.0f / float(QUANTIZE_KV_SEQUENCE));
  }
}

)";

static double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return (values[(values.size() - 1) / 2] + values[values.size() / 2]) * 0.5;
}

int main(int argc, char** argv)
{
  setbuf(stdout, nullptr);
  const uint32_t T = argc > 1 ? atoi(argv[1]) : 32768;
  const uint32_t H = argc > 2 ? atoi(argv[2]) : 56;
  const uint32_t N = argc > 3 ? atoi(argv[3]) : 1;
  const int rounds = argc > 4 ? atoi(argv[4]) : 20;
  const uint32_t D = argc > 5 ? atoi(argv[5]) : 128;
  const std::string precision = argc > 6 ? argv[6] : "16F";
  const bool full = argc > 7 && atoi(argv[7]);
  const uint32_t R = argc > 8 ? atoi(argv[8]) : T;
  if (!R || R > T) return 2;
  if ((precision != "16F" && precision != "16BF" && precision != "32F") || !D || D > 256) return 2;
  if (!T || T > 1048576 || !H || H > 256 || !N || N > 8 || rounds < 1 || uint64_t(N) * T * H * D > UINT32_MAX)
    return 2;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) return 2;
  auto context = ccv_nnc_init_mfa_context(device.get());
  if (!ccv_nnc_mfa_context_supported(context) || !ccv_nnc_mfa_has_neural_accelerators(context)) return 2;
  auto queue = NS::TransferPtr(device->newCommandQueue());
  NAInt8AttentionDescriptor descriptor;
  descriptor.matrixDimensions = { R, T, D };
  descriptor.batchDimension = N; descriptor.Hq = H; descriptor.Hk = H;
  descriptor.ioPrecision = precision == "32F" ? GEMMOperandPrecision::FP32 : precision == "16BF" ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16;
  descriptor.scale = 1; descriptor.lowPrecisionIntermediates = precision == "16F";
  if (N > 1) {
    descriptor.batchStrides[AttentionOperand::Q] = R * H * D;
    descriptor.batchStrides[AttentionOperand::K] = T * H * D;
    descriptor.batchStrides[AttentionOperand::V] = T * H * D;
    descriptor.batchStrides[AttentionOperand::O] = R * H * D;
  }
  auto value = context->kernel_cache.findKernel<NAInt8AttentionKernel, NAInt8AttentionDescriptor, NAInt8AttentionKernelDescriptor>(descriptor, device.get(), DeviceProperties());
  auto kernel = value->kernel;
  std::string reference = kernel->source;
  const size_t begin = reference.find("kernel void compute_v_mean(");
  const size_t end = reference.find("kernel void quantize_v(", begin);
  if (begin == std::string::npos || end == std::string::npos) return 2;
  std::string oldMean = D % 4 == 0 ? referenceMean : referenceMeanScalar;
  for (size_t pos = 0; (pos = oldMean.find("128", pos)) != std::string::npos; pos += std::to_string(D).size())
    oldMean.replace(pos, 3, std::to_string(D));
  if (precision != "16F") {
    const size_t pos = oldMean.find("const half *src");
    oldMean.replace(pos, 15, precision == "32F" ? "const float *src" : "const bfloat *src");
  }
  reference.replace(begin, end - begin, oldMean);
  NS::Error* error = nullptr;
  auto library = NS::TransferPtr(device->newLibrary(NS::String::string(reference.c_str(), NS::UTF8StringEncoding), nullptr, &error));
  if (!library) { fprintf(stderr, "%s\n", error->localizedDescription()->utf8String()); return 1; }
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t qTiles = (R + 15) / 16, kvTiles = (T + 63) / 64;
  const uint32_t stride = N > 1 ? T * H * D : 0;
  const uint32_t dimensions[] = { R, T, H, H, 16, 64, qTiles, kvTiles, N > 1 ? R * H * D : 0, stride, stride, N > 1 ? H * qTiles : 0, N > 1 ? H * kvTiles : 0 };
  for (uint32_t i = 0; i < 13; ++i) constants->setConstantValue(&dimensions[i], MTL::DataTypeUInt, 900 + i);
  auto function = NS::TransferPtr(library->newFunction(NS::String::string("compute_v_mean", NS::UTF8StringEncoding), constants.get(), &error));
  if (!function) { fprintf(stderr, "%s\n", error->localizedDescription()->utf8String()); return 1; }
  auto baseline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  if (!baseline) { fprintf(stderr, "%s\n", error->localizedDescription()->utf8String()); return 1; }
  const size_t count = size_t(N) * T * H * D, meanBytes = size_t(N) * H * D * sizeof(float);
  auto input = NS::TransferPtr(device->newBuffer(count * (precision == "32F" ? sizeof(float) : sizeof(uint16_t)), MTL::ResourceStorageModeShared));
  std::array<NS::SharedPtr<MTL::Buffer>, 2> outputs;
  for (auto& output : outputs) output = NS::TransferPtr(device->newBuffer(meanBytes, MTL::ResourceStorageModeShared));
  if (!input || !outputs[0] || !outputs[1]) return 2;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> random(-3, 3);
  for (size_t i = 0; i < count; ++i) {
    const float f = random(rng);
    if (precision == "32F") static_cast<float*>(input->contents())[i] = f;
    else if (precision == "16F") static_cast<_Float16*>(input->contents())[i] = f;
    else {
      uint32_t bits;
      memcpy(&bits, &f, sizeof(bits));
      static_cast<uint16_t*>(input->contents())[i] = (bits + 0x7fff + ((bits >> 16) & 1)) >> 16;
    }
  }
  const size_t elementBytes = precision == "32F" ? 4 : 2;
  const size_t qCount = size_t(N) * R * H * D;
  std::array<NS::SharedPtr<MTL::Buffer>, 3> quantized, scales;
  std::array<NS::SharedPtr<MTL::Buffer>, 2> qkInputs, attentionOutputs;
  NS::SharedPtr<MTL::Buffer> lBuffer;
  if (full) {
    for (int i = 0; i < 3; ++i) {
      quantized[i] = NS::TransferPtr(device->newBuffer(i == 0 ? qCount : count, MTL::ResourceStorageModeShared));
      scales[i] = NS::TransferPtr(device->newBuffer(size_t(N) * H * (i == 0 ? qTiles : kvTiles) * sizeof(float), MTL::ResourceStorageModeShared));
      if (!quantized[i] || !scales[i]) return 2;
    }
    for (int i = 0; i < 2; ++i) {
      qkInputs[i] = NS::TransferPtr(device->newBuffer((i == 0 ? qCount : count) * elementBytes, MTL::ResourceStorageModeShared));
      attentionOutputs[i] = NS::TransferPtr(device->newBuffer(qCount * elementBytes, MTL::ResourceStorageModeShared));
      if (!qkInputs[i] || !attentionOutputs[i]) return 2;
      // Rotate the deterministic random V input to get distinct Q and K values.
      const size_t n = i == 0 ? qCount : count;
      const size_t offset = (i + 1) * D;
      const auto inputBytes = static_cast<const char*>(input->contents());
      const auto outputBytes = static_cast<char*>(qkInputs[i]->contents());
      for (size_t j = 0; j < n; ++j)
        memcpy(outputBytes + j * elementBytes, inputBytes + ((j + offset) % count) * elementBytes, elementBytes);
    }
    lBuffer = NS::TransferPtr(device->newBuffer(size_t(N) * R * H * (precision == "16F" ? 2 : 4), MTL::ResourceStorageModeShared));
    if (!lBuffer) return 2;
  }
  uint32_t vectorsPadded = 1;
  while (vectorsPadded < (D % 4 == 0 ? D / 4 : D)) vectorsPadded *= 2;
  uint32_t headsPadded = 1;
  while (headsPadded < H) headsPadded *= 2;
  printf("device=%s shape=[%u,%u,%u,%u] precision=%s full=%d R=%u rounds=%d production_threads=%u production_tg_bytes=%zu device_tg_cap=%zu\n", device->name()->utf8String(), N, T, H, D, precision.c_str(), full, R, rounds, kernel->vMeanThreadgroupSize(), size_t(value->fifth->staticThreadgroupMemoryLength()), size_t(device->maxThreadgroupMemoryLength()));
  std::array<std::vector<double>, 2> times;
  std::vector<double> ratios;
  for (int round = -4; round < rounds; ++round) {
    double elapsed[2];
    for (int step = 0; step < 2; ++step) {
      const int variant = (round % 2 == 0) ? step : 1 - step;
      auto iterationPool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
      auto commandBuffer = queue->commandBuffer();
      auto batch = ccv_nnc_start_command_batch_from_command_buffer(commandBuffer, 0);
      if (full) {
        for (int operand = 0; operand < 2; ++operand) {
          auto encoder = batch->startCommand();
          encoder->setComputePipelineState(operand == 0 ? value->second.get() : value->third.get());
          encoder->setBuffer(qkInputs[operand].get(), 0, 0);
          encoder->setBuffer(quantized[operand].get(), 0, 1);
          encoder->setBuffer(scales[operand].get(), 0, 2);
          encoder->dispatchThreadgroups(MTL::Size(operand == 0 ? qTiles : kvTiles, H, N), MTL::Size(operand == 0 ? kernel->qQuantizeThreads : kernel->kvQuantizeThreads, 1, 1));
          batch->finishCommand(encoder);
        }
      }
      auto encoder = batch->startCommand();
      encoder->setComputePipelineState(variant ? value->fifth.get() : baseline.get());
      encoder->setBuffer(input.get(), 0, 0);
      encoder->setBuffer(outputs[variant].get(), 0, 1);
      encoder->dispatchThreadgroups(variant ? kernel->vMeanThreadgroupsPerGrid(N) : MTL::Size(vectorsPadded * headsPadded, 1, N), MTL::Size(variant ? kernel->vMeanThreadgroupSize() : kernel->vMeanThreads, 1, 1));
      batch->finishCommand(encoder);
      if (full) {
        encoder = batch->startCommand();
        encoder->setComputePipelineState(value->fourth.get());
        encoder->setBuffer(input.get(), 0, 0);
        encoder->setBuffer(quantized[2].get(), 0, 1);
        encoder->setBuffer(scales[2].get(), 0, 2);
        encoder->setBuffer(outputs[variant].get(), 0, 3);
        encoder->dispatchThreadgroups(MTL::Size(kvTiles, H, N), MTL::Size(kernel->kvQuantizeThreads, 1, 1));
        batch->finishCommand(encoder);
        encoder = batch->startCommand();
        encoder->setComputePipelineState(value->pipeline.get());
        encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation(), 0);
        for (int operand = 0; operand < 3; ++operand) {
          encoder->setBuffer(quantized[operand].get(), 0, operand);
          encoder->setBuffer(scales[operand].get(), 0, 10 + operand);
        }
        encoder->setBuffer(attentionOutputs[variant].get(), 0, 3);
        encoder->setBuffer(lBuffer.get(), 0, 4);
        encoder->setBuffer(outputs[variant].get(), 0, 14);
        encoder->dispatchThreadgroups(kernel->threadgroupsPerGrid(N, R), MTL::Size(kernel->threadgroupSize(value->pipeline.get()), 1, 1));
        batch->finishCommand(encoder);
      }
      ccv_nnc_finish_command_batch(batch);
      commandBuffer->commit(); commandBuffer->waitUntilCompleted();
      if (commandBuffer->status() == MTL::CommandBufferStatusError) { fprintf(stderr, "%s\n", commandBuffer->error()->localizedDescription()->utf8String()); return 1; }
      elapsed[variant] = (commandBuffer->GPUEndTime() - commandBuffer->GPUStartTime()) * 1000;
      if (round >= 0) times[variant].push_back(elapsed[variant]);
    }
    if (round >= 0) ratios.push_back(elapsed[0] / elapsed[1]);
    printf("%s=%d old_ms=%.5f production_ms=%.5f speedup=%.5f\n", round < 0 ? "warmup" : "round", round < 0 ? round + 4 : round, elapsed[0], elapsed[1], elapsed[0] / elapsed[1]);
  }
  const bool equal = memcmp(outputs[0]->contents(), outputs[1]->contents(), meanBytes) == 0 &&
      (!full || memcmp(attentionOutputs[0]->contents(), attentionOutputs[1]->contents(), qCount * elementBytes) == 0);
  printf("RESULT old_ms=%.5f production_ms=%.5f paired_speedup=%.5f byte_equal=%d\n", median(times[0]), median(times[1]), median(ratios), equal);
  ccv_nnc_deinit_mfa_context(context);
  return equal ? 0 : 1;
}
