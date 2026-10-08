// Production decode timing, randomized CPU validation, and optional stage profiling.
// Profile overrides are benchmark-only; they never change production selection.
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" {
#include "ccv.h"
}
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/kernels/AttentionR1Descriptor.hpp"
#include "nnc/mfa/kernels/AttentionR1Kernel.hpp"
#include "nnc/mfa/kernels/ShaderCache.hpp"

int main(int argc, char** argv)
{
  if (argc < 11 || argc > 18) {
    fprintf(stderr, "Usage: %s R C D B Hq Hkv warmup samples causal dynamic_flags [precision] [profile] [partitions] [reduce_simds] [heads_per_group] [direct_simds] [heads_per_simd]\n", argv[0]);
    return 2;
  }
  for (int i = 1; i < argc; ++i) {
    char* end = nullptr;
    errno = 0;
    const long value = strtol(argv[i], &end, 10);
    if (errno || !*argv[i] || *end || value < 0 || value > INT_MAX) return 2;
  }
  const int r = atoi(argv[1]), c = atoi(argv[2]), d = atoi(argv[3]), b = atoi(argv[4]);
  const int hq = atoi(argv[5]), hk = atoi(argv[6]), warmup = atoi(argv[7]), timed = atoi(argv[8]);
  const int causal = atoi(argv[9]), flags = atoi(argv[10]);
  const int precision = argc > 11 ? atoi(argv[11]) : 0;
  const int profile = argc > 12 ? atoi(argv[12]) : 0;
  const int partitions = argc > 13 ? atoi(argv[13]) : 0;
  const int reduce_simds = argc > 14 ? atoi(argv[14]) : 0;
  const int heads_per_group = argc > 15 ? atoi(argv[15]) : 0;
  const int direct_simds = argc > 16 ? atoi(argv[16]) : 0;
  const int heads_per_simd = argc > 17 ? atoi(argv[17]) : 0;
  const int upcast = 1;
  if (std::min({r,c,d,b,hq,hk,timed}) <= 0 || hq % hk || hq > USHRT_MAX || d > USHRT_MAX ||
      r > 2 || (d != 128 && d != 256) || causal > 1 || precision > 1 || profile > 1 || partitions > 1024 || reduce_simds > 32 || heads_per_group > 16 || direct_simds > 32 || heads_per_simd > 2 ||
      (direct_simds && (partitions || heads_per_group || heads_per_simd)) ||
      (heads_per_group && ((hq/hk) % heads_per_group || heads_per_group*r > 32)) ||
      (!profile && (partitions || reduce_simds || heads_per_group || direct_simds || heads_per_simd)) || (flags & ~3) || uint64_t(r) * b > INT_MAX || r > INT_MAX - 128 || c > INT_MAX - 128)
    return 2;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) return 2;
  const long double limit = device->maxBufferLength() / sizeof(float);
  if (static_cast<long double>(b)*r*hq*d > limit || static_cast<long double>(b)*c*hk*d > limit)
    return 2;
  auto context = ccv_nnc_init_mfa_context(device.get());
  if (flags & 1) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_GEMM_SPECIALIZING_M);
  if (flags & 2) ccv_nnc_enable_flag(CCV_NNC_DISABLE_MFA_ATTENTION_SPECIALIZING_C);
  auto queue = NS::TransferPtr(device->newCommandQueue());
  const size_t q_count = size_t(b)*r*hq*d, kv_count = size_t(b)*c*hk*d;
  const size_t element_size = sizeof(uint16_t);
  std::vector<float> block(1 << 18);
  std::array<NS::SharedPtr<MTL::Buffer>, 4> buffers;
  const size_t counts[] = {q_count, kv_count, kv_count, q_count};
  for (int i = 0; i < 4; ++i) {
    buffers[i] = NS::TransferPtr(device->newBuffer(counts[i]*element_size, MTL::ResourceStorageModeShared));
    if (!buffers[i]) return 2;
    if (i < 3) {
      for (size_t offset = 0; offset < counts[i]; offset += block.size()) {
        const size_t count = std::min(block.size(), counts[i] - offset);
        for (size_t j = 0; j < count; ++j) {
          const size_t index = offset + j;
          uint32_t value = uint32_t(index)*747796405u + uint32_t(i+1)*2891336453u;
          value = ((value >> ((value >> 28)+4)) ^ value)*277803737u;
          value = (value >> 22) ^ value;
          block[j] = float(int(value % 2047)-1023)/1024;
          if (i == 2) block[j] = block[j]*0.3f + float((index/d) % hk)*0.01f;
        }
        auto dst = static_cast<unsigned char*>(buffers[i]->contents()) + offset*element_size;
        if (precision == 0)
          ccv_float_to_half_precision(block.data(), reinterpret_cast<uint16_t*>(dst), count);
        else
          ccv_float_to_bfloat(block.data(), reinterpret_cast<uint16_t*>(dst), count);
      }
    }
  }
  ccv_nnc_mfa_attention_params_t params = {};
  params.data_type = precision == 1 ? MTL::DataTypeBFloat : MTL::DataTypeHalf;
  params.upcast = upcast;
  params.R = r; params.C = c; params.D = d; params.Hq = hq; params.Hk = hk;
  params.output_rows = r*b; params.K_trans = 1; params.alpha = 1/std::sqrt(float(d));
  params.is_causal = causal; params.batched = b > 1; params.batch_dims_q[0] = b > 1 ? b : 0;
  params.use_neural_accelerators = 1; params.use_quantized_attention = 0;
  printf("device=%s gpu_core_count=%u R=%d C=%d D=%d B=%d Hq=%d Hkv=%d causal=%d dynamic_flags=%d precision=%d upcast=%d\n",
      device->name()->utf8String(), context->device_properties.coreCount, r, c, d, b, hq, hk, causal, flags, precision, upcast);
  const double minimum_warmup = getenv("CCV_NA_WARMUP_SECONDS") ? atof(getenv("CCV_NA_WARMUP_SECONDS")) : 2.0;
  if (!std::isfinite(minimum_warmup) || minimum_warmup < 0) return 2;
  auto descriptor = AttentionR1Descriptor::select(
      precision == 1 ? GEMMOperandPrecision::BF16 : GEMMOperandPrecision::FP16,
      c, hq, hk, d, params.alpha, true, false, r, causal, b, context->device_properties.coreCount);
  if (partitions) {
    if (descriptor.mode == AttentionR1Descriptor::Mode::cooperative) {
      descriptor.simdgroups = r;
      descriptor.headsPerSIMD = 1;
    }
    descriptor.workgroups = partitions;
    descriptor.mode = partitions == 1 ? AttentionR1Descriptor::Mode::direct : AttentionR1Descriptor::Mode::splitReduce;
  }
  if (reduce_simds) descriptor.reductionSIMDGroups = reduce_simds;
  if (heads_per_group || heads_per_simd) {
    if (descriptor.mode == AttentionR1Descriptor::Mode::cooperative) return 2;
    const uint32_t group_heads = heads_per_group ? heads_per_group : descriptor.simdgroups * descriptor.headsPerSIMD / r;
    const uint32_t per_simd = heads_per_simd ? heads_per_simd : descriptor.headsPerSIMD;
    if (group_heads % per_simd) return 2;
    descriptor.simdgroups = group_heads * r / per_simd;
    descriptor.headsPerSIMD = per_simd;
  }
  if (direct_simds) {
    descriptor.simdgroups = direct_simds;
    descriptor.headsPerSIMD = 1;
    descriptor.workgroups = 1;
    descriptor.reductionSIMDGroups = 1;
    descriptor.mode = AttentionR1Descriptor::Mode::cooperative;
  }
  const bool cooperative = descriptor.mode == AttentionR1Descriptor::Mode::cooperative;
  const bool split = descriptor.mode == AttentionR1Descriptor::Mode::splitReduce;
  printf("profile_scratch=production profile=%d overridden=%d partitions=%u heads_per_group=%u reduce_simds=%u cooperative=%d direct_simds=%u heads_per_simd=%u\n",
      profile, !!(partitions || reduce_simds || heads_per_group || direct_simds || heads_per_simd), descriptor.workgroups, cooperative ? 0 : descriptor.simdgroups*descriptor.headsPerSIMD/r, descriptor.reductionSIMDGroups, cooperative, cooperative ? descriptor.simdgroups : 0, descriptor.headsPerSIMD);
  auto pipeline = context->kernel_cache.findKernel<AttentionR1Kernel, AttentionR1Descriptor, AttentionR1KernelDescriptor>(
      descriptor, context->device.get(), context->device_properties);
  unsigned char shader_digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256(pipeline->kernel->source.data(), CC_LONG(pipeline->kernel->source.size()), shader_digest);
  printf("shader_sha256=");
  for (auto byte : shader_digest) printf("%02x", byte);
  printf("\n");
  const size_t scratch_bytes = size_t(b)*r*hq*descriptor.workgroups*(d+2)*sizeof(float);
  if (scratch_bytes > device->maxBufferLength() ||
      pipeline->kernel->threadgroupMemoryAllocation(descriptor) > device->maxThreadgroupMemoryLength() ||
      32*descriptor.simdgroups > pipeline->pipeline->maxTotalThreadsPerThreadgroup() ||
      (split && (32*descriptor.reductionSIMDGroups > pipeline->second->maxTotalThreadsPerThreadgroup() ||
      descriptor.reductionSIMDGroups*d*sizeof(float) > device->maxThreadgroupMemoryLength()))) return 2;
  // Match production's private, capacity-rounded scratch allocation. A separate
  // shared buffer can misrepresent the cost of a selected split geometry.
  auto scratch = NS::RetainPtr(context->request_scratch(scratch_bytes));
  if (!scratch) return 2;
  const uint32_t c_length = c;
  // Standalone stage timings describe bottlenecks; only the full dispatcher
  // timing is compared with complete MLX SDPA. Overrides use the same shaders.
  std::vector<double> samples;
  for (int stage = 0; stage < (profile ? (split ? 4 : 2) : 1); ++stage) {
    double warm_seconds = 0;
    samples.clear();
    for (int iteration = 0; samples.size() < size_t(timed); ++iteration) {
      auto iteration_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
      auto command_buffer = queue->commandBuffer();
      auto batch = ccv_nnc_start_command_batch_from_command_buffer(command_buffer, 0);
      if (stage == 0) {
        MTL::Buffer* tensors[10] = {buffers[0].get(), buffers[1].get(), buffers[2].get(), buffers[3].get()};
        size_t offsets[10] = {};
        ccv_nnc_mfa_encode_attention(context, params, batch, tensors, offsets);
      } else {
        if (stage == 1 || stage == 2) {
          auto encoder = batch->startCommand();
          encoder->setComputePipelineState(pipeline->pipeline.get());
          encoder->setThreadgroupMemoryLength(pipeline->kernel->threadgroupMemoryAllocation(descriptor),0);
          for (int tensor = 0; tensor < 3; ++tensor) {
            encoder->useResource(buffers[tensor].get(), MTL::ResourceUsageRead);
            encoder->setBuffer(buffers[tensor].get(), 0, tensor);
          }
          auto destination = split ? scratch.get() : buffers[3].get();
          encoder->useResource(destination, MTL::ResourceUsageWrite);
          encoder->setBuffer(destination, 0, 3);
          encoder->setBytes(&c_length, sizeof(c_length), 4);
          encoder->dispatchThreadgroups(cooperative ? MTL::Size(hq,b,r) : MTL::Size(hq/(descriptor.simdgroups/r*descriptor.headsPerSIMD),b,descriptor.workgroups), MTL::Size(32*descriptor.simdgroups,1,1));
          batch->finishCommand(encoder);
        }
        if (split && (stage == 1 || stage == 3)) {
          auto encoder = batch->startCommand();
          encoder->setComputePipelineState(pipeline->second.get());
          encoder->setThreadgroupMemoryLength(descriptor.reductionSIMDGroups*d*sizeof(float),0);
          encoder->useResource(scratch.get(), MTL::ResourceUsageRead);
          encoder->useResource(buffers[3].get(), MTL::ResourceUsageWrite);
          encoder->setBuffer(scratch.get(),0,0);
          encoder->setBuffer(buffers[3].get(),0,1);
          encoder->dispatchThreadgroups(MTL::Size(hq,b,r), MTL::Size(32*descriptor.reductionSIMDGroups,1,1));
          batch->finishCommand(encoder);
        }
      }
      ccv_nnc_finish_command_batch(batch);
      command_buffer->commit();
      command_buffer->waitUntilCompleted();
      if (command_buffer->status() != MTL::CommandBufferStatusCompleted) return 3;
      const double seconds = command_buffer->GPUEndTime() - command_buffer->GPUStartTime();
      if (iteration < warmup || warm_seconds < minimum_warmup) warm_seconds += seconds;
      else samples.push_back(seconds*1000);
    }
    printf("stage=%s samples_ms=", stage==0 ? "production" : (stage==1 ? "selected" : (stage==2 ? "partials" : "reduce")));
    for (size_t i=0; i<samples.size(); ++i) printf("%s%.9g", i ? "," : "", samples[i]);
    std::sort(samples.begin(),samples.end());
    printf(" median_ms=%.9g\n", (samples[(samples.size()-1)/2]+samples[samples.size()/2])*0.5);
  }
  const auto q = buffers[0]->contents();
  const auto k = buffers[1]->contents();
  const auto v = buffers[2]->contents();
  const auto output = buffers[3]->contents();
  auto read = [precision](const void* data, size_t i) -> float {
    if (precision == 0) return static_cast<const _Float16*>(data)[i];
    const uint32_t bits = uint32_t(static_cast<const uint16_t*>(data)[i]) << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
  };
  CC_SHA256_CTX hash;
  CC_SHA256_Init(&hash);
  for (size_t offset = 0; offset < q_count; offset += block.size()) {
    const size_t count = std::min(block.size(), q_count - offset);
    for (size_t i = 0; i < count; ++i) {
      block[i] = read(output, offset+i);
      if (!std::isfinite(block[i])) return 4;
    }
    CC_SHA256_Update(&hash, block.data(), CC_LONG(count * sizeof(float)));
  }
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &hash);
  printf("output_sha256=");
  for (auto byte : digest) printf("%02x", byte);
  printf("\n");
  double error = 0, norm = 0, max_error = 0;
  for (int z : {0,b-1}) for (int h : {0,hq-1}) for (int row : {0,r/2,r-1}) {
    const int kh = h/(hq/hk), cols = causal ? std::max(0,std::min(c,row+(c-r)+1)) : c;
    std::vector<double> scores(cols);
    for (int col = 0; col < cols; ++col) {
      double dot = 0;
      for (int dim = 0; dim < d; ++dim)
        dot += read(q, ((size_t(z)*r+row)*hq+h)*d+dim)*read(k, ((size_t(z)*c+col)*hk+kh)*d+dim);
      scores[col] = dot/std::sqrt(double(d));
    }
    const double maximum = cols ? *std::max_element(scores.begin(), scores.end()) : 0;
    double sum = 0;
    for (auto& score : scores) { score = std::exp(score-maximum); sum += score; }
    for (int dim : {0,d/2,d-1}) {
      double expected = 0;
      for (int col = 0; col < cols; ++col)
        expected += scores[col]*read(v, ((size_t(z)*c+col)*hk+kh)*d+dim);
      if (sum > 0) expected /= sum;
      const double diff = read(output, ((size_t(z)*r+row)*hq+h)*d+dim)-expected;
      error += diff*diff; norm += expected*expected; max_error = std::max(max_error,std::abs(diff));
    }
  }
  const double l2 = std::sqrt(error/std::max(norm,1e-30));
  printf("cpu_sample_max_abs=%.9g normalized_l2=%.9g all_finite=1\n", max_error, l2);
  ccv_nnc_deinit_mfa_context(context);
  return l2 <= (precision == 1 ? 0.01 : 0.003) ? 0 : 4;
}
