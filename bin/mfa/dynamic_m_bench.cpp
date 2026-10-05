// Build: make dynamic_m_bench. Reports median GPU time including activation quantization.
// Argument 1 checks layouts/reuse; argument 2 checks register/native transitions
// in both call orders without warmup (correctness only, not performance timing).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulDescriptor.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulKernel.hpp"
#include "nnc/mfa/kernels/NAInt8MatMulKernelDescriptor.hpp"

int main(int argc, char** argv)
{
  const bool require_reuse = argc > 1 && atoi(argv[1]);
  const bool register_cases = argc > 1 && atoi(argv[1]) == 2;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  auto queue = NS::TransferPtr(device->newCommandQueue());
  auto context = ccv_nnc_init_mfa_context(device.get());
  if (const char* cores = std::getenv("CCV_PROBE_CORES"))
    context->device_properties.coreCount = std::atoi(cores);
  printf("device=%s policy_cores=%u\n", device->name()->utf8String(), context->device_properties.coreCount);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> random(-1, 1);
  // Mode 1 keeps the original layout/reuse suite; mode 2 checks production
  // register/native transitions in both call orders with one dispatch per case.
  for (int layout = register_cases ? 3 : 0; layout < (register_cases ? 7 : 3); ++layout) {
    const uint32_t B = layout == 0 || register_cases ? 1 : 3;
    const uint32_t N = layout == 3 || layout == 5 ? 2560 : layout == 4 ? 2176 : layout == 6 ? 1536 : 1024;
    const uint32_t K = layout == 3 ? 9216 : layout == 4 ? 8448 : layout == 5 ? 8192 : layout == 6 ? 6144 : 4096;
    for (int reverse = 0; reverse < (register_cases ? 2 : 1); ++reverse) {
      context->kernel_cache.evict();
      PipelineValue<NAInt8MatMulKernel>* previous = nullptr;
      uint32_t previousM = 0;
      NAInt8MatMulDescriptor previousDescriptor;
      std::vector<uint32_t> rows = register_cases ?
        std::vector<uint32_t>{511, 512, 513, 575, 576, 577, 2457, 2458, 4095, 4096, 4097} :
        std::vector<uint32_t>{7, 16, 17, 512, 1536, 4095, 4096, 4097};
      if (reverse) std::reverse(rows.begin(), rows.end());
      for (uint32_t M : rows) {
        ccv_nnc_mfa_scaled_gemm_params_t p = {};
        p.data_type = MTL::DataTypeHalf;
        p.M = M; p.N = N; p.K = K;
        p.fused_bias = 1; p.use_neural_accelerators = 1;
        p.batch_dimension = B;
        p.batch_stride_a = layout == 2 ? K : M * K;
        p.batch_stride_c = layout == 2 ? N : M * N;
        p.leading_dimension_a = register_cases ? 0 : layout == 2 ? B * K : K;
        p.leading_dimension_c = register_cases ? 0 : layout == 2 ? B * N : N;
        auto a = NS::TransferPtr(device->newBuffer(size_t(B) * M * K * 2, MTL::ResourceStorageModeShared));
        auto b = NS::TransferPtr(device->newBuffer(size_t(N) * K + N * 2, MTL::ResourceStorageModeShared));
        auto c = NS::TransferPtr(device->newBuffer(size_t(B) * M * N * 2, MTL::ResourceStorageModeShared));
        auto bias = NS::TransferPtr(device->newBuffer(N * 2, MTL::ResourceStorageModeShared));
        for (size_t i = 0; i < size_t(B) * M * K; ++i)
          ((_Float16*)a->contents())[i] = random(rng);
        for (size_t i = 0; i < size_t(N) * K; ++i)
          ((int8_t*)b->contents())[i] = int(random(rng) * 127);
        for (uint32_t i = 0; i < N; ++i) {
          ((_Float16*)((char*)b->contents() + size_t(N) * K))[i] = 0.01f;
          ((_Float16*)bias->contents())[i] = random(rng);
        }
        MTL::Buffer* tensors[] = {a.get(), b.get(), c.get(), bias.get(), nullptr};
        size_t offsets[] = {0, 0, 0, 0};
        std::vector<_Float16> reference;
        double times[2];
        for (int dynamic = 0; dynamic < 2; ++dynamic) {
          p.loadM = dynamic;
          std::vector<double> samples;
          for (int iteration = 0; iteration < (register_cases ? 1 : 13); ++iteration) {
            auto cb = queue->commandBuffer();
            auto batch = ccv_nnc_start_command_batch_from_command_buffer(cb, 0);
            for (int repeat = 0; repeat < (register_cases ? 1 : 4); ++repeat)
              ccv_nnc_mfa_encode_scaled_gemm(context, p, batch, tensors, offsets);
            ccv_nnc_finish_command_batch(batch);
            cb->commit(); cb->waitUntilCompleted();
            if (cb->status() == MTL::CommandBufferStatusError) {
              fprintf(stderr, "%s\n", cb->error()->localizedDescription()->utf8String());
              return 1;
            }
            if (register_cases || iteration >= 3)
              samples.push_back((cb->GPUEndTime() - cb->GPUStartTime()) * 1e3 / (register_cases ? 1 : 4));
          }
          std::sort(samples.begin(), samples.end());
          times[dynamic] = (samples[(samples.size() - 1) / 2] + samples[samples.size() / 2]) / 2;
          auto output = (_Float16*)c->contents();
          if (!dynamic)
            reference.assign(output, output + size_t(B) * M * N);
          else if (memcmp(reference.data(), output, reference.size() * 2)) {
            fprintf(stderr, "specialized/dynamic mismatch layout=%d M=%u\n", layout, M);
            return 1;
          }
        }
        // Independently sample CPU int8 dot products, including the last row of each batch.
        for (uint32_t batch = 0; batch < B; ++batch)
          for (uint32_t row : {0u, M - 1}) {
            auto src = (_Float16*)a->contents() + batch * p.batch_stride_a + row * (p.leading_dimension_a ? p.leading_dimension_a : K);
            float max_abs = 0;
            for (uint32_t k = 0; k < K; ++k) max_abs = std::max(max_abs, fabsf(src[k]));
            const _Float16 scale = max_abs / 127.f;
            for (uint32_t col : {0u, N - 1}) {
              int dot = 0;
              for (uint32_t k = 0; k < K; ++k)
                dot += int(rintf(float(src[k]) * (127.f / max_abs))) * ((int8_t*)b->contents())[size_t(col) * K + k];
              float expected = float(dot) * float(scale) * float((_Float16)0.01f) + float(((_Float16*)bias->contents())[col]);
              float actual = reference[batch * p.batch_stride_c + row * (p.leading_dimension_c ? p.leading_dimension_c : N) + col];
              if (fabsf(expected - actual) > 0.002f * std::max(1.f, fabsf(expected))) {
                fprintf(stderr, "CPU mismatch: %g vs %g\n", actual, expected); return 1;
              }
            }
          }
        NAInt8MatMulDescriptor d;
        d.matrixDimensions = simd::uint3{M, N, K}; d.batchDimension = B;
        d.loadM = true; d.useBias = true;
        if (p.leading_dimension_a)
          d.leadingDimensions = simd::uint2{p.leading_dimension_a, p.leading_dimension_c};
        if (B > 1) {
          d.batchStrides = simd::uint4{p.batch_stride_a, 0, p.batch_stride_c, 0};
          d.packedABatchStride = M * K; d.aScaleBatchStride = M;
        }
        auto value = context->kernel_cache.findKernel<NAInt8MatMulKernel, NAInt8MatMulDescriptor, NAInt8MatMulKernelDescriptor>(d, device.get(), context->device_properties);
        const bool reused = previous && value == previous;
        const bool sameRange = previous && (register_cases ? d == previousDescriptor : (M >= 4096) == (previousM >= 4096));
        if (require_reuse && previous && reused != sameRange) { fprintf(stderr, "incorrect M-range pipeline reuse\n"); return 1; }
        previous = value;
        previousDescriptor = d;
        previousM = M;
        auto fixedDescriptor = d; fixedDescriptor.loadM = false;
        const auto expected = NAInt8MatMulKernelDescriptor(fixedDescriptor, context->device_properties);
        if (value->kernel->useRegisterOperands != expected.useRegisterOperands) {
          fprintf(stderr, "cached selection differs from fixed-M policy\n"); return 1;
        }
        printf("int8 layout=%d reverse=%d M=%u specialized_ms=%.6f dynamic_ms=%.6f reused=%d register=%d parity=pass\n",
          layout, reverse, M, times[0], times[1], reused, value->kernel->useRegisterOperands);
        fflush(stdout);
      }
    }
  }
  ccv_nnc_deinit_mfa_context(context);
  return 0;
}
