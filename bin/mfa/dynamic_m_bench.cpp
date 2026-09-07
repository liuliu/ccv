// Build: make dynamic_m_bench. Reports median GPU time including activation quantization.
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
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  auto queue = NS::TransferPtr(device->newCommandQueue());
  auto context = ccv_nnc_init_mfa_context(device.get());
  printf("device=%s\n", device->name()->utf8String());
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> random(-1, 1);
  for (int layout = 0; layout < 3; ++layout) {
    const uint32_t B = layout == 0 ? 1 : 3, N = 1024, K = 4096;
    PipelineValue<NAInt8MatMulKernel>* previous = nullptr;
    for (uint32_t M : {7u, 16u, 17u, 512u, 1536u, 4095u, 4096u}) {
      ccv_nnc_mfa_scaled_gemm_params_t p = {};
      p.data_type = MTL::DataTypeHalf;
      p.M = M; p.N = N; p.K = K;
      p.fused_bias = 1; p.use_neural_accelerators = 1;
      p.batch_dimension = B;
      p.batch_stride_a = layout == 2 ? K : M * K;
      p.batch_stride_c = layout == 2 ? N : M * N;
      p.leading_dimension_a = layout == 2 ? B * K : K;
      p.leading_dimension_c = layout == 2 ? B * N : N;
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
        for (int iteration = 0; iteration < 13; ++iteration) {
          auto cb = queue->commandBuffer();
          auto batch = ccv_nnc_start_command_batch_from_command_buffer(cb, 0);
          for (int repeat = 0; repeat < 4; ++repeat)
            ccv_nnc_mfa_encode_scaled_gemm(context, p, batch, tensors, offsets);
          ccv_nnc_finish_command_batch(batch);
          cb->commit(); cb->waitUntilCompleted();
          if (cb->status() == MTL::CommandBufferStatusError) {
            fprintf(stderr, "%s\n", cb->error()->localizedDescription()->utf8String());
            return 1;
          }
          if (iteration >= 3)
            samples.push_back((cb->GPUEndTime() - cb->GPUStartTime()) * 1e3 / 4);
        }
        std::sort(samples.begin(), samples.end());
        times[dynamic] = (samples[4] + samples[5]) / 2;
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
          auto src = (_Float16*)a->contents() + batch * p.batch_stride_a + row * p.leading_dimension_a;
          float max_abs = 0;
          for (uint32_t k = 0; k < K; ++k) max_abs = std::max(max_abs, fabsf(src[k]));
          const _Float16 scale = max_abs / 127.f;
          for (uint32_t col : {0u, N - 1}) {
            int dot = 0;
            for (uint32_t k = 0; k < K; ++k)
              dot += int(rintf(float(src[k]) * (127.f / max_abs))) * ((int8_t*)b->contents())[size_t(col) * K + k];
            float expected = float(dot) * float(scale) * float((_Float16)0.01f) + float(((_Float16*)bias->contents())[col]);
            float actual = reference[batch * p.batch_stride_c + row * p.leading_dimension_c + col];
            if (fabsf(expected - actual) > 0.002f * std::max(1.f, fabsf(expected))) {
              fprintf(stderr, "CPU mismatch: %g vs %g\n", actual, expected); return 1;
            }
          }
        }
      NAInt8MatMulDescriptor d;
      d.matrixDimensions = simd::uint3{M, N, K}; d.batchDimension = B;
      d.loadM = true; d.useBias = true;
      d.leadingDimensions = simd::uint2{p.leading_dimension_a, p.leading_dimension_c};
      if (B > 1) {
        d.batchStrides = simd::uint4{p.batch_stride_a, 0, p.batch_stride_c, 0};
        d.packedABatchStride = M * K; d.aScaleBatchStride = M;
      }
      auto value = context->kernel_cache.findKernel<NAInt8MatMulKernel, NAInt8MatMulDescriptor, NAInt8MatMulKernelDescriptor>(d, device.get(), DeviceProperties());
      const bool reused = previous && value == previous;
      if (require_reuse && previous && M < 4096 && !reused) { fprintf(stderr, "pipeline not reused\n"); return 1; }
      if (M == 4096 && reused) { fprintf(stderr, "lost 4096 bucket\n"); return 1; }
      previous = value;
      printf("int8 layout=%d M=%u specialized_ms=%.6f dynamic_ms=%.6f reused=%d parity=pass\n", layout, M, times[0], times[1], reused);
      fflush(stdout);
    }
  }
  ccv_nnc_deinit_mfa_context(context);
}
