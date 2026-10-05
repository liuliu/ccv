// Opt-in large-buffer regression; requires roughly 4.3 GiB of shared memory.
// Build: make na_int8_wide_offsets_bench. Run without arguments.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include "nnc/mfa/ccv_nnc_mfa.hpp"

int main(int argc, char** argv)
{
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) return 1;
  const uint32_t maxM = 512, N = 262272, K = 16384;
  const size_t scaleOffset = size_t(N) * K; // Already 128-byte aligned.
  const size_t weightBytes = scaleOffset + size_t(N) * sizeof(_Float16);
  printf("device=%s max_buffer_bytes=%zu weight_bytes=%zu\n",
    device->name()->utf8String(), size_t(device->maxBufferLength()), weightBytes);
  if (weightBytes > device->maxBufferLength()) {
    printf("SKIP: device cannot allocate the >4 GiB weight buffer\n");
    return 77;
  }
  auto a = NS::TransferPtr(device->newBuffer(size_t(maxM) * K * 2, MTL::ResourceStorageModeShared));
  auto b = NS::TransferPtr(device->newBuffer(weightBytes, MTL::ResourceStorageModeShared));
  auto c = NS::TransferPtr(device->newBuffer(size_t(maxM) * N * 2, MTL::ResourceStorageModeShared));
  if (!a || !b || !c) return 1;
  auto av = static_cast<_Float16*>(a->contents());
  auto bv = static_cast<int8_t*>(b->contents());
  auto scales = reinterpret_cast<_Float16*>(bv + scaleOffset);
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> activation(-127, 127), weight(-31, 31);
  for (uint32_t row = 0; row < maxM; ++row) {
    for (uint32_t k = 0; k < K; ++k)
      av[size_t(row) * K + k] = activation(rng) / 128.f;
    // Exact scale 1/128 makes the production quantizer recover the input bytes.
    av[size_t(row) * K] = 127.f / 128;
  }
  std::memset(bv, 0, scaleOffset);
  for (uint32_t col = 0; col < N; ++col) scales[col] = 1.f / 256;
  // Distinct random weight rows at the beginning and on both sides of 4 GiB.
  // Include zero-filled neighboring rows in the CPU checks below.
  const uint32_t columns[] = {0, 1, 127, 128, 131071, 262015, 262016,
    262142, 262143, 262144, 262145, 262270, 262271};
  for (uint32_t col : columns)
    if (col != 128 && col != 262142 && col != 262270)
      for (uint32_t k = 0; k < K; ++k)
        bv[size_t(col) * K + k] = weight(rng);
  auto context = ccv_nnc_init_mfa_context(device.get());
  auto queue = NS::TransferPtr(device->newCommandQueue());
  MTL::Buffer* tensors[] = {a.get(), b.get(), c.get(), nullptr};
  size_t offsets[] = {0, 0, 0, 0};
  int failures = 0;
  // M=511 also exercises the native partial-row store. Keep a fixed-M control.
  for (uint32_t M : {512u, 511u})
    for (int dynamic = 0; dynamic < 2; ++dynamic) {
      ccv_nnc_mfa_scaled_gemm_params_t p = {};
      p.data_type = MTL::DataTypeHalf;
      p.M = M; p.N = N; p.K = K;
      p.batch_dimension = 1; p.use_neural_accelerators = 1; p.loadM = dynamic;
      std::memset(c->contents(), 0xff, c->length()); // NaN sentinel for missing writes.
      auto cb = NS::RetainPtr(queue->commandBuffer());
      auto batch = ccv_nnc_start_command_batch_from_command_buffer(cb.get(), 0);
      ccv_nnc_mfa_encode_scaled_gemm(context, p, batch, tensors, offsets);
      ccv_nnc_finish_command_batch(batch);
      cb->commit(); cb->waitUntilCompleted();
      if (cb->status() != MTL::CommandBufferStatusCompleted) {
        fprintf(stderr, "GPU command failed\n");
        ccv_nnc_deinit_mfa_context(context);
        return 1;
      }
      const auto output = static_cast<const _Float16*>(c->contents());
      int errors = 0, checked = 0;
      for (uint32_t row : {0u, 63u, 127u, 255u, M - 1})
        for (uint32_t col : columns) {
          int32_t dot = 0;
          for (uint32_t k = 0; k < K; ++k)
            dot += int32_t(float(av[size_t(row) * K + k]) * 128) * bv[size_t(col) * K + k];
          const float expected = _Float16(float(dot) / 32768);
          const float actual = output[size_t(row) * N + col];
          if (!std::isfinite(actual) || actual != expected) {
            if (errors < 4)
              printf("mismatch M=%u dynamic=%d row=%u col=%u expected=%g actual=%g\n",
                M, dynamic, row, col, expected, actual);
            ++errors;
          }
          ++checked;
        }
      printf("wide_offsets M=%u N=%u K=%u dynamic=%d checked=%d errors=%d %s\n",
        M, N, K, dynamic, checked, errors, errors ? "FAIL" : "PASS");
      fflush(stdout);
      failures += errors;
    }
  ccv_nnc_deinit_mfa_context(context);
  return failures ? 1 : 0;
}
