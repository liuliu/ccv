// Build: make na_int8_sol_attention_bench
// Usage: ./na_int8_sol_attention_bench [T=32768] [H=56] [rounds=6] [raw-prefix] [begin=470] [scale=1]
// Raw inputs are contiguous FP16 [1,T,H,128] in prefix.{q,k,v}.bin.
// With no capture, deterministic random inputs measure shapes, not model quality.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "nnc/mfa/ccv_nnc_mfa.hpp"

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
  const int rounds = argc > 3 ? atoi(argv[3]) : 6;
  const char* raw = argc > 4 && strcmp(argv[4], "-") != 0 ? argv[4] : nullptr;
  const uint32_t begin = argc > 5 ? atoi(argv[5]) : std::min(470u, T);
  const float scale = argc > 6 ? atof(argv[6]) : 1;
  if (T == 0 || T > 1048576 || H == 0 || H > 256 || rounds < 1 || begin > T || !std::isfinite(scale)) return 2;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) return 2;
  auto context = ccv_nnc_init_mfa_context(device.get());
  if (!ccv_nnc_mfa_context_supported(context) || !ccv_nnc_mfa_has_neural_accelerators(context)) {
    fprintf(stderr, "This benchmark requires neural matrix accelerators.\n"); return 2;
  }
  auto queue = NS::TransferPtr(device->newCommandQueue());
  printf("device=%s input=%s shape=[1,%u,%u,128] B=64 Q=64 radius=1 tau=0.5 span=[%u,%u) scale=%g rounds=%d\n",
    device->name()->utf8String(), raw ? raw : "synthetic-uniform-seed42", T, H, begin, T, scale, rounds);
  const size_t count = size_t(T) * H * 128, bytes = count * 2;
  std::array<NS::SharedPtr<MTL::Buffer>, 6> buffers;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> random(-1, 1);
  const char* suffix[] = { "q", "k", "v" };
  for (int i = 0; i < 6; ++i) {
    buffers[i] = NS::TransferPtr(device->newBuffer(bytes, MTL::ResourceStorageModeShared));
    if (!buffers[i]) { fprintf(stderr, "Tensor allocation failed.\n"); return 2; }
    if (i < 3) {
      if (raw) {
        std::ifstream file(std::string(raw) + "." + suffix[i] + ".bin", std::ios::binary);
        file.read(static_cast<char*>(buffers[i]->contents()), bytes);
        if (size_t(file.gcount()) != bytes || file.peek() != std::ifstream::traits_type::eof()) {
          fprintf(stderr, "Capture size must match the requested shape.\n"); return 2;
        }
      } else {
        auto data = static_cast<_Float16*>(buffers[i]->contents());
        for (size_t j = 0; j < count; ++j) data[j] = random(rng) * (i == 0 ? 0.3f : 3.f);
      }
    }
  }
  ccv_nnc_mfa_sol_attention_params_t sol = {};
  sol.N = 1; sol.T = T; sol.H = H; sol.block_size = 64; sol.query_block_size = 64;
  sol.approximation_start = begin; sol.approximation_end = T;
  sol.scale = scale; sol.tau = 0.5f; sol.local_block_radius = 1;
  ccv_nnc_mfa_attention_params_t native = {};
  native.data_type = MTL::DataTypeHalf;
  native.R = T; native.C = T; native.Hq = H; native.Hk = H; native.D = 128;
  native.output_rows = T; native.K_trans = 1; native.alpha = scale;
  native.use_neural_accelerators = 1; native.use_quantized_attention = 1;
  ccv_nnc_mfa_prepare_attention(context, native);
  const uint32_t J = (T + 63) / 64, JP = (J + 63) / 64 * 64;
  const size_t route_offset = size_t(H) * J * 128 * 4 + size_t(H) * JP * 128 * 6 + size_t(H) * 128 * 8;
  const size_t route_count = size_t(H) * J * J;
  auto route_copy = NS::TransferPtr(device->newBuffer(route_count, MTL::ResourceStorageModeShared));
  if (!route_copy) return 2;
  std::array<std::vector<double>, 3> times;
  std::vector<double> exact_ratios, sparse_ratios;
  for (int round = -3; round < rounds; ++round) {
    double elapsed[3] = {};
    // Alternate order to balance thermal and cache effects within each round.
    for (int step = 0; step < 3; ++step) {
      const int variant = ((round + 3) % 2 == 0) ? step : 2 - step;
      auto iteration_pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
      auto cb = queue->commandBuffer();
      auto batch = ccv_nnc_start_command_batch_from_command_buffer(cb, 0);
      MTL::Buffer* tensors[10] = { buffers[0].get(), buffers[1].get(), buffers[2].get(), buffers[3 + variant].get() };
      size_t offsets[10] = {};
      if (variant == 0)
        ccv_nnc_mfa_encode_attention(context, native, batch, tensors, offsets);
      else {
        auto params = sol;
        if (variant == 1) params.local_block_radius = J; // Actual all-exact kernel, no native bypass.
        ccv_nnc_mfa_encode_sol_attention(context, params, batch, tensors, offsets);
      }
      ccv_nnc_finish_command_batch(batch);
      cb->commit(); cb->waitUntilCompleted();
      if (cb->status() == MTL::CommandBufferStatusError) {
        fprintf(stderr, "%s\n", cb->error()->localizedDescription()->utf8String()); return 1;
      }
      elapsed[variant] = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000;
      if (round >= 0) times[variant].push_back(elapsed[variant]);
      // Validate every output, including warmups, to catch intermittent failures.
      const auto output = static_cast<const _Float16*>(buffers[3 + variant]->contents());
      for (size_t i = 0; i < count; ++i)
        if (!std::isfinite(float(output[i]))) {
          fprintf(stderr, "Nonfinite output: round=%d variant=%d index=%zu.\n", round, variant, i);
          return 1;
        }
      if (variant == 2 && round == rounds - 1) {
        // Inspect actual routing outside the timed attention command buffer.
        auto copy_cb = queue->commandBuffer();
        auto blit = copy_cb->blitCommandEncoder();
        blit->copyFromBuffer(context->scratch.get(), route_offset, route_copy.get(), 0, route_count);
        blit->endEncoding(); copy_cb->commit(); copy_cb->waitUntilCompleted();
        if (copy_cb->status() == MTL::CommandBufferStatusError) return 1;
      }
    }
    printf("%s=%d native_ms=%.4f exact_ms=%.4f sparse_ms=%.4f ratios=%.4f,%.4f\n",
      round < 0 ? "warmup" : "round", round < 0 ? round + 3 : round, elapsed[0], elapsed[1], elapsed[2], elapsed[0] / elapsed[1], elapsed[0] / elapsed[2]);
    if (round >= 0) { exact_ratios.push_back(elapsed[0] / elapsed[1]); sparse_ratios.push_back(elapsed[0] / elapsed[2]); }
  }
  auto dense_output = static_cast<const _Float16*>(buffers[3]->contents());
  auto exact_output = static_cast<const _Float16*>(buffers[4]->contents());
  auto sparse_output = static_cast<const _Float16*>(buffers[5]->contents());
  double exact_error = 0, sparse_error = 0, norm = 0, protected_error = 0, protected_norm = 0;
  for (size_t i = 0; i < count; ++i) {
    const double d = dense_output[i], e = exact_output[i], s = sparse_output[i];
    if (!std::isfinite(d) || !std::isfinite(e) || !std::isfinite(s)) { fprintf(stderr, "Nonfinite output at %zu.\n", i); return 1; }
    exact_error += (d - e) * (d - e); sparse_error += (d - s) * (d - s); norm += d * d;
    const uint32_t row = i / (H * 128);
    if ((row / 64) * 64 < begin) { protected_error += (d - s) * (d - s); protected_norm += d * d; }
  }
  size_t exact_blocks = 0;
  auto routes = static_cast<const uint8_t*>(route_copy->contents());
  for (size_t i = 0; i < route_count; ++i) exact_blocks += routes[i] != 0;
  const double exact_l2 = sqrt(exact_error / std::max(norm, 1e-30));
  const double protected_l2 = sqrt(protected_error / std::max(protected_norm, 1e-30));
  printf("RESULT native_ms=%.4f exact_ms=%.4f sparse_ms=%.4f paired_exact_speedup=%.4f paired_sparse_speedup=%.4f exact_block_fraction=%.6f exact_relative_l2=%.8g protected_relative_l2=%.8g sparse_relative_l2=%.8g scratch_bytes=%zu\n",
    median(times[0]), median(times[1]), median(times[2]), median(exact_ratios), median(sparse_ratios), double(exact_blocks) / route_count,
    exact_l2, protected_l2, sqrt(sparse_error / std::max(norm, 1e-30)), size_t(context->scratch->length()));
  ccv_nnc_deinit_mfa_context(context);
  return exact_l2 < 1e-4 && protected_l2 < 1e-4 ? 0 : 1;
}
