#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>
#include "nnc/mfa/ccv_nnc_mfa.hpp"

void encode_sol_separate(ccv_nnc_mfa_context_t*, ccv_nnc_mfa_sol_attention_params_t, mtl_command_batch_t*, mtl_buffer_t**, size_t*);
void encode_sol_baseline(ccv_nnc_mfa_context_t*, ccv_nnc_mfa_sol_attention_params_t, mtl_command_batch_t*, mtl_buffer_t**, size_t*);
void encode_sol_experiment(ccv_nnc_mfa_context_t*, ccv_nnc_mfa_sol_attention_params_t, mtl_command_batch_t*, mtl_buffer_t**, size_t*, int);

static double median(std::vector<double> values)
{
  std::sort(values.begin(), values.end());
  return (values[(values.size() - 1) / 2] + values[values.size() / 2]) * 0.5;
}

int main(int argc, char** argv)
{
  setbuf(stdout, nullptr);
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  auto queue = NS::TransferPtr(device->newCommandQueue());
  const int fusion = getenv("SOL_FUSION") ? atoi(getenv("SOL_FUSION")) : 0;
  const int variants = fusion == 3 ? 3 : 2;
  std::array<ccv_nnc_mfa_context_t*, 3> contexts{};
  for (int i = 0; i < variants; ++i) contexts[i] = ccv_nnc_init_mfa_context(device.get());
  printf("COMPARE device=%s fusion=%d (0: 8 vs 7 launches; 1: 7 vs 5; 2: 7 vs 4; 3: 7 vs 5 vs 4; 4: prototype 5 vs production 5)\n", device->name()->utf8String(), fusion);
  const bool record = getenv("SOL_RECORD"), replay = getenv("SOL_REPLAY");
  const char* fixtures = getenv("SOL_FIXTURES");
  if ((record || replay) && (!fixtures || (record && replay))) return 2;
  const bool verify = argc < 2 || strcmp(argv[1], "--verify") == 0;
  const int rounds = verify ? (getenv("SOL_REPEATS") ? atoi(getenv("SOL_REPEATS")) : 1) : (argc > 3 ? atoi(argv[3]) : 6);
  if (rounds <= 0 || fusion < 0 || fusion > 4) return 2;
  using Shape = std::array<uint32_t, 5>;
  const std::vector<Shape> shapes = getenv("SOL_LONG_TAIL") ? std::vector<Shape>{{1,32769,2,64,64}} : verify ? std::vector<Shape>{
    {2,257,3,64,64}, {1,197,3,32,16}, {1,133,2,16,16}, {1,20501,2,64,64},
    {1,32769,2,64,64}, {2,193,3,64,32}, {1,129,3,64,16}, {1,20480,2,64,64}
  } : std::vector<Shape>{{1, uint32_t(atoi(argv[1])), argc > 2 ? uint32_t(atoi(argv[2])) : 56, 64, 64}};
  for (const auto& shape : shapes) {
    ccv_nnc_mfa_sol_attention_params_t p = {shape[0], shape[1], shape[2], shape[3],
      verify ? 17 : std::min(470u, shape[1]), shape[1] - (verify ? 7 : 0), 1, 0.5, shape[4], 1};
    if (getenv("SOL_SCALE")) p.scale = atof(getenv("SOL_SCALE"));
    const size_t inputOffset = getenv("SOL_UNALIGNED_INPUT") ? 258 : 256;
    const size_t count = size_t(p.N) * p.T * p.H * 128, bytes = count * 2;
    std::array<NS::SharedPtr<MTL::Buffer>, 6> buffers;
    std::mt19937 rng(getenv("SOL_SEED") ? atoi(getenv("SOL_SEED")) : 42);
    std::uniform_real_distribution<float> random(-1, 1);
    for (int i = 0; i < 3 + variants; ++i) {
      buffers[i] = NS::TransferPtr(device->newBuffer(bytes + 512, MTL::ResourceStorageModeShared));
      if (!buffers[i]) return 2;
      memset(buffers[i]->contents(), 0xa5, bytes + 512);
      if (i < 3) {
        const float range = i == 0 ? (getenv("SOL_ZERO_Q") ? 0 : 0.3f) : (getenv("SOL_LARGE_V") && i == 2 ? 65504.f : 3.f);
        auto values = (_Float16*)((char*)buffers[i]->contents() + inputOffset);
        for (size_t j = 0; j < count; ++j) values[j] = random(rng) * range;
      }
    }
    for (int dense = 0; dense < 2; ++dense) {
      p.local_block_radius = dense ? (p.T + p.block_size - 1) / p.block_size : 1;
      printf("SHAPE N=%u T=%u H=%u B=%u Q=%u dense=%d rounds=%d begin=%u end=%u scale=%g\n", p.N, p.T, p.H, p.block_size, p.query_block_size, dense, rounds, p.approximation_start, p.approximation_end, p.scale);
      char name[256];
      snprintf(name, sizeof(name), "/%u-%u-%u-%u-%u-%d-%g-%d-%d.bin", p.N, p.T, p.H, p.block_size, p.query_block_size, dense, p.scale, bool(getenv("SOL_ZERO_Q")), bool(getenv("SOL_LARGE_V")));
      const std::string fixture = fixtures ? std::string(fixtures) + name : "";
      if (replay) {
        FILE* file = fopen(fixture.c_str(), "rb");
        if (!file || fread((char*)buffers[3]->contents() + 256, 1, bytes, file) != bytes) return 2;
        fclose(file);
      }
      std::array<std::vector<double>, 3> times, ratios;
      for (int round = verify ? 0 : -3; round < rounds; ++round) {
        std::array<double, 3> t{};
        for (int step = 0; step < variants; ++step) {
          const int variant = (step + round + 6) % variants;
          if ((record && variant > 0) || (replay && variant == 0)) {
            if (round >= 0) times[variant].push_back(0);
            continue;
          }
          auto innerPool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
          auto cb = queue->commandBuffer();
          auto batch = ccv_nnc_start_command_batch_from_command_buffer(cb, 0);
          MTL::Buffer* tensors[] = {buffers[0].get(), buffers[1].get(), buffers[2].get(), buffers[3 + variant].get()};
          size_t offsets[] = {inputOffset, inputOffset, inputOffset, 256};
          if (fusion == 4 && variant == 0)
            encode_sol_experiment(contexts[variant], p, batch, tensors, offsets, 1);
          else if (fusion == 4)
            ccv_nnc_mfa_encode_sol_attention(contexts[variant], p, batch, tensors, offsets);
          else if (fusion && variant)
            encode_sol_experiment(contexts[variant], p, batch, tensors, offsets, fusion == 3 ? variant : fusion);
          else if (!fusion && !variant)
            encode_sol_baseline(contexts[variant], p, batch, tensors, offsets);
          else
            encode_sol_separate(contexts[variant], p, batch, tensors, offsets);
          ccv_nnc_finish_command_batch(batch);
          cb->commit(); cb->waitUntilCompleted();
          if (cb->status() == MTL::CommandBufferStatusError) {
            fprintf(stderr, "%s\n", cb->error()->localizedDescription()->utf8String()); return 1;
          }
          t[variant] = (cb->GPUEndTime() - cb->GPUStartTime()) * 1000;
          if (round >= 0) times[variant].push_back(t[variant]);
          // Check every execution, including warmups, to catch intermittent failures.
          const auto output = (const _Float16*)((char*)buffers[3 + variant]->contents() + 256);
          for (size_t i = 0; i < count; ++i)
            if (!std::isfinite(float(output[i]))) {
              fprintf(stderr, "NONFINITE round=%d variant=%d i=%zu value=%g\n", round, variant, i, float(output[i])); return 1;
            }
        }
        if (!record && !replay && fusion == 4) {
          const auto expected = (const _Float16*)((char*)buffers[3]->contents() + 256);
          const auto actual = (const _Float16*)((char*)buffers[4]->contents() + 256);
          double error2 = 0, norm2 = 0, maxRelative = 0;
          for (size_t i = 0; i < count; ++i) {
            const double x = expected[i], y = actual[i], e = x - y;
            error2 += e * e; norm2 += x * x;
            maxRelative = std::max(maxRelative, fabs(e) / std::max(1.0, std::max(fabs(x), fabs(y))));
          }
          if (sqrt(error2 / std::max(norm2, 1e-30)) > 1e-5 || maxRelative > 0.002) return 1;
        }
        printf("round=%d baseline_ms=%.5f", round, t[0]);
        for (int variant = 1; variant < variants; ++variant) {
          const double speedup = t[variant] > 0 ? t[0] / t[variant] : 0;
          printf(" variant%d_ms=%.5f speedup%d=%.5f", variant, t[variant], variant, speedup);
          if (round >= 0) ratios[variant].push_back(speedup);
        }
        printf("\n");
      }
      if (record) {
        FILE* file = fopen(fixture.c_str(), "wb");
        if (!file || fwrite((char*)buffers[3]->contents() + 256, 1, bytes, file) != bytes) return 2;
        fclose(file);
        printf("RECORDED %s\n", fixture.c_str());
        continue;
      }
      const auto expected = (const _Float16*)((char*)buffers[3]->contents() + 256);
      for (int variant = 1; variant < variants; ++variant) {
        const auto actual = (const _Float16*)((char*)buffers[3 + variant]->contents() + 256);
        size_t differences = 0;
        double error2 = 0, norm2 = 0, maxError = 0, maxRelative = 0;
        for (size_t i = 0; i < count; ++i) {
          const float x = expected[i], y = actual[i];
          if (!std::isfinite(x) || !std::isfinite(y)) {
            fprintf(stderr, "NONFINITE variant=%d i=%zu baseline=%g actual=%g\n", variant, i, x, y); return 1;
          }
          differences += memcmp(expected + i, actual + i, 2) != 0;
          const double e = double(x) - y;
          error2 += e * e; norm2 += double(x) * x;
          maxError = std::max(maxError, fabs(e));
          maxRelative = std::max(maxRelative, fabs(e) / std::max(1.0, std::max(fabs(double(x)), fabs(double(y)))));
        }
        const double relativeL2 = sqrt(error2 / std::max(norm2, 1e-30));
        printf("RESULT variant=%d baseline_ms=%.5f new_ms=%.5f paired_speedup=%.5f differences=%zu relative_l2=%.9g max_abs=%.9g baseline_scratch=%zu new_scratch=%zu\n", variant, median(times[0]), median(times[variant]), median(ratios[variant]), differences, relativeL2, maxError, size_t(contexts[0]->scratch->length()), size_t(contexts[variant]->scratch->length()));
        if ((!fusion && differences) || relativeL2 > 1e-5 || maxRelative > 0.002) return 1;
      }
      for (int j = 3; j < 3 + variants; ++j)
        for (size_t i = 0; i < 256; ++i)
          if (((uint8_t*)buffers[j]->contents())[i] != 0xa5 || ((uint8_t*)buffers[j]->contents())[256 + bytes + i] != 0xa5) return 1;
    }
  }
  for (int i = 0; i < variants; ++i) ccv_nnc_deinit_mfa_context(contexts[i]);
}
