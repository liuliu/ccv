// Replay original FP16 Q/K snapshots through the production centered quantizers.
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "nnc/mfa/ccv_nnc_mfa.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionDescriptor.hpp"
#include "nnc/mfa/kernels/NAInt8AttentionKernel.hpp"

int main(int argc, char** argv)
{
  if (argc != 8) {
    fprintf(stderr, "Usage: %s capture.bin R C Hq Hk D output_prefix\n", argv[0]);
    return 2;
  }
  std::array<uint32_t, 5> sizes{};
  for (int i = 0; i < 5; ++i) {
    char* end = nullptr;
    errno = 0;
    const unsigned long value = strtoul(argv[i + 2], &end, 10);
    if (errno || argv[i + 2][0] == '-' || !*argv[i + 2] || *end || !value || value > 65535) return 2;
    sizes[i] = value;
  }
  const uint32_t R = sizes[0], C = sizes[1], Hq = sizes[2], Hk = sizes[3], D = sizes[4];
  if (Hq % Hk || D < 8 || D > 256 || D % 8) return 2;
  auto pool = NS::TransferPtr(NS::AutoreleasePool::alloc()->init());
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device) return 2;
  auto context = ccv_nnc_init_mfa_context(device.get());
  if (!ccv_nnc_mfa_has_neural_accelerators(context)) return 2;
  const size_t q_count = size_t(R) * Hq * D, k_count = size_t(C) * Hk * D;
  const uint32_t q_tiles = (R + 15) / 16, k_tiles = (C + 63) / 64;
  const size_t bytes[] = {q_count * 2, k_count * 2, q_count, k_count,
      size_t(Hq) * q_tiles * 4, size_t(Hk) * k_tiles * 4, size_t(Hk) * D * 4, size_t(Hk) * D * 4};
  std::array<NS::SharedPtr<MTL::Buffer>, 9> buffers;
  for (int i = 0; i < 8; ++i) {
    buffers[i] = NS::TransferPtr(device->newBuffer(bytes[i], MTL::ResourceStorageModeShared));
    if (!buffers[i]) return 2;
  }
  FILE* input = fopen(argv[1], "rb");
  if (!input) return 2;
  const bool read_ok = fread(buffers[0]->contents(), 1, bytes[0], input) == bytes[0] &&
      fread(buffers[1]->contents(), 1, bytes[1], input) == bytes[1];
  fclose(input);
  if (!read_ok) return 2;
  NAInt8AttentionDescriptor descriptor;
  descriptor.matrixDimensions = simd::uint3{R, C, D};
  descriptor.Hq = Hq; descriptor.Hk = Hk;
  descriptor.scale = 1.0f / sqrtf(D);
  descriptor.qkHadamard = true;
  descriptor.loadR = descriptor.loadC = true;
  auto p = context->kernel_cache.findKernel<NAInt8AttentionKernel, NAInt8AttentionDescriptor,
      NAInt8AttentionKernelDescriptor>(descriptor, device.get(), context->device_properties);
  // Per-chunk mean sums for sequences longer than one reduction chunk.
  buffers[8] = NS::TransferPtr(device->newBuffer(std::max<size_t>(16, p->kernel->vMeanPartialBytes(1, C)), MTL::ResourceStorageModeShared));
  if (!buffers[8]) return 2;
  const uint32_t dimensions[] = {R, C, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto queue = NS::TransferPtr(device->newCommandQueue());
  auto cb = queue->commandBuffer();
  auto dispatch = [&](MTL::ComputePipelineState* pipeline, MTL::Size grid, MTL::Size threads,
                      std::initializer_list<int> bindings) {
    auto encoder = cb->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline);
    encoder->setBytes(dimensions, sizeof(dimensions), 21);
    int index = 0;
    for (int buffer : bindings) encoder->setBuffer(buffers[buffer].get(), 0, index++);
    encoder->dispatchThreadgroups(grid, threads);
    encoder->endEncoding();
  };
  {
    // K stands in for V; only the K mean (buffer 6) is used.
    auto encoder = cb->computeCommandEncoder();
    encoder->setBytes(dimensions, sizeof(dimensions), 21);
    int index = 0;
    for (int buffer : {1, 7, 1, 6}) encoder->setBuffer(buffers[buffer].get(), 0, index++);
    p->kernel->encodeVMean(encoder, p->fifth.get(), p->seventh.get(), buffers[8].get(), 0, 1, C);
    encoder->endEncoding();
  }
  dispatch(p->second.get(), MTL::Size(q_tiles, Hq, 1), MTL::Size(128, 1, 1), {0, 2, 4, 6});
  dispatch(p->third.get(), MTL::Size(k_tiles, Hk, 1), MTL::Size(256, 1, 1), {1, 3, 5, 6});
  cb->commit(); cb->waitUntilCompleted();
  if (cb->status() != MTL::CommandBufferStatusCompleted) return 3;
  const char* names[] = {"had_q", "had_k", "had_q_scale", "had_k_scale", "k_mean"};
  for (int i = 0; i < 5; ++i) {
    const std::string path = std::string(argv[7]) + "-" + names[i] + ".bin";
    FILE* output = fopen(path.c_str(), "wb");
    if (!output) return 2;
    const bool write_ok = fwrite(buffers[i + 2]->contents(), 1, bytes[i + 2], output) == bytes[i + 2];
    fclose(output);
    if (!write_ok) return 2;
  }
  printf("replayed R=%u C=%u Hq=%u Hk=%u D=%u\n", R, C, Hq, Hk, D);
  ccv_nnc_deinit_mfa_context(context);
  return 0;
}
