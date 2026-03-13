#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <string>

#include "nnc/mfa/kernels/DeviceProperties.hpp"
#include "nnc/mfa/kernels/GEMMDescriptor.hpp"
#include "nnc/mfa/kernels/GEMMKernel.hpp"
#include "nnc/mfa/kernels/GEMMKernelDescriptor.hpp"
#include "nnc/mfa/kernels/ShaderCache.hpp"

namespace {

struct Conv3DShape {
  uint32_t batch;
  uint32_t input_d;
  uint32_t input_h;
  uint32_t input_w;
  uint32_t input_c;
  uint32_t output_c;
  uint32_t kernel_d;
  uint32_t kernel_h;
  uint32_t kernel_w;
};

GEMMDescriptor make_descriptor(uint32_t m, uint32_t n, uint32_t k)
{
  GEMMDescriptor descriptor;
  descriptor.matrixDimensions = simd::uint3{m, n, k};
  descriptor.memoryPrecisions = {
      .A = GEMMOperandPrecision::FP16,
      .B = GEMMOperandPrecision::FP16,
      .C = GEMMOperandPrecision::FP16,
      .bias = GEMMOperandPrecision::FP16,
  };
  descriptor.registerPrecisionC = std::nullopt;
  descriptor.leadingDimensions = std::nullopt;
  descriptor.batchStrides = std::nullopt;
  descriptor.transposeState = simd::uchar3{0, 1, 0};
  descriptor.loadPreviousC = false;
  descriptor.useBias = false;
  descriptor.loadM = false;
  descriptor.supportIndirectCommandBuffers = false;
  return descriptor;
}

const char* bool_string(bool value)
{
  return value ? "true" : "false";
}

} // namespace

int main()
{
  auto* pool = NS::AutoreleasePool::alloc()->init();
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  if (!device)
  {
    std::fprintf(stderr, "Metal device unavailable.\n");
    std::_Exit(1);
  }

  const Conv3DShape shape{
      .batch = 1,
      .input_d = 4,
      .input_h = 6,
      .input_w = 10,
      .input_c = 4,
      .output_c = 64,
      .kernel_d = 3,
      .kernel_h = 3,
      .kernel_w = 3,
  };
  const uint32_t output_d = shape.input_d - shape.kernel_d + 1;
  const uint32_t output_h = shape.input_h - shape.kernel_h + 1;
  const uint32_t output_w = shape.input_w - shape.kernel_w + 1;
  const uint32_t gemm_m = shape.batch * output_d * output_h * output_w;
  const uint32_t gemm_n = shape.output_c;
  const uint32_t gemm_k = shape.kernel_d * shape.kernel_h * shape.kernel_w * shape.input_c;

  DeviceProperties dprops{};
  ShaderCache shader_cache;
  auto pipeline_value =
      shader_cache.findKernel<GEMMKernel, GEMMDescriptor, GEMMKernelDescriptor>(
          make_descriptor(gemm_m, gemm_n, gemm_k), device.get(), dprops);
  auto* kernel = pipeline_value->kernel;

  const std::string dump_path = "gemm_kernel_dump.metal";
  std::ofstream dump(dump_path);
  dump << kernel->source;
  dump.close();

  std::printf("device: %s\n", device->name()->utf8String());
  std::printf(
      "conv3d->gemm: output=%ux%ux%u M=%u N=%u K=%u transposeB=true\n",
      output_d,
      output_h,
      output_w,
      gemm_m,
      gemm_n,
      gemm_k);
  std::printf(
      "gemm kernel: block=%u x %u x %u leading_block=%u x %u x %u splits=%u x %u "
      "register=%u x %u threadgroup_size=%u tgm_bytes=%u async_load=%s async_store=%s "
      "disable_async_copy=%s apple9_or_later=%s\n",
      static_cast<unsigned>(kernel->blockDimensions[0]),
      static_cast<unsigned>(kernel->blockDimensions[1]),
      static_cast<unsigned>(kernel->blockDimensions[2]),
      static_cast<unsigned>(kernel->leadingBlockDimensions[0]),
      static_cast<unsigned>(kernel->leadingBlockDimensions[1]),
      static_cast<unsigned>(kernel->leadingBlockDimensions[2]),
      static_cast<unsigned>(kernel->splits[0]),
      static_cast<unsigned>(kernel->splits[1]),
      static_cast<unsigned>(kernel->registerM),
      static_cast<unsigned>(kernel->registerN),
      static_cast<unsigned>(kernel->threadgroupSize),
      static_cast<unsigned>(kernel->threadgroupMemoryAllocation),
      bool_string(kernel->preferAsyncLoad),
      bool_string(kernel->preferAsyncStore),
      bool_string(kernel->disableAsyncCopy),
      bool_string(device->supportsFamily(MTL::GPUFamily(1009))));
  std::printf("kernel source dump: %s\n", dump_path.c_str());
  std::printf(
      "source anchors: multiply_accumulate=%zu createLoadC/direct-store path searchable in dump\n",
      kernel->source.find("METAL_FUNC void multiply_accumulate("));
  std::fflush(stdout);
  std::fflush(stderr);
  pool->drain();
  std::_Exit(0);
}
