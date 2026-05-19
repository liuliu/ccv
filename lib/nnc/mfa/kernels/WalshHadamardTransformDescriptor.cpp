#include "WalshHadamardTransformDescriptor.hpp"
#include "WalshHadamardTransformKernel.hpp"
#include "../ccv_nnc_mfa_hash.hpp"
#include "../ccv_nnc_mfa_error.hpp"

#include <algorithm>

static inline uint32_t _ccv_nnc_mfa_ilog2(const uint32_t x)
{
  return 31 - __builtin_clz(x);
}

static inline uint32_t _ccv_nnc_mfa_wht_rows_per_threadgroup(MTL::Device* const device, const uint32_t strategy, const uint32_t dim)
{
  if (strategy != 2)
    return 1;
  const uint32_t maxThreads = (uint32_t)device->maxThreadsPerThreadgroup().width;
  return std::max<uint32_t>(1, std::min<uint32_t>(8, maxThreads / dim));
}

bool WalshHadamardTransformDescriptor::operator==(const WalshHadamardTransformDescriptor& rhs) const {
  return
  memoryPrecision == rhs.memoryPrecision &&
  rowCount == rhs.rowCount &&
  dim == rhs.dim &&
  scale == rhs.scale;
}

std::size_t std::hash<WalshHadamardTransformDescriptor>::operator()(const WalshHadamardTransformDescriptor& hash) const noexcept {
  using namespace ccv::nnc::mfa::hash;
  std::size_t seed = 0;
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.memoryPrecision.value, hash.rowCount }));
  combine_64(seed, pack_64(simd::uint2 { hash.dim, *reinterpret_cast<const uint32_t*>(&hash.scale) }));
  return seed;
}

std::pair<WalshHadamardTransformKernelDescriptor, PipelineValue<WalshHadamardTransformKernel>*> WalshHadamardTransformDescriptor::findKernel(MTL::Device* const device, const DeviceProperties&, NS::Array* const, MTL::BinaryArchive* const, const std::string&, std::unordered_map<WalshHadamardTransformKernelDescriptor, std::unique_ptr<WalshHadamardTransformKernel>>* const libraryCache) const noexcept {
  auto createKernel =
  [=](WalshHadamardTransformKernelDescriptor descriptor) -> WalshHadamardTransformKernel* {
    auto iterator = libraryCache->find(descriptor);
    if (iterator != libraryCache->end()) {
      return iterator->second.get();
    } else {
      WalshHadamardTransformKernel* kernel = new WalshHadamardTransformKernel(descriptor, device);
      (*libraryCache)[descriptor] = std::unique_ptr<WalshHadamardTransformKernel>(kernel);
      return kernel;
    }
  };

  WalshHadamardTransformKernelDescriptor kernelDesc;
  kernelDesc.memoryPrecision = memoryPrecision;

  auto createPipeline =
  [=](MTL::Library* library) -> MTL::ComputePipelineState* {
    auto constants = NS::TransferPtr
    (MTL::FunctionConstantValues::alloc()->init());
    uint32_t maxRadix = 1;
    uint32_t numSteps = 0;
    uint32_t finalRadix = 1;
    if (dim > 1) {
      maxRadix = std::min<uint32_t>(dim, 16);
      const uint32_t logDim = _ccv_nnc_mfa_ilog2(dim);
      const uint32_t logRadix = _ccv_nnc_mfa_ilog2(maxRadix);
      numSteps = logDim / logRadix;
      finalRadix = 1u << (logDim % logRadix);
    }
    const uint32_t strategy = (dim >= 32 && dim <= 128) ? 2 : ((dim <= 128) ? 1 : 0);
    const uint32_t rowsPerThreadgroup = _ccv_nnc_mfa_wht_rows_per_threadgroup(device, strategy, dim);
    const uint32_t numThreads = (strategy == 2) ? dim * rowsPerThreadgroup : ((strategy == 1) ? dim : std::max<uint32_t>(dim / maxRadix, 1));
    constants->setConstantValue(&dim, MTL::DataTypeUInt, NS::UInteger(0));
    constants->setConstantValue(&maxRadix, MTL::DataTypeUInt, NS::UInteger(1));
    constants->setConstantValue(&numThreads, MTL::DataTypeUInt, NS::UInteger(2));
    constants->setConstantValue(&numSteps, MTL::DataTypeUInt, NS::UInteger(3));
    constants->setConstantValue(&finalRadix, MTL::DataTypeUInt, NS::UInteger(4));
    constants->setConstantValue(&scale, MTL::DataTypeFloat, NS::UInteger(5));
    constants->setConstantValue(&rowCount, MTL::DataTypeUInt, NS::UInteger(6));
    constants->setConstantValue(&strategy, MTL::DataTypeUInt, NS::UInteger(7));
    constants->setConstantValue(&rowsPerThreadgroup, MTL::DataTypeUInt, NS::UInteger(8));

    NS::String* swiftName = NS::String::string("walsh_hadamard_transform", NS::UTF8StringEncoding);
    NS::Error* error = nil;

    auto function = NS::TransferPtr
    (library->newFunction(swiftName, constants.get(), &error));
    CCV_NNC_MFA_CHECK_ERROR(error);

    auto pipeline = device->newComputePipelineState(function.get(), &error);
    CCV_NNC_MFA_CHECK_ERROR(error);
    return pipeline;
  };
  WalshHadamardTransformKernel* kernel = createKernel(kernelDesc);
  auto pipeline = NS::TransferPtr(createPipeline(kernel->library.get()));

  PipelineValue<WalshHadamardTransformKernel>* output = new PipelineValue<WalshHadamardTransformKernel> { kernel, pipeline };
  return std::make_pair(kernelDesc, output);
}
