#include "NAInt8SolAttentionDescriptor.hpp"
#include "NAInt8SolAttentionKernel.hpp"
#include "../ccv_nnc_mfa_error.hpp"

bool NAInt8SolAttentionDescriptor::operator==(const NAInt8SolAttentionDescriptor& rhs) const
{
  return stage == rhs.stage && blockSize == rhs.blockSize && N == rhs.N && T == rhs.T && H == rhs.H && queryBlockSize == rhs.queryBlockSize;
}

size_t std::hash<NAInt8SolAttentionDescriptor>::operator()(const NAInt8SolAttentionDescriptor& d) const noexcept
{
  size_t hash = uint32_t(d.stage);
  for (uint32_t dimension : { d.blockSize, d.N, d.T, d.H, d.queryBlockSize })
    hash = hash * 31 + dimension;
  return hash;
}

std::pair<NAInt8SolAttentionKernelDescriptor, PipelineValue<NAInt8SolAttentionKernel>*> NAInt8SolAttentionDescriptor::findKernel(MTL::Device* const device, const DeviceProperties&, NS::Array*, MTL::BinaryArchive*, const std::string&, std::unordered_map<NAInt8SolAttentionKernelDescriptor, std::unique_ptr<NAInt8SolAttentionKernel>>* cache) const noexcept
{
  const NAInt8SolAttentionKernelDescriptor key = { blockSize };
  auto& kernel = (*cache)[key];
  if (!kernel)
    kernel = std::make_unique<NAInt8SolAttentionKernel>(key, device);
  const char* names[] = { "sol_v_mean", "sol_quantize", "sol_pool", "sol_prepare_summaries", "sol_route", "sol_attention" };
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  const uint32_t dimensions[] = { N, T, H, queryBlockSize };
  for (NS::UInteger i = 0; i < 4; ++i)
    constants->setConstantValue(&dimensions[i], MTL::DataTypeUInt, i);
  NS::Error* error = nullptr;
  auto function = NS::TransferPtr(kernel->library->newFunction(NS::String::string(names[uint32_t(stage)], NS::UTF8StringEncoding), constants.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  auto pipeline = NS::TransferPtr(device->newComputePipelineState(function.get(), &error));
  CCV_NNC_MFA_CHECK_ERROR(error);
  CCV_NNC_MFA_PRECONDITION(pipeline->staticThreadgroupMemoryLength() <= device->maxThreadgroupMemoryLength());
  CCV_NNC_MFA_PRECONDITION(kernel->threadgroupSize(*this) <= pipeline->maxTotalThreadsPerThreadgroup());
  return { key, new PipelineValue<NAInt8SolAttentionKernel>{ kernel.get(), pipeline } };
}
