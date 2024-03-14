#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_add(mfa::context* context, ccv_nnc_mfa_add_params_t params)
{
  context->add_cache.prepare(context, mfa::add::hash(params));
}

void ccv_nnc_mfa_encode_add(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_add_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  mfa::add::hash hash(params);
  auto iterator = context->add_cache.map.find(hash);
  if (iterator == context->add_cache.map.end()) {
    mfa::precondition_failure("add hash not cached.", __LINE__, __FILE__, __FUNCTION__);
  }
  
  auto* pipeline = iterator->second;
  auto encoder = command_batch->startCommand();
  
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 3);
  
  encoder->setComputePipelineState(pipeline->add_pso.get());
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[2], MTL::ResourceUsageWrite);

  auto grid_size = pipeline->grid_size;
  CCV_NNC_MFA_PRECONDITION(grid_size.depth > 0);
  encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
  command_batch->finishCommand(encoder);
}

// MARK: - C++

mfa::add::hash::hash(ccv_nnc_mfa_add_params_t params) {
  data_type = params.data_type;
  length = params.length;
}

bool mfa::add::hash::operator==(const mfa::add::hash& hash) const {
  return (data_type == hash.data_type) && (length == hash.length);
}

std::ostream& operator<<(std::ostream& os, const mfa::add::hash& hash) {
  os << "mfa::add::hash {";
  os << " .data_type = " << hash.data_type << ',';
  os << " .length = " << hash.length << " ";
  os << "}";
  return os;
}

std::size_t std::hash<mfa::add::hash>::operator()(const mfa::add::hash& hash) const noexcept {
  std::size_t seed = 0;
  using namespace mfa::hash;
  combine_64(seed, hash.data_type);
  combine_32(seed, hash.length);
  return seed;
}

mfa::add::pipeline::pipeline(mfa::context* context, mfa::add::hash hash) {
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf))
  
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  std::string shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void add(
  device const real4 *src0 [[buffer(0)]],
  device const real4 *src1 [[buffer(1)]],
  device real4 *dst [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  dst[idx] = src0[idx] + src1[idx];
}
    )";

  std::string defines = "";
  if (hash.data_type == MTL::DataTypeFloat) {
    defines += std::string("typedef float4 real4;");
    defines += "\n";
  } else {
    defines += std::string("typedef half4 real4;");
    defines += "\n";
  }

  defines += "constant uint count = ";
  CCV_NNC_MFA_PRECONDITION(hash.length % 4 == 0)
  const unsigned int count = hash.length / 4;
  defines += std::to_string(count) + ";";
  defines += "\n";
  this->group_size = MTL::Size(256, 1, 1);
  const int num_blocks = (count + 255) / 256;
  this->grid_size = MTL::Size(num_blocks, 1, 1);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  NS::SharedPtr<MTL::ComputePipelineState>* pso = &add_pso;

  std::string source = defines;
  if (METAL_LOG_LEVEL(context) >= 4) {
    std::cerr << source << std::endl;
  }
  source += shader;

  NS::Error *error = nullptr;
  auto swift_source = NS::String::string(source.c_str(),
  NS::UTF8StringEncoding);
  auto library = NS::TransferPtr(context->device->newLibrary(swift_source, nullptr, &error));
  if (!library) {
    CCV_NNC_MFA_CHECK_ERROR(error)
  }
    
  auto swift_name = NS::String::string("add", NS::UTF8StringEncoding);
  auto function = NS::TransferPtr(library->newFunction(swift_name, constants.get(), &error));
  if (!function) {
    CCV_NNC_MFA_CHECK_ERROR(error)
  }
    
  *pso = NS::TransferPtr(context->device->newComputePipelineState(function.get(), &error));
  if (!*pso) {
    CCV_NNC_MFA_CHECK_ERROR(error)
  }
  
  pool->drain();
}
