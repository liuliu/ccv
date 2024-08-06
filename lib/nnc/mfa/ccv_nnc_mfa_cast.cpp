#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_cast(mfa::context* context, ccv_nnc_mfa_cast_params_t params)
{
  context->cast_cache.prepare(context, mfa::cast::hash(params));
}

void ccv_nnc_mfa_encode_cast(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_cast_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  mfa::cast::hash hash(params);
  auto iterator = context->cast_cache.map.find(hash);
  if (iterator == context->cast_cache.map.end()) {
    mfa::precondition_failure("cast hash not cached.", __LINE__, __FILE__, __FUNCTION__);
  }
  
  auto* pipeline = iterator->second;
  auto encoder = command_batch->startCommand();
  
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 2);
  
  encoder->setComputePipelineState(pipeline->cast_pso.get());
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageWrite);

  auto grid_size = pipeline->grid_size;
  CCV_NNC_MFA_PRECONDITION(grid_size.depth > 0);
  encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
  command_batch->finishCommand(encoder);
}

// MARK: - C++

mfa::cast::hash::hash(ccv_nnc_mfa_cast_params_t params) {
  original_data_type = params.original_data_type;
  data_type = params.data_type;
  length = params.length;
}

bool mfa::cast::hash::operator==(const mfa::cast::hash& hash) const {
  return
  (original_data_type == hash.original_data_type) &&
  (data_type == hash.data_type) &&
  (length == hash.length);
}

std::ostream& operator<<(std::ostream& os, const mfa::cast::hash& hash) {
  os << "mfa::cast::hash {";
  os << " .original_data_type = " << hash.original_data_type << ',';
  os << " .data_type = " << hash.data_type << ',';
  os << " .length = " << hash.length << " ";
  os << "}";
  return os;
}

std::size_t std::hash<mfa::cast::hash>::operator()(const mfa::cast::hash& hash) const noexcept {
  std::size_t seed = 0;
  using namespace mfa::hash;
  combine_64(seed, hash.original_data_type);
  combine_64(seed, hash.data_type);
  combine_32(seed, hash.length);
  return seed;
}

mfa::cast::pipeline::pipeline(mfa::context* context, mfa::cast::hash hash) {
  CCV_NNC_MFA_PRECONDITION((hash.original_data_type == MTL::DataTypeFloat) || (hash.original_data_type == MTL::DataTypeHalf))
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf))
  
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  std::string shader;
  // In this case, we can igore the boundary check.
  if (hash.length % (4 * 256) == 0) {
    shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void cast(
  device original_real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  destination[idx] = (real4)(src[idx]);
}
    )";
  } else if (hash.length % 4 == 0) {
    shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void cast(
  device original_real4 *src [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = (real4)(src[idx]);
}
    )";
  } else {
    shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void cast(
  device original_real *src [[buffer(0)]],
  device real *destination [[buffer(1)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= count)
    return;
  destination[idx] = (real)(src[idx]);
}
    )";
  }

  std::string defines = "";
  if (hash.data_type == MTL::DataTypeFloat) {
    defines += std::string("typedef float real;");
    defines += "\n";
    defines += std::string("typedef float4 real4;");
    defines += "\n";
  } else {
    defines += std::string("typedef half real;");
    defines += "\n";
    defines += std::string("typedef half4 real4;");
    defines += "\n";
  }

  if (hash.original_data_type == MTL::DataTypeFloat) {
    defines += std::string("typedef float original_real;");
    defines += "\n";
    defines += std::string("typedef float4 original_real4;");
    defines += "\n";
  } else {
    defines += std::string("typedef half original_real;");
    defines += "\n";
    defines += std::string("typedef half4 original_real4;");
    defines += "\n";
  }

  unsigned int count;
  if (hash.length % 4 == 0) {
    count = hash.length / 4;
  } else {
    count = hash.length;
  }
  if (hash.length % (4 * 256) != 0) {
    defines += "constant uint count = ";
    defines += std::to_string(count) + ";";
    defines += "\n";
  }
  this->group_size = MTL::Size(256, 1, 1);
  const int num_blocks = (count + 255) / 256;
  this->grid_size = MTL::Size(num_blocks, 1, 1);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  NS::SharedPtr<MTL::ComputePipelineState>* pso = &cast_pso;

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
    
  auto swift_name = NS::String::string("cast", NS::UTF8StringEncoding);
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
