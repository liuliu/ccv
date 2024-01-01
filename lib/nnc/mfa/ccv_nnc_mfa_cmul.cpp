#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_cmul(mfa::context* context, ccv_nnc_mfa_cmul_params_t params)
{
  context->cmul_cache.prepare(context, mfa::cmul::hash(params));
}

void ccv_nnc_mfa_encode_cmul(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_cmul_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  mfa::cmul::hash hash(params);
  auto iterator = context->cmul_cache.map.find(hash);
  if (iterator == context->cmul_cache.map.end()) {
    mfa::precondition_failure("cmul hash not cached.", __LINE__, __FILE__, __FUNCTION__);
  }
  
  auto* pipeline = iterator->second;
  auto encoder = command_batch->startCommand();
  
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 3);
  
  encoder->setComputePipelineState(pipeline->cmul_pso.get());
  if (tensors[0] == tensors[2])
  {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  } else if (tensors[1] == tensors[2]) {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[0], MTL::ResourceUsageRead);
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
  }

  auto grid_size = pipeline->grid_size;
  CCV_NNC_MFA_PRECONDITION(grid_size.depth > 0);
  encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
  command_batch->finishCommand(encoder);
}

// MARK: - C++

mfa::cmul::hash::hash(ccv_nnc_mfa_cmul_params_t params) {
  data_type = params.data_type;
  memcpy(astride, params.astride, sizeof(params.astride));
  memcpy(bstride, params.bstride, sizeof(params.bstride));
  memcpy(cstride, params.cstride, sizeof(params.cstride));
  memcpy(dim, params.dim, sizeof(params.dim));
}

bool mfa::cmul::hash::operator==(const mfa::cmul::hash& hash) const {
  return
  (data_type == hash.data_type) &&
  (astride[0] == hash.astride[0]) &&
  (astride[1] == hash.astride[1]) &&
  (astride[2] == hash.astride[2]) &&
  (bstride[0] == hash.bstride[0]) &&
  (bstride[1] == hash.bstride[1]) &&
  (bstride[2] == hash.bstride[2]) &&
  (cstride[0] == hash.cstride[0]) &&
  (cstride[1] == hash.cstride[1]) &&
  (cstride[2] == hash.cstride[2]) &&
  (dim[0] == hash.dim[0]) &&
  (dim[1] == hash.dim[1]) &&
  (dim[2] == hash.dim[2]) &&
  (dim[3] == hash.dim[3]);
}

std::ostream& operator<<(std::ostream& os, const mfa::cmul::hash& hash) {
  os << "mfa::cmul::hash {";
  os << " .data_type = " << hash.data_type << ',';
  os << " .astride[0] = " << hash.astride[0] << ',';
  os << " .astride[1] = " << hash.astride[1] << ',';
  os << " .astride[2] = " << hash.astride[2] << ',';
  os << " .bstride[0] = " << hash.bstride[0] << ',';
  os << " .bstride[1] = " << hash.bstride[1] << ',';
  os << " .bstride[2] = " << hash.bstride[2] << ',';
  os << " .cstride[0] = " << hash.cstride[0] << ',';
  os << " .cstride[1] = " << hash.cstride[1] << ',';
  os << " .cstride[2] = " << hash.cstride[2] << ',';
  os << " .dim[0] = " << hash.dim[0] << ',';
  os << " .dim[1] = " << hash.dim[1] << ',';
  os << " .dim[2] = " << hash.dim[2] << ',';
  os << " .dim[3] = " << hash.dim[3] << " ";
  os << "}";
  return os;
}

std::size_t std::hash<mfa::cmul::hash>::operator()(const mfa::cmul::hash& hash) const noexcept {
  std::size_t seed = 0;
  using namespace mfa::hash;
  combine_64(seed, hash.data_type);
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.astride[0], (unsigned int)hash.astride[1] }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.astride[2], (unsigned int)hash.bstride[0] }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.bstride[1], (unsigned int)hash.bstride[2] }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.cstride[0], (unsigned int)hash.cstride[1] }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.cstride[2], (unsigned int)hash.dim[0] }));
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.dim[1], (unsigned int)hash.dim[2] }));
  combine_32(seed, (unsigned int)hash.dim[3]);
  return seed;
}

mfa::cmul::pipeline::pipeline(mfa::context* context, mfa::cmul::hash hash) {
  // FlashNorm not supported for group cmul yet.
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf))
  
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  std::string shader;
  if (hash.dim[3] == 0 && hash.dim[2] == 0 && hash.dim[1] == 0) {
    shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void cmul(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint idx = tpig.x;
  if (idx >= dim0)
    return;
  const float a0 = (float)src0[idx * 2];
  const float a1 = (float)src0[idx * 2 + 1];
  const float b0 = (float)src1[idx * 2];
  const float b1 = (float)src1[idx * 2 + 1];
  destination[idx * 2] = (real)(a0 * b0 - a1 * b1);
  destination[idx * 2 + 1] = (real)(a0 * b1 + a1 * b0);
}
    )";
  } else if (hash.dim[3] == 0 && hash.dim[2] == 0) {
    shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void cmul(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint x = tpig.x;
  const uint y = tpig.y;
  if (y >= dim1 || x >= dim0)
    return;
  const uint ida = y * astride0 + x * 2;
  const uint idb = y * bstride0 + x * 2;
  const uint idc = y * cstride0 + x * 2;
  const float a0 = (float)src0[ida];
  const float a1 = (float)src0[ida + 1];
  const float b0 = (float)src1[idb];
  const float b1 = (float)src1[idb + 1];
  destination[idc] = (real)(a0 * b0 - a1 * b1);
  destination[idc + 1] = (real)(a0 * b1 + a1 * b0);
}
    )";
  } else if (hash.dim[3] == 0) {
    shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void cmul(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint x = tpig.x;
  const uint y = tpig.y;
  const uint z = tpig.z;
  if (y >= dim1 || x >= dim0)
    return;
  const uint ida = z * astride1 + y * astride0 + x * 2;
  const uint idb = z * bstride1 + y * bstride0 + x * 2;
  const uint idc = z * cstride1 + y * cstride0 + x * 2;
  const float a0 = (float)src0[ida];
  const float a1 = (float)src0[ida + 1];
  const float b0 = (float)src1[idb];
  const float b1 = (float)src1[idb + 1];
  destination[idc] = (real)(a0 * b0 - a1 * b1);
  destination[idc + 1] = (real)(a0 * b1 + a1 * b0);
}
    )";
  } else {
    shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void cmul(
  device real *src0 [[buffer(0)]],
  device real *src1 [[buffer(1)]],
  device real *destination [[buffer(2)]],

  uint3 tpig [[thread_position_in_grid]]
) {
  const uint x = tpig.x;
  const uint y = tpig.y;
  const uint z = tpig.z;
  if (y >= dim1 || x >= dim0)
    return;
  const int u = z % dim2;
  const int v = z / dim2;
  const uint ida = v * astride2 + u * astride1 + y * astride0 + x * 2;
  const uint idb = v * bstride2 + u * bstride1 + y * bstride0 + x * 2;
  const uint idc = v * cstride2 + u * cstride1 + y * cstride0 + x * 2;
  const float a0 = (float)src0[ida];
  const float a1 = (float)src0[ida + 1];
  const float b0 = (float)src1[idb];
  const float b1 = (float)src1[idb + 1];
  destination[idc] = (real)(a0 * b0 - a1 * b1);
  destination[idc + 1] = (real)(a0 * b1 + a1 * b0);
}
    )";
  }

  std::string defines = "";
  if (hash.data_type == MTL::DataTypeFloat) {
    defines += std::string("typedef float real;");
    defines += "\n";
  } else {
    defines += std::string("typedef half real;");
    defines += "\n";
  }

  defines += "constant uint dim0 = ";
  defines += std::to_string(hash.dim[0] / 2) + ";";
  defines += "\n";
  if (hash.dim[1] > 0)
  {
    defines += "constant uint dim1 = ";
    defines += std::to_string(hash.dim[1]) + ";";
    defines += "\n";
    defines += "constant uint astride0 = ";
    defines += std::to_string(hash.astride[0]) + ";";
    defines += "\n";
    defines += "constant uint bstride0 = ";
    defines += std::to_string(hash.bstride[0]) + ";";
    defines += "\n";
    defines += "constant uint cstride0 = ";
    defines += std::to_string(hash.cstride[0]) + ";";
    defines += "\n";
  }
  if (hash.dim[2] > 0)
  {
    defines += "constant uint astride1 = ";
    defines += std::to_string(hash.astride[1]) + ";";
    defines += "\n";
    defines += "constant uint bstride1 = ";
    defines += std::to_string(hash.bstride[1]) + ";";
    defines += "\n";
    defines += "constant uint cstride1 = ";
    defines += std::to_string(hash.cstride[1]) + ";";
    defines += "\n";
  }
  if (hash.dim[3] > 0 && hash.dim[2] > 0)
  {
    defines += "constant uint dim2 = ";
    defines += std::to_string(hash.dim[2]) + ";";
    defines += "\n";
    defines += "constant uint astride2 = ";
    defines += std::to_string(hash.astride[2]) + ";";
    defines += "\n";
    defines += "constant uint bstride2 = ";
    defines += std::to_string(hash.bstride[2]) + ";";
    defines += "\n";
    defines += "constant uint cstride2 = ";
    defines += std::to_string(hash.cstride[2]) + ";";
    defines += "\n";
  }
  if (hash.dim[3] == 0 && hash.dim[2] == 0 && hash.dim[1] == 0)
  {
    this->group_size = MTL::Size(256, 1, 1);
    const int num_blocks = (hash.dim[0] / 2 + 255) / 256;
    this->grid_size = MTL::Size(num_blocks, 1, 1);
  } else if (hash.dim[3] == 0 && hash.dim[2] == 0) {
    this->group_size = MTL::Size(32, 8, 1);
    this->grid_size = MTL::Size((hash.dim[0] / 2 + 31) / 32, (hash.dim[1] + 7) / 8, 1);
  } else if (hash.dim[3] == 0) {
    this->group_size = MTL::Size(32, 8, 1);
    this->grid_size = MTL::Size((hash.dim[0] / 2 + 31) / 32, (hash.dim[1] + 7) / 8, hash.dim[2]);
  } else {
    this->group_size = MTL::Size(32, 8, 1);
    this->grid_size = MTL::Size((hash.dim[0] / 2 + 31) / 32, (hash.dim[1] + 7) / 8, hash.dim[2] * hash.dim[3]);
  }

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  NS::SharedPtr<MTL::ComputePipelineState>* pso = &cmul_pso;

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
    
  auto swift_name = NS::String::string("cmul", NS::UTF8StringEncoding);
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
