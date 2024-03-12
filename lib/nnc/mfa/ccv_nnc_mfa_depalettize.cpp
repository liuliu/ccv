#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_depalettize(mfa::context* context, ccv_nnc_mfa_depalettize_params_t params)
{
  context->depalettize_cache.prepare(context, mfa::depalettize::hash(params));
}

void ccv_nnc_mfa_encode_depalettize(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_depalettize_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  mfa::depalettize::hash hash(params);
  auto iterator = context->depalettize_cache.map.find(hash);
  if (iterator == context->depalettize_cache.map.end()) {
    mfa::precondition_failure("Depalettize hash not cached.", __LINE__, __FILE__, __FUNCTION__);
  }
  
  auto* pipeline = iterator->second;
  auto encoder = command_batch->startCommand();
  
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 2);
  
  encoder->setComputePipelineState(pipeline->depalettize_pso.get());
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageWrite);
  
  auto grid_size = pipeline->grid_size;
  grid_size.depth = 1;
  CCV_NNC_MFA_PRECONDITION(grid_size.depth > 0);
  encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
  command_batch->finishCommand(encoder);
}

// MARK: - C++

mfa::depalettize::hash::hash(ccv_nnc_mfa_depalettize_params_t params) {
  data_type = params.data_type;
  qbits = params.qbits;
  number_in_blocks = params.number_in_blocks;
  length = params.length;
}

bool mfa::depalettize::hash::operator==(const mfa::depalettize::hash& hash) const {
  return
  (data_type == hash.data_type) &&
  (qbits == hash.qbits) &&
  (number_in_blocks == hash.number_in_blocks) &&
  (length == hash.length);
}

std::ostream& operator<<(std::ostream& os, const mfa::depalettize::hash& hash) {
  os << "mfa::depalettize::hash {";
  os << " .data_type = " << hash.data_type << ',';
  os << " .qbits = " << hash.qbits << ',';
  os << " .number_in_blocks = " << hash.number_in_blocks << ',';
  os << " .length = " << hash.length << " ";
  os << "}";
  return os;
}

std::size_t std::hash<mfa::depalettize::hash>::operator()(const mfa::depalettize::hash& hash) const noexcept {
  std::size_t seed = 0;
  using namespace mfa::hash;
  combine_64(seed, hash.data_type);
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.qbits, (unsigned int)hash.number_in_blocks }));
  combine_64(seed, hash.length);
  return seed;
}

mfa::depalettize::pipeline::pipeline(mfa::context* context, mfa::depalettize::hash hash) {
  // FlashNorm not supported for group depalettize yet.
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf))
  
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  std::string shader;
  if (hash.qbits == 6) {
    shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void depalettize(
  device uchar *source [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  device const uchar *ui0 = source + (sizeof(real) * palette_size + number_in_blocks * 3) * tgid.y;
  threadgroup real palette[palette_size];
  if (lid < palette_size) {
    palette[lid] = ((device real*)ui0)[lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint x = tgid.x * threadgroup_size + lid;
  device const packed_uchar3 *ui1 = (device const packed_uchar3*)(ui0 + sizeof(real) * palette_size);
  const packed_uchar3 u = ui1[x];
  const real4 d = real4(palette[u.x >> 2], palette[((u.x & 3) << 4) | (u.y >> 4)], palette[((u.y & 15) << 2) | (u.z >> 6)], palette[u.z & 63]);
  destination[number_in_blocks * tgid.y + x] = d;
}
    )";
  } else if (hash.qbits == 8) {
    if (hash.length % hash.number_in_blocks == 0) {
      shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void depalettize(
  device uchar *source [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  device const uchar *ui0 = source + (sizeof(real) * palette_size + number_in_blocks * 4) * tgid.y;
  threadgroup real palette[palette_size];
  if (lid < palette_size) {
    palette[lid] = ((device real*)ui0)[lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint x = tgid.x * threadgroup_size + lid;
  device const uchar4 *ui1 = (device const uchar4*)(ui0 + sizeof(real) * palette_size);
  const uchar4 u = ui1[x];
  const real4 d = real4(palette[u.x], palette[u.y], palette[u.z], palette[u.w]);
  destination[number_in_blocks * tgid.y + x] = d;
}
      )";
    } else {
      shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void depalettize(
  device uchar *source [[buffer(0)]],
  device real4 *destination [[buffer(1)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  device const uchar *ui0 = source + (sizeof(real) * palette_size + number_in_blocks * 4) * tgid.y;
  threadgroup real palette[palette_size];
  if (lid < palette_size) {
    palette[lid] = ((device real*)ui0)[lid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint x = tgid.x * threadgroup_size + lid;
  if (x + number_in_blocks * tgid.y >= number_of_elements) {
    return;
  }
  device const uchar4 *ui1 = (device const uchar4*)(ui0 + sizeof(real) * palette_size);
  const uchar4 u = ui1[x];
  const real4 d = real4(palette[u.x], palette[u.y], palette[u.z], palette[u.w]);
  destination[number_in_blocks * tgid.y + x] = d;
}
      )";
    }
  }
  
  std::string defines = "";
  if (hash.data_type == MTL::DataTypeFloat) {
    defines += std::string("typedef float real;");
    defines += std::string("typedef float4 real4;");
    defines += "\n";
  } else {
    defines += std::string("typedef half real;");
    defines += std::string("typedef half4 real4;");
    defines += "\n";
  }
  
  uint16_t threadgroup_size = 256;
  defines += "constant ushort threadgroup_size = ";
  defines += std::to_string(threadgroup_size) + ";";
  defines += "\n";
  this->group_size = MTL::Size(threadgroup_size, 1, 1);
  CCV_NNC_MFA_PRECONDITION(hash.qbits == 8 || hash.qbits == 6);

  if (hash.qbits == 6) {
    CCV_NNC_MFA_PRECONDITION((hash.length % hash.number_in_blocks) == 0);
    defines += "constant ushort palette_size = 64;\n";

    defines += "constant uint number_in_blocks = ";
    defines += std::to_string(hash.number_in_blocks / 4) + ";";
    defines += "\n";
    const int num_blocks = hash.length / hash.number_in_blocks;
    CCV_NNC_MFA_PRECONDITION((hash.number_in_blocks % (256 * 4)) == 0);
    const int repeat_4 = hash.number_in_blocks / (256 * 4);
    this->grid_size = MTL::Size(repeat_4, num_blocks, 1);
  } else if (hash.qbits == 8) {
    defines += "constant ushort palette_size = 256;\n";

    defines += "constant uint number_in_blocks = ";
    defines += std::to_string(hash.number_in_blocks / 4) + ";";
    defines += "\n";
    if (hash.length % hash.number_in_blocks != 0) {
      defines += "constant uint number_of_elements = ";
      defines += std::to_string(hash.length / 4) + ";";
      defines += "\n";
    }
    const int num_blocks = (hash.length + hash.number_in_blocks - 1) / hash.number_in_blocks;
    CCV_NNC_MFA_PRECONDITION((hash.number_in_blocks % (256 * 4)) == 0);
    const int repeat_4 = hash.number_in_blocks / (256 * 4);
    this->grid_size = MTL::Size(repeat_4, num_blocks, 1);
    CCV_NNC_MFA_PRECONDITION((hash.length % 4) == 0);
  }
  
  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  NS::SharedPtr<MTL::ComputePipelineState>* pso = &depalettize_pso;

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
    
  auto swift_name = NS::String::string("depalettize", NS::UTF8StringEncoding);
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
