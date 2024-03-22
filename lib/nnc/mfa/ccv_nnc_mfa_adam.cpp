#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>
#include <sstream>
#include <iomanip>

// MARK: - C

void ccv_nnc_mfa_prepare_adam(mfa::context* context, ccv_nnc_mfa_adam_params_t params)
{
  context->adam_cache.prepare(context, mfa::adam::hash(params));
}

void ccv_nnc_mfa_encode_adam(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_adam_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  mfa::adam::hash hash(params);
  auto iterator = context->adam_cache.map.find(hash);
  if (iterator == context->adam_cache.map.end()) {
    mfa::precondition_failure("adam hash not cached.", __LINE__, __FILE__, __FUNCTION__);
  }
  
  auto* pipeline = iterator->second;
  auto encoder = command_batch->startCommand();
  
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  if (hash.amsgrad)
  {
    CCV_NNC_MFA_PRECONDITION(num_tensors == 9);
  } else {
    CCV_NNC_MFA_PRECONDITION(num_tensors == 7);
  }
  const float rate_inv_bias_correction1 = params.rate / (1 - powf(params.beta1, params.step));
  const float inv_bias_correction2 = 1. / (1 - powf(params.beta2, params.step));
  float values[2] = { rate_inv_bias_correction1, inv_bias_correction2 };
  encoder->setBytes(values, sizeof(float) * 2, 10);
  
  encoder->setComputePipelineState(pipeline->adam_pso.get());
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  if (tensors[1] != tensors[2])
  {
    encoder->useResource(tensors[1], MTL::ResourceUsageRead);
    encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[1], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  }
  if (tensors[3] != tensors[5])
  {
    encoder->useResource(tensors[3], MTL::ResourceUsageRead);
    encoder->useResource(tensors[5], MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[3], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  }
  if (tensors[4] != tensors[6])
  {
    encoder->useResource(tensors[4], MTL::ResourceUsageRead);
    encoder->useResource(tensors[6], MTL::ResourceUsageWrite);
  } else {
    encoder->useResource(tensors[4], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
  }
  if (num_tensors == 9)
  {
    if (tensors[7] != tensors[8])
    {
      encoder->useResource(tensors[7], MTL::ResourceUsageRead);
      encoder->useResource(tensors[8], MTL::ResourceUsageWrite);
    } else {
      encoder->useResource(tensors[7], MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
    }
  }
  
  auto grid_size = pipeline->grid_size;
  grid_size.depth = 1;
  CCV_NNC_MFA_PRECONDITION(grid_size.depth > 0);
  encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
  command_batch->finishCommand(encoder);
}

// MARK: - C++

mfa::adam::hash::hash(ccv_nnc_mfa_adam_params_t params) {
  data_type = params.data_type;
  adamw = params.adamw;
  amsgrad = params.amsgrad;
  rate = params.rate;
  scale = params.scale;
  beta1 = params.beta1;
  beta2 = params.beta2;
  decay = params.decay;
  epsilon = params.epsilon;
  length = params.length;
}

bool mfa::adam::hash::operator==(const mfa::adam::hash& hash) const {
  return
  (data_type == hash.data_type) &&
  (adamw == hash.adamw) &&
  (amsgrad == hash.amsgrad) &&
  (rate == hash.rate) &&
  (scale == hash.scale) &&
  (beta1 == hash.beta1) &&
  (beta2 == hash.beta2) &&
  (decay == hash.decay) &&
  (epsilon == hash.epsilon) &&
  (length == hash.length);
}

std::ostream& operator<<(std::ostream& os, const mfa::adam::hash& hash) {
  os << "mfa::adam::hash {";
  os << " .data_type = " << hash.data_type << ',';
  os << " .adamw = " << hash.adamw << ',';
  os << " .amsgrad = " << hash.amsgrad << ',';
  os << " .rate = " << hash.rate << ',';
  os << " .scale = " << hash.scale << ',';
  os << " .beta1 = " << hash.beta1 << ',';
  os << " .beta2 = " << hash.beta2 << ',';
  os << " .decay = " << hash.decay << ',';
  os << " .epsilon = " << hash.epsilon << ',';
  os << " .length = " << hash.length << " ";
  os << "}";
  return os;
}

std::size_t std::hash<mfa::adam::hash>::operator()(const mfa::adam::hash& hash) const noexcept {
  std::size_t seed = 0;
  using namespace mfa::hash;
  combine_64(seed, hash.data_type);
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.adamw, (unsigned int)hash.amsgrad }));
  combine_64(seed, pack_64(simd::uint2 { *reinterpret_cast<const uint32_t*>(&hash.rate), *reinterpret_cast<const uint32_t*>(&hash.scale) }));
  combine_64(seed, pack_64(simd::uint2 { *reinterpret_cast<const uint32_t*>(&hash.beta1), *reinterpret_cast<const uint32_t*>(&hash.beta2) }));
  combine_64(seed, pack_64(simd::uint2 { *reinterpret_cast<const uint32_t*>(&hash.decay), *reinterpret_cast<const uint32_t*>(&hash.epsilon) }));
  combine_64(seed, hash.length);
  return seed;
}

static std::string high_precision_to_string(float value) {
    std::ostringstream oss;
    oss << std::setprecision(std::numeric_limits<float>::max_digits10) << value;
    return oss.str();
}

mfa::adam::pipeline::pipeline(mfa::context* context, mfa::adam::hash hash) {
  // FlashNorm not supported for group adam yet.
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf))
  
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  std::string shader;
  if (hash.adamw) {
    if (hash.amsgrad) {
      shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void adam(
  device real *gradient [[buffer(0)]],
  device real *source [[buffer(1)]],
  device real *destination [[buffer(2)]],
  device real *mom [[buffer(3)]],
  device real *vel [[buffer(4)]],
  device real *new_mom [[buffer(5)]],
  device real *new_vel [[buffer(6)]],
  device real *vm [[buffer(7)]],
  device real *new_vm [[buffer(8)]],
  constant float *values [[buffer(10)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint idx = tgid.x * threadgroup_size + lid;
  if (idx >= tensor_length)
    return;
  float rate_inv_bias_correction1 = values[0];
  float inv_bias_correction2 = values[1];
  float grad = scale * (float)gradient[idx];
  const float a = (float)source[idx];
  const float m = beta1 * (float)mom[idx] + (1 - beta1) * grad;
  const float v = beta2 * (float)vel[idx] + (1 - beta2) * grad * grad;
  const float vel_hat = v * inv_bias_correction2;
  const float vel_max_hat = max((float)vm[idx], vel_hat);
  destination[idx] = (real)(a - rate_decay * a - (m * rate_inv_bias_correction1) / (sqrt(vel_max_hat) + epsilon));
  new_mom[idx] = (real)m;
  new_vel[idx] = (real)v;
  new_vm[idx] = (real)vel_max_hat;
}
      )";
    } else {
      shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void adam(
  device real *gradient [[buffer(0)]],
  device real *source [[buffer(1)]],
  device real *destination [[buffer(2)]],
  device real *mom [[buffer(3)]],
  device real *vel [[buffer(4)]],
  device real *new_mom [[buffer(5)]],
  device real *new_vel [[buffer(6)]],
  constant float *values [[buffer(10)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint idx = tgid.x * threadgroup_size + lid;
  if (idx >= tensor_length)
    return;
  float rate_inv_bias_correction1 = values[0];
  float inv_bias_correction2 = values[1];
  float grad = scale * (float)gradient[idx];
  const float a = (float)source[idx];
  const float m = beta1 * (float)mom[idx] + (1 - beta1) * grad;
  const float v = beta2 * (float)vel[idx] + (1 - beta2) * grad * grad;
  destination[idx] = (real)(a - rate_decay * a - (m * rate_inv_bias_correction1) / (sqrt(v * inv_bias_correction2) + epsilon));
  new_mom[idx] = (real)m;
  new_vel[idx] = (real)v;
}
      )";
    }
  } else {
    if (hash.amsgrad) {
      shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void adam(
  device real *gradient [[buffer(0)]],
  device real *source [[buffer(1)]],
  device real *destination [[buffer(2)]],
  device real *mom [[buffer(3)]],
  device real *vel [[buffer(4)]],
  device real *new_mom [[buffer(5)]],
  device real *new_vel [[buffer(6)]],
  device real *vm [[buffer(7)]],
  device real *new_vm [[buffer(8)]],
  constant float *values [[buffer(10)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint idx = tgid.x * threadgroup_size + lid;
  if (idx >= tensor_length)
    return;
  float rate_inv_bias_correction1 = values[0];
  float inv_bias_correction2 = values[1];
  float grad = scale * (float)gradient[idx];
  const float a = (float)source[idx];
  grad += decay * a;
  const float m = beta1 * (float)mom[idx] + (1 - beta1) * grad;
  const float v = beta2 * (float)vel[idx] + (1 - beta2) * grad * grad;
  const float vel_hat = v * inv_bias_correction2;
  const float vel_max_hat = max((float)vm[idx], vel_hat);
  destination[idx] = (real)(a - (m * rate_inv_bias_correction1) / (sqrt(vel_max_hat) + epsilon));
  new_mom[idx] = (real)m;
  new_vel[idx] = (real)v;
  new_vm[idx] = (real)vel_max_hat;
}
      )";
    } else {
      shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void adam(
  device real *gradient [[buffer(0)]],
  device real *source [[buffer(1)]],
  device real *destination [[buffer(2)]],
  device real *mom [[buffer(3)]],
  device real *vel [[buffer(4)]],
  device real *new_mom [[buffer(5)]],
  device real *new_vel [[buffer(6)]],
  constant float *values [[buffer(10)]],

  uint3 tgid [[threadgroup_position_in_grid]],
  ushort lid [[thread_index_in_threadgroup]]
) {
  const uint idx = tgid.x * threadgroup_size + lid;
  if (idx >= tensor_length)
    return;
  float rate_inv_bias_correction1 = values[0];
  float inv_bias_correction2 = values[1];
  float grad = scale * (float)gradient[idx];
  const float a = (float)source[idx];
  grad += decay * a;
  const float m = beta1 * (float)mom[idx] + (1 - beta1) * grad;
  const float v = beta2 * (float)vel[idx] + (1 - beta2) * grad * grad;
  destination[idx] = (real)(a - (m * rate_inv_bias_correction1) / (sqrt(v * inv_bias_correction2) + epsilon));
  new_mom[idx] = (real)m;
  new_vel[idx] = (real)v;
}
      )";
    }
  }

  std::string defines = "";
  if (hash.data_type == MTL::DataTypeFloat) {
    defines += std::string("typedef float real;");
    defines += "\n";
  } else {
    defines += std::string("typedef half real;");
    defines += "\n";
  }
  
  uint16_t threadgroup_size = 256;
  defines += "constant ushort threadgroup_size = ";
  defines += std::to_string(threadgroup_size) + ";";
  defines += "\n";
  defines += "constant uint tensor_length = ";
  defines += std::to_string(hash.length) + ";";
  defines += "\n";
  if (hash.adamw)
  {
    defines += "constant float rate_decay = ";
    defines += high_precision_to_string(hash.rate * hash.decay) + ";";
    defines += "\n";
  } else {
    defines += "constant float decay = ";
    defines += high_precision_to_string(hash.decay) + ";";
    defines += "\n";
  }
  defines += "constant float scale = ";
  defines += high_precision_to_string(hash.scale) + ";";
  defines += "\n";
  defines += "constant float beta1 = ";
  defines += high_precision_to_string(hash.beta1) + ";";
  defines += "\n";
  defines += "constant float beta2 = ";
  defines += high_precision_to_string(hash.beta2) + ";";
  defines += "\n";
  defines += "constant float epsilon = ";
  defines += high_precision_to_string(hash.epsilon) + ";";
  defines += "\n";
  this->group_size = MTL::Size(threadgroup_size, 1, 1);

  const int num_blocks = (hash.length + threadgroup_size - 1) / threadgroup_size;
  this->grid_size = MTL::Size(num_blocks, 1, 1);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  NS::SharedPtr<MTL::ComputePipelineState>* pso = &adam_pso;

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
    
  auto swift_name = NS::String::string("adam", NS::UTF8StringEncoding);
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
