#include "ccv_nnc_mfa.hpp"
#include "ccv_nnc_mfa_hash.hpp"
#include <simd/simd.h>
using namespace ccv::nnc;

#include <string>

// MARK: - C

void ccv_nnc_mfa_prepare_gemv(mfa::context* context, ccv_nnc_mfa_gemv_params_t params)
{
  context->gemv_cache.prepare(context, mfa::gemv::hash(params));
}

void ccv_nnc_mfa_encode_gemv(ccv_nnc_mfa_context_t* context, ccv_nnc_mfa_gemv_params_t params, mtl_command_batch_t* command_batch, mtl_buffer_t** tensors, size_t* tensor_offsets)
{
  mfa::gemv::hash hash(params);
  auto iterator = context->gemv_cache.map.find(hash);
  if (iterator == context->gemv_cache.map.end()) {
    mfa::precondition_failure("gemv hash not cached.", __LINE__, __FILE__, __FUNCTION__);
  }
  
  auto* pipeline = iterator->second;
  auto encoder = command_batch->startCommand();
  
  int num_tensors = 0;
  while (tensors[num_tensors] != nullptr) {
    encoder->setBuffer(tensors[num_tensors], tensor_offsets[num_tensors], NS::UInteger(num_tensors));
    num_tensors += 1;
  }
  CCV_NNC_MFA_PRECONDITION(num_tensors == 3 || num_tensors == 4);
  
  encoder->setComputePipelineState(pipeline->gemv_pso.get());
  encoder->useResource(tensors[0], MTL::ResourceUsageRead);
  encoder->useResource(tensors[1], MTL::ResourceUsageRead);
  encoder->useResource(tensors[2], MTL::ResourceUsageWrite);
  if (num_tensors == 4) {
    encoder->useResource(tensors[3], MTL::ResourceUsageRead);
  }

  auto grid_size = pipeline->grid_size;
  CCV_NNC_MFA_PRECONDITION(grid_size.depth > 0);
  encoder->dispatchThreadgroups(grid_size, pipeline->group_size);
  command_batch->finishCommand(encoder);
}

// MARK: - C++

mfa::gemv::hash::hash(ccv_nnc_mfa_gemv_params_t params) {
  data_type = params.data_type;
  nrows = params.nrows;
  ncols = params.ncols;
  fused_bias = params.fused_bias;
}

bool mfa::gemv::hash::operator==(const mfa::gemv::hash& hash) const {
  return
  (data_type == hash.data_type) &&
  (nrows == hash.nrows) &&
  (ncols == hash.ncols) &&
  (fused_bias == hash.fused_bias);
}

std::ostream& operator<<(std::ostream& os, const mfa::gemv::hash& hash) {
  os << "mfa::gemv::hash {";
  os << " .data_type = " << hash.data_type << ',';
  os << " .nrows = " << hash.nrows << ',';
  os << " .ncols = " << hash.ncols << ',';
  os << " .fused_bias = " << bool(hash.fused_bias) << " ";
  os << "}";
  return os;
}

std::size_t std::hash<mfa::gemv::hash>::operator()(const mfa::gemv::hash& hash) const noexcept {
  std::size_t seed = 0;
  using namespace mfa::hash;
  combine_64(seed, hash.data_type);
  combine_64(seed, pack_64(simd::uint2 { (unsigned int)hash.nrows, (unsigned int)hash.ncols }));
  combine_32(seed, pack_32(simd::uchar4 { hash.fused_bias, 0, 0, 0 }));
  return seed;
}

mfa::gemv::pipeline::pipeline(mfa::context* context, mfa::gemv::hash hash) {
  // FlashNorm not supported for group gemv yet.
  CCV_NNC_MFA_PRECONDITION((hash.data_type == MTL::DataTypeFloat) || (hash.data_type == MTL::DataTypeHalf) || (hash.data_type == MTL::DataTypeBFloat))
  
  auto* pool = NS::AutoreleasePool::alloc()->init();
  
  std::string shader;
  if (hash.fused_bias) {
    shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void gemv(
  device const real *src0 [[buffer(0)]],
  device const real *src1 [[buffer(1)]],
  device real *dst [[buffer(2)]],
  device const real *bias [[buffer(3)]],

  uint tgpig[[threadgroup_position_in_grid]],
  uint tiisg[[thread_index_in_simdgroup]]) {

  const int64_t rb = tgpig * N;
  device const real* y = (device const real*)src1;

  if (ncols < 128) {
    for (uint row = 0; row < N; ++row) {
      uint r1 = rb + row;
      if (r1 >= nrows) {
        break;
      }
      device const real* x = (device const real*)src0 + r1 * ncols;
      float sumf = 0;
      for (uint i = tiisg; i < ncols; i += 32) {
        sumf += (real)x[i] * (real)y[i];
      }

      float all_sum = simd_sum(sumf);
      if (tiisg == 0) {
        dst[r1] = bias[r1] + (real)all_sum;
      }
    }
  } else {
    device const real4* y4 = (device const real4*)y;
    for (uint row = 0; row < N; ++row) {
      uint r1 = rb + row;
      if (r1 >= nrows) {
        break;
      }

      device const real* x = (device const real*)src0 + r1 * ncols;
      device const real4* x4 = (device const real4*)x;

      float sumf = 0;
      for (uint i = tiisg; i < ncols / 4; i += 32) {
        sumf += (real)x4[i][0] * y4[i][0];
        sumf += (real)x4[i][1] * y4[i][1];
        sumf += (real)x4[i][2] * y4[i][2];
        sumf += (real)x4[i][3] * y4[i][3];
      }

      float all_sum = simd_sum(sumf);
      if (tiisg == 0) {
        dst[r1] = bias[r1] + (real)all_sum;
      }
    }
  }
}
    )";
  } else {
    shader = R"(
#include <metal_stdlib>
using namespace metal;

kernel void gemv(
  device const real *src0 [[buffer(0)]],
  device const real *src1 [[buffer(1)]],
  device real *dst [[buffer(2)]],

  uint tgpig[[threadgroup_position_in_grid]],
  uint tiisg[[thread_index_in_simdgroup]]) {

  const int64_t rb = tgpig * N;
  device const real* y = (device const real*)src1;

  if (ncols < 128) {
    for (uint row = 0; row < N; ++row) {
      uint r1 = rb + row;
      if (r1 >= nrows) {
        break;
      }
      device const real* x = (device const real*)src0 + r1 * ncols;
      float sumf = 0;
      for (uint i = tiisg; i < ncols; i += 32) {
        sumf += (real)x[i] * (real)y[i];
      }

      float all_sum = simd_sum(sumf);
      if (tiisg == 0) {
        dst[r1] = (real)all_sum;
      }
    }
  } else {
    device const real4* y4 = (device const real4*)y;
    for (uint row = 0; row < N; ++row) {
      uint r1 = rb + row;
      if (r1 >= nrows) {
        break;
      }

      device const real* x = (device const real*)src0 + r1 * ncols;
      device const real4* x4 = (device const real4*)x;

      float sumf = 0;
      for (uint i = tiisg; i < ncols / 4; i += 32) {
        sumf += (real)x4[i][0] * y4[i][0];
        sumf += (real)x4[i][1] * y4[i][1];
        sumf += (real)x4[i][2] * y4[i][2];
        sumf += (real)x4[i][3] * y4[i][3];
      }

      float all_sum = simd_sum(sumf);
      if (tiisg == 0) {
        dst[r1] = (real)all_sum;
      }
    }
  }
}
    )";
  }

  std::string defines = "";
  if (hash.data_type == MTL::DataTypeFloat) {
    defines += std::string("typedef float real;");
    defines += "\n";
    defines += std::string("typedef float4 real4;");
    defines += "\n";
  } else if (hash.data_type == MTL::DataTypeBFloat) {
    defines += std::string("typedef bfloat real;");
    defines += "\n";
    defines += std::string("typedef bfloat4 real4;");
    defines += "\n";
  } else {
    defines += std::string("typedef half real;");
    defines += "\n";
    defines += std::string("typedef half4 real4;");
    defines += "\n";
  }

  unsigned int N = 8;
  if (strstr(context->device->name()->utf8String(), "M1") != 0 ||
      strstr(context->device->name()->utf8String(), "M2") != 0 ||
      strstr(context->device->name()->utf8String(), "M3") != 0 ||
      strstr(context->device->name()->utf8String(), "M4") != 0) {
    // If it is M* chip, check if it is Pro / Ultra / Max.
	if (strstr(context->device->name()->utf8String(), "Pro") != 0) {
      N = 2; // For Pro, we set N = 2 to maximize core utilization.
    } else if (strstr(context->device->name()->utf8String(), "Max") != 0 ||
        strstr(context->device->name()->utf8String(), "Ultra") != 0) {
      N = 1; // For Ultra / Max, we set N = 1 to maximize core utilization.
    }
  }
  defines += "constant uint N = ";
  defines += std::to_string(N) + ";";
  defines += "\n";
  defines += "constant uint ncols = ";
  defines += std::to_string(hash.ncols) + ";";
  defines += "\n";
  defines += "constant uint nrows = ";
  defines += std::to_string(hash.nrows) + ";";
  defines += "\n";
  this->group_size = MTL::Size(32, 1, 1);
  this->grid_size = MTL::Size((hash.nrows + N - 1) / N, 1, 1);

  auto constants = NS::TransferPtr(MTL::FunctionConstantValues::alloc()->init());
  NS::SharedPtr<MTL::ComputePipelineState>* pso = &gemv_pso;

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
    
  auto swift_name = NS::String::string("gemv", NS::UTF8StringEncoding);
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
