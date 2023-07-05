#ifndef GUARD_ccv_nnc_mfa_defines_hpp
#define GUARD_ccv_nnc_mfa_defines_hpp

#ifdef __cplusplus
#include "3rdparty/metal-cpp/Metal.hpp"
namespace ccv {
namespace nnc {
namespace mfa {
class context;
} // namespace mfa
} // namespace nnc
} // namespace ccv

typedef ccv::nnc::mfa::context ccv_nnc_mfa_context_t;
typedef MTL::Buffer mtl_buffer_t;
typedef MTL::CommandBuffer mtl_command_buffer_t;
typedef MTL::ComputeCommandEncoder mtl_compute_command_encoder_t;
typedef MTL::CommandQueue mtl_command_queue_t;
typedef MTL::Device mtl_device_t;
#else
typedef void ccv_nnc_mfa_context_t;
typedef void mtl_buffer_t;
typedef void mtl_command_buffer_t;
typedef void mtl_compute_command_encoder_t;
typedef void mtl_command_queue_t;
typedef void mtl_device_t;
#endif // __cplusplus

#ifdef __cplusplus
namespace MTL {
class CommandBatch {
public:
  MTL::CommandBuffer* command_buffer;
  
  // Although labeled `MTL::ComputeCommandEncoder`, this should be used for
  // memcpy and memset as well. Here is a performant reference implementation
  // using custom shaders to bypass the CPU-side latency of switching encoders:
  // https://github.com/philipturner/metal-usm/tree/main/BlitEncoderAlternative
  MTL::ComputeCommandEncoder* command_encoder;
  
  uint16_t batched_command_count = 0;
  
  CommandBatch(MTL::CommandQueue* command_queue) {
    command_buffer = command_queue->commandBuffer();
    command_encoder = command_buffer->computeCommandEncoder();
  }
  
  ~CommandBatch() {
    command_encoder->endEncoding();
    command_buffer->commit();
  }
};
} // namespace MTL

typedef MTL::CommandBatch mtl_command_batch_t;
#else // __cplusplus
typedef struct {
  mtl_command_buffer_t* command_buffer;
  mtl_compute_command_encoder_t* encoder;
  uint16_t batched_command_count;
} MTLCommandBatch;

typedef MTLCommandBatch mtl_command_batch_t;
#endif // __cplusplus

#endif
