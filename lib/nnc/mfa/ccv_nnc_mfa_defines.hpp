#ifndef GUARD_ccv_nnc_mfa_defines_hpp
#define GUARD_ccv_nnc_mfa_defines_hpp

// MARK: - Types

#ifdef __cplusplus
#include "nnc/mfa/3rdparty/metal-cpp/Metal.hpp"
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
  MTL::CommandBuffer* commandBuffer;
  
  // Although labeled `MTL::ComputeCommandEncoder`, this should be used for
  // memcpy and memset as well. Here is a performant reference implementation
  // using custom shaders to bypass the CPU-side latency of switching encoders:
  // https://github.com/philipturner/metal-usm/tree/main/BlitEncoderAlternative
  MTL::ComputeCommandEncoder* commandEncoder;
  
  uint16_t batchedCommandCount = 0;
  uint8_t commandActive = 0;
  
  CommandBatch(MTL::CommandQueue* commandQueue);
  ~CommandBatch();
  
  MTL::ComputeCommandEncoder* startCommand(MTL::ComputePipelineState* pso);
  void finishCommand(MTL::ComputeCommandEncoder* commandEncoder);
};
} // namespace MTL

typedef MTL::CommandBatch mtl_command_batch_t;
#else // __cplusplus
typedef struct {
  mtl_command_buffer_t* command_buffer;
  mtl_compute_command_encoder_t* command_encoder;
  uint16_t batched_command_count;
  uint8_t command_active;
} MTLCommandBatch;

typedef MTLCommandBatch mtl_command_batch_t;
#endif // __cplusplus

// MARK: - Diagnostics

 #ifndef CCV_METAL_LOGGING_ENABLE
 #define CCV_METAL_LOGGING_ENABLE 1
 #endif

 #ifndef CCV_NNC_MFA_EXTERNAL_METALLIB_ENABLE
 #define CCV_NNC_MFA_EXTERNAL_METALLIB_ENABLE 1
 #endif

// 0 - crash reports
// 1 - metallib initialization
// 2 - PSO creation
// 3 - command encoding

#if CCV_METAL_LOGGING_ENABLE

#ifdef __cplusplus
#define METAL_LOG_LEVEL(CONTEXT) CONTEXT->log_level
#else
#define METAL_LOG_LEVEL(CONTEXT) ccv_nnc_mfa_context_log_level(CONTEXT)
#endif // __cplusplus

#else // CCV_NNC_METAL_LOGGING_ENABLE

#define METAL_LOG_LEVEL(CONTEXT) 0

#endif // CCV_NNC_METAL_LOGGING_ENABLE

#endif
