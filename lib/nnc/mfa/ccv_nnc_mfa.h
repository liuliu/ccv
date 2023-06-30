#ifndef GUARD_ccv_nnc_mfa_h
#define GUARD_ccv_nnc_mfa_h

void* ccv_nnc_init_mfa_context(void* device, void* command_queue);
bool ccv_nnc_mfa_context_supported(void* mfa_context);
void ccv_nnc_deinit_mfa_context(void* mfa_context);

#ifdef __cplusplus

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#define CCV_NNC_MFA_ASSERT(error) \
if (error) { ccv_nnc_mfa_fatal_error(error, __LINE__, __COLUMN__, __FILE__, __FUNCTION__); } \

void ccv_nnc_mfa_fatal_error(NS::Error* error, const char* line, const char *column, const char *file_name, const char *function_name);

class ccv_nnc_mfa_context {
public:
  NS::SharedPtr<MTL::Device> device;
  NS::SharedPtr<MTL::CommandQueue> commandQueue;
  NS::SharedPtr<MTL::Library> library;
  
  ccv_nnc_mfa_context(MTL::Device* device, MTL::CommandQueue* command_queue);
  
  bool get_supported() {
    return _supported;
  }
  
private:
  bool _supported;
};

#endif

#endif
