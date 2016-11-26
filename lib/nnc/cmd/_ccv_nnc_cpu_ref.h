#ifndef GUARD_ccv_nnc_cpu_ref_h
#define GUARD_ccv_nnc_cpu_ref_h

#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

void _ccv_nnc_tensor_transfer_cpu_ref(const ccv_nnc_tensor_view_t* a, ccv_nnc_tensor_view_t* b);
void _ccv_nnc_tensor_set_cpu_ref(ccv_nnc_tensor_view_t* a, float b);
int _ccv_nnc_ewsum_forw_cpu_ref(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context);

#endif
