#ifndef GUARD_ccv_nnc_cpu_ref_h
#define GUARD_ccv_nnc_cpu_ref_h

#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

void _ccv_nnc_tensor_transfer_cpu_ref(const ccv_nnc_tensor_view_t* const a, ccv_nnc_tensor_view_t* const b);
void _ccv_nnc_tensor_set_cpu_ref(ccv_nnc_tensor_view_t* const a, const float b);
int _ccv_nnc_ewsum_forw_cpu_ref(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const ccv_nnc_stream_context_t* const stream_context);

#endif
