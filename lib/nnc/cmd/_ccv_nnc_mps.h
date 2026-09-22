#ifndef GUARD_cmd_ccv_nnc_mps_h
#define GUARD_cmd_ccv_nnc_mps_h

#include "ccv.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_internal.h"

int _ccv_nnc_scaled_dot_product_attention_forw_mps(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context);

#endif
