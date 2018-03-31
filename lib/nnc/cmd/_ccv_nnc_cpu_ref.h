#ifndef GUARD_ccv_nnc_cpu_ref_h
#define GUARD_ccv_nnc_cpu_ref_h

#include <ccv.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_internal.h>

void _ccv_nnc_tensor_transfer_cpu_ref(const ccv_nnc_tensor_view_t* const a, ccv_nnc_tensor_view_t* const b);
void _ccv_nnc_tensor_set_cpu_ref(ccv_nnc_tensor_view_t* const a, const float b);
void _ccv_nnc_ewsum_forw_cpu_ref(ccv_nnc_tensor_view_t* const* const inputs, const int input_size, ccv_nnc_tensor_view_t* const* const outputs, const int output_size);
void _ccv_nnc_ewprod_forw_cpu_ref(ccv_nnc_tensor_view_t* const* const inputs, const int input_size, ccv_nnc_tensor_view_t* const* const outputs, const int output_size);
void _ccv_nnc_add_forw_cpu_ref(const float p, const float q, ccv_nnc_tensor_view_t* const a, ccv_nnc_tensor_view_t* const b, ccv_nnc_tensor_view_t* const c);
void _ccv_nnc_mul_forw_cpu_ref(const float p, ccv_nnc_tensor_view_t* const a, ccv_nnc_tensor_view_t* const b, ccv_nnc_tensor_view_t* const c);
void _ccv_nnc_reduce_sum_forw_cpu_ref(ccv_nnc_tensor_view_t* const a, ccv_nnc_tensor_view_t* const b);

#endif
