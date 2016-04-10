/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_cmd_fc_opt_h
#define GUARD_ccv_nnc_cmd_fc_opt_h

#include <ccv.h>
#include <nnc/ccv_nnc.h>

int ccv_nnc_fc_forw_gemm(const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_view_t* w, const ccv_nnc_tensor_view_t* bias, ccv_nnc_tensor_view_t* b);
int ccv_nnc_fc_back_gemm(const ccv_nnc_tensor_view_t* g, const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_view_t* w, ccv_nnc_tensor_view_t* dw, ccv_nnc_tensor_view_t* bias, ccv_nnc_tensor_view_t* h, int flags);
int ccv_nnc_fc_forw_opt(const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_view_t* w, const ccv_nnc_tensor_view_t* bias, ccv_nnc_tensor_view_t* b);
int ccv_nnc_fc_back_opt(const ccv_nnc_tensor_view_t* g, const ccv_nnc_tensor_view_t* a, const ccv_nnc_tensor_view_t* w, ccv_nnc_tensor_view_t* dw, ccv_nnc_tensor_view_t* bias, ccv_nnc_tensor_view_t* h, int flags);

#endif
