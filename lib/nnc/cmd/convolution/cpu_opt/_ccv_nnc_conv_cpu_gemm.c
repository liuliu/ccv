#include <ccv.h>
#include <ccv_internal.h>
#include <nnc/ccv_nnc.h>
#include <nnc/ccv_nnc_easy.h>
#include <nnc/ccv_nnc_internal.h>
#include "../_ccv_nnc_conv_cpu_opt.h"

int _ccv_nnc_conv_forw_gemm_cpu_opt(const ccv_nnc_tensor_view_t* const a, const ccv_nnc_tensor_t* const w, const ccv_nnc_tensor_t* const bias, const ccv_nnc_hint_t hint, ccv_nnc_tensor_view_t* const b)
{
	assert(!CCV_IS_TENSOR_VIEW(a));
	assert(!CCV_IS_TENSOR_VIEW(w));
	assert(!bias || !CCV_IS_TENSOR_VIEW(bias));
	assert(!CCV_IS_TENSOR_VIEW(b));
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(a_nd == CCV_NNC_MAX_DIM + 1 || a_nd == CCV_NNC_MAX_DIM + 2);
	const int* adim = (a_nd == CCV_NNC_MAX_DIM + 1) ? a->info.dim : a->info.dim + 1;
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	assert(b_nd == CCV_NNC_MAX_DIM + 1 || b_nd == CCV_NNC_MAX_DIM + 2);
	const int* bdim = (b_nd == CCV_NNC_MAX_DIM + 1) ? b->info.dim : b->info.dim + 1;
	assert(hint.border.begin[0] == 0 && hint.border.begin[1] == 0);
	assert(hint.border.end[0] == 0 && hint.border.end[1] == 0);
	assert(adim[0] == bdim[0]);
	assert(adim[1] == bdim[1]);
	assert(hint.stride.dim[0] <= 1 && hint.stride.dim[1] <= 1);
	ccv_dense_matrix_t am = ccv_dense_matrix(adim[0] * adim[1], adim[2], CCV_32F | CCV_C1, a->data.u8, 0);
	ccv_dense_matrix_t bm = ccv_dense_matrix(bdim[0] * bdim[1], bdim[2], CCV_32F | CCV_C1, b->data.u8, 0);
	// copy bias into each row.
	int i;
	if (bias)
		for (i = 0; i < bm.rows; i++)
			memcpy(bm.data.f32 + i * bdim[2], bias->data.f32, sizeof(float) * bdim[2]);
	ccv_dense_matrix_t* dbm = &bm;
	ccv_dense_matrix_t wm = ccv_dense_matrix(bdim[2], adim[2], CCV_32F | CCV_C1, w->data.u8, 0);
	if (bias)
		ccv_gemm(&am, &wm, 1, dbm, 1, CCV_B_TRANSPOSE, (ccv_matrix_t**)&dbm, 0); // supply b as matrix C is allowed
	else
		ccv_gemm(&am, &wm, 1, 0, 0, CCV_B_TRANSPOSE, (ccv_matrix_t**)&dbm, 0); // supply b as matrix C is allowed
	return CCV_NNC_EXEC_SUCCESS;
}
