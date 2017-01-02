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
	assert(!CCV_IS_TENSOR_VIEW(bias));
	assert(!CCV_IS_TENSOR_VIEW(b));
	assert(hint.border.begin[1] == 0 && hint.border.begin[2] == 0);
	assert(hint.border.end[1] == 0 && hint.border.end[2] == 0);
	assert(a->info.dim[1] == b->info.dim[1]);
	assert(a->info.dim[2] == b->info.dim[2]);
	assert(hint.stride.dim[1] <= 1 && hint.stride.dim[2] <= 1);
	ccv_dense_matrix_t am = ccv_dense_matrix(a->info.dim[1] * a->info.dim[2], a->info.dim[0], CCV_32F | CCV_C1, a->data.u8, 0);
	ccv_dense_matrix_t bm = ccv_dense_matrix(b->info.dim[1] * b->info.dim[2], b->info.dim[0], CCV_32F | CCV_C1, b->data.u8, 0);
	// copy bias into each row.
	int i;
	for (i = 0; i < bm.rows; i++)
		memcpy(bm.data.f32 + i * b->info.dim[0], bias->data.f32, sizeof(float) * b->info.dim[0]);
	ccv_dense_matrix_t* dbm = &bm;
	ccv_dense_matrix_t wm = ccv_dense_matrix(b->info.dim[0], a->info.dim[0], CCV_32F | CCV_C1, w->data.u8, 0);
	ccv_gemm(&am, &wm, 1, dbm, 1, CCV_B_TRANSPOSE, (ccv_matrix_t**)&dbm, 0); // supply b as matrix C is allowed
	return CCV_NNC_EXEC_SUCCESS;
}
