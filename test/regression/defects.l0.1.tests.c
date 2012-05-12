#include "ccv.h"
#include "case.h"
#include "ccv_case.h"
#include "ccv_internal.h"

/* this file is for all failed cases (a.k.a. bugs) that before 0.1 version */

TEST_CASE("ccv_filter cannot cover full range when the source is small enough")
{
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(20, 14, CCV_32F | CCV_C1, 0, 0);
	ccv_dense_matrix_t* y = ccv_dense_matrix_new(15, 5, CCV_32F | CCV_C1, 0, 0);
	int i;
	for (i = 0; i < 15 * 5; i++)
		y->data.f32[i] = 1;
	for (i = 0; i < 20 * 14; i++)
		x->data.f32[i] = 1;
	ccv_dense_matrix_t* d = ccv_dense_matrix_new(20, 14, CCV_32F | CCV_C1, 0, 0);
	for (i = 0; i < 20 * 14; i++)
		d->data.f32[i] = sqrtf(-1.0f);
	ccv_filter(x, y, &d, 0, CCV_NO_PADDING);
	REQUIRE(!ccv_any_nan(d), "filter result shouldn't contain any nan value");
	ccv_matrix_free(d);
	ccv_matrix_free(y);
	ccv_matrix_free(x);
}

#include "case_main.h"
