#include "ccv.h"
#include "case.h"

TEST_CASE("matrix multiplication")
{ 
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(3, 2, CCV_64F | CCV_C1, 0, 0);
	a->data.db[0] = 0.11;
	a->data.db[1] = 0.12;
	a->data.db[2] = 0.13;
	a->data.db[3] = 0.21;
	a->data.db[4] = 0.22;
	a->data.db[5] = 0.23;
	ccv_dense_matrix_t* b = ccv_dense_matrix_new(3, 2, CCV_64F | CCV_C1, 0, 0);
	b->data.db[0] = 1011;
	b->data.db[1] = 1012;
	b->data.db[2] = 1021;
	b->data.db[3] = 1022;
	b->data.db[4] = 1031;
	b->data.db[5] = 1032;
	ccv_dense_matrix_t* y = 0;
	ccv_gemm(a, b, 1, 0, 0, CCV_A_TRANSPOSE, (ccv_matrix_t**)&y, 0);
	double dy[4];
	memcpy(dy, y->data.db, sizeof(double) * 4);
	ccv_matrix_free(a);
	ccv_matrix_free(b);
	ccv_matrix_free(y);
	ccv_garbage_collect();
	double hy[4] = {470.760000, 471.220000, 572.860000, 573.420000};
	REQUIRE_EQ_ARRAY_WITH_TOLERANCE(double, hy, dy, 4, 1e-6, "2x3, 3x2 matrix multiplication failure");
}

TEST_CASE("vector sum")
{
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(3, 2, CCV_64F | CCV_C1, 0, 0);
	a->data.db[0] = 0.11;
	a->data.db[1] = 0.12;
	a->data.db[2] = 0.13;
	a->data.db[3] = 0.21;
	a->data.db[4] = 0.22;
	a->data.db[5] = 0.23;
	double sum = ccv_sum(a);
	ccv_matrix_free(a);
	ccv_garbage_collect();
	REQUIRE_EQ_WITH_TOLERANCE(sum, 1.02, 1e-6, "3x2 vector sum failure");
}

#include "case-main.h"
