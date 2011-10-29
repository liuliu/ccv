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
	double hy[4] = {470.760000, 471.220000, 572.860000, 573.420000};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(double, hy, y->data.db, 4, 1e-6, "2x3, 3x2 matrix multiplication failure");
	ccv_matrix_free(a);
	ccv_matrix_free(b);
	ccv_matrix_free(y);
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
	REQUIRE_EQ_WITH_TOLERANCE(sum, 1.02, 1e-6, "3x2 vector sum failure");
}

TEST_CASE("vector L2 normalize")
{
	int i;
	ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(1, 10, CCV_32F | CCV_C1, 0, 0);
	for (i = 0; i < 10; i++)
		dmt->data.fl[i] = i;
	ccv_normalize(dmt, (ccv_matrix_t**)&dmt, 0, CCV_L2_NORM);
	float hm[10] = {0.000000, 0.059235, 0.118470, 0.177705, 0.236940, 0.296174, 0.355409, 0.414644, 0.473879, 0.533114};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hm, dmt->data.fl, 10, 1e-6, "10d vector L2 normalize failure");
	ccv_matrix_free(dmt);
}

#include "case_main.h"
