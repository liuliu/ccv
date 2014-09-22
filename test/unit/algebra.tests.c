#include "ccv.h"
#include "case.h"

TEST_CASE("matrix multiplication")
{ 
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(3, 2, CCV_64F | CCV_C1, 0, 0);
	a->data.f64[0] = 0.11;
	a->data.f64[1] = 0.12;
	a->data.f64[2] = 0.13;
	a->data.f64[3] = 0.21;
	a->data.f64[4] = 0.22;
	a->data.f64[5] = 0.23;
	ccv_dense_matrix_t* b = ccv_dense_matrix_new(3, 2, CCV_64F | CCV_C1, 0, 0);
	b->data.f64[0] = 1011;
	b->data.f64[1] = 1012;
	b->data.f64[2] = 1021;
	b->data.f64[3] = 1022;
	b->data.f64[4] = 1031;
	b->data.f64[5] = 1032;
	ccv_dense_matrix_t* y = 0;
	ccv_gemm(a, b, 1, 0, 0, CCV_A_TRANSPOSE, (ccv_matrix_t**)&y, 0);
	double hy[4] = {470.760000, 471.220000, 572.860000, 573.420000};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(double, hy, y->data.f64, 4, 1e-6, "2x3, 3x2 matrix multiplication failure");
	ccv_matrix_free(a);
	ccv_matrix_free(b);
	ccv_matrix_free(y);
}

TEST_CASE("matrix addition")
{
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(3, 2, CCV_64F | CCV_C1, 0, 0);
	a->data.f64[0] = 0.11;
	a->data.f64[1] = 0.12;
	a->data.f64[2] = 0.13;
	a->data.f64[3] = 0.21;
	a->data.f64[4] = 0.22;
	a->data.f64[5] = 0.23;
	ccv_dense_matrix_t* b = ccv_dense_matrix_new(3, 2, CCV_64F | CCV_C1, 0, 0);
	b->data.f64[0] = 1011;
	b->data.f64[1] = 1012;
	b->data.f64[2] = 1021;
	b->data.f64[3] = 1022;
	b->data.f64[4] = 1031;
	b->data.f64[5] = 1032;
	ccv_dense_matrix_t* y = 0;
	ccv_add(a, b, (ccv_matrix_t**)&y, 0);
	double hy[6] = {1011.11, 1012.12, 1021.13, 1022.21, 1031.22, 1032.23};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(double, hy, y->data.f64, 4, 1e-6, "3x2, 3x2 matrix addition failure");
	ccv_matrix_free(a);
	ccv_matrix_free(b);
	ccv_matrix_free(y);
}

TEST_CASE("vector sum")
{
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(3, 2, CCV_64F | CCV_C1, 0, 0);
	a->data.f64[0] = 0.11;
	a->data.f64[1] = 0.12;
	a->data.f64[2] = 0.13;
	a->data.f64[3] = 0.21;
	a->data.f64[4] = 0.22;
	a->data.f64[5] = 0.23;
	double sum = ccv_sum(a, CCV_SIGNED);
	ccv_matrix_free(a);
	REQUIRE_EQ_WITH_TOLERANCE(sum, 1.02, 1e-6, "3x2 vector sum failure");
}

TEST_CASE("vector L2 normalize")
{
	int i;
	ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(1, 10, CCV_32F | CCV_C1, 0, 0);
	for (i = 0; i < 10; i++)
		dmt->data.f32[i] = i;
	ccv_normalize(dmt, (ccv_matrix_t**)&dmt, 0, CCV_L2_NORM);
	float hm[10] = {0.000000, 0.059235, 0.118470, 0.177705, 0.236940, 0.296174, 0.355409, 0.414644, 0.473879, 0.533114};
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(float, hm, dmt->data.f32, 10, 1e-6, "10d vector L2 normalize failure");
	ccv_matrix_free(dmt);
}

TEST_CASE("summed area table without padding")
{
	int i, j;
	ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(5, 4, CCV_8U | CCV_C3, 0, 0);
	unsigned char* ptr = dmt->data.u8;
	for (i = 0; i < dmt->rows; i++)
	{
		for (j = 0; j < dmt->cols; j++)
		{
			ptr[j * 3] = 1;
			ptr[j * 3 + 1] = 2;
			ptr[j * 3 + 2] = 3;
		}
		ptr += dmt->step;
	}
	ccv_dense_matrix_t* b = 0;
	ccv_sat(dmt, &b, 0, CCV_NO_PADDING);
	int sat[60] = {  1,  2,  3,  2,  4,  6,  3,  6,  9,  4,  8, 12,
				     2,  4,  6,  4,  8, 12,  6, 12, 18,  8, 16, 24,
				     3,  6,  9,  6, 12, 18,  9, 18, 27, 12, 24, 36,
				     4,  8, 12,  8, 16, 24, 12, 24, 36, 16, 32, 48,
				     5, 10, 15, 10, 20, 30, 15, 30, 45, 20, 40, 60 };
	REQUIRE_ARRAY_EQ(int, sat, b->data.i32, 60, "4x5 matrix summed area table computation error");
	ccv_matrix_free(dmt);
	ccv_matrix_free(b);
}

TEST_CASE("summed area table with padding")
{
	int i, j;
	ccv_dense_matrix_t* dmt = ccv_dense_matrix_new(5, 3, CCV_8U | CCV_C3, 0, 0);
	unsigned char* ptr = dmt->data.u8;
	for (i = 0; i < dmt->rows; i++)
	{
		for (j = 0; j < dmt->cols; j++)
		{
			ptr[j * 3] = 1;
			ptr[j * 3 + 1] = 2;
			ptr[j * 3 + 2] = 3;
		}
		ptr += dmt->step;
	}
	ccv_dense_matrix_t* b = 0;
	ccv_sat(dmt, &b, 0, CCV_PADDING_ZERO);
	int sat[72] = {  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
					 0,  0,  0,  1,  2,  3,  2,  4,  6,  3,  6,  9,
				     0,  0,  0,  2,  4,  6,  4,  8, 12,  6, 12, 18,
				     0,  0,  0,  3,  6,  9,  6, 12, 18,  9, 18, 27,
				     0,  0,  0,  4,  8, 12,  8, 16, 24, 12, 24, 36,
				     0,  0,  0,  5, 10, 15, 10, 20, 30, 15, 30, 45, };
	REQUIRE_ARRAY_EQ(int, sat, b->data.i32, 72, "3x5 matrix summed area table (with padding) computation error");
	ccv_matrix_free(dmt);
	ccv_matrix_free(b);
}

#include "case_main.h"
