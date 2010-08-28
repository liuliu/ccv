#include "ccv.h"

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(3, 2, CCV_64F | CCV_C1, NULL, NULL);
	a->data.db[0] = 0.11;
	a->data.db[1] = 0.12;
	a->data.db[2] = 0.13;
	a->data.db[3] = 0.21;
	a->data.db[4] = 0.22;
	a->data.db[5] = 0.23;
	ccv_dense_matrix_t* b = ccv_dense_matrix_new(3, 2, CCV_64F | CCV_C1, NULL, NULL);
	b->data.db[0] = 1011;
	b->data.db[1] = 1012;
	b->data.db[2] = 1021;
	b->data.db[3] = 1022;
	b->data.db[4] = 1031;
	b->data.db[5] = 1032;
	ccv_dense_matrix_t* y = NULL;
	ccv_gemm(a, b, 1, NULL, 0, CCV_A_TRANSPOSE, &y);
	printf("%f %f\n%f %f\n", y->data.db[0], y->data.db[1], y->data.db[2], y->data.db[3]);
	ccv_matrix_free(a);
	ccv_matrix_free(b);
	ccv_matrix_free(y);
	ccv_garbage_collect();
	return 0;
}
