#include "ccv.h"

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* image = NULL;
	ccv_unserialize(argv[1], image, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(image->rows, image->cols, CCV_32FC1);
	ccv_dense_matrix_t* b = ccv_dense_matrix_new(a->rows, a->cols, CCV_32FC1);
	int i, j;
	return 0;
}
