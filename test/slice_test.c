#include "ccv.h"

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* image = NULL;
	ccv_unserialize(argv[1], &image, CCV_SERIAL_ANY_FILE);
	ccv_dense_matrix_t* b = NULL;
	ccv_slice(image, &b, 33, 41, 111, 91);
	int len, quality = 95;
	ccv_serialize(b, argv[2], &len, CCV_SERIAL_JPEG_FILE, &quality);
	ccv_matrix_free(image);
	ccv_matrix_free(b);
	ccv_garbage_collect();
	return 0;
}
