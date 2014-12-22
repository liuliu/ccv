#include <ccv.h>

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
	ccv_write(image, argv[2], 0, CCV_IO_PNG_FILE, 0);
	return 0;
}
