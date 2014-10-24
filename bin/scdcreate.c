#include "ccv.h"
#include <ctype.h>
#include <getopt.h>

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_RGB_COLOR | CCV_IO_ANY_FILE);
	ccv_dense_matrix_t* scd = 0;
	ccv_scd(image, &scd, 0);
	ccv_dense_matrix_t* visualize = 0;
	ccv_visualize(scd, (ccv_matrix_t**)&visualize, 0);
	ccv_write(visualize, argv[2], 0, CCV_IO_PNG_FILE, 0);
	ccv_matrix_free(scd);
	ccv_matrix_free(image);
	return 0;
}
