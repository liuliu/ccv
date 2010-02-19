#include "ccv.h"
#include <jpeglib.h>
#include <libpng/png.h>

#include "io/__ccv_io_jpeg.c"
#include "io/__ccv_io_png.c"

void ccv_unserialize(const char* in, ccv_dense_matrix_t** x, int type)
{
	FILE* fd;
	if (type & CCV_SERIAL_ANY_FILE)
		fd = fopen(in, "rb");
	switch (type)
	{
		case CCV_SERIAL_JPEG_FILE:
			__ccv_unserialize_jpeg_fd(fd, x);
			ccv_matrix_generate_signature((char*) (*x)->data.ptr, (*x)->rows * (*x)->step, (*x)->sig, NULL);
			break;
		case CCV_SERIAL_PNG_FILE:
			__ccv_unserialize_png_fd(fd, x);
			ccv_matrix_generate_signature((char*) (*x)->data.ptr, (*x)->rows * (*x)->step, (*x)->sig, NULL);
			break;
	}
	fclose(fd);
}

int ccv_serialize(ccv_dense_matrix_t* mat, char* out, int* len, int type)
{
	return 0;
}
