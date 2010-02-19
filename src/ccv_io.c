#include "ccv.h"
#include <setjmp.h>
#include <jpeglib.h>

#include "io/__ccv_io_jpeg.c"

void ccv_unserialize(const char* in, ccv_dense_matrix_t** x, int type)
{
	switch (type)
	{
		case CCV_SERIAL_JPEG_FILE:
			__ccv_unserialize_jpeg_file(in, x);
			ccv_matrix_generate_signature((char*) (*x)->data.ptr, (*x)->rows * (*x)->step, (*x)->sig, NULL);
			break;
	}
}

int ccv_serialize(ccv_dense_matrix_t* mat, char* out, int* len, int type)
{
	return 0;
}
