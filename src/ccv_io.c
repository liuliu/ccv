#include "ccv.h"
#include <setjmp.h>
#include <jpeglib.h>

ccv_dense_matrix_t* __ccv_unserialize_jpeg_file(const char* file)
{
	FILE* in = fopen(file, "rb");
}

ccv_dense_matrix_t* ccv_unserialize(const char* in, int type)
{
	switch (type)
	{
		case CCV_SERIAL_JPEG_FILE:
			break;
	}
}

int ccv_serialize(ccv_dense_matrix_t* mat, char* out, int* len, int type)
{
}
