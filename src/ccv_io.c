#include "ccv.h"
#include <jpeglib.h>
#include <libpng/png.h>

#include "io/__ccv_io_libjpeg.c"
#include "io/__ccv_io_libpng.c"

void ccv_unserialize(const char* in, ccv_dense_matrix_t** x, int type)
{
	FILE* fd;
	if (type & CCV_SERIAL_ANY_FILE)
		fd = fopen(in, "rb");
	if (type == CCV_SERIAL_ANY_FILE)
	{
		unsigned char sig[8];
		(void) fread(sig, 1, 8, fd);
		if (memcmp(sig, "\x89\x50\x4e\x47\xd\xa\x1a\xa", 8) == 0)
			type = CCV_SERIAL_PNG_FILE;
		else if (memcmp(sig, "\xff\xd8\xff", 3) == 0)
			type = CCV_SERIAL_JPEG_FILE;
		fseek(fd, 0, SEEK_SET);

	}
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
	if (type & CCV_SERIAL_ANY_FILE)
		fclose(fd);
}

int ccv_serialize(ccv_dense_matrix_t* mat, char* out, int* len, int type, void* conf)
{
	FILE* fd;
	if (type & CCV_SERIAL_ANY_FILE)
		fd = fopen(out, "wb");
	switch (type)
	{
		case CCV_SERIAL_JPEG_FILE:
			__ccv_serialize_jpeg_fd(mat, fd, conf);
			*len = 0;
			break;
		case CCV_SERIAL_PNG_FILE:
			__ccv_serialize_png_fd(mat, fd, conf);
			*len = 0;
			break;
	}
	if (type & CCV_SERIAL_ANY_FILE)
		fclose(fd);
	return CCV_SERIAL_FINAL;
}
