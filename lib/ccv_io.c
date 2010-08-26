#include "ccv.h"
#ifdef HAVE_LIBJPEG
#include <jpeglib.h>
#endif
#ifdef HAVE_LIBPNG
#include <libpng/png.h>
#endif
#ifdef HAVE_LIBJPEG
#include "io/__ccv_io_libjpeg.c"
#endif
#ifdef HAVE_LIBPNG
#include "io/__ccv_io_libpng.c"
#endif
#include "io/__ccv_io_bmp.c"

void ccv_unserialize(const char* in, ccv_dense_matrix_t** x, int type)
{
	FILE* fd;
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : 0;
	if (type & CCV_SERIAL_ANY_FILE)
		fd = fopen(in, "rb");
	if ((type & 0XFF) == CCV_SERIAL_ANY_FILE)
	{
		unsigned char sig[8];
		(void) fread(sig, 1, 8, fd);
		if (memcmp(sig, "\x89\x50\x4e\x47\xd\xa\x1a\xa", 8) == 0)
			type = CCV_SERIAL_PNG_FILE;
		else if (memcmp(sig, "\xff\xd8\xff", 3) == 0)
			type = CCV_SERIAL_JPEG_FILE;
		else if (memcmp(sig, "BM", 2) == 0)
			type = CCV_SERIAL_BMP_FILE;
		fseek(fd, 0, SEEK_SET);
	}
	switch (type & 0XFF)
	{
#ifdef HAVE_LIBJPEG
		case CCV_SERIAL_JPEG_FILE:
			__ccv_unserialize_jpeg_fd(fd, x, ctype);
			if (*x != 0)
				(*x)->sig = ccv_matrix_generate_signature((char*) (*x)->data.ptr, (*x)->rows * (*x)->step, 0);
			break;
#endif
#ifdef HAVE_LIBPNG
		case CCV_SERIAL_PNG_FILE:
			__ccv_unserialize_png_fd(fd, x, ctype);
			if (*x != 0)
				(*x)->sig = ccv_matrix_generate_signature((char*) (*x)->data.ptr, (*x)->rows * (*x)->step, 0);
			break;
#endif
		case CCV_SERIAL_BMP_FILE:
			__ccv_unserialize_bmp_fd(fd, x, ctype);
			if (*x != 0)
				(*x)->sig = ccv_matrix_generate_signature((char*) (*x)->data.ptr, (*x)->rows * (*x)->step, 0);
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
#ifdef HAVE_LIBJPEG
		case CCV_SERIAL_JPEG_FILE:
			__ccv_serialize_jpeg_fd(mat, fd, conf);
			*len = 0;
			break;
#endif
#ifdef HAVE_LIBPNG
		case CCV_SERIAL_PNG_FILE:
			__ccv_serialize_png_fd(mat, fd, conf);
			*len = 0;
			break;
#endif
	}
	if (type & CCV_SERIAL_ANY_FILE)
		fclose(fd);
	return CCV_SERIAL_FINAL;
}
