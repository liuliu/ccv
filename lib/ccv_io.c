#include "ccv.h"
#include "ccv_internal.h"
#ifdef HAVE_LIBJPEG
#include <jpeglib.h>
#endif

#ifdef HAVE_LIBPNG
  #ifdef __APPLE__
    #include "TargetConditionals.h"
    #if TARGET_OS_IPHONE
         // iOS
    #elif TARGET_IPHONE_SIMULATOR
        // iOS Simulator
    #elif TARGET_OS_MAC
      #include <zlib.h>
      #include <png.h>
    #else
        // Unsupported platform
    #endif
  #else
    #include <zlib.h>
    #include <png.h>
  #endif
#endif

#ifdef HAVE_LIBJPEG
#include "io/_ccv_io_libjpeg.c"
#endif
#ifdef HAVE_LIBPNG
#include "io/_ccv_io_libpng.c"
#endif
#include "io/_ccv_io_bmp.c"
#include "io/_ccv_io_binary.c"

int ccv_read(const char* in, ccv_dense_matrix_t** x, int type)
{
	FILE* fd = 0;
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : 0;
	if (type & CCV_IO_ANY_FILE)
	{
		fd = fopen(in, "rb");
		if (!fd)
			return CCV_IO_ERROR;
	}
	if ((type & 0XFF) == CCV_IO_ANY_FILE)
	{
		unsigned char sig[8];
		(void) fread(sig, 1, 8, fd);
		if (memcmp(sig, "\x89\x50\x4e\x47\xd\xa\x1a\xa", 8) == 0)
			type = CCV_IO_PNG_FILE;
		else if (memcmp(sig, "\xff\xd8\xff", 3) == 0)
			type = CCV_IO_JPEG_FILE;
		else if (memcmp(sig, "BM", 2) == 0)
			type = CCV_IO_BMP_FILE;
		else if (memcmp(sig, "CCVBINDM", 8) == 0)
			type = CCV_IO_BINARY_FILE;
		fseek(fd, 0, SEEK_SET);
	}
	switch (type & 0XFF)
	{
#ifdef HAVE_LIBJPEG
		case CCV_IO_JPEG_FILE:
			_ccv_read_jpeg_fd(fd, x, ctype);
			break;
#endif
#ifdef HAVE_LIBPNG
		case CCV_IO_PNG_FILE:
			_ccv_read_png_fd(fd, x, ctype);
			break;
#endif
		case CCV_IO_BMP_FILE:
			_ccv_read_bmp_fd(fd, x, ctype);
			break;
		case CCV_IO_BINARY_FILE:
			_ccv_read_binary_fd(fd, x, ctype);
	}
	if (*x != 0)
		ccv_make_matrix_immutable(*x);
	if (type & CCV_IO_ANY_FILE)
		fclose(fd);
	return CCV_IO_FINAL;
}

int ccv_write(ccv_dense_matrix_t* mat, char* out, int* len, int type, void* conf)
{
	FILE* fd = 0;
	if (type & CCV_IO_ANY_FILE)
	{
		fd = fopen(out, "wb");
		if (!fd)
			return CCV_IO_ERROR;
	}
	switch (type)
	{
#ifdef HAVE_LIBJPEG
		case CCV_IO_JPEG_FILE:
			_ccv_write_jpeg_fd(mat, fd, conf);
			if (len != 0)
				*len = 0;
			break;
#endif
#ifdef HAVE_LIBPNG
		case CCV_IO_PNG_FILE:
			_ccv_write_png_fd(mat, fd, conf);
			if (len != 0)
				*len = 0;
			break;
#endif
		case CCV_IO_BINARY_FILE:
			_ccv_write_binary_fd(mat, fd, conf);
			if (len != 0)
				*len = 0;
			break;
	}
	if (type & CCV_IO_ANY_FILE)
		fclose(fd);
	return CCV_IO_FINAL;
}
