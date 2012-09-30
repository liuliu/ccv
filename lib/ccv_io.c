#include "ccv.h"
#include "ccv_internal.h"
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
#include <libpng/png.h>
#endif
#include "io/_ccv_io_libpng.c"
#endif
#ifdef HAVE_LIBJPEG
#include <jpeglib.h>
#include "io/_ccv_io_libjpeg.c"
#endif
#include "io/_ccv_io_bmp.c"
#include "io/_ccv_io_binary.c"
#include "io/_ccv_io_raw.c"

static int _ccv_read_and_close_fd(FILE* fd, ccv_dense_matrix_t** x, int type)
{
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : 0;
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

static int _ccv_read_raw(ccv_dense_matrix_t** x, void* data, int type, int rows, int cols, int scanline)
{
	assert(rows > 0 && cols > 0 && scanline > 0);
	if (type & CCV_IO_NO_COPY)
	{
		// there is no conversion that we can apply if it is NO_COPY mode
		assert((type && 0xFF) == CCV_IO_ANY_RAW);
		*x = ccv_dense_matrix_new(rows, cols, type, data, 0);
		(*x)->step = scanline;
	} else {
		switch (type & 0xFF)
		{
			case CCV_IO_RGB_RAW:
				_ccv_read_rgb_raw(x, data, type, rows, cols, scanline);
				break;
			case CCV_IO_RGBA_RAW:
				break;
			case CCV_IO_ARGB_RAW:
				break;
			case CCV_IO_BGR_RAW:
				break;
			case CCV_IO_BGRA_RAW:
				break;
			case CCV_IO_ABGR_RAW:
				break;
			case CCV_IO_GRAY_RAW:
				_ccv_read_gray_raw(x, data, type, rows, cols, scanline);
				break;
		}
	}
	if (*x != 0)
		ccv_make_matrix_immutable(*x);
	return CCV_IO_FINAL;
}

int ccv_read_impl(const char* in, ccv_dense_matrix_t** x, int type, int rows, int cols, int scanline)
{
	FILE* fd = 0;
	if (type & CCV_IO_ANY_FILE)
	{
		fd = fopen(in, "rb");
		if (!fd)
			return CCV_IO_ERROR;
		return _ccv_read_and_close_fd(fd, x, type);
	} else if (type & CCV_IO_ANY_STREAM) {
		assert(rows > 8);
		assert((type & 0xFF) == CCV_IO_DEFLATE_STREAM); // deflate stream (compressed stream) is not supported yet
		fd = fmemopen((void*)in, (size_t)rows, "rb");
		if (!fd)
			return CCV_IO_ERROR;
		type = (type & ~0x10) | 0x20;
		return _ccv_read_and_close_fd(fd, x, type);
	} else if (type & CCV_IO_ANY_RAW) {
		return _ccv_read_raw(x, (void*)in /* it can be modifiable if it is NO_COPY mode */, type, rows, cols, scanline);
	}
	return CCV_IO_UNKNOWN;
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
