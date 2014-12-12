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
#include <zlib.h>
#include <png.h>
#endif
#include "io/_ccv_io_libpng.c"
#endif
#ifdef HAVE_LIBJPEG
#include <jpeglib.h>
#include "io/_ccv_io_libjpeg.c"
#endif
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <sys/param.h>
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
		// NO_COPY mode generate an "unreusable" matrix, which requires you to
		// manually release its data block (which is, in fact the same data
		// block you passed in)
		int ctype = CCV_8U | CCV_C1;
		switch (type & 0xFF)
		{
			case CCV_IO_RGB_RAW:
			case CCV_IO_BGR_RAW:
				ctype = CCV_8U | CCV_C3;
				break;
			case CCV_IO_RGBA_RAW:
			case CCV_IO_ARGB_RAW:
			case CCV_IO_BGRA_RAW:
			case CCV_IO_ABGR_RAW:
				ctype = CCV_8U | CCV_C4;
				break;
			case CCV_IO_GRAY_RAW:
			default:
				/* default one */
				break;
		}
		*x = ccv_dense_matrix_new(rows, cols, ctype | CCV_NO_DATA_ALLOC, data, 0);
		(*x)->step = scanline;
	} else {
		switch (type & 0xFF)
		{
			case CCV_IO_RGB_RAW:
				_ccv_read_rgb_raw(x, data, type, rows, cols, scanline);
				break;
			case CCV_IO_RGBA_RAW:
				_ccv_read_rgba_raw(x, data, type, rows, cols, scanline);
				break;
			case CCV_IO_ARGB_RAW:
				_ccv_read_argb_raw(x, data, type, rows, cols, scanline);
				break;
			case CCV_IO_BGR_RAW:
				_ccv_read_bgr_raw(x, data, type, rows, cols, scanline);
				break;
			case CCV_IO_BGRA_RAW:
				_ccv_read_bgra_raw(x, data, type, rows, cols, scanline);
				break;
			case CCV_IO_ABGR_RAW:
				_ccv_read_abgr_raw(x, data, type, rows, cols, scanline);
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

#if defined(__APPLE__) || defined(BSD)
typedef struct {
	char* buffer;
	fpos_t pos;
	size_t size;
} ccv_io_mem_t;

static int readfn(void* context, char* buf, int size)
{
	ccv_io_mem_t* mem = (ccv_io_mem_t*)context;
	if (size + mem->pos > mem->size)
		size = mem->size - mem->pos;
	memcpy(buf, mem->buffer + mem->pos, size);
	mem->pos += size;
	return size;
}

static fpos_t seekfn(void* context, fpos_t off, int whence)
{
	ccv_io_mem_t* mem = (ccv_io_mem_t*)context;
	fpos_t pos;
	switch (whence)
	{
		case SEEK_SET:
			pos = off;
			break;
		case SEEK_CUR:
			pos = mem->pos + off;
			break;
		case SEEK_END:
			pos = mem->size + off;
			break;
	}
	if (pos >= mem->size)
		return -1;
	mem->pos = pos;
	return pos;
}
#endif

int ccv_read_impl(const void* in, ccv_dense_matrix_t** x, int type, int rows, int cols, int scanline)
{
	FILE* fd = 0;
	if (type & CCV_IO_ANY_FILE)
	{
		assert(rows == 0 && cols == 0 && scanline == 0);
		fd = fopen((const char*)in, "rb");
		if (!fd)
			return CCV_IO_ERROR;
		return _ccv_read_and_close_fd(fd, x, type);
	} else if (type & CCV_IO_ANY_STREAM) {
		assert(rows > 8 && cols == 0 && scanline == 0);
		assert((type & 0xFF) != CCV_IO_DEFLATE_STREAM); // deflate stream (compressed stream) is not supported yet
#if _XOPEN_SOURCE >= 700 || _POSIX_C_SOURCE >= 200809L || defined(__APPLE__) || defined(BSD)
		// this is only supported by glibc
#if _XOPEN_SOURCE >= 700 || _POSIX_C_SOURCE >= 200809L
		fd = fmemopen((void*)in, (size_t)rows, "rb");
#else
		ccv_io_mem_t mem = {
			.size = rows,
			.pos = 0,
			.buffer = (char*)in,
		};
		fd = funopen(&mem, readfn, 0, seekfn, 0);
#endif
		if (!fd)
			return CCV_IO_ERROR;
		// mimicking itself as a "file"
		type = (type & ~0x10) | 0x20;
		return _ccv_read_and_close_fd(fd, x, type);
#endif
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
		case CCV_IO_JPEG_FILE:
#ifdef HAVE_LIBJPEG
			_ccv_write_jpeg_fd(mat, fd, conf);
			if (len != 0)
				*len = 0;
#else
			assert(0 && "ccv_write requires libjpeg support for JPEG format");
#endif
			break;
		case CCV_IO_PNG_FILE:
#ifdef HAVE_LIBPNG
			_ccv_write_png_fd(mat, fd, conf);
			if (len != 0)
				*len = 0;
#else
			assert(0 && "ccv_write requires libpng support for PNG format");
#endif
			break;
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
