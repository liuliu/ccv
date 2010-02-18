#include "ccv.h"
#include <setjmp.h>
#include <jpeglib.h>

/* adapted from OpenCV's grfmt_jpeg.cpp file */
typedef struct
{
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
} ccv_jpeg_error_mgr_t;

METHODDEF(void) error_exit(j_common_ptr cinfo)
{
    ccv_jpeg_error_mgr_t* err_mgr = (ccv_jpeg_error_mgr_t*)(cinfo->err);

    /* Return control to the setjmp point */
    longjmp(err_mgr->setjmp_buffer, 1);
}

void __ccv_unserialize_jpeg_file(const char* file, ccv_dense_matrix_t** x)
{
	FILE* in = fopen(file, "rb");
	if (in == NULL)
		return NULL;
	struct jpeg_decompress_struct cinfo;
	struct my_error_mgr jerr;
	JSAMPARRAY buffer;
	int row_stride;
	cinfo.err = jpeg_std_error(&jerr.pub);
	jerr.pub.error_exit = error_exit;
	if (setjmp(jerr.setjmp_buffer))
	{
		jpeg_destroy_decompress(&cinfo);
		fclose(in);
		return NULL;
	}
	jpeg_create_decompress(&cinfo);

	jpeg_stdio_src(&cinfo, in);

	(void) jpeg_read_header(&cinfo, TRUE);
	
	ccv_dense_matrix_t* im = *x;
	if (im == NULL)
		*x = im = ccv_dense_matrix_new(cinfo.output_height, cinfo.output_width, CCV_8U | CCV_C3, NULL, NULL);

	(void) jpeg_start_decompress(&cinfo);
	row_stride = cinfo.output_width * cinfo.output_components;
	buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

	while (cinfo.output_scanline < cinfo.output_height)
	{
		(void) jpeg_read_scanlines(&cinfo, buffer, 1);
		// put_scanline_someplace(buffer[0], row_stride);
	}

	(void) jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);

	fclose(in);
}

void ccv_unserialize(const char* in, ccv_dense_matrix_t** x, int type)
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
