#include <setjmp.h>

typedef struct ccv_jpeg_error_mgr_t
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

/***************************************************************************
 * following code is for supporting MJPEG image files
 * based on a message of Laurent Pinchart on the video4linux mailing list
 ***************************************************************************/

/* JPEG DHT Segment for YCrCb omitted from MJPEG data */
static
unsigned char _ccv_jpeg_odml_dht[0x1a4] = {
	0xff, 0xc4, 0x01, 0xa2,

	0x00, 0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00,
	0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,

	0x01, 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
	0x00, 0x00, 0x00, 0x00, 0x00,
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b,

	0x10, 0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04,
	0x04, 0x00, 0x00, 0x01, 0x7d,
	0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
	0x13, 0x51, 0x61, 0x07,
	0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1,
	0x15, 0x52, 0xd1, 0xf0,
	0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a,
	0x25, 0x26, 0x27, 0x28,
	0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
	0x46, 0x47, 0x48, 0x49,
	0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65,
	0x66, 0x67, 0x68, 0x69,
	0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85,
	0x86, 0x87, 0x88, 0x89,
	0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3,
	0xa4, 0xa5, 0xa6, 0xa7,
	0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba,
	0xc2, 0xc3, 0xc4, 0xc5,
	0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8,
	0xd9, 0xda, 0xe1, 0xe2,
	0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4,
	0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa,

	0x11, 0x00, 0x02, 0x01, 0x02, 0x04, 0x04, 0x03, 0x04, 0x07, 0x05, 0x04,
	0x04, 0x00, 0x01, 0x02, 0x77,
	0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41,
	0x51, 0x07, 0x61, 0x71,
	0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09,
	0x23, 0x33, 0x52, 0xf0,
	0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17,
	0x18, 0x19, 0x1a, 0x26,
	0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
	0x45, 0x46, 0x47, 0x48,
	0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
	0x65, 0x66, 0x67, 0x68,
	0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83,
	0x84, 0x85, 0x86, 0x87,
	0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a,
	0xa2, 0xa3, 0xa4, 0xa5,
	0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8,
	0xb9, 0xba, 0xc2, 0xc3,
	0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6,
	0xd7, 0xd8, 0xd9, 0xda,
	0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4,
	0xf5, 0xf6, 0xf7, 0xf8,
	0xf9, 0xfa
};

/*
 * Parse the DHT table.
 * This code comes from jpeg6b (jdmarker.c).
 */
static int _ccv_jpeg_load_dht(struct jpeg_decompress_struct *info, unsigned char *dht, JHUFF_TBL *ac_tables[], JHUFF_TBL *dc_tables[])
{
	unsigned int length = (dht[2] << 8) + dht[3] - 2;
	unsigned int pos = 4;
	unsigned int count, i;
	int index;

	JHUFF_TBL **hufftbl;
	unsigned char bits[17];
	unsigned char huffval[256];

	while (length > 16)
	{
		bits[0] = 0;
		index = dht[pos++];
		count = 0;
		for (i = 1; i <= 16; ++i)
		{
			bits[i] = dht[pos++];
			count += bits[i];
		}
		length -= 17;

		if (count > 256 || count > length)
			return -1;

		for (i = 0; i < count; ++i)
			huffval[i] = dht[pos++];
		length -= count;

		if (index & 0x10)
		{
			index -= 0x10;
			hufftbl = &ac_tables[index];
		}
		else
			hufftbl = &dc_tables[index];

		if (index < 0 || index >= NUM_HUFF_TBLS)
			return -1;

		if (*hufftbl == 0)
			*hufftbl = jpeg_alloc_huff_table ((j_common_ptr)info);
		if (*hufftbl == 0)
			return -1;

		memcpy((*hufftbl)->bits, bits, sizeof (*hufftbl)->bits);
		memcpy((*hufftbl)->huffval, huffval, sizeof (*hufftbl)->huffval);
	}

	if (length != 0)
		return -1;

	return 0;
}

/***************************************************************************
 * end of code for supportting MJPEG image files
 * based on a message of Laurent Pinchart on the video4linux mailing list
 ***************************************************************************/

static void _ccv_read_jpeg_fd(FILE* in, ccv_dense_matrix_t** x, int type)
{
	struct jpeg_decompress_struct cinfo;
	struct ccv_jpeg_error_mgr_t jerr;
	JSAMPARRAY buffer;
	int row_stride;
	cinfo.err = jpeg_std_error(&jerr.pub);
	jerr.pub.error_exit = error_exit;
	if (setjmp(jerr.setjmp_buffer))
	{
		jpeg_destroy_decompress(&cinfo);
		return;
	}
	jpeg_create_decompress(&cinfo);

	jpeg_stdio_src(&cinfo, in);

	jpeg_read_header(&cinfo, TRUE);
	
	ccv_dense_matrix_t* im = *x;
	if (im == 0)
		*x = im = ccv_dense_matrix_new(cinfo.image_height, cinfo.image_width, (type) ? type : CCV_8U | ((cinfo.num_components > 1) ? CCV_C3 : CCV_C1), 0, 0);

	/* yes, this is a mjpeg image format, so load the correct huffman table */
	if (cinfo.ac_huff_tbl_ptrs[0] == 0 && cinfo.ac_huff_tbl_ptrs[1] == 0 && cinfo.dc_huff_tbl_ptrs[0] == 0 && cinfo.dc_huff_tbl_ptrs[1] == 0)
		_ccv_jpeg_load_dht(&cinfo, _ccv_jpeg_odml_dht, cinfo.ac_huff_tbl_ptrs, cinfo.dc_huff_tbl_ptrs);

	if(cinfo.num_components != 4)
	{
		if (cinfo.num_components > 1)
		{
			cinfo.out_color_space = JCS_RGB;
			cinfo.out_color_components = 3;
		} else {
			cinfo.out_color_space = JCS_GRAYSCALE;
			cinfo.out_color_components = 1;
		}
	} else {
		cinfo.out_color_space = JCS_CMYK;
		cinfo.out_color_components = 4;
	}

	jpeg_start_decompress(&cinfo);
	row_stride = cinfo.output_width * 4;
	buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

	unsigned char* ptr = im->data.u8;
	int i;
	int ch = CCV_GET_CHANNEL(im->type);
	if(cinfo.num_components != 4)
	{
		if ((cinfo.num_components > 1 && ch == CCV_C3) || (cinfo.num_components == 1 && ch == CCV_C1))
		{
			/* no format coversion, direct copy */
			if (im->cols * ch < im->step)
			{
				size_t extra = im->step - im->cols * ch;
				// empty the padding
				while (cinfo.output_scanline < cinfo.output_height)
				{
					jpeg_read_scanlines(&cinfo, buffer, 1);
					memcpy(ptr, buffer[0], im->step);
					memset(ptr + im->cols * ch, 0, extra);
					ptr += im->step;
				}
			} else {
				while (cinfo.output_scanline < cinfo.output_height)
				{
					jpeg_read_scanlines(&cinfo, buffer, 1);
					memcpy(ptr, buffer[0], im->step);
					ptr += im->step;
				}
			}
		} else {
			if (cinfo.num_components > 1 && CCV_GET_CHANNEL(im->type) == CCV_C1)
			{
				/* RGB to gray */
				while (cinfo.output_scanline < cinfo.output_height)
				{
					jpeg_read_scanlines(&cinfo, buffer, 1);
					unsigned char* g = ptr;
					unsigned char* rgb = (unsigned char*)buffer[0];
					for(i = 0; i < im->cols; i++, rgb += 3, g++)
						*g = (unsigned char)((rgb[0] * 6969 + rgb[1] * 23434 + rgb[2] * 2365) >> 15);
					ptr += im->step;
				}
			} else if (cinfo.num_components == 1 && CCV_GET_CHANNEL(im->type) == CCV_C3) {
				/* gray to RGB */
				while (cinfo.output_scanline < cinfo.output_height)
				{
					jpeg_read_scanlines(&cinfo, buffer, 1);
					unsigned char* g = (unsigned char*)buffer[0];
					unsigned char* rgb = ptr;
					for(i = 0; i < im->cols; i++, rgb += 3, g++)
						rgb[0] = rgb[1] = rgb[2] = *g;
					ptr += im->step;
				}
			}
			// empty out the padding
			if (im->cols * ch < im->step)
			{
				size_t extra = im->step - im->cols * ch;
				unsigned char* ptr = im->data.u8 + im->cols * ch;
				for (i = 0; i < im->rows; i++, ptr += im->step)
					memset(ptr, 0, extra);
			}
		}
	} else {
		if (CCV_GET_CHANNEL(im->type) == CCV_C1)
		{
			/* CMYK to gray */
			while (cinfo.output_scanline < cinfo.output_height)
			{
				jpeg_read_scanlines(&cinfo, buffer, 1);
				unsigned char* cmyk = (unsigned char*)buffer[0];
				unsigned char* g = ptr;
				for(i = 0; i < im->cols; i++, g++, cmyk += 4)
				{
					int c = cmyk[0], m = cmyk[1], y = cmyk[2], k = cmyk[3];
					c = k - ((255 - c) * k >> 8);
					m = k - ((255 - m) * k >> 8);
					y = k - ((255 - y) * k >> 8);
					*g = (unsigned char)((c * 6969 + m * 23434 + y * 2365) >> 15);
				}
				ptr += im->step;
			}
		} else if (CCV_GET_CHANNEL(im->type) == CCV_C3) {
			/* CMYK to RGB */
			while (cinfo.output_scanline < cinfo.output_height)
			{
				jpeg_read_scanlines(&cinfo, buffer, 1);
				unsigned char* cmyk = (unsigned char*)buffer[0];
				unsigned char* rgb = ptr;
				for(i = 0; i < im->cols; i++, rgb += 3, cmyk += 4)
				{
					int c = cmyk[0], m = cmyk[1], y = cmyk[2], k = cmyk[3];
					c = k - ((255 - c) * k >> 8);
					m = k - ((255 - m) * k >> 8);
					y = k - ((255 - y) * k >> 8);
					rgb[0] = (unsigned char)c;
					rgb[1] = (unsigned char)m;
					rgb[2] = (unsigned char)y;
				}
				ptr += im->step;
			}
		}
		// empty out the padding
		if (im->cols * ch < im->step)
		{
			size_t extra = im->step - im->cols * ch;
			unsigned char* ptr = im->data.u8 + im->cols * ch;
			for (i = 0; i < im->rows; i++, ptr += im->step)
				memset(ptr, 0, extra);
		}
	}

	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
}

static void _ccv_write_jpeg_fd(ccv_dense_matrix_t* mat, FILE* fd, void* conf)
{
	struct jpeg_compress_struct cinfo;
	struct ccv_jpeg_error_mgr_t jerr;
	jpeg_create_compress(&cinfo);
    cinfo.err = jpeg_std_error(&jerr.pub);
	jerr.pub.error_exit = error_exit;
	jpeg_stdio_dest(&cinfo, fd);
	if (setjmp(jerr.setjmp_buffer))
	{
		jpeg_destroy_compress(&cinfo);
		return;
	}
	cinfo.image_width = mat->cols;
	cinfo.image_height = mat->rows;
	cinfo.input_components = (CCV_GET_CHANNEL(mat->type) == CCV_C1) ? 1 : 3;
	cinfo.in_color_space = (CCV_GET_CHANNEL(mat->type) == CCV_C1) ? JCS_GRAYSCALE : JCS_RGB;
	jpeg_set_defaults(&cinfo);
	if (conf == 0)
		jpeg_set_quality(&cinfo, 95, TRUE);
	else
		jpeg_set_quality(&cinfo, *(int*)conf, TRUE);
	jpeg_start_compress(&cinfo, TRUE);
	int i;
	unsigned char* ptr = mat->data.u8;
	for (i = 0; i < mat->rows; i++)
	{
		jpeg_write_scanlines(&cinfo, &ptr, 1);
		ptr += mat->step;
	}
	jpeg_finish_compress(&cinfo);
	jpeg_destroy_compress(&cinfo);
}
