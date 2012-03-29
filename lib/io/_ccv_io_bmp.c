static void _ccv_read_bmp_fd(FILE* in, ccv_dense_matrix_t** x, int type)
{
	fseek(in, 10, SEEK_SET);
	int offset;
	(void) fread(&offset, 4, 1, in);
	int size;
	(void) fread(&size, 4, 1, in);
	int width = 0, height = 0, bpp = 0, rle_code = 0, clrused = 0;
	if (size >= 36)
	{
		(void) fread(&width, 4, 1, in);
		(void) fread(&height, 4, 1, in);
		(void) fread(&bpp, 4, 1, in);
		bpp = bpp >> 16;
		(void) fread(&rle_code, 4, 1, in);
		fseek(in, 12, SEEK_CUR);
		(void) fread(&clrused, 4, 1, in);
		fseek(in, size - 36, SEEK_CUR);
		/* only support 24-bit bmp */
	} else if (size == 12) {
		(void) fread(&width, 4, 1, in);
		(void) fread(&height, 4, 1, in);
		(void) fread(&bpp, 4, 1, in);
		bpp = bpp >> 16;
		/* TODO: not finished */
	}
	if (width == 0 || height == 0 || bpp == 0)
		return;
	ccv_dense_matrix_t* im = *x;
	if (im == 0)
		*x = im = ccv_dense_matrix_new(height, width, (type) ? type : CCV_8U | ((bpp > 8) ? CCV_C3 : CCV_C1), 0, 0);
	fseek(in, offset, SEEK_SET);
	int i, j;
	unsigned char* ptr = im->data.u8 + (im->rows - 1) * im->step;
	if ((bpp == 8 && CCV_GET_CHANNEL(im->type) == CCV_C1) || (bpp == 24 && CCV_GET_CHANNEL(im->type) == CCV_C3))
	{
		if (CCV_GET_CHANNEL(im->type) == CCV_C1)
		{
			for (i = 0; i < im->rows; i++)
			{
				(void) fread(ptr, 1, im->step, in);
				ptr -= im->step;
			}
		} else {
			for (i = 0; i < im->rows; i++)
			{
				(void) fread(ptr, 1, im->step, in);
				for (j = 0; j < im->cols * 3; j += 3)
				{
					unsigned char t = ptr[j];
					ptr[j] = ptr[j + 2];
					ptr[j + 2] = t;
				}
				ptr -= im->step;
			}
		}
	} else {
		if (bpp == 24 && CCV_GET_CHANNEL(im->type) == CCV_C1)
		{
			int bufstep = (im->cols * 3 + 3) & -4;
			unsigned char* buffer = (unsigned char*)alloca(bufstep);
			for (i = 0; i < im->rows; i++)
			{
				(void) fread(buffer, 1, bufstep, in);
				unsigned char* rgb = buffer;
				unsigned char* g = ptr;
				for(j = 0; j < im->cols; j++, rgb += 3, g++)
					*g = (unsigned char)((rgb[2] * 6969 + rgb[1] * 23434 + rgb[0] * 2365) >> 15);
				ptr -= im->step;
			}
		} else if (bpp == 8 && CCV_GET_CHANNEL(im->type) == CCV_C3) {
			int bufstep = (im->cols + 3) & -4;
			unsigned char* buffer = (unsigned char*)alloca(bufstep);
			for (i = 0; i < im->rows; i++)
			{
				(void) fread(buffer, 1, bufstep, in);
				unsigned char* g = buffer;
				unsigned char* rgb = ptr;
				for(j = 0; j < im->cols; j++, rgb += 3, g++)
					rgb[2] = rgb[1] = rgb[0] = *g;
				ptr -= im->step;
			}
		}
	}
}
