static void _ccv_read_rgb_raw(ccv_dense_matrix_t** x, const void* data, int type, int rows, int cols, int scanline)
{
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : CCV_8U | CCV_C3;
	ccv_dense_matrix_t* dx = *x = ccv_dense_matrix_new(rows, cols, ctype, 0, 0);
	int i, j;
	unsigned char* x_ptr = dx->data.u8;
	unsigned char* a_ptr = (unsigned char*)data;
	if (type & CCV_IO_GRAY)
	{
		for (i = 0; i < rows; i++)
		{
			for (j = 0; j < cols; j++)
				x_ptr[j] = (unsigned char)((a_ptr[j * 3] * 6969 + a_ptr[j * 3 + 1] * 23434 + a_ptr[j * 3 + 2] * 2365) >> 15);
			a_ptr += scanline;
			x_ptr += dx->step;
		}
	} else {
		for (i = 0; i < rows; i++)
		{
			memcpy(x_ptr, a_ptr, cols * 3);
			a_ptr += scanline;
			x_ptr += dx->step;
		}
	}
}

static void _ccv_read_gray_raw(ccv_dense_matrix_t** x, const void* data, int type, int rows, int cols, int scanline)
{
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : CCV_8U | CCV_C1;
	ccv_dense_matrix_t* dx = *x = ccv_dense_matrix_new(rows, cols, ctype, 0, 0);
	int i, j;
	unsigned char* x_ptr = dx->data.u8;
	unsigned char* a_ptr = (unsigned char*)data;
	if (type & CCV_IO_GRAY)
	{
		for (i = 0; i < rows; i++)
		{
			memcpy(x_ptr, a_ptr, cols);
			a_ptr += scanline;
			x_ptr += dx->step;
		}
	} else {
		for (i = 0; i < rows; i++)
		{
			for (j = 0; j < cols; j++)
				x_ptr[j * 3] = x_ptr[j * 3 + 1] = x_ptr[j * 3 + 2] = a_ptr[j];
			a_ptr += scanline;
			x_ptr += dx->step;
		}
	}
}
