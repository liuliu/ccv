static void _ccv_read_rgb_raw(ccv_dense_matrix_t** x, const void* data, int type, int rows, int cols, int scanline)
{
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : CCV_8U | CCV_C3;
	ccv_dense_matrix_t* dx = *x = ccv_dense_matrix_new(rows, cols, ctype, 0, 0);
	int i, j;
	switch (type & 0xF00)
	{
		case CCV_IO_GRAY:
		{
			unsigned char* g = dx->data.u8;
			unsigned char* rgb = (unsigned char*)data;
			int rgb_padding = scanline - cols * 3;
			assert(rgb_padding >= 0);
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < cols; j++)
					g[j] = (unsigned char)((rgb[0] * 6969 + rgb[1] * 23434 + rgb[2] * 2365) >> 15), rgb += 3;
				rgb += rgb_padding;
				g += dx->step;
			}
			break;
		} 
		case CCV_IO_RGB_COLOR:
		default:
		{
			unsigned char* x_ptr = dx->data.u8;
			unsigned char* a_ptr = (unsigned char*)data;
			assert(scanline >= cols * 3);
			for (i = 0; i < rows; i++)
			{
				memcpy(x_ptr, a_ptr, cols * 3);
				a_ptr += scanline;
				x_ptr += dx->step;
			}
			break;
		}
	}
}

static void _ccv_read_rgba_raw(ccv_dense_matrix_t** x, const void* data, int type, int rows, int cols, int scanline)
{
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : CCV_8U | CCV_C3;
	ccv_dense_matrix_t* dx = *x = ccv_dense_matrix_new(rows, cols, ctype, 0, 0);
	int i, j;
	switch (type & 0xF00)
	{
		case CCV_IO_GRAY:
		{
			unsigned char* g = dx->data.u8;
			unsigned char* rgba = (unsigned char*)data;
			int rgba_padding = scanline - cols * 4;
			assert(rgba_padding >= 0);
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < cols; j++)
					g[j] = (unsigned char)((rgba[0] * 6969 + rgba[1] * 23434 + rgba[2] * 2365) >> 15), rgba += 4;
				rgba += rgba_padding;
				g += dx->step;
			}
			break;
		}
		case CCV_IO_RGB_COLOR:
		{
			unsigned char* rgb = dx->data.u8;
			int rgb_padding = dx->step - cols * 3;
			assert(rgb_padding >= 0);
			unsigned char* rgba = (unsigned char*)data;
			int rgba_padding = scanline - cols * 4;
			assert(rgba_padding >= 0);
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < cols; j++)
					rgb[0] = rgba[0], rgb[1] = rgba[1], rgb[2] = rgba[2],
						rgb += 3, rgba += 4;
				rgba += rgba_padding;
				rgb += rgb_padding;
			}
			break;
		}
		default:
		{
			unsigned char* x_ptr = dx->data.u8;
			unsigned char* a_ptr = (unsigned char*)data;
			assert(scanline >= cols * 4);
			for (i = 0; i < rows; i++)
			{
				memcpy(x_ptr, a_ptr, cols * 4);
				a_ptr += scanline;
				x_ptr += dx->step;
			}
			break;
		}
	}
}

static void _ccv_read_argb_raw(ccv_dense_matrix_t** x, const void* data, int type, int rows, int cols, int scanline)
{
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : CCV_8U | CCV_C3;
	ccv_dense_matrix_t* dx = *x = ccv_dense_matrix_new(rows, cols, ctype, 0, 0);
	int i, j;
	switch (type & 0xF00)
	{
		case CCV_IO_GRAY:
		{
			unsigned char* g = dx->data.u8;
			unsigned char* argb = (unsigned char*)data;
			int argb_padding = scanline - cols * 4;
			assert(argb_padding >= 0);
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < cols; j++)
					g[j] = (unsigned char)((argb[1] * 6969 + argb[2] * 23434 + argb[3] * 2365) >> 15), argb += 4;
				argb += argb_padding;
				g += dx->step;
			}
			break;
		}
		case CCV_IO_RGB_COLOR:
		{
			unsigned char* rgb = dx->data.u8;
			int rgb_padding = dx->step - cols * 3;
			assert(rgb_padding >= 0);
			unsigned char* argb = (unsigned char*)data;
			int argb_padding = scanline - cols * 4;
			assert(argb_padding >= 0);
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < cols; j++)
					rgb[0] = argb[1], rgb[1] = argb[2], rgb[2] = argb[3],
						rgb += 3, argb += 4;
				argb += argb_padding;
				rgb += rgb_padding;
			}
			break;
		}
		default:
		{
			unsigned char* x_ptr = dx->data.u8;
			unsigned char* a_ptr = (unsigned char*)data;
			assert(scanline >= cols * 4);
			for (i = 0; i < rows; i++)
			{
				memcpy(x_ptr, a_ptr, cols * 4);
				a_ptr += scanline;
				x_ptr += dx->step;
			}
			break;
		}
	}
}

static void _ccv_read_bgr_raw(ccv_dense_matrix_t** x, const void* data, int type, int rows, int cols, int scanline)
{
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : CCV_8U | CCV_C3;
	ccv_dense_matrix_t* dx = *x = ccv_dense_matrix_new(rows, cols, ctype, 0, 0);
	int i, j;
	switch (type & 0xF00)
	{
		case CCV_IO_GRAY:
		{
			unsigned char* g = dx->data.u8;
			unsigned char* bgr = (unsigned char*)data;
			int bgr_padding = scanline - cols * 3;
			assert(bgr_padding >= 0);
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < cols; j++)
					g[j] = (unsigned char)((bgr[2] * 6969 + bgr[1] * 23434 + bgr[0] * 2365) >> 15), bgr += 3;
				bgr += bgr_padding;
				g += dx->step;
			}
			break;
		}
		case CCV_IO_RGB_COLOR:
		{
			unsigned char* rgb = dx->data.u8;
			int rgb_padding = dx->step - cols * 3;
			assert(rgb_padding >= 0);
			unsigned char* bgr = (unsigned char*)data;
			int bgr_padding = scanline - cols * 3;
			assert(bgr_padding >= 0);
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < cols; j++)
					rgb[0] = bgr[2], rgb[1] = bgr[1], rgb[2] = bgr[0],
						rgb += 3, bgr += 3;
				bgr += bgr_padding;
				rgb += rgb_padding;
			}
			break;
		}
		default:
		{
			unsigned char* x_ptr = dx->data.u8;
			unsigned char* a_ptr = (unsigned char*)data;
			assert(scanline >= cols * 3);
			for (i = 0; i < rows; i++)
			{
				memcpy(x_ptr, a_ptr, cols * 3);
				a_ptr += scanline;
				x_ptr += dx->step;
			}
			break;
		}
	}
}

static void _ccv_read_bgra_raw(ccv_dense_matrix_t** x, const void* data, int type, int rows, int cols, int scanline)
{
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : CCV_8U | CCV_C3;
	ccv_dense_matrix_t* dx = *x = ccv_dense_matrix_new(rows, cols, ctype, 0, 0);
	int i, j;
	switch (type & 0xF00)
	{
		case CCV_IO_GRAY:
		{
			unsigned char* g = dx->data.u8;
			unsigned char* bgra = (unsigned char*)data;
			int bgra_padding = scanline - cols * 4;
			assert(bgra_padding >= 0);
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < cols; j++)
					g[j] = (unsigned char)((bgra[2] * 6969 + bgra[1] * 23434 + bgra[0] * 2365) >> 15), bgra += 4;
				bgra += bgra_padding;
				g += dx->step;
			}
			break;
		}
		case CCV_IO_RGB_COLOR:
		{
			unsigned char* rgb = dx->data.u8;
			int rgb_padding = dx->step - cols * 3;
			assert(rgb_padding >= 0);
			unsigned char* bgra = (unsigned char*)data;
			int bgra_padding = scanline - cols * 4;
			assert(bgra_padding >= 0);
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < cols; j++)
					rgb[0] = bgra[2], rgb[1] = bgra[1], rgb[2] = bgra[0],
						rgb += 3, bgra += 4;
				bgra += bgra_padding;
				rgb += rgb_padding;
			}
			break;
		}
		default:
		{
			unsigned char* x_ptr = dx->data.u8;
			unsigned char* a_ptr = (unsigned char*)data;
			assert(scanline >= cols * 4);
			for (i = 0; i < rows; i++)
			{
				memcpy(x_ptr, a_ptr, cols * 4);
				a_ptr += scanline;
				x_ptr += dx->step;
			}
			break;
		}
	}
}

static void _ccv_read_abgr_raw(ccv_dense_matrix_t** x, const void* data, int type, int rows, int cols, int scanline)
{
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : CCV_8U | CCV_C3;
	ccv_dense_matrix_t* dx = *x = ccv_dense_matrix_new(rows, cols, ctype, 0, 0);
	int i, j;
	switch (type & 0xF00)
	{
		case CCV_IO_GRAY:
		{
			unsigned char* g = dx->data.u8;
			unsigned char* abgr = (unsigned char*)data;
			int abgr_padding = scanline - cols * 4;
			assert(abgr_padding >= 0);
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < cols; j++)
					g[j] = (unsigned char)((abgr[3] * 6969 + abgr[2] * 23434 + abgr[1] * 2365) >> 15), abgr += 4;
				abgr += abgr_padding;
				g += dx->step;
			}
			break;
		}
		case CCV_IO_RGB_COLOR:
		{
			unsigned char* rgb = dx->data.u8;
			int rgb_padding = dx->step - cols * 3;
			assert(rgb_padding >= 0);
			unsigned char* abgr = (unsigned char*)data;
			int abgr_padding = scanline - cols * 4;
			assert(abgr_padding >= 0);
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < cols; j++)
					rgb[0] = abgr[3], rgb[1] = abgr[2], rgb[2] = abgr[1],
						rgb += 3, abgr += 4;
				abgr += abgr_padding;
				rgb += rgb_padding;
			}
			break;
		}
		default:
		{
			unsigned char* x_ptr = dx->data.u8;
			unsigned char* a_ptr = (unsigned char*)data;
			assert(scanline >= cols * 4);
			for (i = 0; i < rows; i++)
			{
				memcpy(x_ptr, a_ptr, cols * 4);
				a_ptr += scanline;
				x_ptr += dx->step;
			}
			break;
		}
	}
}

static void _ccv_read_gray_raw(ccv_dense_matrix_t** x, const void* data, int type, int rows, int cols, int scanline)
{
	int ctype = (type & 0xF00) ? CCV_8U | ((type & 0xF00) >> 8) : CCV_8U | CCV_C1;
	ccv_dense_matrix_t* dx = *x = ccv_dense_matrix_new(rows, cols, ctype, 0, 0);
	int i, j;
	switch (type & 0xF00)
	{
		case CCV_IO_RGB_COLOR:
		{
			unsigned char* rgb = dx->data.u8;
			unsigned char* g = (unsigned char*)data;
			int rgb_padding = dx->step - cols * 3;
			assert(rgb_padding >= 0);
			for (i = 0; i < rows; i++)
			{
				for (j = 0; j < cols; j++)
					rgb[0] = rgb[1] = rgb[2] = g[j], rgb += 3;
				g += scanline;
				rgb += rgb_padding;
			}
			break;
		}
		case CCV_IO_GRAY:
		default:
		{
			unsigned char* x_ptr = dx->data.u8;
			unsigned char* a_ptr = (unsigned char*)data;
			assert(scanline >= cols);
			for (i = 0; i < rows; i++)
			{
				memcpy(x_ptr, a_ptr, cols);
				a_ptr += scanline;
				x_ptr += dx->step;
			}
			break;
		}
	}
}
