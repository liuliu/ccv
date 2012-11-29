static void _ccv_read_png_fd(FILE* in, ccv_dense_matrix_t** x, int type)
{
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		png_destroy_read_struct(&png_ptr, &info_ptr, 0);
		return;
	}
	png_init_io(png_ptr, in);
	png_read_info(png_ptr, info_ptr);
	png_uint_32 width, height;
	int bit_depth, color_type;
	png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, 0, 0, 0);

	ccv_dense_matrix_t* im = *x;
	if (im == 0)
		*x = im = ccv_dense_matrix_new((int) height, (int) width, (type) ? type : CCV_8U | (((color_type & PNG_COLOR_MASK_COLOR) == PNG_COLOR_TYPE_GRAY) ? CCV_C1 : CCV_C3), 0, 0);

	png_set_strip_16(png_ptr);
	png_set_strip_alpha(png_ptr);
	if (color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png_ptr);
	if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
		png_set_expand_gray_1_2_4_to_8(png_ptr);
	if (CCV_GET_CHANNEL(im->type) == CCV_C3)
		png_set_gray_to_rgb(png_ptr);
	else if (CCV_GET_CHANNEL(im->type) == CCV_C1)
		png_set_rgb_to_gray(png_ptr, 1, -1, -1);

	png_read_update_info(png_ptr, info_ptr);

	unsigned char** row_vectors = (unsigned char**)alloca(im->rows * sizeof(unsigned char*));
	int i;
	for (i = 0; i < im->rows; i++)
		row_vectors[i] = im->data.u8 + i * im->step;
	png_read_image(png_ptr, row_vectors);
	png_read_end(png_ptr, 0);
	int ch = CCV_GET_CHANNEL(im->type);
	// empty out the padding
	if (im->cols * ch < im->step)
	{
		size_t extra = im->step - im->cols * ch;
		unsigned char* ptr = im->data.u8 + im->cols * ch;
		for (i = 0; i < im->rows; i++, ptr += im->step)
			memset(ptr, 0, extra);
	}

	png_destroy_read_struct(&png_ptr, &info_ptr, 0);
}

static void _ccv_write_png_fd(ccv_dense_matrix_t* mat, FILE* fd, void* conf)
{
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		png_destroy_write_struct(&png_ptr, &info_ptr);
		return;
	}
	png_init_io(png_ptr, fd);
	int compression_level = 0;
	if (conf != 0)
		compression_level = ccv_clamp(*(int*)conf, 0, MAX_MEM_LEVEL);
	if(compression_level > 0)
	{
		png_set_compression_mem_level(png_ptr, compression_level);
	} else {
		// tune parameters for speed
		// (see http://wiki.linuxquestions.org/wiki/Libpng)
		png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
		png_set_compression_level(png_ptr, Z_BEST_SPEED);
	}
	png_set_compression_strategy(png_ptr, Z_HUFFMAN_ONLY);
	png_set_IHDR(png_ptr, info_ptr, mat->cols, mat->rows, (mat->type & CCV_8U) ? 8 : 16, (CCV_GET_CHANNEL(mat->type) == CCV_C1) ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	
	unsigned char** row_vectors = (unsigned char**)alloca(mat->rows * sizeof(unsigned char*));
	int i;
	for (i = 0; i < mat->rows; i++)
		row_vectors[i] = mat->data.u8 + i * mat->step;
	png_write_info(png_ptr, info_ptr);
	png_write_image(png_ptr, row_vectors);
	png_write_end(png_ptr, info_ptr);
	png_destroy_write_struct(&png_ptr, &info_ptr);
}
