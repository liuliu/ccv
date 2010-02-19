static void __ccv_unserialize_png_file(FILE* in, ccv_dense_matrix_t** x)
{
	png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	png_infop end_info = png_create_info_struct(png_ptr);
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
		return;
	}
	png_init_io(png_ptr, in);
	png_read_info(png_ptr, info_ptr);
	png_uint_32 width, height;
	int bit_depth, color_type;
	png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type, 0, 0, 0);
	
	ccv_dense_matrix_t* im = *x;
	if (im == NULL)
		*x = im = ccv_dense_matrix_new((int) height, (int) width, CCV_8U | ((color_type == PNG_COLOR_TYPE_GRAY) ? CCV_C1 : CCV_C3), NULL, NULL);

	if (color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png_ptr);
	if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
		png_set_expand_gray_1_2_4_to_8(png_ptr);
	png_set_strip_alpha(png_ptr);
	unsigned char** row_vectors = (unsigned char**)alloca(im->rows * sizeof(unsigned char*));
	int i;
	for (i = 0; i < im->rows; i++)
		row_vectors[i] = im->data.ptr + i * im->step;
	png_read_image(png_ptr, row_vectors);
	png_read_end(png_ptr, end_info);

	png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
}
