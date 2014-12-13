#include "ccv.h"
#include <sys/time.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

unsigned char colors[6][3] = {{0,0,255},{0,255,0},{255,0,0},{255,255,0},{255,0,255},{0,255,255}};

int main(int argc, char** argv)
{
	assert(argc == 3);
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_RGB_COLOR | CCV_IO_ANY_FILE);
	ccv_mser_param_t params = {
		.min_area = 60,
		.max_area = (int)(image->rows * image->cols * 0.3 + 0.5),
		.min_diversity = 0.2,
		.area_threshold = 1.01,
		.min_margin = 0.003,
		.max_evolution = 200,
		.edge_blur_sigma = sqrt(3.0),
		.delta = 5,
		.max_variance = 0.25,
		.direction = CCV_DARK_TO_BRIGHT,
	};
	if (image)
	{
		ccv_dense_matrix_t* yuv = 0;
		// ccv_color_transform(image, &yuv, 0, CCV_RGB_TO_YUV);
		ccv_read(argv[1], &yuv, CCV_IO_GRAY | CCV_IO_ANY_FILE);
		unsigned int elapsed_time = get_current_time();
		ccv_dense_matrix_t* canny = 0;
		ccv_canny(yuv, &canny, 0, 3, 175, 320);
		ccv_dense_matrix_t* outline = 0;
		ccv_close_outline(canny, &outline, 0);
		ccv_matrix_free(canny);
		ccv_dense_matrix_t* mser = 0;
		ccv_array_t* mser_keypoint = ccv_mser(yuv, outline, &mser, 0, params);
		elapsed_time = get_current_time() - elapsed_time;
		ccv_matrix_free(outline);
		printf("total : %d in time %dms\n", mser_keypoint->rnum, elapsed_time);
		ccv_array_free(mser_keypoint);
		ccv_make_matrix_mutable(image);
		int i, j;
		for (i = 0; i < image->rows; i++)
			for (j = 0; j < image->cols; j++)
			{
				if (mser->data.i32[i * mser->cols + j])
				{
					image->data.u8[i * image->step + j * 3] = colors[mser->data.i32[i * mser->cols + j] % 6][0];
					image->data.u8[i * image->step + j * 3 + 1] = colors[mser->data.i32[i * mser->cols + j] % 6][1];
					image->data.u8[i * image->step + j * 3 + 2] = colors[mser->data.i32[i * mser->cols + j] % 6][2];
				}
			}
		ccv_write(image, argv[2], 0, CCV_IO_PNG_FILE, 0);
		ccv_matrix_free(yuv);
		ccv_matrix_free(image);
	}
	ccv_disable_cache();
	return 0;
}
