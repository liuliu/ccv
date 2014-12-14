#include "ccv.h"
#include <ctype.h>
#include <getopt.h>
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif

#ifdef HAVE_GSL
static ccv_dense_matrix_t* _ccv_aflw_slice_with_rect(gsl_rng* rng, ccv_dense_matrix_t* image, ccv_rect_t rect, ccv_size_t size, ccv_margin_t margin, float deform_angle, float deform_scale, float deform_shift)
{
	ccv_dense_matrix_t* resize = 0;
	ccv_slice(image, (ccv_matrix_t**)&resize, 0, rect.y, rect.x, rect.height, rect.width);
	assert(rect.width == rect.height);
	float scale = gsl_rng_uniform(rng);
	// to make the scale evenly distributed, for example, when deforming of 1/2 ~ 2, we want it to distribute around 1, rather than any average of 1/2 ~ 2
	scale = (1 + deform_scale * scale) / (1 + deform_scale * (1 - scale));
	int new_width = (int)(rect.width * scale + 0.5);
	int new_height = (int)(rect.height * scale + 0.5);
	ccv_point_t offset = ccv_point((int)((deform_shift * 2 * gsl_rng_uniform(rng) - deform_shift) * rect.width + 0.5 + (rect.width - new_width) * 0.5), (int)((deform_shift * 2 * gsl_rng_uniform(rng) - deform_shift) * rect.height + 0.5 + (rect.height - new_height) * 0.5));
	rect.x += offset.x;
	rect.y += offset.y;
	ccv_dense_matrix_t* b = 0;
	if (size.width > rect.width)
		ccv_resample(resize, &b, 0, size.height + margin.top + margin.bottom, size.width + margin.left + margin.right, CCV_INTER_CUBIC);
	else
		ccv_resample(resize, &b, 0, size.height + margin.top + margin.bottom, size.width + margin.left + margin.right, CCV_INTER_AREA);
	ccv_matrix_free(resize);
	return b;
}
#endif

int main(int argc, char** argv)
{
#ifdef HAVE_GSL
	assert(argc == 4);
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	FILE* r = fopen(argv[1], "r");
	char* base_dir = argv[2];
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	char* file = (char*)malloc(1024);
	int i = 0;
	ccv_rect_t rect;
	ccv_decimal_pose_t pose;
	// rect.x, rect.y, rect.width, rect.height roll pitch yaw
	while (fscanf(r, "%s %d %d %d %d %f %f %f", file, &rect.x, &rect.y, &rect.width, &rect.height, &pose.roll, &pose.pitch, &pose.yaw) != EOF)
	{
		if (pose.pitch < CCV_PI * 22.5 / 180 && pose.pitch > -CCV_PI * 22.5 / 180 &&
			pose.roll < CCV_PI * 22.5 / 180 && pose.roll > -CCV_PI * 22.5 / 180 &&
			pose.yaw < CCV_PI * 20 / 180 && pose.yaw > -CCV_PI * 20 / 180 &&
			rect.width >= 15 && rect.height >= 15)
		{
			// resize to a more proper sizes
			char* filename = (char*)malloc(1024);
			strncpy(filename, base_dir, 1024);
			filename[dirlen - 1] = '/';
			strncpy(filename + dirlen, file, 1024 - dirlen);
			ccv_dense_matrix_t* image = 0;
			ccv_read(filename, &image, CCV_IO_ANY_FILE | CCV_IO_GRAY);
			char* savefile = (char*)malloc(1024);
			ccv_dense_matrix_t* b = _ccv_aflw_slice_with_rect(rng, image, rect, ccv_size(48, 48), ccv_margin(0, 0, 0, 0), 10, 0.1, 0.05);
			snprintf(savefile, 1024, "%s/aflw-%07d-bw.png", argv[3], i);
			ccv_write(b, savefile, 0, CCV_IO_PNG_FILE, 0);
			ccv_matrix_free(b);
			ccv_matrix_free(image);
			image = 0;
			ccv_read(filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
			b = _ccv_aflw_slice_with_rect(rng, image, rect, ccv_size(48, 48), ccv_margin(0, 0, 0, 0), 10, 0.1, 0.05);
			snprintf(savefile, 1024, "%s/aflw-%07d-rgb.png", argv[3], i);
			ccv_write(b, savefile, 0, CCV_IO_PNG_FILE, 0);
			ccv_matrix_free(b);
			ccv_matrix_free(image);
			i++;
			free(savefile);
			free(filename);
		}
	}
	fclose(r);
	free(file);
	gsl_rng_free(rng);
#else
	assert(0 && "aflw requires GSL library support");
#endif
	return 0;
}
