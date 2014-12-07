#include "ccv.h"
#include <ctype.h>
#include <getopt.h>
#ifdef HAVE_GSL
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#endif

#ifdef HAVE_GSL
static ccv_dense_matrix_t* _ccv_scd_slice_with_distortion(gsl_rng* rng, ccv_dense_matrix_t* image, ccv_decimal_pose_t pose, ccv_size_t size, ccv_margin_t margin, float deform_angle, float deform_scale, float deform_shift)
{
	float rotate_x = 0; // (deform_angle * 2 * gsl_rng_uniform(rng) - deform_angle) * CCV_PI / 180 + pose.pitch;
	float rotate_y = 0; // (deform_angle * 2 * gsl_rng_uniform(rng) - deform_angle) * CCV_PI / 180 + pose.yaw;
	float rotate_z = (deform_angle * 2 * gsl_rng_uniform(rng) - deform_angle) * CCV_PI / 180; // + pose.roll;
	float scale = gsl_rng_uniform(rng);
	// to make the scale evenly distributed, for example, when deforming of 1/2 ~ 2, we want it to distribute around 1, rather than any average of 1/2 ~ 2
	scale = (1 + deform_scale * scale) / (1 + deform_scale * (1 - scale));
	float scale_ratio = sqrtf((float)(size.width * size.height) / (pose.a * pose.b * 4));
	float m00 = cosf(rotate_z) * scale;
	float m01 = cosf(rotate_y) * sinf(rotate_z) * scale;
	float m02 = (deform_shift * 2 * gsl_rng_uniform(rng) - deform_shift) / scale_ratio + pose.x + (margin.right - margin.left) / scale_ratio - image->cols * 0.5;
	float m10 = (sinf(rotate_y) * cosf(rotate_z) - cosf(rotate_x) * sinf(rotate_z)) * scale;
	float m11 = (sinf(rotate_y) * sinf(rotate_z) + cosf(rotate_x) * cosf(rotate_z)) * scale;
	float m12 = (deform_shift * 2 * gsl_rng_uniform(rng) - deform_shift) / scale_ratio + pose.y + (margin.bottom - margin.top) / scale_ratio - image->rows * 0.5;
	float m20 = (sinf(rotate_y) * cosf(rotate_z) + sinf(rotate_x) * sinf(rotate_z)) * scale;
	float m21 = (sinf(rotate_y) * sinf(rotate_z) - sinf(rotate_x) * cosf(rotate_z)) * scale;
	float m22 = cosf(rotate_x) * cosf(rotate_y);
	ccv_dense_matrix_t* b = 0;
	ccv_perspective_transform(image, &b, 0, m00, m01, m02, m10, m11, m12, m20, m21, m22);
	ccv_dense_matrix_t* resize = 0;
	ccv_size_t scale_size = {
		.width = (int)((size.width + margin.left + margin.right) / scale_ratio + 0.5),
		.height = (int)((size.height + margin.top + margin.bottom) / scale_ratio + 0.5),
	};
	assert(scale_size.width > 0 && scale_size.height > 0);
	ccv_slice(b, (ccv_matrix_t**)&resize, 0, (int)(b->rows * 0.5 - (size.height + margin.top + margin.bottom - 15 /* offset for aflw data */) / scale_ratio * 0.5 + 0.5), (int)(b->cols * 0.5 - (size.width + margin.left + margin.right) / scale_ratio * 0.5 + 0.5), scale_size.height, scale_size.width);
	ccv_matrix_free(b);
	b = 0;
	if (scale_ratio > 1)
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
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	FILE* r = fopen(argv[1], "r");
	char* base_dir = argv[2];
	int dirlen = (base_dir != 0) ? strlen(base_dir) + 1 : 0;
	char* file = (char*)malloc(1024);
	char* file_id = (char*)malloc(1024);
	int face_id;
	ccv_decimal_pose_t pose;
	int i = 0;
	// roll pitch yaw
	while (fscanf(r, "%s %s %d %f %f %f %f %f %f %f", file_id, file, &face_id, &pose.x, &pose.y, &pose.a, &pose.b, &pose.roll, &pose.pitch, &pose.yaw) != EOF)
	{
		if (pose.pitch < CCV_PI * 22.5 / 180 && pose.pitch > -CCV_PI * 22.5 / 180 &&
			pose.roll < CCV_PI * 22.5 / 180 && pose.roll > -CCV_PI * 22.5 / 180 &&
			pose.yaw < CCV_PI * 20 / 180 && pose.yaw > -CCV_PI * 20 / 180)
		{
			// resize to a more proper sizes
			pose.a *= 0.9;
			pose.b *= 0.9;
			char* filename = (char*)malloc(1024);
			strncpy(filename, base_dir, 1024);
			filename[dirlen - 1] = '/';
			strncpy(filename + dirlen, file, 1024 - dirlen);
			ccv_dense_matrix_t* image = 0;
			ccv_read(filename, &image, CCV_IO_ANY_FILE | CCV_IO_GRAY);
			char* savefile = (char*)malloc(1024);
			ccv_dense_matrix_t* b = _ccv_scd_slice_with_distortion(rng, image, pose, ccv_size(40, 40), ccv_margin(0, 0, 0, 0), CCV_PI * 5 / 180, 0.1, 0.05);
			snprintf(savefile, 1024, "/home/liu/Data/facepos/aflw-%07d-bw.png", i);
			ccv_write(b, savefile, 0, CCV_IO_PNG_FILE, 0);
			ccv_matrix_free(b);
			ccv_matrix_free(image);
			image = 0;
			ccv_read(filename, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
			b = _ccv_scd_slice_with_distortion(rng, image, pose, ccv_size(40, 40), ccv_margin(0, 0, 0, 0), CCV_PI * 5 / 180, 0.1, 0.05);
			snprintf(savefile, 1024, "/home/liu/Data/facepos/aflw-%07d-rgb.png", i);
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
	free(file_id);
	gsl_rng_free(rng);
#else
	assert(0 && "aflw requires GSL library support");
#endif
	return 0;
}
