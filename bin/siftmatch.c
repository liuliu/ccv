#include "ccv.h"
#include <sys/time.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	assert(argc == 3);
	ccv_dense_matrix_t* object = 0;
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize(argv[1], &object, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	ccv_unserialize(argv[2], &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	unsigned int elapsed_time = get_current_time();
	ccv_sift_param_t param;
	param.noctaves = 3;
	param.nlevels = 6;
	param.upsample = 0;
	param.edge_threshold = 10;
	param.norm_threshold = 0;
	param.peak_threshold = 0;
	ccv_array_t* obj_keypoints = 0;
	ccv_dense_matrix_t* obj_desc = 0;
	ccv_sift(object, &obj_keypoints, &obj_desc, 0, param);
	ccv_array_t* image_keypoints = 0;
	ccv_dense_matrix_t* image_desc = 0;
	ccv_sift(image, &image_keypoints, &image_desc, 0, param);
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	int i, j, k;
	ccv_dense_matrix_t* imx = ccv_dense_matrix_new(image->rows, image->cols, CCV_8U | CCV_C1, 0, 0);
	memset(imx->data.ptr, 0, imx->rows * imx->step);
	for (i = 0; i < obj_keypoints->rnum; i++)
	{
		float* odesc = obj_desc->data.fl + i * 128;
		int minj = -1;
		double mind = 1e6, mind2 = 1e6;
		for (j = 0; j < image_keypoints->rnum; j++)
		{
			float* idesc = image_desc->data.fl + j * 128;
			double d = 0;
			for (k = 0; k < 128; k++)
			{
				d += (odesc[k] - idesc[k]) * (odesc[k] - idesc[k]);
				if (d > mind2)
					break;
			}
			if (d < mind)
			{
				mind2 = mind;
				mind = d;
				minj = j;
			} else if (d < mind2) {
				mind2 = d;
			}
		}
		if (mind < mind2 * 0.6)
		{
			ccv_keypoint_t* kp = (ccv_keypoint_t*)ccv_array_get(image_keypoints, minj);
			int ix = (int)(kp->x + 0.5);
			int iy = (int)(kp->y + 0.5);
			imx->data.ptr[ix + iy * imx->step] = 255;
		}
	}
	int len;
	ccv_serialize(imx, "match.png", &len, CCV_SERIAL_PNG_FILE, 0);
	ccv_array_free(obj_keypoints);
	ccv_array_free(image_keypoints);
	ccv_matrix_free(obj_desc);
	ccv_matrix_free(image_desc);
	ccv_matrix_free(object);
	ccv_matrix_free(image);
	ccv_garbage_collect();
	return 0;
}

