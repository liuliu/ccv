#include "ccv.h"
#include <sys/time.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

float __ccv_mod_2pi(float x)
{
	while (x > 2 * CCV_PI)
		x -= 2 * CCV_PI;
	while (x < 0)
		x += 2 * CCV_PI;
	return x;
}

int main(int argc, char** argv)
{
	ccv_dense_matrix_t* image = 0;
	ccv_unserialize(argv[1], &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	unsigned int elapsed_time = get_current_time();
	ccv_sift_param_t param;
	param.noctaves = 3;
	param.nlevels = 6;
	param.up2x = 1;
	param.edge_threshold = 10;
	param.norm_threshold = 0;
	param.peak_threshold = 0;
	ccv_array_t* keypoints = 0;
	ccv_sift(image, &keypoints, 0, 0, param);
	printf("%d\n", keypoints->rnum);
	ccv_dense_matrix_t* imx = ccv_dense_matrix_new(image->rows, image->cols, CCV_8U | CCV_C1, 0, 0);
	memset(imx->data.ptr, 0, imx->rows * imx->step);
	int i;
	for (i = 0; i < keypoints->rnum; i++)
	{
		ccv_keypoint_t* kp = (ccv_keypoint_t*)ccv_array_get(keypoints, i);
		int ix = (int)(kp->x + 0.5);
		int iy = (int)(kp->y + 0.5);
		imx->data.ptr[ix + iy * imx->step] = 255;
	}
	int len;
	ccv_serialize(imx, "keypoint.png", &len, CCV_SERIAL_PNG_FILE, 0);
	ccv_matrix_free(imx);
	printf("elpased time : %d\n", get_current_time() - elapsed_time);
	ccv_matrix_free(image);
	ccv_garbage_collect();
	image = 0;
	ccv_unserialize("keypoint.png", &image, CCV_SERIAL_GRAY | CCV_SERIAL_ANY_FILE);
	memset(image->data.ptr, 0, image->step * image->rows);
	FILE* frame = fopen("box.frame", "r");
	float x, y, s0, s1;
	ccv_array_t* gtkp = ccv_array_new(10, sizeof(ccv_keypoint_t));
	/*
	int gt, dl;
	fscanf(frame, "%d %d", &gt, &dl);
	for (i = 0; i < gt; i++)
	{
		fscanf(frame, "%f %f %f %f", &y, &x, &s0, &s1);
		int j;
		float dummy;
		for (j = 0; j < dl; j++)
			fscanf(frame, "%f", &dummy);
		ccv_keypoint_t nkp;
		nkp.x = x;
		nkp.y = y;
		nkp.regular.scale = s0;
		nkp.regular.angle = s1;
		image->data.ptr[(int)(y + 0.5) * image->step + (int)(x + 0.5)] = 255;
		ccv_array_push(gtkp, &nkp);
	}
	*/
	while (fscanf(frame, "%f %f %f %f", &x, &y, &s0, &s1) != EOF)
	{
		image->data.ptr[(int)(y + 0.5) * image->step + (int)(x + 0.5)] = 255;
		ccv_keypoint_t nkp;
		nkp.x = x;
		nkp.y = y;
		nkp.regular.scale = s0;
		nkp.regular.angle = s1;
		ccv_array_push(gtkp, &nkp);
	}
	fclose(frame);
	ccv_serialize(image, "mixkp.png", &len, CCV_SERIAL_PNG_FILE, 0);
	int match = 0, angle_match = 0;
	for (i = 0; i < keypoints->rnum; i++)
	{
		ccv_keypoint_t* kp = (ccv_keypoint_t*)ccv_array_get(keypoints, i);
		double mind = 10000;
		int j, minj = -1;
		for (j = 0; j < gtkp->rnum; j++)
		{
			ccv_keypoint_t* gk = (ccv_keypoint_t*)ccv_array_get(gtkp, j);
			if ((gk->x - kp->x) * (gk->x - kp->x) + (gk->y - kp->y) * (gk->y - kp->y) + __ccv_mod_2pi(gk->regular.angle - kp->regular.angle) < mind)
			{
				minj = j;
				mind = (gk->x - kp->x) * (gk->x - kp->x) + (gk->y - kp->y) * (gk->y - kp->y) + __ccv_mod_2pi(gk->regular.angle - kp->regular.angle);
			}
		}
		ccv_keypoint_t* gk = (ccv_keypoint_t*)ccv_array_get(gtkp, minj);
		mind = (gk->x - kp->x) * (gk->x - kp->x) + (gk->y - kp->y) * (gk->y - kp->y);
		if (mind < 0.05)
		{
			match++;
			if (__ccv_mod_2pi(gk->regular.angle - kp->regular.angle) < 0.1)
				angle_match++;
		}
	}
	printf("%.2f%% keypoint matched within 0.05 pixel\n", (float)match * 100.0 / (float)keypoints->rnum);
	printf("%.2f%% angle matched within 0.1 radius\n", (float)angle_match * 100.0 / (float)match);
	ccv_array_free(keypoints);
	ccv_array_free(gtkp);
	ccv_matrix_free(image);
	ccv_garbage_collect();
	return 0;
}

