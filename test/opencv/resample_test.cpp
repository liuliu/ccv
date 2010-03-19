#include "cv.h"
#include "highgui.h"
#include <assert.h>
#include <stdio.h>
#include <sys/time.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	assert(argc == 3);
	IplImage* image = cvLoadImage(argv[1]);
	CvMat* x = cvCreateMat((int)(image->height / 1.414), (int)(image->width / 1.414), CV_8UC3);
	unsigned int elapsed_time = get_current_time();
	cvResize(image, x, CV_INTER_AREA);
	printf("elapsed time : %d\n", get_current_time() - elapsed_time);
	cvSaveImage(argv[2], x);
	cvReleaseMat(&x);
	cvReleaseImage(&image);
	return 0;
}


