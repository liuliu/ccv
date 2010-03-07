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
	CvMat* x = cvCreateMat(image->height / 2, image->width / 2, CV_8UC3);
	unsigned int elapsed_time = get_current_time();
	cvPyrDown(image, x);
	printf("elapsed time : %d\n", get_current_time() - elapsed_time);
	cvSaveImage(argv[2], x);
	cvReleaseMat(&x);
	cvReleaseImage(&image);
	return 0;
}
