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
	CvMat* gray = cvCreateMat(image->height, image->width, CV_8UC1);
	CvMat* x = cvCreateMat(image->height, image->width, CV_32FC1);
	cvCvtColor(image, gray, CV_BGR2GRAY);
	unsigned int elapsed_time = get_current_time();
	cvSobel(gray, x, 1, 0, 1);
	printf("elapsed time : %d\n", get_current_time() - elapsed_time);
	cvSaveImage(argv[2], x);
	cvReleaseMat(&gray);
	cvReleaseImage(&image);
	return 0;
}


