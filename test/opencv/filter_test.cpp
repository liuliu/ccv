#include "cv.h"
#include "highgui.h"
#include <assert.h>
#include <stdio.h>

int main(int argc, char** argv)
{
	assert(argc == 3);
	IplImage* image = cvLoadImage(argv[1]);
	CvMat* gray = cvCreateMat(image->height, image->width, CV_32FC1);
	CvMat* x = cvCreateMat(image->height, image->width, CV_32FC1);
	CvMat* kernel = cvCreateMat(101, 101, CV_32FC1);
	int i, j;
	for (i = 0; i < image->height; i++)
		for (j = 0; j < image->width; j++)
			gray->data.fl[i * gray->cols + j] = image->imageData[i * image->widthStep + j * 3] * 0.1 + image->imageData[i * image->widthStep + j * 3 + 1] * 0.61 + image->imageData[i * image->widthStep + j * 3 + 2] * 0.29;
	for (i = 0; i < kernel->rows; i++)
		for (j = 0; j < kernel->cols; j++)
			kernel->data.fl[i * kernel->cols + j] = exp(-((i - kernel->rows / 2) * (i - kernel->rows / 2) + (j - kernel->cols / 2) * (j - kernel->cols / 2)) / 100);
	cvFilter2D(gray, x, kernel);
	cvSaveImage(argv[2], x);
	cvReleaseMat(&gray);
	cvReleaseImage(&image);
	return 0;
}

