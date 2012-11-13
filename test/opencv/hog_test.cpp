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
#define HOG_BORDER_SIZE (2)

static void icvCreateHOG( CvMat* img, int* i32c8 )
{
	CvMat* dx = cvCreateMat( img->rows, img->cols, CV_32FC1 );
	CvMat* dy = cvCreateMat( img->rows, img->cols, CV_32FC1 );
	CvMat* angle = cvCreateMat( img->rows, img->cols, CV_32FC1 );
	CvMat* magnitude = cvCreateMat( img->rows, img->cols, CV_32FC1 );
	CvMat* bin = cvCreateMat( img->rows, img->cols, CV_32SC1 );

	cvSobel( img, dx, 1, 0, 1 );
	cvSobel( img, dy, 0, 1, 1 );
	cvCartToPolar( dx, dy, magnitude, angle, 1 );
	cvConvertScale( angle, bin, 8.0 / 360.0, -0.5 );

	int &rows = img->rows, &cols = img->cols;//, &step = img->step;
	int x, y, i, j;
	float* magptr = magnitude->data.fl + HOG_BORDER_SIZE + HOG_BORDER_SIZE * cols;
	int* binptr = bin->data.i + HOG_BORDER_SIZE + HOG_BORDER_SIZE * cols;
	int* iptr = i32c8;
//	uchar* imgptr = img->data.ptr + step * HOG_BORDER_SIZE + HOG_BORDER_SIZE;
	for ( y = HOG_BORDER_SIZE; y < rows - HOG_BORDER_SIZE; y++ )
	{
		float hog[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
		for ( i = -HOG_BORDER_SIZE; i <= HOG_BORDER_SIZE; i++ )
			for ( j = -HOG_BORDER_SIZE; j <= HOG_BORDER_SIZE; j++ )
			{
				int k = i * cols + j;
				hog[binptr[k]] += magptr[k];
			}
		for ( i = 0; i < 8; ++i )
			iptr[i] = (int) hog[i];
//		memcpy( fptr, hog, 4 * 8 );
/*
		for ( i = 0; i < 8; ++i )
			fptr[i] = *imgptr;
		fptr[8] = *imgptr;
		++imgptr;
*/
		iptr += 8;
		++binptr;
		++magptr;
		for ( x = HOG_BORDER_SIZE + 1; x < cols - HOG_BORDER_SIZE; x++ )
		{
			for ( i = -HOG_BORDER_SIZE; i <= HOG_BORDER_SIZE; i++ )
			{
				int k = i * cols - HOG_BORDER_SIZE - 1;
				hog[binptr[k]] -= magptr[k];
				hog[binptr[k + HOG_BORDER_SIZE * 2 + 1]] += magptr[k + HOG_BORDER_SIZE * 2 + 1];
			}
			for ( i = 0; i < 8; ++i )
				iptr[i] = (int) hog[i];
//			memcpy( fptr, hog, 4 * 8 );
/*
			for ( i = 0; i < 8; ++i )
				fptr[i] = *imgptr;
			fptr[8] = *imgptr;
			++imgptr;
*/
			iptr += 8;
			++binptr;
			++magptr;
		}
		binptr += HOG_BORDER_SIZE * 2;
		magptr += HOG_BORDER_SIZE * 2;
//		imgptr += step - cols + HOG_BORDER_SIZE * 2;
	}

	cvReleaseMat( &bin );
	cvReleaseMat( &magnitude );
	cvReleaseMat( &angle );
	cvReleaseMat( &dx );
	cvReleaseMat( &dy );
}

int main(int argc, char** argv)
{
	assert(argc == 3);
	IplImage* image = cvLoadImage(argv[1]);
	CvMat* gray = cvCreateMat(image->height, image->width, CV_8UC1);
	CvMat* x = cvCreateMat(image->height - 4, image->width - 4, CV_32SC1);
	cvCvtColor(image, gray, CV_BGR2GRAY);
	int* hog = (int*)malloc(sizeof(int) * (image->width - 4) * (image->height - 4) * 8);
	unsigned int elapsed_time = get_current_time();
	icvCreateHOG(gray, hog);
	printf("elapsed time : %d\n", get_current_time() - elapsed_time);
	int i, j;
	for (i = 0; i < x->rows; i++)
		for (j = 0; j < x->cols; j++)
			x->data.i[i * x->cols + j] = hog[i * x->cols * 8 + j * 8];
	cvSaveImage(argv[2], x);
	cvReleaseMat(&gray);
	cvReleaseImage(&image);
	return 0;
}
