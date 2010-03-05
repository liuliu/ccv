#include "ccv.h"
#include <sys/time.h>
#include <xmmintrin.h>

void __ccv_atan2(int* x, int* y, int* angle, int len)
{
	int i = 0;
	for(; i < len; i++)
	{
		int x2 = x[i] * x[i];
		int y2 = y[i] * y[i];
		if (x[i] == 0 && y[i] == 0) {
			angle[i] = 0;
		} else if (y2 <= x2) {
			angle[i] = x[i] * y[i] * 1440 / (x2 * 25 + 7 * y2);
			angle[i] = (x[i] < 0 ? angle[i] + 180 : y[i] >= 0 ? angle[i] : 360 + angle[i]);
		} else {
			angle[i] = x[i] * y[i] * 1440 / (y2 * 25 + 7 * x2);
			angle[i] = (y[i] >= 0 ? 90 - angle[i] : 270 - angle[i]);
		}
	}
}

int ccv_atan2(int y, int x)
{
	int a;
	if (x == 0 && y == 0)
		return 0;
	int x2 = x * x, y2 = y * y;
	if (y2 <= x2) 
	{
		a = x * y * 1440 / (x2 * 25 + 7 * y2); /* x, y < 1221, otherwise overflow */
		return (x < 0 ? a + 180 : y >= 0 ? a : 360 + a);
	} else {
		a = x * y * 1440 / (y2 * 25 + 7 * x2);
		return (y >= 0 ? 90 - a : 270 - a);
	}
}

float fastAtan2( float y, float x )
{
	double a, x2 = (double)x*x, y2 = (double)y*y;
	if( y2 <= x2 )
	{
		a = (180./3.141592653)*x*y/(x2 + 0.28*y2 + 1e-6);
		return (float)(x < 0 ? a + 180 : y >= 0 ? a : 360+a);
	}
	a = (180./3.141592653)*x*y/(y2 + 0.28*x2 + 1e-6);
	return (float)(y >= 0 ? 90 - a : 270 - a);
}

#define CV_PI (3.141592654)
#define DBL_EPSILON (1e-6)

static int FastAtan2_32f(const float *Y, const float *X, float *angle, int len )
{
	int i = 0;
	float scale = (float)(180/CV_PI);

	static const int iabsmask[] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
	__m128 eps = _mm_set1_ps((float)DBL_EPSILON), absmask = _mm_load_ps((const float*)iabsmask);
	__m128 _90 = _mm_set1_ps((float)(CV_PI*0.5)), _180 = _mm_set1_ps((float)CV_PI), _360 = _mm_set1_ps((float)(CV_PI*2));
	__m128 zero = _mm_setzero_ps(), _0_28 = _mm_set1_ps(0.28f), scale4 = _mm_set1_ps(scale);
	
	for( ; i <= len - 4; i += 4 )
	{
		__m128 x4 = _mm_loadu_ps(X + i), y4 = _mm_loadu_ps(Y + i);
		__m128 xq4 = _mm_mul_ps(x4, x4), yq4 = _mm_mul_ps(y4, y4);
		__m128 xly = _mm_cmplt_ps(xq4, yq4);
		__m128 z4 = _mm_div_ps(_mm_mul_ps(x4, y4), _mm_add_ps(_mm_add_ps(_mm_max_ps(xq4, yq4),
			_mm_mul_ps(_mm_min_ps(xq4, yq4), _0_28)), eps));
		
		// a4 <- x < y ? 90 : 0;
		__m128 a4 = _mm_and_ps(xly, _90);
		// a4 <- (y < 0 ? 360 - a4 : a4) == ((x < y ? y < 0 ? 270 : 90) : (y < 0 ? 360 : 0))
		__m128 mask = _mm_cmplt_ps(y4, zero);
		a4 = _mm_or_ps(_mm_and_ps(_mm_sub_ps(_360, a4), mask), _mm_andnot_ps(mask, a4));
		// a4 <- (x < 0 && !(x < y) ? 180 : a4)
		mask = _mm_andnot_ps(xly, _mm_cmplt_ps(x4, zero));
		a4 = _mm_or_ps(_mm_and_ps(_180, mask), _mm_andnot_ps(mask, a4));
		
		// a4 <- (x < y ? a4 - z4 : a4 + z4)
		a4 = _mm_mul_ps(_mm_add_ps(_mm_xor_ps(z4, _mm_andnot_ps(absmask, xly)), a4), scale4);
		_mm_storeu_ps(angle + i, a4);
	}
	
	for( ; i < len; i++ )
	{
		float x = X[i], y = Y[i];
		float a, x2 = x*x, y2 = y*y;
		if( y2 <= x2 )
			a = x*y/(x2 + 0.28f*y2 + (float)DBL_EPSILON) + (float)(x < 0 ? CV_PI : y >= 0 ? 0 : CV_PI*2);
		else
			a = (float)(y >= 0 ? CV_PI*0.5 : CV_PI*1.5) - x*y/(y2 + 0.28f*x2 + (float)DBL_EPSILON);
		angle[i] = a*scale;
	}

    return 0;
}

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
	int x, y, i;
	int angle = 0; //, miss = 0;
	/*
	for (y = 1; y < 1024; y++)
		for (x = y + 1; x < 1024; x++)
		{
			int angle1 = (int)(fastAtan2(y, x) + 0.5);
			int angle2 = (int)(atan2(y, x) * 180 / 3.141592654 + 0.5);
			if (abs(angle1 - angle2) > 0)
			{
			//	printf("%d %d : %d %d\n", y, x, angle1, angle2);
				miss++;
			}
		}
	printf("missing rate : %d\n", miss);
	miss = 0;
	for (y = 1; y < 1024; y++)
		for (x = y + 1; x < 1024; x++)
		{
			int angle1;
		//	angle1 = x * y * 600 / (x * x * 10 + 3 * y * y);
		//	if (y > 1024 || x > 1024)
		//	{
				angle1 = x * y * 180 / (x * x * 3 + y * y);
		//	} else {
		//		angle1 = x * y * 1440 / (x * x * 25 + 7 * y * y);
		//		angle1 = x * y * 18000 / (x * x * 314 + 88 * y * y);
		//	}
			int angle2 = (int)(atan2(y, x) * 180 / 3.141592654 + 0.5);
			if (abs(angle1 - angle2) > 0)
			{
			//	printf("%d %d : %d %d\n", y, x, angle1, angle2);
				miss++;
			}
		}
	printf("missing rate : %d\n", miss);
	
	for (y = -1024; y <= 1024; y++)
		for (x = -1024; x <= 1024; x++)
		{
			int angle1 = ccv_atan2(y, x);
			int angle2 = (int)(atan2(y, x) * 180 / 3.141592654 + 0.5);
			if (angle2 < 0)
				angle2 += 360;
			if (abs(angle1 - angle2) > 2)
			{
				printf("%d %d : %d %d\n", y, x, angle1, angle2);
			}
		}
	*/
	unsigned int elpased_time;

	//int* atanTable = malloc(sizeof(int) * 1024 * 1024);
	//for (y = 0; y < 1024; y++)
	//	for (x = 0; x < 1024; x++)
	//		atanTable[y * 1024 + x] = (int)(atan2(y, x) * 180 / 3.141592654 + 0.5);
	//elpased_time = get_current_time();
	//for (i = 0; i < 100; i++)
	//for (y = 0; y < 1024; y++)
	//	for (x = 0; x < 1024; x++)
	//		angle += atanTable[x * 1024 + y];
	//printf("lut : %dms\n", get_current_time() - elpased_time);
	float* xf = (float*)malloc(sizeof(float) * 1024 * 1024);
	float* yf = (float*)malloc(sizeof(float) * 1024 * 1024);
	for (y = 0; y < 1024; y++)
		for (x = 0; x < 1024; x++)
		{
			xf[y * 1024 + x] = x;
			yf[y * 1024 + x] = y;
		}
	float* af = (float*)malloc(sizeof(float) * 1024 * 1024);
	elpased_time = get_current_time();
	for (i = 0; i < 100; i++)
		FastAtan2_32f(xf, yf, af, 1024 * 1024);
	for (y = 0; y < 1024; y++)
		for (x = 0; x < 1024; x++)
			angle += af[y * 1024 + x];
	printf("FastAtan2_32f : %dms\n", get_current_time() - elpased_time);
	free(xf); free(yf); free(af);
	int* xi = (int*)malloc(sizeof(int) * 1024 * 1024);
	int* yi = (int*)malloc(sizeof(int) * 1024 * 1024);
	for (y = 0; y < 1024; y++)
		for (x = 0; x < 1024; x++)
		{
			xi[y * 1024 + x] = x;
			yi[y * 1024 + x] = y;
		}
	int* ai = (int*)malloc(sizeof(int) * 1024 * 1024);
	elpased_time = get_current_time();
	for (i = 0; i < 100; i++)
		__ccv_atan2(xi, yi, ai, 1024 * 1024);
	for (y = 0; y < 1024; y++)
		for (x = 0; x < 1024; x++)
			angle += ai[y * 1024 + x];
	printf("__ccv_atan2 : %dms\n", get_current_time() - elpased_time);
	free(xi); free(yi); free(ai);
	elpased_time = get_current_time();
	for (y = 0; y < 1024; y++)
		for (x = 0; x < 1024; x++)
			angle += (int)(atan2(y, x) * 180 / 3.141592654 + 0.5);
	printf("atan2 : %dms\n", get_current_time() - elpased_time);
	elpased_time = get_current_time();
	for (i = 0; i < 100; i++)
	for (y = 0; y < 1024; y++)
		for (x = 0; x < 1024; x++)
			angle += (int)(fastAtan2(y, x) + 0.5);
	printf("fastAtan2 : %dms\n", get_current_time() - elpased_time);
	printf("%d\n", angle);

	return 0;
}
