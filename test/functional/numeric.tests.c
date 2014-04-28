#include "ccv.h"
#include "case.h"
#include "ccv_case.h"
#include "3rdparty/dsfmt/dSFMT.h"

/* numeric tests are more like functional tests rather than unit tests:
 * the following tests contain:
 * 1. compute eigenvectors / eigenvalues on a random symmetric matrix and verify these are eigenvectors / eigenvalues;
 * 2. minimization of the famous rosenbrock function;
 * 3. compute ssd with ccv_filter, and compare the result with naive method
 * 4. compare the result from ccv_distance_transform (linear time) with reference implementation from voc-release4 (O(nlog(n))) */

TEST_CASE("compute eigenvectors and eigenvalues of a symmetric matrix")
{
	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, 0xdead);
	dsfmt_genrand_close_open(&dsfmt);
	ccv_dense_matrix_t* a = ccv_dense_matrix_new(4, 4, CCV_64F | CCV_C1, 0, 0);
	int i, j, k;
	for (i = 0; i < 4; i++)
		for (j = i; j < 4; j++)
			a->data.f64[i * 4 + j] = dsfmt_genrand_close_open(&dsfmt) * 10;
	for (i = 0; i < 4; i++)
		for (j = 0; j < i; j++)
			a->data.f64[i * 4 + j] = a->data.f64[j * 4 + i];
	ccv_dense_matrix_t* evec = 0;
	ccv_dense_matrix_t* eval = 0;
	ccv_eigen(a, &evec, &eval, 0, 1e-6);
	for (k = 0; k < 4; k++)
	{
		double veca[4] = {
			0, 0, 0, 0,
		};
		for (i = 0; i < 4; i++)
			for (j = 0; j < 4; j++)
				veca[i] += a->data.f64[i * 4 + j] * evec->data.f64[k * 4 + j];
		double vece[4];
		for (i = 0; i < 4; i++)
			vece[i] = eval->data.f64[k] * evec->data.f64[k * 4 + i];
		REQUIRE_ARRAY_EQ_WITH_TOLERANCE(double, veca, vece, 4, 1e-6, "verify %d(th) eigenvectors and eigenvalues with Ax = rx", k + 1);
	}
	ccv_matrix_free(a);
	ccv_matrix_free(evec);
	ccv_matrix_free(eval);
}

int rosenbrock(const ccv_dense_matrix_t* x, double* f, ccv_dense_matrix_t* df, void* data)
{
	int* steps = (int*)data;
	(*steps)++;
	int i;
	double rf = 0;
	double* x_vec = x->data.f64;
	for (i = 0; i < 1; i++)
		rf += 100 * (x_vec[i + 1] - x_vec[i] * x_vec[i]) * (x_vec[i + 1] - x_vec[i] * x_vec[i]) + (1 - x_vec[i]) * (1 - x_vec[i]);
	*f = rf;
	double* df_vec = df->data.f64;
	ccv_zero(df);
	df_vec[0] = df_vec[1] = 0;
	for (i = 0; i < 1; i++)
		df_vec[i] = -400 * x_vec[i] * (x_vec[i+1] - x_vec[i] * x_vec[i]) - 2 * (1 - x_vec[i]);
	for (i = 1; i < 2; i++)
		df_vec[i] += 200 * (x_vec[i] - x_vec[i - 1] * x_vec[i - 1]);
	return 0;
}

TEST_CASE("minimize rosenbrock")
{
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(1, 2, CCV_64F | CCV_C1, 0, 0);
	ccv_zero(x);
	int steps = 0;
	ccv_minimize_param_t params;
	params.interp = 0.1;
	params.extrap = 3.0;
	params.max_iter = 20;
	params.ratio = 10.0;
	params.sig = 0.1;
	params.rho = 0.05;
	ccv_minimize(x, 25, 1.0, rosenbrock, params, &steps);
	double dx[2] = { 1, 1 };
	REQUIRE_ARRAY_EQ_WITH_TOLERANCE(double, x->data.f64, dx, 2, 1e-6, "the global minimal should be at (1.0, 1.0)");
	ccv_matrix_free(x);
}

double gaussian(double x, double y, void* data)
{
	return exp(-(x * x + y * y) / 20) / sqrt(CCV_PI * 20);
}

TEST_CASE("Gaussian blur with kernel size even & odd")
{
	ccv_dense_matrix_t* image = 0;
	ccv_read("../../samples/street.png", &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
	ccv_dense_matrix_t* kernel = ccv_dense_matrix_new(100, 100, CCV_32F | CCV_GET_CHANNEL(image->type), 0, 0);
	ccv_filter_kernel(kernel, gaussian, 0);
	ccv_normalize(kernel, (ccv_matrix_t**)&kernel, 0, CCV_L1_NORM);
	ccv_dense_matrix_t* x = 0;
	ccv_filter(image, kernel, &x, CCV_32F, 0);
	ccv_matrix_free(kernel);
	kernel = ccv_dense_matrix_new(101, 101, CCV_32F | CCV_GET_CHANNEL(image->type), 0, 0);
	ccv_filter_kernel(kernel, gaussian, 0);
	ccv_normalize(kernel, (ccv_matrix_t**)&kernel, 0, CCV_L1_NORM);
	ccv_dense_matrix_t* y = 0;
	ccv_filter(image, kernel, &y, CCV_32F, 0);
	ccv_matrix_free(kernel);
	ccv_matrix_free(image);
	REQUIRE_MATRIX_FILE_EQ(x, "data/street.g100.bin", "should be Gaussian blur of 100x100 (even) on street.png");
	ccv_matrix_free(x);
	REQUIRE_MATRIX_FILE_EQ(y, "data/street.g101.bin", "should be Gaussian blur of 101x101 (odd) on street.png");
	ccv_matrix_free(y);
}

TEST_CASE("ccv_filter centre point for even number window size, hint: (size - 1) / 2")
{
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(10, 10, CCV_32F | CCV_C1, 0, 0);
	ccv_dense_matrix_t* y = ccv_dense_matrix_new(10, 10, CCV_32F | CCV_C1, 0, 0);
	float sum = 0;
	int i;
	for (i = 0; i < 100; i++)
	{
		x->data.f32[i] = y->data.f32[99 - i] = i;
		sum += (99 - i) * i;
	}
	ccv_dense_matrix_t* d = 0;
	ccv_filter(x, y, &d, 0, CCV_NO_PADDING);
	REQUIRE_EQ_WITH_TOLERANCE(d->data.f32[4 * 10 + 4], sum, 0.1, "filter centre value should match the sum value computed by a for loop");
	ccv_matrix_free(d);
	ccv_matrix_free(y);
	ccv_matrix_free(x);
}

TEST_CASE("ccv_filter centre point for odd number window size, hint: (size - 1) / 2")
{
	ccv_dense_matrix_t* x = ccv_dense_matrix_new(11, 11, CCV_32F | CCV_C1, 0, 0);
	ccv_dense_matrix_t* y = ccv_dense_matrix_new(11, 11, CCV_32F | CCV_C1, 0, 0);
	float sum = 0;
	int i;
	for (i = 0; i < 121; i++)
	{
		x->data.f32[i] = y->data.f32[120 - i] = i;
		sum += (120 - i) * i;
	}
	ccv_dense_matrix_t* d = 0;
	ccv_filter(x, y, &d, 0, CCV_NO_PADDING);
	REQUIRE_EQ_WITH_TOLERANCE(d->data.f32[5 * 11 + 5], sum, 0.1, "filter centre value should match the sum value computed by a for loop");
	ccv_matrix_free(d);
	ccv_matrix_free(y);
	ccv_matrix_free(x);
}

#include "ccv_internal.h"

static void naive_ssd(ccv_dense_matrix_t* image, ccv_dense_matrix_t* template, ccv_dense_matrix_t* out)
{
	int thw = template->cols / 2;
	int thh = template->rows / 2;
	int i, j, k, x, y, ch = CCV_GET_CHANNEL(image->type);
	unsigned char* i_ptr = image->data.u8 + thh * image->step;
	double* o = out->data.f64 + out->cols * thh;
	ccv_zero(out);
	for (i = thh; i < image->rows - thh - 1; i++)
	{
		for (j = thw; j < image->cols - thw - 1; j++)
		{
			unsigned char* t_ptr = template->data.u8;
			unsigned char* j_ptr = i_ptr - thh * image->step;
			o[j] = 0;
			for (y = -thh; y <= thh; y++)
			{
				for (x = -thw; x <= thw; x++)
					for (k = 0; k < ch; k++)
						o[j] += (j_ptr[(x + j) * ch + k] - t_ptr[(x + thw) * ch + k]) * (j_ptr[(x + j) * ch + k] - t_ptr[(x + thw) * ch + k]);
				t_ptr += template->step;
				j_ptr += image->step;
			}
		}
		i_ptr += image->step;
		o += out->cols;
	}
}

TEST_CASE("convolution ssd (sum of squared differences) v.s. naive ssd")
{
	ccv_dense_matrix_t* street = 0;
	ccv_dense_matrix_t* pedestrian = 0;
	ccv_read("../../samples/pedestrian.png", &pedestrian, CCV_IO_ANY_FILE);
	ccv_read("../../samples/street.png", &street, CCV_IO_ANY_FILE);
	ccv_dense_matrix_t* result = 0;
	ccv_filter(street, pedestrian, &result, CCV_64F, 0);
	ccv_dense_matrix_t* square = 0;
	ccv_multiply(street, street, (ccv_matrix_t**)&square, 0);
	ccv_dense_matrix_t* sat = 0;
	ccv_sat(square, &sat, 0, CCV_PADDING_ZERO);
	ccv_matrix_free(square);
	double sum[] = {0, 0, 0};
	int i, j, k;
	int ch = CCV_GET_CHANNEL(street->type);
	unsigned char* p_ptr = pedestrian->data.u8;
#define for_block(_, _for_get) \
	for (i = 0; i < pedestrian->rows; i++) \
	{ \
		for (j = 0; j < pedestrian->cols; j++) \
			for (k = 0; k < ch; k++) \
				sum[k] += _for_get(p_ptr, j * ch + k, 0) * _for_get(p_ptr, j * ch + k, 0); \
		p_ptr += pedestrian->step; \
	}
	ccv_matrix_getter(pedestrian->type, for_block);
#undef for_block
	int phw = pedestrian->cols / 2;
	int phh = pedestrian->rows / 2;
	ccv_dense_matrix_t* output = ccv_dense_matrix_new(street->rows, street->cols, CCV_64F | CCV_C1, 0, 0);
	ccv_zero(output);
	unsigned char* s_ptr = sat->data.u8 + sat->step * phh;
	unsigned char* r_ptr = result->data.u8 + result->step * phh;
	double* o_ptr = output->data.f64 + output->cols * phh;
#define for_block(_for_get_s, _for_get_r) \
	for (i = phh; i < output->rows - phh - 1; i++) \
	{ \
		for (j = phw; j < output->cols - phw - 1; j++) \
		{ \
			o_ptr[j] = 0; \
			for (k = 0; k < ch; k++) \
			{ \
				o_ptr[j] += (_for_get_s(s_ptr + sat->step * ccv_min(phh + 1, sat->rows - i - 1), ccv_min(j + phw + 1, sat->cols - 1) * ch + k, 0) \
						  - _for_get_s(s_ptr + sat->step * ccv_min(phh + 1, sat->rows - i - 1), ccv_max(j - phw, 0) * ch + k, 0) \
						  + _for_get_s(s_ptr + sat->step * ccv_max(-phh, -i), ccv_max(j - phw, 0) * ch + k, 0) \
						  - _for_get_s(s_ptr + sat->step * ccv_max(-phh, -i), ccv_min(j + phw + 1, sat->cols - 1) * ch + k, 0)) \
						  + sum[k] - 2.0 * _for_get_r(r_ptr, j * ch + k, 0); \
			} \
		} \
		s_ptr += sat->step; \
		r_ptr += result->step; \
		o_ptr += output->cols; \
	}
	ccv_matrix_getter(sat->type, ccv_matrix_getter_a, result->type, for_block);
#undef for_block
	ccv_matrix_free(result);
	ccv_matrix_free(sat);
	ccv_dense_matrix_t* final = 0;
	ccv_slice(output, (ccv_matrix_t**)&final, 0, phh, phw, output->rows - phh * 2, output->cols - phw * 2);
	ccv_zero(output);
	naive_ssd(street, pedestrian, output);
	ccv_dense_matrix_t* ref = 0;
	ccv_slice(output, (ccv_matrix_t**)&ref, 0, phh, phw, output->rows - phh * 2, output->cols - phw * 2);
	ccv_matrix_free(output);
	ccv_matrix_free(pedestrian);
	ccv_matrix_free(street);
	REQUIRE_MATRIX_EQ(ref, final, "ssd computed by convolution doesn't match the one computed by naive method");
	ccv_matrix_free(final);
	ccv_matrix_free(ref);
}

// divide & conquer method for distance transform (copied directly from dpm-matlab (voc-release4)

static inline int square(int x) { return x*x; }

// dt helper function
void dt_min_helper(float *src, float *dst, int *ptr, int step, 
	       int s1, int s2, int d1, int d2, float a, float b) {
 if (d2 >= d1) {
   int d = (d1+d2) >> 1;
   int s = s1;
   int p;
   for (p = s1+1; p <= s2; p++)
     if (src[s*step] + a*square(d-s) + b*(d-s) > 
	 src[p*step] + a*square(d-p) + b*(d-p))
	s = p;
   dst[d*step] = src[s*step] + a*square(d-s) + b*(d-s);
   ptr[d*step] = s;
   dt_min_helper(src, dst, ptr, step, s1, s, d1, d-1, a, b);
   dt_min_helper(src, dst, ptr, step, s, s2, d+1, d2, a, b);
 }
}

// dt of 1d array
void dt_min1d(float *src, float *dst, int *ptr, int step, int n, 
	  float a, float b) {
  dt_min_helper(src, dst, ptr, step, 0, n-1, 0, n-1, a, b);
}

void daq_min_distance_transform(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, double dx, double dy, double dxx, double dyy)
{
	ccv_dense_matrix_t* dc = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | CCV_C1, 0, 0);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | CCV_C1, 0, 0);
	unsigned char* a_ptr = a->data.u8;
	float* b_ptr = db->data.f32;
	int i, j;
#define for_block(_, _for_get) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
			b_ptr[j] = _for_get(a_ptr, j, 0); \
		b_ptr += db->cols; \
		a_ptr += a->step; \
	}
	ccv_matrix_getter(a->type, for_block);
#undef for_block
	int* ix = (int*)calloc(a->cols * a->rows, sizeof(int));
	int* iy = (int*)calloc(a->cols * a->rows, sizeof(int));
	b_ptr = db->data.f32;
	float* c_ptr = dc->data.f32;
	for (i = 0; i < a->rows; i++)
		dt_min1d(b_ptr + i * a->cols, c_ptr + i * a->cols, ix + i * a->cols, 1, a->cols, dxx, dx);
	for (j = 0; j < a->cols; j++)
		dt_min1d(c_ptr + j, b_ptr + j, iy + j, a->cols, a->rows, dyy, dy);
	free(ix);
	free(iy);
	ccv_matrix_free(dc);
}

TEST_CASE("ccv_distance_transform (linear time) v.s. distance transform using divide & conquer (O(nlog(n)))")
{
	ccv_dense_matrix_t* geometry = 0;
	ccv_read("../../samples/geometry.png", &geometry, CCV_IO_GRAY | CCV_IO_ANY_FILE);
	ccv_dense_matrix_t* distance = 0;
	double dx = 0;
	double dy = 0;
	double dxx = 1;
	double dyy = 1;
	ccv_distance_transform(geometry, &distance, 0, 0, 0, 0, 0, dx, dy, dxx, dyy, CCV_GSEDT);
	ccv_dense_matrix_t* ref = 0;
	daq_min_distance_transform(geometry, &ref, dx, dy, dxx, dyy);
	ccv_matrix_free(geometry);
	REQUIRE_MATRIX_EQ(distance, ref, "distance transform computed by ccv_distance_transform doesn't match the one computed by divide & conquer (voc-release4)");
	ccv_matrix_free(ref);
	ccv_matrix_free(distance);
}

// dt helper function
void dt_max_helper(float *src, float *dst, int *ptr, int step, 
	       int s1, int s2, int d1, int d2, float a, float b) {
 if (d2 >= d1) {
   int d = (d1+d2) >> 1;
   int s = s1;
   int p;
   for (p = s1+1; p <= s2; p++)
     if (src[s*step] - a*square(d-s) - b*(d-s) <
	 src[p*step] - a*square(d-p) - b*(d-p))
	s = p;
   dst[d*step] = src[s*step] - a*square(d-s) - b*(d-s);
   ptr[d*step] = s;
   dt_max_helper(src, dst, ptr, step, s1, s, d1, d-1, a, b);
   dt_max_helper(src, dst, ptr, step, s, s2, d+1, d2, a, b);
 }
}

// dt of 1d array
void dt_max1d(float *src, float *dst, int *ptr, int step, int n, 
	  float a, float b) {
  dt_max_helper(src, dst, ptr, step, 0, n-1, 0, n-1, a, b);
}

void daq_max_distance_transform(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, double dx, double dy, double dxx, double dyy)
{
	ccv_dense_matrix_t* dc = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | CCV_C1, 0, 0);
	ccv_dense_matrix_t* db = *b = ccv_dense_matrix_new(a->rows, a->cols, CCV_32F | CCV_C1, 0, 0);
	unsigned char* a_ptr = a->data.u8;
	float* b_ptr = db->data.f32;
	int i, j;
#define for_block(_, _for_get) \
	for (i = 0; i < a->rows; i++) \
	{ \
		for (j = 0; j < a->cols; j++) \
			b_ptr[j] = _for_get(a_ptr, j, 0); \
		b_ptr += db->cols; \
		a_ptr += a->step; \
	}
	ccv_matrix_getter(a->type, for_block);
#undef for_block
	int* ix = (int*)calloc(a->cols * a->rows, sizeof(int));
	int* iy = (int*)calloc(a->cols * a->rows, sizeof(int));
	b_ptr = db->data.f32;
	float* c_ptr = dc->data.f32;
	for (i = 0; i < a->rows; i++)
		dt_max1d(b_ptr + i * a->cols, c_ptr + i * a->cols, ix + i * a->cols, 1, a->cols, dxx, dx);
	for (j = 0; j < a->cols; j++)
		dt_max1d(c_ptr + j, b_ptr + j, iy + j, a->cols, a->rows, dyy, dy);
	free(ix);
	free(iy);
	ccv_matrix_free(dc);
}

TEST_CASE("ccv_distance_transform to compute max distance")
{
	ccv_dense_matrix_t* geometry = 0;
	ccv_read("../../samples/geometry.png", &geometry, CCV_IO_GRAY | CCV_IO_ANY_FILE);
	ccv_dense_matrix_t* distance = 0;
	double dx = 1;
	double dy = 1;
	double dxx = 0.4;
	double dyy = 0.4;
	ccv_distance_transform(geometry, &distance, 0, 0, 0, 0, 0, dx, dy, dxx, dyy, CCV_NEGATIVE | CCV_GSEDT);
	ccv_dense_matrix_t* ref = 0;
	daq_max_distance_transform(geometry, &ref, dx, dy, dxx, dyy);
	ccv_matrix_free(geometry);
	int i;
	for (i = 0; i < distance->rows * distance->cols; i++)
		distance->data.f32[i] = -distance->data.f32[i];
	REQUIRE_MATRIX_EQ(distance, ref, "maximum distance transform computed by negate ccv_distance_transform doesn't match the one computed by divide & conquer");
	ccv_matrix_free(ref);
	ccv_matrix_free(distance);
}

#include "case_main.h"
