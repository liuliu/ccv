/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

#ifndef GUARD_ccv_h
#define GUARD_ccv_h

#include <unistd.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <float.h>
#include <math.h>
#ifdef HAVE_SSE2
#include <xmmintrin.h>
#endif
#include <assert.h>
#include <alloca.h>

#define CCV_PI (3.141592653589793)
#define ccmalloc malloc
#define ccrealloc realloc
#define ccfree free

enum {
	CCV_8U  = 0x0100,
	CCV_32S = 0x0200,
	CCV_32F = 0x0400,
	CCV_64S = 0x0800,
	CCV_64F = 0x1000,
};

enum {
	CCV_C1 = 0x01,
	CCV_C2 = 0x02,
	CCV_C3 = 0x03,
	CCV_C4 = 0x04,
};

static const int _ccv_get_data_type_size[] = { -1, 1, 4, -1, 4, -1, -1, -1, 8, -1, -1, -1, -1, -1, -1, -1, 8 };

#define CCV_GET_DATA_TYPE(x) ((x) & 0xFF00)
#define CCV_GET_DATA_TYPE_SIZE(x) _ccv_get_data_type_size[CCV_GET_DATA_TYPE(x) >> 8]
#define CCV_MAX_CHANNEL (0xFF)
#define CCV_GET_CHANNEL(x) ((x) & 0xFF)
#define CCV_ALL_DATA_TYPE (CCV_8U | CCV_32S | CCV_32F | CCV_64S | CCV_64F)

enum {
	CCV_MATRIX_DENSE  = 0x010000,
	CCV_MATRIX_SPARSE = 0x020000,
	CCV_MATRIX_CSR    = 0x040000,
	CCV_MATRIX_CSC    = 0x080000,
};

enum {
	CCV_GARBAGE       = 0x80000000, // matrix is in cache (not used by any functions)
	CCV_REUSABLE      = 0x40000000, // matrix can be recycled
	CCV_UNMANAGED     = 0x20000000, // matrix is allocated by user, therefore, cannot be freed by ccv_matrix_free/ccv_matrix_free_immediately
	CCV_NO_DATA_ALLOC = 0x10000000, // matrix is allocated as header only, but with no data section, therefore, you have to free the data section separately
};

typedef union {
	unsigned char* u8;
	int* i32;
	float* f32;
	int64_t* i64;
	double* f64;
} ccv_matrix_cell_t;

typedef struct {
	int type;
	uint64_t sig;
	int refcount;
	int rows;
	int cols;
	int step;
	union {
		unsigned char u8;
		int i32;
		float f32;
		int64_t i64;
		double f64;
		void* p;
	} tag;
	ccv_matrix_cell_t data;
} ccv_dense_matrix_t;

enum {
	CCV_SPARSE_VECTOR = 0x00100000,
	CCV_DENSE_VECTOR  = 0x00200000,
};

typedef struct ccv_dense_vector_t {
	int step;
	int length;
	int index;
	int prime;
	int load_factor;
	ccv_matrix_cell_t data;
	int* indice;
	struct ccv_dense_vector_t* next;
} ccv_dense_vector_t;

enum {
	CCV_SPARSE_ROW_MAJOR = 0x00,
	CCV_SPARSE_COL_MAJOR = 0x01,
};

typedef struct {
	int type;
	uint64_t sig;
	int refcount;
	int rows;
	int cols;
	int major;
	int prime;
	int load_factor;
	union {
		unsigned char chr;
		int i;
		float fl;
		int64_t l;
		double db;
	} tag;
	ccv_dense_vector_t* vector;
} ccv_sparse_matrix_t;

extern int _ccv_get_sparse_prime[];
#define CCV_GET_SPARSE_PRIME(x) _ccv_get_sparse_prime[(x)]

typedef void ccv_matrix_t;

/* the explicit cache mechanism ccv_cache.c */
/* the new cache is radix tree based, but has a strict memory usage upper bound
 * so that you don't have to explicitly call ccv_drain_cache() every time */

typedef void(*ccv_cache_index_free_f)(void*);

typedef union {
	struct {
		uint64_t bitmap;
		uint64_t set;
		uint64_t age;
	} branch;
	struct {
		uint64_t sign;
		uint64_t off;
		uint64_t type;
	} terminal;
} ccv_cache_index_t;

typedef struct {
	ccv_cache_index_t origin;
	uint32_t rnum;
	uint32_t age;
	size_t up;
	size_t size;
	ccv_cache_index_free_f ffree[16];
} ccv_cache_t;

/* I made it as generic as possible */

void ccv_cache_init(ccv_cache_t* cache, size_t up, int cache_types, ccv_cache_index_free_f ffree, ...);
void* ccv_cache_get(ccv_cache_t* cache, uint64_t sign, uint8_t* type);
int ccv_cache_put(ccv_cache_t* cache, uint64_t sign, void* x, uint32_t size, uint8_t type);
void* ccv_cache_out(ccv_cache_t* cache, uint64_t sign, uint8_t* type);
int ccv_cache_delete(ccv_cache_t* cache, uint64_t sign);
void ccv_cache_cleanup(ccv_cache_t* cache);
void ccv_cache_close(ccv_cache_t* cache);

/* deprecated methods, often these implemented in another way and no longer suitable for newer computer architecture */
/* 0 */

typedef struct {
	int type;
	uint64_t sig;
	int refcount;
	int rows;
	int cols;
	int nnz;
	union {
		unsigned char chr;
		int i;
		float fl;
		int64_t l;
		double db;
	} tag;
	int* index;
	int* offset;
	ccv_matrix_cell_t data;
} ccv_compressed_sparse_matrix_t;

#define ccv_clamp(x, a, b) (((x) < (a)) ? (a) : (((x) > (b)) ? (b) : (x)))
#define ccv_min(a, b) (((a) < (b)) ? (a) : (b))
#define ccv_max(a, b) (((a) > (b)) ? (a) : (b))

/* matrix memory operations ccv_memory.c */
#define ccv_compute_dense_matrix_size(rows, cols, type) (sizeof(ccv_dense_matrix_t) + (((cols) * CCV_GET_DATA_TYPE_SIZE(type) * CCV_GET_CHANNEL(type) + 3) & -4) * (rows))
ccv_dense_matrix_t* __attribute__((warn_unused_result)) ccv_dense_matrix_renew(ccv_dense_matrix_t* x, int rows, int cols, int types, int prefer_type, uint64_t sig);
ccv_dense_matrix_t* __attribute__((warn_unused_result)) ccv_dense_matrix_new(int rows, int cols, int type, void* data, uint64_t sig);
ccv_dense_matrix_t ccv_dense_matrix(int rows, int cols, int type, void* data, uint64_t sig);
void ccv_make_matrix_mutable(ccv_matrix_t* mat);
void ccv_make_matrix_immutable(ccv_matrix_t* mat);
ccv_sparse_matrix_t* __attribute__((warn_unused_result)) ccv_sparse_matrix_new(int rows, int cols, int type, int major, uint64_t sig);
void ccv_matrix_free_immediately(ccv_matrix_t* mat);
void ccv_matrix_free(ccv_matrix_t* mat);

uint64_t ccv_cache_generate_signature(const char* msg, int len, uint64_t sig_start, ...);

#define CCV_DEFAULT_CACHE_SIZE (1024 * 1024 * 64)

void ccv_drain_cache(void);
void ccv_disable_cache(void);
void ccv_enable_default_cache(void);
void ccv_enable_cache(size_t size);

#define ccv_get_dense_matrix_cell_by(type, x, row, col, ch) \
	(((type) & CCV_32S) ? (void*)((x)->data.i32 + ((row) * (x)->cols + (col)) * CCV_GET_CHANNEL(type) + (ch)) : \
	(((type) & CCV_32F) ? (void*)((x)->data.f32+ ((row) * (x)->cols + (col)) * CCV_GET_CHANNEL(type) + (ch)) : \
	(((type) & CCV_64S) ? (void*)((x)->data.i64+ ((row) * (x)->cols + (col)) * CCV_GET_CHANNEL(type) + (ch)) : \
	(((type) & CCV_64F) ? (void*)((x)->data.f64 + ((row) * (x)->cols + (col)) * CCV_GET_CHANNEL(type) + (ch)) : \
	(void*)((x)->data.u8 + (row) * (x)->step + (col) * CCV_GET_CHANNEL(type) + (ch))))))

#define ccv_get_dense_matrix_cell(x, row, col, ch) ccv_get_dense_matrix_cell_by((x)->type, x, row, col, ch)

/* this is for simplicity in code, I am sick of x->data.f64[i * x->cols + j] stuff, this is clearer, and compiler
 * can optimize away the if structures */
#define ccv_get_dense_matrix_cell_value_by(type, x, row, col, ch) \
	(((type) & CCV_32S) ? (x)->data.i32[((row) * (x)->cols + (col)) * CCV_GET_CHANNEL(type) + (ch)] : \
	(((type) & CCV_32F) ? (x)->data.f32[((row) * (x)->cols + (col)) * CCV_GET_CHANNEL(type) + (ch)] : \
	(((type) & CCV_64S) ? (x)->data.i64[((row) * (x)->cols + (col)) * CCV_GET_CHANNEL(type) + (ch)] : \
	(((type) & CCV_64F) ? (x)->data.f64[((row) * (x)->cols + (col)) * CCV_GET_CHANNEL(type) + (ch)] : \
	(x)->data.u8[(row) * (x)->step + (col) * CCV_GET_CHANNEL(type) + (ch)]))))

#define ccv_get_dense_matrix_cell_value(x, row, col, ch) ccv_get_dense_matrix_cell_value_by((x)->type, x, row, col, ch)

#define ccv_get_value(type, ptr, i) \
	(((type) & CCV_32S) ? ((int*)(ptr))[(i)] : \
	(((type) & CCV_32F) ? ((float*)(ptr))[(i)] : \
	(((type) & CCV_64S) ? ((int64_t*)(ptr))[(i)] : \
	(((type) & CCV_64F) ? ((double*)(ptr))[(i)] : \
	((unsigned char*)(ptr))[(i)]))))

#define ccv_set_value(type, ptr, i, value, factor) switch (CCV_GET_DATA_TYPE((type))) { \
	case CCV_32S: ((int*)(ptr))[(i)] = (int)(value) >> factor; break; \
	case CCV_32F: ((float*)(ptr))[(i)] = (float)value; break; \
	case CCV_64S: ((int64_t*)(ptr))[(i)] = (int64_t)(value) >> factor; break; \
	case CCV_64F: ((double*)(ptr))[(i)] = (double)value; break; \
	default: ((unsigned char*)(ptr))[(i)] = ccv_clamp((int)(value) >> factor, 0, 255); }

/* basic io ccv_io.c */

enum {
	// modifier for converting to gray-scale
	CCV_IO_GRAY      = 0x100,
	// modifier for converting to color
	CCV_IO_RGB_COLOR = 0x300,
};

enum {
	// modifier for not copy the data over when read raw in-memory data
	CCV_IO_NO_COPY = 0x10000,
};

enum {
	// read self-describe in-memory data
	CCV_IO_ANY_STREAM     = 0x010,
	CCV_IO_BMP_STREAM     = 0x011,
	CCV_IO_JPEG_STREAM    = 0x012,
	CCV_IO_PNG_STREAM     = 0x013,
	CCV_IO_PLAIN_STREAM   = 0x014,
	CCV_IO_DEFLATE_STREAM = 0x015,
	// read self-describe on-disk data
	CCV_IO_ANY_FILE       = 0x020,
	CCV_IO_BMP_FILE       = 0x021,
	CCV_IO_JPEG_FILE      = 0x022,
	CCV_IO_PNG_FILE       = 0x023,
	CCV_IO_BINARY_FILE    = 0x024,
	// read not-self-describe in-memory data (a.k.a. raw data)
	// you need to specify rows, cols, or scanline for these data
	CCV_IO_ANY_RAW        = 0x040,
	CCV_IO_RGB_RAW        = 0x041,
	CCV_IO_RGBA_RAW       = 0x042,
	CCV_IO_ARGB_RAW       = 0x043,
	CCV_IO_BGR_RAW        = 0x044,
	CCV_IO_BGRA_RAW       = 0x045,
	CCV_IO_ABGR_RAW       = 0x046,
	CCV_IO_GRAY_RAW       = 0x047,
};

enum {
	CCV_IO_FINAL = 0x00,
	CCV_IO_CONTINUE,
	CCV_IO_ERROR,
	CCV_IO_ATTEMPTED,
	CCV_IO_UNKNOWN,
};

int ccv_read_impl(const void* in, ccv_dense_matrix_t** x, int type, int rows, int cols, int scanline);
#define ccv_read_n(in, x, type, rows, cols, scanline, ...) \
	ccv_read_impl(in, x, type, rows, cols, scanline)
#define ccv_read(in, x, type, ...) \
	ccv_read_n(in, x, type, ##__VA_ARGS__, 0, 0, 0)
// this is a way to implement function-signature based dispatch, you can call either
// ccv_read(in, x, type) or ccv_read(in, x, type, rows, cols, scanline)
// notice that you can implement this with va_* functions, but that is not type-safe
int ccv_write(ccv_dense_matrix_t* mat, char* out, int* len, int type, void* conf);

/* basic algebra algorithms ccv_algebra.c */

double ccv_trace(ccv_matrix_t* mat);

enum {
	CCV_L1_NORM  = 0x01, // |dx| + |dy|
	CCV_L2_NORM  = 0x02, // sqrt(dx^2 + dy^2)
	CCV_GSEDT    = 0x04, // Generalized Squared Euclidean Distance Transform:
						 // a * dx + b * dy + c * dx^2 + d * dy^2, when combined with CCV_L1_NORM:
						 // a * |dx| + b * |dy| + c * dx^2 + d * dy^2
	CCV_NEGATIVE = 0x08, // negative distance computation (from positive (min) to negative (max))
	CCV_POSITIVE = 0x00, // positive distance computation (the default)
};

enum {
	CCV_NO_PADDING = 0x00,
	CCV_PADDING_ZERO = 0x01,
	CCV_PADDING_EXTEND = 0x02,
	CCV_PADDING_MIRROR = 0x04,
};

enum {
	CCV_SIGNED = 0x00,
	CCV_UNSIGNED = 0x01,
};

double ccv_norm(ccv_matrix_t* mat, int type);
double ccv_normalize(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int flag);
void ccv_sat(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int padding_pattern);
double ccv_dot(ccv_matrix_t* a, ccv_matrix_t* b);
double ccv_sum(ccv_matrix_t* mat, int flag);
double ccv_variance(ccv_matrix_t* mat);
void ccv_multiply(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** c, int type);
void ccv_subtract(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** c, int type);

enum {
	CCV_A_TRANSPOSE = 0x01,
	CCV_B_TRANSPOSE = 0X02,
	CCV_C_TRANSPOSE = 0X04,
};

void ccv_gemm(ccv_matrix_t* a, ccv_matrix_t* b, double alpha, ccv_matrix_t* c, double beta, int transpose, ccv_matrix_t** d, int type);

/* matrix build blocks / utility functions ccv_util.c */

ccv_dense_matrix_t* ccv_get_dense_matrix(ccv_matrix_t* mat);
ccv_sparse_matrix_t* ccv_get_sparse_matrix(ccv_matrix_t* mat);
ccv_dense_vector_t* ccv_get_sparse_matrix_vector(ccv_sparse_matrix_t* mat, int index);
ccv_matrix_cell_t ccv_get_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col);
void ccv_set_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col, void* data);
void ccv_compress_sparse_matrix(ccv_sparse_matrix_t* mat, ccv_compressed_sparse_matrix_t** csm);
void ccv_decompress_sparse_matrix(ccv_compressed_sparse_matrix_t* csm, ccv_sparse_matrix_t** smt);

void ccv_move(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int y, int x);
int ccv_matrix_eq(ccv_matrix_t* a, ccv_matrix_t* b);
void ccv_slice(ccv_matrix_t* a, ccv_matrix_t** b, int type, int y, int x, int rows, int cols);
void ccv_visualize(ccv_matrix_t* a, ccv_dense_matrix_t** b, int type);
void ccv_flatten(ccv_matrix_t* a, ccv_matrix_t** b, int type, int flag);
void ccv_zero(ccv_matrix_t* mat);
void ccv_shift(ccv_matrix_t* a, ccv_matrix_t** b, int type, int lr, int rr);
int ccv_any_nan(ccv_matrix_t *a);

/* basic data structures ccv_util.c */

typedef struct {
	int width;
	int height;
} ccv_size_t;

inline static ccv_size_t ccv_size(int width, int height)
{
	ccv_size_t size;
	size.width = width;
	size.height = height;
	return size;
}

inline static int ccv_size_is_zero(ccv_size_t size)
{
	return size.width == 0 && size.height == 0;
}

typedef struct {
	int x;
	int y;
	int width;
	int height;
} ccv_rect_t;

inline static ccv_rect_t ccv_rect(int x, int y, int width, int height)
{
	ccv_rect_t rect;
	rect.x = x;
	rect.y = y;
	rect.width = width;
	rect.height = height;
	return rect;
}

inline static int ccv_rect_is_zero(ccv_rect_t rect)
{
	return rect.x == 0 && rect.y == 0 && rect.width == 0 && rect.height == 0;
}

typedef struct {
	int type;
	uint64_t sig;
	int refcount;
	int rnum;
	int size;
	int rsize;
	void* data;
} ccv_array_t;

ccv_array_t* __attribute__((warn_unused_result)) ccv_array_new(int rsize, int rnum, uint64_t sig);
void ccv_array_push(ccv_array_t* array, void* r);
typedef int(*ccv_array_group_f)(const void*, const void*, void*);
int ccv_array_group(ccv_array_t* array, ccv_array_t** index, ccv_array_group_f gfunc, void* data);
void ccv_make_array_immutable(ccv_array_t* array);
void ccv_make_array_mutable(ccv_array_t* array);
void ccv_array_zero(ccv_array_t* array);
void ccv_array_clear(ccv_array_t* array);
void ccv_array_free_immediately(ccv_array_t* array);
void ccv_array_free(ccv_array_t* array);

#define ccv_array_get(a, i) (((char*)((a)->data)) + (a)->rsize * (i))

typedef struct {
	int x, y;
} ccv_point_t;

inline static ccv_point_t ccv_point(int x, int y)
{
	ccv_point_t point;
	point.x = x;
	point.y = y;
	return point;
}

typedef struct {
	float x, y;
} ccv_decimal_point_t;

inline static ccv_decimal_point_t ccv_decimal_point(float x, float y)
{
	ccv_decimal_point_t point;
	point.x = x;
	point.y = y;
	return point;
}

typedef struct {
	float x, y, a, b;
	float roll, pitch, yaw;
} ccv_decimal_pose_t;

inline static ccv_decimal_pose_t ccv_decimal_pose(float x, float y, float a, float b, float roll, float pitch, float yaw)
{
	ccv_decimal_pose_t pose;
	pose.x = x;
	pose.y = y;
	pose.a = a;
	pose.b = b;
	pose.roll = roll;
	pose.pitch = pitch;
	pose.yaw = yaw;
	return pose;
}

typedef struct {
	ccv_rect_t rect;
	int size;
	ccv_array_t* set;
	long m10, m01, m11, m20, m02;
} ccv_contour_t;

ccv_contour_t* ccv_contour_new(int set);
void ccv_contour_push(ccv_contour_t* contour, ccv_point_t point);
void ccv_contour_free(ccv_contour_t* contour);

/* numerical algorithms ccv_numeric.c */

/* clarification about algebra and numerical algorithms:
 * when using the word "algebra", I assume the operation is well established in Mathematic sense
 * and can be calculated with a straight-forward, finite sequence of operation. The "numerical"
 * in other word, refer to a class of algorithm that can only approximate/or iteratively found the
 * solution. Thus, "invert" would be classified as numerical because of the sense that in some case,
 * it can only be "approximate" (in least-square sense), so to "solve". */

void ccv_invert(ccv_matrix_t* a, ccv_matrix_t** b, int type);
void ccv_solve(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d, int type);
void ccv_eigen(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d, int type);

typedef struct {
	double interp;
	double extrap;
	int max_iter;
	double ratio;
	double rho;
	double sig;
} ccv_minimize_param_t;

typedef int(*ccv_minimize_f)(const ccv_dense_matrix_t* x, double* f, ccv_dense_matrix_t* df, void*);
void ccv_minimize(ccv_dense_matrix_t* x, int length, double red, ccv_minimize_f func, ccv_minimize_param_t params, void* data);

void ccv_filter(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_dense_matrix_t** d, int type, int padding_pattern);
typedef double(*ccv_filter_kernel_f)(double x, double y, void*);
void ccv_filter_kernel(ccv_dense_matrix_t* x, ccv_filter_kernel_f func, void* data);

/* modern numerical algorithms */

void ccv_distance_transform(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, ccv_dense_matrix_t** x, int x_type, ccv_dense_matrix_t** y, int y_type, double dx, double dy, double dxx, double dyy, int flag);
void ccv_sparse_coding(ccv_matrix_t* x, int k, ccv_matrix_t** A, int typeA, ccv_matrix_t** y, int typey);
void ccv_compressive_sensing_reconstruct(ccv_matrix_t* a, ccv_matrix_t* x, ccv_matrix_t** y, int type);

/* basic computer vision algorithms / or build blocks ccv_basic.c */

void ccv_sobel(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int dx, int dy);
void ccv_gradient(ccv_dense_matrix_t* a, ccv_dense_matrix_t** theta, int ttype, ccv_dense_matrix_t** m, int mtype, int dx, int dy);

enum {
	CCV_FLIP_X = 0x01,
	CCV_FLIP_Y = 0x02,
};

void ccv_flip(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int btype, int type);
void ccv_blur(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, double sigma);

enum {
	CCV_RGB_TO_YUV = 0x01,
};

void ccv_color_transform(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int flag);

/* resample algorithms ccv_resample.c */

enum {
	CCV_INTER_AREA    = 0x01,
	CCV_INTER_LINEAR  = 0X02,
	CCV_INTER_CUBIC   = 0X04,
	CCV_INTER_LANCZOS = 0X08,
};

void ccv_resample(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int btype, int rows, int cols, int type);
void ccv_sample_down(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int src_x, int src_y);
void ccv_sample_up(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int src_x, int src_y);

/* transformation algorithms ccv_transform.c */

void ccv_decimal_slice(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, float y, float x, int rows, int cols);
ccv_decimal_point_t ccv_perspective_transform_apply(ccv_decimal_point_t point, ccv_size_t size, float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22);
void ccv_perspective_transform(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, float m00, float m01, float m02, float m10, float m11, float m12, float m20, float m21, float m22);

/* classic computer vision algorithms ccv_classic.c */

void ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int b_type, int sbin, int size);
void ccv_canny(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int size, double low_thresh, double high_thresh);
void ccv_close_outline(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type);
/* range: exclusive, return value: inclusive (i.e., threshold = 5, 0~5 is background, 6~range-1 is foreground */
int ccv_otsu(ccv_dense_matrix_t* a, double* outvar, int range);

typedef struct {
	ccv_decimal_point_t point;
	uint8_t status;
} ccv_decimal_point_with_status_t;

void ccv_optical_flow_lucas_kanade(ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_array_t* point_a, ccv_array_t** point_b, ccv_size_t win_size, int level, double min_eigen);

/* modern computer vision algorithms */
/* SIFT, DAISY, SWT, MSER, DPM, BBF, SGF, SSD, FAST */

/* daisy related methods */
typedef struct {
	double radius;
	int rad_q_no;
	int th_q_no;
	int hist_th_q_no;
	float normalize_threshold;
	int normalize_method;
} ccv_daisy_param_t;

enum {
	CCV_DAISY_NORMAL_PARTIAL = 0x01,
	CCV_DAISY_NORMAL_FULL    = 0x02,
	CCV_DAISY_NORMAL_SIFT    = 0x03,
};

void ccv_daisy(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, ccv_daisy_param_t params);

/* sift related methods */
typedef struct {
	float x, y;
	int octave;
	int level;
	union {
		struct {
			double a, b;
			double c, d;
		} affine;
		struct {
			double scale;
			double angle;
		} regular;
	};
} ccv_keypoint_t;

typedef struct {
	int up2x;
	int noctaves;
	int nlevels;
	float edge_threshold;
	float peak_threshold;
	float norm_threshold;
} ccv_sift_param_t;

extern const ccv_sift_param_t ccv_sift_default_params;

void ccv_sift(ccv_dense_matrix_t* a, ccv_array_t** keypoints, ccv_dense_matrix_t** desc, int type, ccv_sift_param_t params);

/* mser related method */

typedef struct {
	/* parameters for MSER */
	int delta;
	int min_area; /* default: 60 */
	int direction; /* default: 0, 0 for both, -1 for bright to dark, 1 for dark to bright */
	int max_area;
	double max_variance;
	double min_diversity;
	int range; /* from 0 to range, inclusive */
	/* parameters for MSCR */
	double area_threshold; /* default: 1.01 */
	double min_margin; /* default: 0.003 */
	int max_evolution;
	double edge_blur_sigma; /* default: 1.0 */
} ccv_mser_param_t;

typedef struct {
	ccv_rect_t rect;
	int size;
	long m10, m01, m11, m20, m02;
	ccv_point_t keypoint;
} ccv_mser_keypoint_t;

enum {
	CCV_BRIGHT_TO_DARK = -1,
	CCV_DARK_TO_BRIGHT = 1,
};

ccv_array_t* __attribute__((warn_unused_result)) ccv_mser(ccv_dense_matrix_t* a, ccv_dense_matrix_t* h, ccv_dense_matrix_t** b, int type, ccv_mser_param_t params);

/* swt related method: stroke width transform is relatively new, typically used in text detection */
typedef struct {
	int interval; // for scale invariant option
	int min_neighbors; // minimal neighbors to make a detection valid, this is for scale-invariant version
	int scale_invariant; // enable scale invariant swt (to scale to different sizes and then combine the results)
	int direction;
	double same_word_thresh[2]; // overlapping more than 0.1 of the bigger one (0), and 0.9 of the smaller one (1)
	/* canny parameters */
	int size;
	int low_thresh;
	int high_thresh;
	/* geometry filtering parameters */
	int max_height;
	int min_height;
	int min_area;
	int letter_occlude_thresh;
	double aspect_ratio;
	double std_ratio;
	/* grouping parameters */
	double thickness_ratio;
	double height_ratio;
	int intensity_thresh;
	double distance_ratio;
	double intersect_ratio;
	double elongate_ratio;
	int letter_thresh;
	/* break textline into words */
	int breakdown;
	double breakdown_ratio;
} ccv_swt_param_t;

extern const ccv_swt_param_t ccv_swt_default_params;

void ccv_swt(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, ccv_swt_param_t params);
ccv_array_t* __attribute__((warn_unused_result)) ccv_swt_detect_words(ccv_dense_matrix_t* a, ccv_swt_param_t params);

/* it makes sense now to include a simple data structure that encapsulate the common idiom of
 * having file name with a bounding box */

typedef struct {
	char* filename;
	union {
		ccv_rect_t box;
		ccv_decimal_pose_t pose;
	};
} ccv_file_info_t;

/* I'd like to include Deformable Part Models as a general object detection method in here
 * The difference between BBF and DPM:
 * ~ BBF is for rigid object detection: banners, box, faces etc.
 * ~ DPM is more generalized, can detect people, car, bike (larger inner-class difference) etc.
 * ~ BBF is blazing fast (few milliseconds), DPM is relatively slow (around 1 seconds or so) */

#define CCV_DPM_PART_MAX (10)

typedef struct {
	ccv_rect_t rect;
	int neighbors;
	int id;
	float confidence;
} ccv_comp_t;

typedef struct {
	ccv_rect_t rect;
	int neighbors;
	int id;
	float confidence;
	int pnum;
	ccv_comp_t part[CCV_DPM_PART_MAX];
} ccv_root_comp_t;

typedef struct {
	ccv_dense_matrix_t* w;
	double dx, dy, dxx, dyy;
	int x, y, z;
	int counterpart;
	float alpha[6];
} ccv_dpm_part_classifier_t;

typedef struct {
	int count;
	ccv_dpm_part_classifier_t root;
	ccv_dpm_part_classifier_t* part;
	float alpha[3], beta;
} ccv_dpm_root_classifier_t;

typedef struct {
	int count;
	ccv_dpm_root_classifier_t* root;
} ccv_dpm_mixture_model_t;

typedef struct {
	int interval;
	int min_neighbors;
	int flags;
	float threshold;
} ccv_dpm_param_t;

typedef struct {
	int components;
	int parts;
	int grayscale;
	int symmetric;
	int min_area; // 3000
	int max_area; // 5000
	int iterations;
	int data_minings;
	int root_relabels;
	int relabels;
	int discard_estimating_constant; // 1
	int negative_cache_size; // 1000
	double include_overlap; // 0.7
	double alpha;
	double alpha_ratio; // 0.85
	double balance; // 1.5
	double C;
	double percentile_breakdown; // 0.05
	ccv_dpm_param_t detector;
} ccv_dpm_new_param_t;

enum {
	CCV_DPM_NO_NESTED = 0x10000000,
};

extern const ccv_dpm_param_t ccv_dpm_default_params;

void ccv_dpm_mixture_model_new(char** posfiles, ccv_rect_t* bboxes, int posnum, char** bgfiles, int bgnum, int negnum, const char* dir, ccv_dpm_new_param_t params);
ccv_array_t* __attribute__((warn_unused_result)) ccv_dpm_detect_objects(ccv_dense_matrix_t* a, ccv_dpm_mixture_model_t** model, int count, ccv_dpm_param_t params);
ccv_dpm_mixture_model_t* __attribute__((warn_unused_result)) ccv_load_dpm_mixture_model(const char* directory);
void ccv_dpm_mixture_model_free(ccv_dpm_mixture_model_t* model);

/* this is open source implementation of object detection algorithm: brightness binary feature
 * it is an extension/modification of original HAAR-like feature with Adaboost, featured faster
 * computation and higher accuracy (current highest accuracy close-source face detector is based
 * on the same algorithm) */

#define CCV_BBF_POINT_MAX (8)
#define CCV_BBF_POINT_MIN (3)

typedef struct {
	int size;
	int px[CCV_BBF_POINT_MAX];
	int py[CCV_BBF_POINT_MAX];
	int pz[CCV_BBF_POINT_MAX];
	int nx[CCV_BBF_POINT_MAX];
	int ny[CCV_BBF_POINT_MAX];
	int nz[CCV_BBF_POINT_MAX];
} ccv_bbf_feature_t;

typedef struct {
	int count;
	float threshold;
	ccv_bbf_feature_t* feature;
	float* alpha;
} ccv_bbf_stage_classifier_t;

typedef struct {
	int count;
	ccv_size_t size;
	ccv_bbf_stage_classifier_t* stage_classifier;
} ccv_bbf_classifier_cascade_t;

enum {
	CCV_BBF_GENETIC_OPT = 0x01,
	CCV_BBF_FLOAT_OPT = 0x02
};

typedef struct {
	int interval;
	int min_neighbors;
	int flags;
	int accurate;
	ccv_size_t size;
} ccv_bbf_param_t;

typedef struct {
	double pos_crit;
	double neg_crit;
	double balance_k;
	int layer;
	int feature_number;
	int optimizer;
	ccv_bbf_param_t detector;
} ccv_bbf_new_param_t;

enum {
	CCV_BBF_NO_NESTED = 0x10000000,
};

extern const ccv_bbf_param_t ccv_bbf_default_params;

void ccv_bbf_classifier_cascade_new(ccv_dense_matrix_t** posimg, int posnum, char** bgfiles, int bgnum, int negnum, ccv_size_t size, const char* dir, ccv_bbf_new_param_t params);
ccv_array_t* __attribute__((warn_unused_result)) ccv_bbf_detect_objects(ccv_dense_matrix_t* a, ccv_bbf_classifier_cascade_t** cascade, int count, ccv_bbf_param_t params);
ccv_bbf_classifier_cascade_t* __attribute__((warn_unused_result)) ccv_load_bbf_classifier_cascade(const char* directory);
ccv_bbf_classifier_cascade_t* __attribute__((warn_unused_result)) ccv_bbf_classifier_cascade_read_binary(char* s);
int ccv_bbf_classifier_cascade_write_binary(ccv_bbf_classifier_cascade_t* cascade, char* s, int slen);
void ccv_bbf_classifier_cascade_free(ccv_bbf_classifier_cascade_t* cascade);

/* Ferns classifier: this is a fern implementation that specifically used for TLD
 * see: http://cvlab.epfl.ch/alumni/oezuysal/ferns.html for more about ferns */

typedef struct {
	int structs;
	int features;
	int scales;
	int posteriors;
	float threshold;
	int cnum[2];
	int* rnum;
	float* posterior;
	// decided to go flat organized fern so that we can profiling different memory layout impacts the performance
	ccv_point_t fern[1];
} ccv_ferns_t;

ccv_ferns_t* __attribute__((warn_unused_result)) ccv_ferns_new(int structs, int features, int scales, ccv_size_t* sizes);
void ccv_ferns_feature(ccv_ferns_t* ferns, ccv_dense_matrix_t* a, int scale, uint32_t* fern);
void ccv_ferns_correct(ccv_ferns_t* ferns, uint32_t* fern, int c, int repeat);
float ccv_ferns_predict(ccv_ferns_t* ferns, uint32_t* fern);
void ccv_ferns_free(ccv_ferns_t* ferns);

/* TLD: Track-Learn-Detection is a long-term object tracking framework, which achieved very high
 * tracking accuracy, this is the tracking algorithm of choice ccv implements */

typedef struct {
	/* short-term lucas-kanade tracking parameters */
	ccv_size_t win_size;
	int level;
	float min_eigen;
	float min_forward_backward_error;
	/* image pyramid (different resolution) generation parameters */
	int interval;
	float shift;
	/* samples generation parameters */
	int min_win;
	float include_overlap;
	float exclude_overlap;
	/* fern classifier setting */
	int structs;
	int features;
	/* nearest neighbor thresholds */
	float validate_set; // 0.5 for conservative confidence
	float nnc_same; // the same object
	float nnc_thres; // highly correlated
	float nnc_verify; // correlated with tracking
	float nnc_beyond; // this is the cap of nnc_thres
	float nnc_collect; // modest correlated, worth to collect as negative example
	int bad_patches; // number of bad patches
	/* deformation round */
	int new_deform;
	int track_deform;
	float new_deform_angle;
	float track_deform_angle;
	float new_deform_scale;
	float track_deform_scale;
	float new_deform_shift;
	float track_deform_shift;
	/* top detections */
	int top_n;
	/* speed up technique, instead of running slide window at
	 * every frame, we will rotate them, for example, slide window 1
	 * only gets examined at frame % rotation == 1 */
	int rotation;
} ccv_tld_param_t;

extern const ccv_tld_param_t ccv_tld_default_params;

typedef struct {
	ccv_tld_param_t params;
	ccv_comp_t box; // tracking comp
	ccv_ferns_t* ferns; // ferns classifier
	ccv_array_t* sv[2]; // example-based classifier
	ccv_size_t patch; // resized to patch for example-based classifier
	int found; // if the last time found a valid box
	int verified; // the last frame is verified, therefore, a successful tracking is verified too
	ccv_array_t* top; // top matches
	float ferns_thres; // computed dynamically from negative examples
	float nnc_thres; // computed dynamically from negative examples
	float nnc_verify_thres; // computed dynamically from negative examples
	double var_thres; // computed dynamically from the supplied same
	uint64_t frame_signature;
	int count;
	void* sfmt;
	void* dsfmt;
	uint32_t fern_buffer[1]; // fetched ferns from image, this is a buffer
} ccv_tld_t;

typedef struct {
	int perform_track;
	int perform_learn;
	int track_success;
	int ferns_detects;
	int nnc_detects;
	int clustered_detects;
	int confident_matches;
	int close_matches;
} ccv_tld_info_t;

ccv_tld_t* __attribute__((warn_unused_result)) ccv_tld_new(ccv_dense_matrix_t* a, ccv_rect_t box, ccv_tld_param_t params);
ccv_comp_t ccv_tld_track_object(ccv_tld_t* tld, ccv_dense_matrix_t* a, ccv_dense_matrix_t* b, ccv_tld_info_t* info);
void ccv_tld_free(ccv_tld_t* tld);

/* ICF: Integrate Channels Features, this is a theorized framework that retrospectively incorporates the original
 * Viola-Jones detection method with various enhancement later. Specifically, this implementation is after:
 * Pedestrian detection at 100 frames per second, Rodrigo Benenson, Markus Mathias, Radu Timofte and Luc Van Gool
 * With WFS (width first search) tree from:
 * High-Performance Rotation Invariant Multiview Face Detection, Chang Huang, Haizhou Ai, Yuan Li and Shihong Lao */

#define CCV_ICF_SAT_MAX (8)

typedef struct {
	int count;
	int channel[CCV_ICF_SAT_MAX];
	ccv_point_t sat[CCV_ICF_SAT_MAX * 2];
	float alpha[CCV_ICF_SAT_MAX];
	float beta;
	float weigh[2];
} ccv_icf_feature_t;

typedef struct {
	int index;
	float threshold;
} ccv_icf_threshold_t;

typedef struct {
	int count;
	ccv_size_t size;
	ccv_icf_threshold_t* thresholds;
	ccv_icf_feature_t* features;
} ccv_icf_classifier_cascade_t;

typedef struct {
	int interval;
	ccv_icf_classifier_cascade_t* cascade;
} ccv_icf_multiscale_classifier_cascade_t;

typedef struct {
	int min_neighbors;
	int flags;
	float threshold;
} ccv_icf_param_t;

typedef struct {
	ccv_icf_param_t detector;
	int interval;
	ccv_size_t size;
	int feature_size;
	int select_feature_size;
	float deform_angle;
	float deform_scale;
	float deform_shift;
	double C;
	double weight_trimming;
} ccv_icf_new_param_t;

void ccv_icf(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type);
ccv_icf_multiscale_classifier_cascade_t* __attribute__((warn_unused_result)) ccv_icf_classifier_cascade_new(ccv_array_t* posfiles, int posnum, ccv_array_t* bgfiles, int negnum, const char* dir, ccv_icf_new_param_t params);
ccv_array_t* __attribute__((warn_unused_result)) ccv_icf_detect_objects(ccv_dense_matrix_t* a, ccv_icf_multiscale_classifier_cascade_t** multiscale_cascade, int count, ccv_icf_param_t params);
ccv_icf_multiscale_classifier_cascade_t* __attribute__((warn_unused_result)) ccv_icf_read_classifier_cascade(const char* directory);
void ccv_icf_write_classifier_cascade(ccv_icf_multiscale_classifier_cascade_t* classifier, const char* directory);
void ccv_icf_classifier_cascade_free(ccv_icf_multiscale_classifier_cascade_t* classifier);

#endif
