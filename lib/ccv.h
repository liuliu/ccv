/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

#ifndef GUARD_ccv_h
#define GUARD_ccv_h

#ifndef _MSC_VER
#include <unistd.h>
#include <stdint.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <xmmintrin.h>
#include <assert.h>
#ifdef _WIN32
#include <malloc.h>
#else
#include <alloca.h>
#endif

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
	CCV_GARBAGE   = 0x80000000, // matrix is in cache (not used by any functions)
	CCV_REUSABLE  = 0x40000000, // matrix can be recycled
	CCV_UNMANAGED = 0x20000000, // matrix is allocated by user, therefore, cannot be freed by ccv_matrix_free/ccv_matrix_free_immediately
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
		uint32_t age;
	} branch;
	struct {
		uint64_t sign;
		uint64_t off;
		uint64_t age_and_size;
	} terminal;
} ccv_cache_index_t;

typedef struct {
	ccv_cache_index_t origin;
	uint32_t rnum;
	uint32_t age;
	size_t up;
	size_t size;
	ccv_cache_index_free_f ffree;
} ccv_cache_t;

/* I made it as generic as possible */

void ccv_cache_init(ccv_cache_t* cache, ccv_cache_index_free_f ffree, size_t up);
void* ccv_cache_get(ccv_cache_t* cache, uint64_t sign);
int ccv_cache_put(ccv_cache_t* cache, uint64_t sign, void* x, uint32_t size);
void* ccv_cache_out(ccv_cache_t* cache, uint64_t sign);
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
ccv_dense_matrix_t* ccv_dense_matrix_renew(ccv_dense_matrix_t* x, int rows, int cols, int types, int prefer_type, uint64_t sig);
ccv_dense_matrix_t* ccv_dense_matrix_new(int rows, int cols, int type, void* data, uint64_t sig);
ccv_dense_matrix_t ccv_dense_matrix(int rows, int cols, int type, void* data, uint64_t sig);
ccv_sparse_matrix_t* ccv_sparse_matrix_new(int rows, int cols, int type, int major, uint64_t sig);
void ccv_matrix_free_immediately(ccv_matrix_t* mat);
void ccv_matrix_free(ccv_matrix_t* mat);

uint64_t ccv_matrix_generate_signature(const char* msg, int len, uint64_t sig_start, ...);

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
	CCV_IO_GRAY           = 0x100,
	CCV_IO_COLOR          = 0x300,
	CCV_IO_ANY_STREAM     = 0x010,
	CCV_IO_PLAIN_STREAM   = 0x011,
	CCV_IO_DEFLATE_STREAM = 0x012,
	CCV_IO_JPEG_STREAM    = 0x013,
	CCV_IO_PNG_STREAM     = 0x014,
	CCV_IO_ANY_FILE       = 0x020,
	CCV_IO_BMP_FILE       = 0x021,
	CCV_IO_JPEG_FILE      = 0x022,
	CCV_IO_PNG_FILE       = 0x023,
	CCV_IO_BINARY_FILE    = 0x024,
};

enum {
	CCV_IO_FINAL = 0x00,
	CCV_IO_CONTINUE,
	CCV_IO_ERROR,
	CCV_IO_ATTEMPTED,
};

int ccv_read(const char* in, ccv_dense_matrix_t** x, int type);
int ccv_write(ccv_dense_matrix_t* mat, char* out, int* len, int type, void* conf);

/* basic algebra algorithms ccv_algebra.c */

double ccv_trace(ccv_matrix_t* mat);

enum {
	CCV_L2_NORM  = 0x01, // |dx| + |dy|
	CCV_L1_NORM  = 0x02, // sqrt(dx^2 + dy^2)
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

double ccv_norm(ccv_matrix_t* mat, int type);
double ccv_normalize(ccv_matrix_t* a, ccv_matrix_t** b, int btype, int flag);
void ccv_sat(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int padding_pattern);
double ccv_dot(ccv_matrix_t* a, ccv_matrix_t* b);
double ccv_sum(ccv_matrix_t* mat);
void ccv_multiply(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** c, int type);
void ccv_substract(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** c, int type);

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

typedef struct {
	int rnum;
	int size;
	int rsize;
	void* data;
} ccv_array_t;

ccv_array_t* ccv_array_new(int rnum, int rsize);
void ccv_array_push(ccv_array_t* array, void* r);
typedef int(*ccv_array_group_f)(const void*, const void*, void*);
int ccv_array_group(ccv_array_t* array, ccv_array_t** index, ccv_array_group_f gfunc, void* data);
void ccv_array_zero(ccv_array_t* array);
void ccv_array_clear(ccv_array_t* array);
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
	ccv_rect_t rect;
	int size;
	ccv_array_t* set;
	long m10, m01, m11, m20, m02;
} ccv_contour_t;

ccv_contour_t* ccv_contour_new(int set);
void ccv_contour_push(ccv_contour_t* contour, ccv_point_t point);
void ccv_contour_free(ccv_contour_t* contour);
/* range: exclusive, return value: inclusive (i.e., threshold = 5, 0~5 is background, 6~range-1 is foreground */
int ccv_otsu(ccv_dense_matrix_t* a, double* outvar, int range);

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
void ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int b_type, int sbin, int size);
void ccv_canny(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int size, double low_thresh, double high_thresh);

enum {
	CCV_INTER_AREA    = 0x01,
	CCV_INTER_LINEAR  = 0X02,
	CCV_INTER_CUBIC   = 0X03,
	CCV_INTER_LANCZOS = 0X04,
};

void ccv_resample(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int btype, int rows, int cols, int type);
void ccv_sample_down(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int src_x, int src_y);
void ccv_sample_up(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, int src_x, int src_y);

enum {
	CCV_FLIP_X = 0x01,
	CCV_FLIP_Y = 0x02,
};

void ccv_flip(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int btype, int type);
void ccv_blur(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, double sigma);

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

void ccv_sift(ccv_dense_matrix_t* a, ccv_array_t** keypoints, ccv_dense_matrix_t** desc, int type, ccv_sift_param_t params);

/* swt related method: stroke width transform is relatively new, typically used in text detection */
typedef struct {
	int up2x;
	int direction;
	/* canny parameters */
	int size;
	double low_thresh;
	double high_thresh;
	/* geometry filtering parameters */
	int max_height;
	int min_height;
	double aspect_ratio;
	double variance_ratio;
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

void ccv_swt(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type, ccv_swt_param_t params);
ccv_array_t* ccv_swt_detect_words(ccv_dense_matrix_t* a, ccv_swt_param_t params);

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
} ccv_dpm_part_classifier_t;

typedef struct {
	int count;
	ccv_dpm_part_classifier_t root;
	ccv_dpm_part_classifier_t* part;
	double beta;
} ccv_dpm_root_classifier_t;

typedef struct {
	int count;
	ccv_dpm_root_classifier_t* root;
} ccv_dpm_mixture_model_t;

typedef struct {
	int interval;
	int min_neighbors;
	int flags;
	double threshold;
} ccv_dpm_param_t;

typedef struct {
} ccv_dpm_new_param_t;

enum {
	CCV_DPM_NO_NESTED = 0x10000000,
};

void ccv_dpm_classifier_lsvm_new(ccv_dense_matrix_t** posimgs, int posnum, char** bgfiles, int bgnum, int negnum, const char* dir, ccv_dpm_new_param_t params);
ccv_array_t* ccv_dpm_detect_objects(ccv_dense_matrix_t* a, ccv_dpm_mixture_model_t** model, int count, ccv_dpm_param_t params);
ccv_dpm_mixture_model_t* ccv_load_dpm_mixture_model(const char* directory);

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

void ccv_bbf_classifier_cascade_new(ccv_dense_matrix_t** posimg, int posnum, char** bgfiles, int bgnum, int negnum, ccv_size_t size, const char* dir, ccv_bbf_new_param_t params);
ccv_array_t* ccv_bbf_detect_objects(ccv_dense_matrix_t* a, ccv_bbf_classifier_cascade_t** cascade, int count, ccv_bbf_param_t params);
ccv_bbf_classifier_cascade_t* ccv_load_bbf_classifier_cascade(const char* directory);
ccv_bbf_classifier_cascade_t* ccv_bbf_classifier_cascade_read_binary(char* s);
int ccv_bbf_classifier_cascade_write_binary(ccv_bbf_classifier_cascade_t* cascade, char* s, int slen);
void ccv_bbf_classifier_cascade_free(ccv_bbf_classifier_cascade_t* cascade);

/* following is proprietary implementation of sparse gradient feature, another object detection algorithm
 * which should have better accuracy to shape focused object (pedestrian, vehicle etc.) but still as fast as bbf */

#define CCV_SGF_POINT_MAX (5)
#define CCV_SGF_POINT_MIN (3)

typedef struct {
	int size;
	int px[CCV_SGF_POINT_MAX];
	int py[CCV_SGF_POINT_MAX];
	int pz[CCV_SGF_POINT_MAX];
	int nx[CCV_SGF_POINT_MAX];
	int ny[CCV_SGF_POINT_MAX];
	int nz[CCV_SGF_POINT_MAX];
} ccv_sgf_feature_t;

typedef struct {
	int count;
	float threshold;
	ccv_sgf_feature_t* feature;
	float* alpha;
} ccv_sgf_stage_classifier_t;

typedef struct {
	int count;
	ccv_size_t size;
	ccv_sgf_stage_classifier_t* stage_classifier;
} ccv_sgf_classifier_cascade_t;

typedef struct {
	double pos_crit;
	double neg_crit;
	double balance_k;
	int layer;
	int feature_number;
} ccv_sgf_param_t;

typedef struct {
	ccv_rect_t rect;
	int neighbors;
	int id;
	float confidence;
} ccv_sgf_comp_t;

enum {
	CCV_SGF_NO_NESTED = 0x10000000,
};

void ccv_sgf_classifier_cascade_new(ccv_dense_matrix_t** posimg, int posnum, char** bgfiles, int bgnum, int negnum, ccv_size_t size, const char* dir, ccv_sgf_param_t params);
ccv_array_t* ccv_sgf_detect_objects(ccv_dense_matrix_t* a, ccv_sgf_classifier_cascade_t** _cascade, int count, int min_neighbors, int flags, ccv_size_t min_size);
ccv_sgf_classifier_cascade_t* ccv_load_sgf_classifier_cascade(const char* directory);
ccv_sgf_classifier_cascade_t* ccv_sgf_classifier_cascade_read_binary(char* s);
int ccv_sgf_classifier_cascade_write_binary(ccv_sgf_classifier_cascade_t* cascade, char* s, int slen);
void ccv_sgf_classifier_cascade_free(ccv_sgf_classifier_cascade_t* cascade);

/* modern machine learning algorithms */
/* RBM, LLE, APCluster */

#endif
