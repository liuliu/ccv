/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

#ifndef _GUARD_ccv_h_
#define _GUARD_ccv_h_

#include <unistd.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <alloca.h>
#include <math.h>

enum {
	CCV_8U  = 0x0100,
	CCV_32S = 0x0200,
	CCV_32F = 0x0400,
	CCV_64F = 0x0800,
};

enum {
	CCV_C1 = 0x01,
	CCV_C2 = 0x02,
	CCV_C3 = 0x04,
	CCV_C4 = 0x08,
};

static const int __ccv_get_data_type_size[] = { -1, 1, 4, -1, 4, -1, -1, -1, 8 };
static const int __ccv_get_channel_num[] = { -1, 1, 2, -1, 3, -1, -1, -1, 4 };

#define CCV_GET_DATA_TYPE(x) ((x) & 0xFF00)
#define CCV_GET_DATA_TYPE_SIZE(x) __ccv_get_data_type_size[CCV_GET_DATA_TYPE(x) >> 8]
#define CCV_GET_CHANNEL(x) ((x) & 0xFF)
#define CCV_GET_CHANNEL_NUM(x) __ccv_get_channel_num[CCV_GET_CHANNEL(x)]

enum {
	CCV_DENSE  = 0x010000,
	CCV_SPARSE = 0x020000,
};

#define CCV_GARBAGE (0x80000000)

typedef union {
	unsigned char* ptr;
	int* i;
	float* fl;
	double* db;
} ccv_matrix_cell_t;

typedef struct {
	int type;
	int sig[5];
	int refcount;
	int rows;
	int cols;
	int step;
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
	int sig[5];
	int refcount;
	int rows;
	int cols;
	int major;
	int prime;
	int load_factor;
	ccv_dense_vector_t* vector;
} ccv_sparse_matrix_t;

static int __ccv_get_sparse_prime[] = { 53, 97, 193, 389, 769, 1543, 3079, 6151, 12289, 24593, 49157, 98317, 196613, 393241, 786433, 1572869 };
#define CCV_GET_SPARSE_PRIME(x) __ccv_get_sparse_prime[(x)]

#define CCV_IS_EMPTY_SIGNATURE(x) ((x)->sig[0] == 0 && (x)->sig[1] == 0 && (x)->sig[2] == 0 && (x)->sig[3] == 0)

typedef void ccv_matrix_t;

typedef struct {
} ccv_array_t;

#define ccv_clamp(x, a, b) (((x) < (a)) ? (a) : (((x) > (b)) ? (b) : (x)))
#define ccv_min(a, b) (((a) < (b)) ? (a) : (b))
#define ccv_max(a, b) (((a) > (b)) ? (a) : (b))

/* matrix operations */
ccv_dense_matrix_t* ccv_dense_matrix_new(int rows, int cols, int type, void* data, int* sig);
ccv_sparse_matrix_t* ccv_sparse_matrix_new(int rows, int cols, int type, int major, int* sig);
void ccv_matrix_generate_signature(const char* msg, int len, int* sig, int* sig_start, ...);
void ccv_matrix_free(ccv_matrix_t* mat);
void ccv_garbage_collect();

#define ccv_get_dense_matrix_cell(x, row, col) \
	((((x)->type) & CCV_32S) ? (void*)((x)->data.i + (row) * (x)->cols + (col)) : \
	((((x)->type) & CCV_32F) ? (void*)((x)->data.fl+ (row) * (x)->cols + (col)) : \
	((((x)->type) & CCV_64F) ? (void*)((x)->data.db + (row) * (x)->cols + (col)) : \
	(void*)((x)->data.ptr + (row) * (x)->step + (col)))))

#define ccv_get_dense_matrix_cell_value(x, row, col) \
	((((x)->type) & CCV_32S) ? (x)->data.i[(row) * (x)->cols + (col)] : \
	((((x)->type) & CCV_32F) ? (x)->data.fl[(row) * (x)->cols + (col)] : \
	((((x)->type) & CCV_64F) ? (x)->data.db[(row) * (x)->cols + (col)] : \
	(x)->data.ptr[(row) * (x)->step + (col)])))

#define ccv_get_value(type, ptr, i) \
	(((type) & CCV_32S) ? ((int*)(ptr))[(i)] : \
	(((type) & CCV_32F) ? ((float*)(ptr))[(i)] : \
	(((type) & CCV_64F) ? ((double*)(ptr))[(i)] : \
	((unsigned char*)(ptr))[(i)])))

#define ccv_set_value(type, ptr, i, value) switch (CCV_GET_DATA_TYPE((type))) { \
	case CCV_32S: ((int*)(ptr))[(i)] = (int)(value + 0.5); break; \
	case CCV_32F: ((float*)(ptr))[(i)] = (float)value; break; \
	case CCV_64F: ((double*)(ptr))[(i)] = (double)value; break; \
	default: ((unsigned char*)(ptr))[(i)] = ccv_clamp((int)(value + 0.5), 0, 255); }

/* basic io */
enum {
	CCV_SERIAL_ANY_STREAM     = 0x10,
	CCV_SERIAL_PLAIN_STREAM   = 0x11,
	CCV_SERIAL_DEFLATE_STREAM = 0x12,
	CCV_SERIAL_JPEG_STREAM    = 0x13,
	CCV_SERIAL_PNG_STREAM     = 0x14,
	CCV_SERIAL_ANY_FILE       = 0x20,
	CCV_SERIAL_BMP_FILE       = 0x21,
	CCV_SERIAL_JPEG_FILE      = 0x22,
	CCV_SERIAL_PNG_FILE       = 0x23,	
};

enum {
	CCV_SERIAL_CONTINUE = 0x01,
	CCV_SERIAL_FINAL,
	CCV_SERIAL_ERROR,
};

void ccv_unserialize(const char* in, ccv_dense_matrix_t** x, int type);
int ccv_serialize(ccv_dense_matrix_t* mat, char* out, int* len, int type, void* conf);

/* basic algebra algorithm */
double ccv_trace(ccv_matrix_t* mat);
double ccv_norm(ccv_matrix_t* mat, int type);
void ccv_gemm(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t* c, int transpose, ccv_matrix_t** d);

/* matrix build blocks */
ccv_dense_matrix_t* ccv_get_dense_matrix(ccv_matrix_t* mat);
ccv_sparse_matrix_t* ccv_get_sparse_matrix(ccv_matrix_t* mat);
ccv_dense_vector_t* ccv_get_sparse_matrix_vector(ccv_sparse_matrix_t* mat, int index);
ccv_matrix_cell_t ccv_get_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col);
void ccv_set_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col, void* data);
void ccv_matrix_assert(ccv_matrix_t* mat, int type, int rows_lt, int rows_gt, int cols_lt, int cols_gt);
void ccv_convert(ccv_matrix_t* x, ccv_matrix_t* y, int type);

/* numerical algorithms */
void ccv_invert(ccv_matrix_t* a, ccv_matrix_t** b);
void ccv_solve(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d);
void ccv_eigen(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d);
void ccv_minimize(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d);
void ccv_filter(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d);

/* modern numerical algorithms */
void ccv_sparse_coding(ccv_matrix_t* x, int k, ccv_matrix_t** A, ccv_matrix_t** y);
void ccv_compressive_sensing_reconstruct(ccv_matrix_t* a, ccv_matrix_t* x, ccv_matrix_t** y);

/* modern computer vision algorithms */
/* SIFT, DAISY, SURF, MSER, SGF, SSD, FAST */

/* modern machine learning algorithms */
/* RBM, LLE, APCluster */

#endif
