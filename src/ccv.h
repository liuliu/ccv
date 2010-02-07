/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

#ifndef _GUARD_ccv_h_
#define _GUARD_ccv_h_

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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

#define CCV_GET_DATA_TYPE_SIZE(x) __ccv_get_data_type_size[(x) >> 8]
#define CCV_GET_CHANNEL_NUM(x) __ccv_get_channel_num[(x)]

enum {
	CCV_DENSE  = 0x010000,
	CCV_SPARSE = 0x020000,
};

typedef struct ccv_dense_matrix_t {
	int type;
	int sig[5];
	int refcount;
	int rows;
	int cols;
	int step;
	union {
		unsigned char* ptr;
		int* i;
		float* fl;
		double* db;
	} data;
} ccv_dense_matrix_t;

typedef struct {
	int type;
	int sig[5];
} ccv_sparse_matrix_t;

#define CCV_IS_EMPTY_SIGNATURE(x) ((x)->sig[0] == 0 && (x)->sig[1] == 0 && (x)->sig[2] == 0 && (x)->sig[3] == 0)

typedef void ccv_matrix_t;

/* matrix operations */
ccv_dense_matrix_t* ccv_dense_matrix_new(int rows, int cols, int type, void* data, int* sig);
ccv_sparse_matrix_t* ccv_sparse_matrix_new(int rows, int cols, int type, void* data, int* sig);
void ccv_matrix_generate_signature(const char* msg, int len, int* sig, int* sig1, int* sig2, int* sig3, int* sig4);
void ccv_matrix_free(ccv_matrix_t* mat);
void ccv_garbage_collect();

/* basic algebra algorithm */
double ccv_trace(ccv_matrix_t* mat);
double ccv_norm(ccv_matrix_t* mat, int type);
void ccv_gemm(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t* c, ccv_matrix_t** d);

/* matrix build blocks */
ccv_dense_matrix_t* ccv_get_dense_matrix(ccv_matrix_t* mat);
ccv_sparse_matrix_t* ccv_get_sparse_matrix(ccv_matrix_t* mat);
int ccv_matrix_assert(ccv_matrix_t* mat, int type, int rows_lt, int rows_gt, int cols_lt, int cols_gt);

/* numerical algorithms */
void ccv_solve(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t* x);
void ccv_eigen(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t* x);
void ccv_minimize(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t* x);
void ccv_filter(ccv_matrix_t* a, ccv_matrix_t* b);

/* modern numerical algorithms */
void ccv_sparse_coding(ccv_matrix_t* x, int k);
void ccv_reconstruct(ccv_matrix_t* a, ccv_matrix_t* x, ccv_matrix_t* y);

/* modern computer vision algorithms */
/* SIFT, DAISY, SURF, MSER, SGF, SSD, FAST */

/* modern machine learning algorithms */
/* RBM, LLE, APCluster */

#endif
