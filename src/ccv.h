/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

#ifndef _GUARD_ccv_h_
#define _GUARD_ccv_h_

enum {
	CCV_8U  = 0x0100,
	CCV_32S = 0x0200,
	CCV_32F = 0x0300,
	CCV_64F = 0x0400,
};

enum {
	CCV_C1 = 0x01,
	CCV_C2 = 0x02,
	CCV_C3 = 0x03,
};

typedef struct {
	int sig[4];
	int rows;
	int cols;
	int type;
	union {
		char* ptr;
		int* i;
		float* fl;
		double* db;
	} data;
} ccv_dense_matrix_t;

typedef struct {
	int sig[4];
	int type;
} ccv_sparse_matrix_t;

typedef ccv_matrix_t void;

/* matrix operations */
ccv_dense_matrix_t* ccv_dense_matrix_new(int rows, int cols, int type, void* data = NULL, int* sig = NULL);
ccv_sparse_matrix_t* ccv_sparse_matrix_new(int rows, int cols, int type, void* data = NULL, int* sig = NULL);
ccv_dense_matrix_t* ccv_get_dense_matrix(ccv_matrix_t* mat);
ccv_sparse_matrix_t* ccv_get_sparse_matrix(ccv_matrix_t* mat);
void ccv_matrix_free(ccv_matrix_t* mat);
double ccv_trace(ccv_matrix_t* mat);
double ccv_norm(ccv_matrix_t* mat, int type);
void ccv_gemm(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t* d, ccv_matrix_t* c = NULL);

/* matrix build blocks */


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
