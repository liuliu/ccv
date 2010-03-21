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
#include <xmmintrin.h>
#include <assert.h>

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
	CCV_MATRIX_DENSE  = 0x010000,
	CCV_MATRIX_SPARSE = 0x020000,
	CCV_MATRIX_CSR    = 0x040000,
	CCV_MATRIX_CSC    = 0x080000,
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
	int type;
	int sig[5];
	int refcount;
	int rows;
	int cols;
	int nnz;
	int* index;
	int* offset;
	ccv_matrix_cell_t data;
} ccv_compressed_sparse_matrix_t;

#define ccv_clamp(x, a, b) (((x) < (a)) ? (a) : (((x) > (b)) ? (b) : (x)))
#define ccv_min(a, b) (((a) < (b)) ? (a) : (b))
#define ccv_max(a, b) (((a) > (b)) ? (a) : (b))

/* matrix operations */
ccv_dense_matrix_t* ccv_dense_matrix_new(int rows, int cols, int type, void* data, int* sig);
ccv_dense_matrix_t ccv_dense_matrix(int rows, int cols, int type, void* data, int* sig);
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
	case CCV_32S: ((int*)(ptr))[(i)] = (int)(value); break; \
	case CCV_32F: ((float*)(ptr))[(i)] = (float)value; break; \
	case CCV_64F: ((double*)(ptr))[(i)] = (double)value; break; \
	default: ((unsigned char*)(ptr))[(i)] = ccv_clamp((int)(value), 0, 255); }

/* unswitch for loop macros */
#define __ccv_get_32s_value(ptr, i) ((int*)(ptr))[(i)]
#define __ccv_get_32f_value(ptr, i) ((float*)(ptr))[(i)]
#define __ccv_get_64f_value(ptr, i) ((double*)(ptr))[(i)]
#define __ccv_get_8u_value(ptr, i) ((unsigned char*)(ptr))[(i)]
#define ccv_matrix_getter(type, block ,rest...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(rest, __ccv_get_32s_value); break; } \
	case CCV_32F: { block(rest, __ccv_get_32f_value); break; } \
	case CCV_64F: { block(rest, __ccv_get_64f_value); break; } \
	default: { block(rest, __ccv_get_8u_value); } } }

#define ccv_matrix_getter_a(type, block ,rest...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(rest, __ccv_get_32s_value); break; } \
	case CCV_32F: { block(rest, __ccv_get_32f_value); break; } \
	case CCV_64F: { block(rest, __ccv_get_64f_value); break; } \
	default: { block(rest, __ccv_get_8u_value); } } }

#define ccv_matrix_getter_b(type, block ,rest...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(rest, __ccv_get_32s_value); break; } \
	case CCV_32F: { block(rest, __ccv_get_32f_value); break; } \
	case CCV_64F: { block(rest, __ccv_get_64f_value); break; } \
	default: { block(rest, __ccv_get_8u_value); } } }

#define __ccv_set_32s_value(ptr, i, value) ((int*)(ptr))[(i)] = (int)(value)
#define __ccv_set_32f_value(ptr, i, value) ((float*)(ptr))[(i)] = (float)(value)
#define __ccv_set_64f_value(ptr, i, value) ((double*)(ptr))[(i)] = (double)(value)
#define __ccv_set_8u_value(ptr, i, value) ((unsigned char*)(ptr))[(i)] = ccv_clamp((int)(value), 0, 255)
#define ccv_matrix_setter(type, block ,rest...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(rest, __ccv_set_32s_value); break; } \
	case CCV_32F: { block(rest, __ccv_set_32f_value); break; } \
	case CCV_64F: { block(rest, __ccv_set_64f_value); break; } \
	default: { block(rest, __ccv_set_8u_value); } } }

#define ccv_matrix_setter_a(type, block ,rest...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(rest, __ccv_set_32s_value); break; } \
	case CCV_32F: { block(rest, __ccv_set_32f_value); break; } \
	case CCV_64F: { block(rest, __ccv_set_64f_value); break; } \
	default: { block(rest, __ccv_set_8u_value); } } }

#define ccv_matrix_setter_b(type, block ,rest...) { switch (CCV_GET_DATA_TYPE(type)) { \
	case CCV_32S: { block(rest, __ccv_set_32s_value); break; } \
	case CCV_32F: { block(rest, __ccv_set_32f_value); break; } \
	case CCV_64F: { block(rest, __ccv_set_64f_value); break; } \
	default: { block(rest, __ccv_set_8u_value); } } }

/* basic io */
enum {
	CCV_SERIAL_GRAY           = 0x100,
	CCV_SERIAL_COLOR          = 0x300,
	CCV_SERIAL_ANY_STREAM     = 0x010,
	CCV_SERIAL_PLAIN_STREAM   = 0x011,
	CCV_SERIAL_DEFLATE_STREAM = 0x012,
	CCV_SERIAL_JPEG_STREAM    = 0x013,
	CCV_SERIAL_PNG_STREAM     = 0x014,
	CCV_SERIAL_ANY_FILE       = 0x020,
	CCV_SERIAL_BMP_FILE       = 0x021,
	CCV_SERIAL_JPEG_FILE      = 0x022,
	CCV_SERIAL_PNG_FILE       = 0x023,
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

enum {
	CCV_L2_NORM = 0x00,
	CCV_L1_NORM = 0x01,
};

double ccv_norm(ccv_matrix_t* mat, int type);
double ccv_dot(ccv_matrix_t* a, ccv_matrix_t* b);
double ccv_sum(ccv_matrix_t* mat);

enum {
	CCV_A_TRANSPOSE = 0x01,
	CCV_B_TRANSPOSE = 0X02,
	CCV_C_TRANSPOSE = 0X04,
};

void ccv_gemm(ccv_matrix_t* a, ccv_matrix_t* b, double alpha, ccv_matrix_t* c, double beta, int transpose, ccv_matrix_t** d);

/* matrix build blocks */
ccv_dense_matrix_t* ccv_get_dense_matrix(ccv_matrix_t* mat);
ccv_sparse_matrix_t* ccv_get_sparse_matrix(ccv_matrix_t* mat);
ccv_dense_vector_t* ccv_get_sparse_matrix_vector(ccv_sparse_matrix_t* mat, int index);
ccv_matrix_cell_t ccv_get_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col);
void ccv_set_sparse_matrix_cell(ccv_sparse_matrix_t* mat, int row, int col, void* data);
void ccv_compress_sparse_matrix(ccv_sparse_matrix_t* mat, ccv_compressed_sparse_matrix_t** csm);
void ccv_decompress_sparse_matrix(ccv_compressed_sparse_matrix_t* csm, ccv_sparse_matrix_t** smt);
void ccv_convert(ccv_matrix_t* a, ccv_matrix_t** b, int type);
void ccv_slice(ccv_matrix_t* a, ccv_matrix_t** b, int y, int x, int rows, int cols);

/* basic data structures */

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
	int size;
	int rsize;
	int rnum;
	void* data;
} ccv_array_t;

ccv_array_t* ccv_array_new(int rnum, int rsize);
void ccv_array_push(ccv_array_t* array, void* r);
typedef int(*ccv_array_group_func)(const void*, const void*, void*);
int ccv_array_group(ccv_array_t* array, ccv_array_t** index, ccv_array_group_func gfunc, void* data);
void ccv_array_clear(ccv_array_t* array);
void ccv_array_free(ccv_array_t* array);

#define ccv_array_get(a, i) (((char*)((a)->data)) + (a)->rsize * (i))

/* numerical algorithms */
/* clarification about algebra and numerical algorithms:
 * when using the word "algebra", I assume the operation is well established in Mathematic sense
 * and can be calculated with a straight-forward, finite sequence of operation. The "numerical"
 * in other word, refer to a class of algorithm that can only approximate/or iteratively found the
 * solution. Thus, "invert" would be classified as numercial because of the sense that in some case,
 * it can only be "approximate" (in least-square sense), so to "solve". */
void ccv_invert(ccv_matrix_t* a, ccv_matrix_t** b);
void ccv_solve(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d);
void ccv_eigen(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d);
void ccv_minimize(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d);
void ccv_filter(ccv_matrix_t* a, ccv_matrix_t* b, ccv_matrix_t** d);
typedef double(*ccv_filter_kernel_func)(double x, double y, void*);
void ccv_filter_kernel(ccv_dense_matrix_t* x, ccv_filter_kernel_func func, void* data);

/* modern numerical algorithms */
void ccv_sparse_coding(ccv_matrix_t* x, int k, ccv_matrix_t** A, ccv_matrix_t** y);
void ccv_compressive_sensing_reconstruct(ccv_matrix_t* a, ccv_matrix_t* x, ccv_matrix_t** y);

/* basic computer vision algorithms / or build blocks */
void ccv_sobel(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int dx, int dy);
void ccv_hog(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int size);

enum {
	CCV_INTER_AREA   = 0x01,
	CCV_INTER_LINEAR = 0X02,
	CCV_INTER_CUBIC  = 0X03,
	CCV_INTER_LACZOS = 0X04,
};

void ccv_resample(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int rows, int cols, int type);
void ccv_sample_down(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b);
void ccv_sample_up(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b);

enum {
	CCV_FLIP_X = 0x01,
	CCV_FLIP_Y = 0x02,
};

void ccv_flip(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, int type);

/* modern computer vision algorithms */
/* SIFT, DAISY, SURF, MSER, SGF, SSD, FAST */
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

void ccv_daisy(ccv_dense_matrix_t* a, ccv_dense_matrix_t** b, ccv_daisy_param_t params);
void ccv_sift(ccv_dense_matrix_t* a);

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
} ccv_sgf_params_t;

typedef struct {
	ccv_rect_t rect;
	int neighbors;
	int id;
	float confidence;
} ccv_sgf_comp_t;

enum {
	CCV_SGF_NO_NESTED = 0x10000000,
};

void ccv_sgf_classifier_cascade_new(ccv_dense_matrix_t** posimg, int posnum, char** bgfiles, int bgnum, int negnum, ccv_size_t size, const char* dir, ccv_sgf_params_t params);
ccv_array_t* ccv_sgf_detect_objects(ccv_dense_matrix_t* a, ccv_sgf_classifier_cascade_t** _cascade, int count, int min_neighbors, int flags, ccv_size_t min_size);
ccv_sgf_classifier_cascade_t* ccv_load_sgf_classifier_cascade(const char* directory);
void ccv_sgf_classifier_cascade_free(ccv_sgf_classifier_cascade_t* cascade);

/* modern machine learning algorithms */
/* RBM, LLE, APCluster */

/****************************************************************************************\

  Generic implementation of QuickSort algorithm.
  ----------------------------------------------
  Using this macro user can declare customized sort function that can be much faster
  than built-in qsort function because of lower overhead on elements
  comparison and exchange. The macro takes less_than (or LT) argument - a macro or function
  that takes 2 arguments returns non-zero if the first argument should be before the second
  one in the sorted sequence and zero otherwise.

  Example:

    Suppose that the task is to sort points by ascending of y coordinates and if
    y's are equal x's should ascend.

    The code is:
    ------------------------------------------------------------------------------
           #define cmp_pts( pt1, pt2 ) \
               ((pt1).y < (pt2).y || ((pt1).y < (pt2).y && (pt1).x < (pt2).x))

           [static] CV_IMPLEMENT_QSORT( icvSortPoints, CvPoint, cmp_pts )
    ------------------------------------------------------------------------------

    After that the function "void icvSortPoints( CvPoint* array, size_t total, int aux );"
    is available to user.

  aux is an additional parameter, which can be used when comparing elements.
  The current implementation was derived from *BSD system qsort():

    * Copyright (c) 1992, 1993
    *  The Regents of the University of California.  All rights reserved.
    *
    * Redistribution and use in source and binary forms, with or without
    * modification, are permitted provided that the following conditions
    * are met:
    * 1. Redistributions of source code must retain the above copyright
    *    notice, this list of conditions and the following disclaimer.
    * 2. Redistributions in binary form must reproduce the above copyright
    *    notice, this list of conditions and the following disclaimer in the
    *    documentation and/or other materials provided with the distribution.
    * 3. All advertising materials mentioning features or use of this software
    *    must display the following acknowledgement:
    *  This product includes software developed by the University of
    *  California, Berkeley and its contributors.
    * 4. Neither the name of the University nor the names of its contributors
    *    may be used to endorse or promote products derived from this software
    *    without specific prior written permission.
    *
    * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
    * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
    * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
    * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
    * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
    * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
    * SUCH DAMAGE.

\****************************************************************************************/

#define CCV_SWAP(a,b,t) ((t) = (a), (a) = (b), (b) = (t))

#define CCV_IMPLEMENT_QSORT_EX(func_name, T, LT, swap_func, user_data_type)                     \
void func_name(T *array, size_t total, user_data_type aux)                                      \
{                                                                                               \
    int isort_thresh = 7;                                                                       \
    T t;                                                                                        \
    int sp = 0;                                                                                 \
                                                                                                \
    struct                                                                                      \
    {                                                                                           \
        T *lb;                                                                                  \
        T *ub;                                                                                  \
    }                                                                                           \
    stack[48];                                                                                  \
                                                                                                \
    if( total <= 1 )                                                                            \
        return;                                                                                 \
                                                                                                \
    stack[0].lb = array;                                                                        \
    stack[0].ub = array + (total - 1);                                                          \
                                                                                                \
    while( sp >= 0 )                                                                            \
    {                                                                                           \
        T* left = stack[sp].lb;                                                                 \
        T* right = stack[sp--].ub;                                                              \
                                                                                                \
        for(;;)                                                                                 \
        {                                                                                       \
            int i, n = (int)(right - left) + 1, m;                                              \
            T* ptr;                                                                             \
            T* ptr2;                                                                            \
                                                                                                \
            if( n <= isort_thresh )                                                             \
            {                                                                                   \
            insert_sort:                                                                        \
                for( ptr = left + 1; ptr <= right; ptr++ )                                      \
                {                                                                               \
                    for( ptr2 = ptr; ptr2 > left && LT(ptr2[0],ptr2[-1], aux); ptr2--)          \
                        swap_func( ptr2[0], ptr2[-1], array, aux, t );                          \
                }                                                                               \
                break;                                                                          \
            }                                                                                   \
            else                                                                                \
            {                                                                                   \
                T* left0;                                                                       \
                T* left1;                                                                       \
                T* right0;                                                                      \
                T* right1;                                                                      \
                T* pivot;                                                                       \
                T* a;                                                                           \
                T* b;                                                                           \
                T* c;                                                                           \
                int swap_cnt = 0;                                                               \
                                                                                                \
                left0 = left;                                                                   \
                right0 = right;                                                                 \
                pivot = left + (n/2);                                                           \
                                                                                                \
                if( n > 40 )                                                                    \
                {                                                                               \
                    int d = n / 8;                                                              \
                    a = left, b = left + d, c = left + 2*d;                                     \
                    left = LT(*a, *b, aux) ? (LT(*b, *c, aux) ? b : (LT(*a, *c, aux) ? c : a))  \
                                      : (LT(*c, *b, aux) ? b : (LT(*a, *c, aux) ? a : c));      \
                                                                                                \
                    a = pivot - d, b = pivot, c = pivot + d;                                    \
                    pivot = LT(*a, *b, aux) ? (LT(*b, *c, aux) ? b : (LT(*a, *c, aux) ? c : a)) \
                                      : (LT(*c, *b, aux) ? b : (LT(*a, *c, aux) ? a : c));      \
                                                                                                \
                    a = right - 2*d, b = right - d, c = right;                                  \
                    right = LT(*a, *b, aux) ? (LT(*b, *c, aux) ? b : (LT(*a, *c, aux) ? c : a)) \
                                      : (LT(*c, *b, aux) ? b : (LT(*a, *c, aux) ? a : c));      \
                }                                                                               \
                                                                                                \
                a = left, b = pivot, c = right;                                                 \
                pivot = LT(*a, *b, aux) ? (LT(*b, *c, aux) ? b : (LT(*a, *c, aux) ? c : a))     \
                                   : (LT(*c, *b, aux) ? b : (LT(*a, *c, aux) ? a : c));         \
                if( pivot != left0 )                                                            \
                {                                                                               \
                    swap_func( *pivot, *left0, array, aux, t );                                 \
                    pivot = left0;                                                              \
                }                                                                               \
                left = left1 = left0 + 1;                                                       \
                right = right1 = right0;                                                        \
                                                                                                \
                for(;;)                                                                         \
                {                                                                               \
                    while( left <= right && !LT(*pivot, *left, aux) )                           \
                    {                                                                           \
                        if( !LT(*left, *pivot, aux) )                                           \
                        {                                                                       \
                            if( left > left1 )                                                  \
                                swap_func( *left1, *left, array, aux, t );                      \
                            swap_cnt = 1;                                                       \
                            left1++;                                                            \
                        }                                                                       \
                        left++;                                                                 \
                    }                                                                           \
                                                                                                \
                    while( left <= right && !LT(*right, *pivot, aux) )                          \
                    {                                                                           \
                        if( !LT(*pivot, *right, aux) )                                          \
                        {                                                                       \
                            if( right < right1 )                                                \
                                swap_func( *right1, *right, array, aux, t );                    \
                            swap_cnt = 1;                                                       \
                            right1--;                                                           \
                        }                                                                       \
                        right--;                                                                \
                    }                                                                           \
                                                                                                \
                    if( left > right )                                                          \
                        break;                                                                  \
                    swap_func( *left, *right, array, aux, t );                                  \
                    swap_cnt = 1;                                                               \
                    left++;                                                                     \
                    right--;                                                                    \
                }                                                                               \
                                                                                                \
                if( swap_cnt == 0 )                                                             \
                {                                                                               \
                    left = left0, right = right0;                                               \
                    goto insert_sort;                                                           \
                }                                                                               \
                                                                                                \
                n = ccv_min( (int)(left1 - left0), (int)(left - left1) );                       \
                for( i = 0; i < n; i++ )                                                        \
                    swap_func( left0[i], left[i-n], array, aux, t );                            \
                                                                                                \
                n = ccv_min( (int)(right0 - right1), (int)(right1 - right) );                   \
                for( i = 0; i < n; i++ )                                                        \
                    swap_func( left[i], right0[i-n+1], array, aux, t );                         \
                n = (int)(left - left1);                                                        \
                m = (int)(right1 - right);                                                      \
                if( n > 1 )                                                                     \
                {                                                                               \
                    if( m > 1 )                                                                 \
                    {                                                                           \
                        if( n > m )                                                             \
                        {                                                                       \
                            stack[++sp].lb = left0;                                             \
                            stack[sp].ub = left0 + n - 1;                                       \
                            left = right0 - m + 1, right = right0;                              \
                        }                                                                       \
                        else                                                                    \
                        {                                                                       \
                            stack[++sp].lb = right0 - m + 1;                                    \
                            stack[sp].ub = right0;                                              \
                            left = left0, right = left0 + n - 1;                                \
                        }                                                                       \
                    }                                                                           \
                    else                                                                        \
                        left = left0, right = left0 + n - 1;                                    \
                }                                                                               \
                else if( m > 1 )                                                                \
                    left = right0 - m + 1, right = right0;                                      \
                else                                                                            \
                    break;                                                                      \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
}

#define __ccv_qsort_default_swap(a, b, array, aux, t) CCV_SWAP((a), (b), (t))

#define CCV_IMPLEMENT_QSORT(func_name, T, cmp)  \
    CCV_IMPLEMENT_QSORT_EX(func_name, T, cmp, __ccv_qsort_default_swap, int)

#endif
