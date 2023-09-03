/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

/**
 * This header is included into main ccv.h
 * such that enables toll-free bridging between
 * ccv_nnc_tensor_t and ccv_dense_matrix_t
 * In effect, ccv_dense_matrix_t will be a specialized
 * version of ccv_nnc_tensor_t. We are taking some penalties
 * from this change though, namely, the size of ccv_dense_matrix_t
 * will be bigger now.
 */

#ifndef GUARD_ccv_nnc_tfb_h
#define GUARD_ccv_nnc_tfb_h

#include <unistd.h>
#include <stdint.h>

enum {
	CCV_TENSOR_FORMAT_NCHW = 0x01,
	CCV_TENSOR_FORMAT_NHWC = 0x02,
	CCV_TENSOR_FORMAT_CHWN = 0x04,
};

enum {
	CCV_TENSOR_CPU_MEMORY = 0x1,
	CCV_TENSOR_GPU_MEMORY = 0x2,
};

enum {
	CCV_COMPUTE_DEVICE_000 = 0x00000,
	CCV_COMPUTE_DEVICE_001 = 0x00100,
	CCV_COMPUTE_DEVICE_002 = 0x00200,
	CCV_COMPUTE_DEVICE_003 = 0x00300,
	CCV_COMPUTE_DEVICE_004 = 0x00400,
	CCV_COMPUTE_DEVICE_005 = 0x00500,
	CCV_COMPUTE_DEVICE_006 = 0x00600,
	CCV_COMPUTE_DEVICE_007 = 0x00700,
	CCV_COMPUTE_DEVICE_ANY = 0xfff00, // The optimal allocation will be found by the algorithm.
};

#define CCV_TENSOR_GET_MEMORY(type) ((type) & 0x3)
#define CCV_TENSOR_GET_DEVICE(type) ((type) & 0xfff00)
#define CCV_TENSOR_GET_DEVICE_ID(type) (CCV_TENSOR_GET_DEVICE(type) >> 8)
#define CCV_TENSOR_SET_DEVICE_ID(type, device_id) (type) = (((type) & ~0xfff00) | (((device_id) & 0xfff) << 8))

enum {
	CCV_TENSOR_VIEW       = 0x01000000,
	CCV_TENSOR_MULTIVIEW  = 0x02000000,
	CCV_TENSOR_PINNED_MEM = 0x04000000, // tensor is pinned in CUDA. This matches CCV_PINNED_MEM.
};

typedef struct {
	short v;
} ccv_float16_t; // In this way, you cannot perform any operations on float16.

typedef union ccv_numeric_data_u {
	char* i8;
	unsigned char* u8;
	int* i32;
	ccv_float16_t* f16;
	float* f32;
	int64_t* i64;
	uint64_t* u64;
	double* f64;
	void* ptr;
} ccv_numeric_data_t;

#define CCV_NNC_MAX_DIM_ALLOC (12)
#define CCV_NNC_MAX_DIM (2)

typedef struct {
	int type;
	int format;
	int datatype;
	int reserved; // This is reserved, for quantized tensor, this might be used in some ways.
	int dim[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_tensor_param_t;

typedef struct {
	int type;
	int refcount;
	ccv_numeric_data_t data;
	off_t dataof; // Offset.
	uintptr_t alias_ref;
	uint64_t data_size;
	uint64_t sig;
	ccv_nnc_tensor_param_t info;
} ccv_nnc_tensor_t;

typedef struct {
	int type;
	int refcount;
	ccv_numeric_data_t data;
	off_t dataof; // Offset for the data.
	uintptr_t alias_ref;
	uint64_t data_size;
	uint64_t sig;
	ccv_nnc_tensor_param_t info;
	/* tensor view and tensor shares the same data structure besides the following. */
	int contiguous; // If we have other flags later, we can make this a generic flag.
	off_t off; // Offset from the original tensor. This could be different from dataof.
	int stride[CCV_NNC_MAX_DIM_ALLOC]; /**< "stride" or, length */
} ccv_nnc_tensor_view_t;

#define CCV_IS_TENSOR_VIEW(x) ((*(int*)(x)) & CCV_TENSOR_VIEW)
#define CCV_IS_TENSOR_MULTIVIEW(x) ((*(int*)(x)) & CCV_TENSOR_MULTIVIEW)
#define CCV_IS_TENSOR_CONTIGUOUS(x) (!CCV_IS_TENSOR_VIEW(x) || (((ccv_nnc_tensor_view_t*)x)->contiguous == 1))

#if CCV_NNC_TENSOR_TFB
#define CCV_TENSOR_IS_DENSE_MATRIX(x) (((x) & 0xFFF) > 0) // has channel components
typedef struct {
	int type;
	int refcount;
	ccv_numeric_data_t data;
	off_t reserved0;
	uintptr_t reserved1;
	uint64_t reserved2;
	uint64_t sig;
	// This is used for toll-free bridging between ccv_dense_matrix_t and ccv_nnc_tensor_t
	// Note that this is bigger than it is needed, we carefully structured this
	// such that bit is reused as much, but still some wasted spaces.
	union {
		struct {
			int resides;
			int format;
			int datatype;
			int reserved3;
			int rows;
			int cols;
			int channels;
			int reserved4; /* This reserved bit need to be zero'ed such that later dim is not cared. */
			int step;
			union {
				unsigned char u8;
				int i32;
				ccv_float16_t f16;
				float f32;
				int64_t i64;
				double f64;
				void* p;
			} tb;
		};
		ccv_nnc_tensor_param_t info;
	}; 
} ccv_dense_matrix_t;
#else
#define CCV_TENSOR_IS_DENSE_MATRIX(x) (0)
typedef struct {
	int type;
	int refcount;
	uint64_t sig;
	int cols;
	int rows;
	int step;
	union {
		unsigned char u8;
		int i32;
		ccv_float16_t f16;
		float f32;
		int64_t i64;
		double f64;
		void* p;
	} tb;
	ccv_numeric_data_t data;
} ccv_dense_matrix_t;
#endif

#endif
