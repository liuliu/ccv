/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_h
#define GUARD_ccv_nnc_h

#include "../ccv.h"

enum {
	CCV_TENSOR_FORMAT_NCHW = 0x01,
	CCV_TENSOR_FORMAT_NHWC = 0x02,
};

enum {
	CCV_TENSOR_CPU_MEMORY = 0x1,
	CCV_TENSOR_GPU_MEMORY = 0x2,
};

typedef struct {
	int type;
	int format;
	int rows;
	int cols;
	int channels;
	struct {
		int rows;
		int cols;
		int channels;
	} stride;
} ccv_nnc_tensor_param_t;

typedef struct {
	int type;
	ccv_nnc_tensor_param_t params;
	union {
		unsigned char* u8;
		int* i32;
		float* f32;
		int64_t* i64;
		double* f64;
	} data;
} ccv_nnc_tensor_t;

enum {
	CCV_NNC_TYPE_CONVOLUTIONAL,
	CCV_NNC_TYPE_FULL_CONNECT,
	CCV_NNC_TYPE_MAX_POOL,
	CCV_NNC_TYPE_AVERAGE_POOL,
	CCV_NNC_TYPE_LOCAL_RESPONSE_NORM,
	CCV_NNC_TYPE_COUNT,
};

typedef struct {
	union {
		struct {
			int count; /**< [convolutional.count] The number of filters for convolutional layer. */
			int strides; /**< [convolutional.strides] The strides for convolutional filter. */
			int border; /**< [convolutional.border] The padding border size for the input matrix. */
			int rows; /**< [convolutional.rows] The number of rows for convolutional filter. */
			int cols; /**< [convolutional.cols] The number of columns for convolutional filter. */
			int channels; /**< [convolutional.channels] The number of channels for convolutional filter. */
			int partition; /**< [convolutional.partition] The number of partitions for convolutional filter. */
		} convolutional;
		struct {
			int strides; /**< [pool.strides] The strides for pooling layer. */
			int size; /**< [pool.size] The window size for pooling layer. */
			int border; /**< [pool.border] The padding border size for the input matrix. */
		} pool;
		struct {
			int size; /**< [rnorm.size] The size of local response normalization layer. */
			float kappa; /**< [rnorm.kappa] As of b[i] = a[i] / (rnorm.kappa + rnorm.alpha * sum(a, i - rnorm.size / 2, i + rnorm.size / 2)) ^ rnorm.beta */
			float alpha; /**< [rnorm.alpha] See **rnorm.kappa**. */
			float beta; /**< [rnorm.beta] See **rnorm.kappa**. */
		} rnorm;
		struct {
			int relu; /**< [full_connect.relu] 0 - ReLU, 1 - no ReLU */
			int count; /**< [full_connect.count] The number of output nodes for full connect layer. */
		} full_connect;
	};
} ccv_nnc_net_param_t;

typedef struct {
	int type;
	int provider;
	ccv_nnc_net_param_t params;
} ccv_nnc_net_t;

typedef void(*ccv_nnc_net_inference_f)(ccv_nnc_net_t* net, ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b);
typedef void(*ccv_nnc_net_backprop_f)(ccv_nnc_net_t* net, ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, ccv_nnc_tensor_t* c, ccv_nnc_tensor_t* d, ccv_nnc_tensor_t* w, ccv_nnc_tensor_t* bias);

typedef struct {
	int type;
	int formats; /**< [formats] The supported formats for this API implementation. */
	ccv_nnc_net_inference_f inference;
	ccv_nnc_net_backprop_f backprop;
} ccv_nnc_api_t;

typedef struct {
	ccv_nnc_api_t convolutional;
	ccv_nnc_api_t full_connect;
	ccv_nnc_api_t max_pool;
	ccv_nnc_api_t average_pool;
	ccv_nnc_api_t local_response_nrom;
} ccv_nnc_api_provider_t;

/**
 * Level-0 API
 */
void ccv_nnc_init(void);

/**
 * Level-1 API
 */
CCV_WARN_UNUSED(ccv_nnc_tensor_t*) ccv_nnc_tensor_new(const void* ptr, ccv_nnc_tensor_param_t params, int flags);
void ccv_nnc_tensor_free(ccv_nnc_tensor_t* tensor);
CCV_WARN_UNUSED(ccv_nnc_net_t*) ccv_nnc_net_new(const void* ptr, ccv_nnc_net_param_t params, int flags);
void ccv_nnc_net_free(ccv_nnc_net_t* net);
void ccv_nnc_net_inference(ccv_nnc_net_t* net, ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b);
void ccv_nnc_net_backprop(ccv_nnc_net_t* net, ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, ccv_nnc_tensor_t* c, ccv_nnc_tensor_t* d, ccv_nnc_tensor_t* w, ccv_nnc_tensor_t* bias);

/**
 * Level-2 API
 */
typedef struct {
} ccv_nnc_solver_t;

#endif
