/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_h
#define GUARD_ccv_nnc_h

#include <ccv.h>

enum {
	CCV_TENSOR_FORMAT_NCHW = 0x01,
	CCV_TENSOR_FORMAT_NHWC = 0x02,
};

enum {
	CCV_TENSOR_CPU_MEMORY = 0x1,
	CCV_TENSOR_GPU_MEMORY = 0x2,
};

#define CCV_NNC_MAX_DIM (2)

typedef struct {
	int type;
	int format;
	int dim[CCV_NNC_MAX_DIM];
	int channels;
} ccv_nnc_tensor_param_t;

typedef struct {
	int type;
	ccv_nnc_tensor_param_t meta;
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
	struct {
		int dim[CCV_NNC_MAX_DIM];
	} size; /**< [size] The window size for the layer. For full connect layer, it is 1 because it is 1x1 convolutional layer with count of filters */
	union {
		struct {
			int count; /**< [convolutional.count] The number of filters for convolutional layer. */
			int channels; /**< [convolutional.channels] The number of channels for convolutional filter. */
		} convolutional;
		struct {
		} pool;
		struct {
			float kappa; /**< [rnorm.kappa] As of b[i] = a[i] / (rnorm.kappa + rnorm.alpha * sum(a, i - rnorm.size / 2, i + rnorm.size / 2)) ^ rnorm.beta */
			float alpha; /**< [rnorm.alpha] See **rnorm.kappa**. */
			float beta; /**< [rnorm.beta] See **rnorm.kappa**. */
		} rnorm;
		struct {
			int count; /**< [full_connect.count] The number of output nodes for full connect layer. */
		} full_connect;
	};
} ccv_nnc_net_param_t;

typedef struct {
	struct {
		int dim[CCV_NNC_MAX_DIM];
	} stride;
	struct {
		int front[CCV_NNC_MAX_DIM];
		int back[CCV_NNC_MAX_DIM];
	} border;
} ccv_nnc_net_hint_t;

typedef struct {
	int type;
	int provide;
	ccv_nnc_net_param_t meta;
	ccv_nnc_net_hint_t hint;
} ccv_nnc_net_t;

typedef void(*ccv_nnc_net_inference_f)(const ccv_nnc_net_t* net, const ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, const ccv_nnc_tensor_t* w, const ccv_nnc_tensor_t* bias);
typedef void(*ccv_nnc_net_backprop_f)(const ccv_nnc_net_t* net, const ccv_nnc_tensor_t* a, const ccv_nnc_tensor_t* b, const ccv_nnc_tensor_t* c, ccv_nnc_tensor_t* d, ccv_nnc_tensor_t* w, ccv_nnc_tensor_t* bias);

typedef struct {
	int tensor_formats; /**< [formats] The supported formats for this API implementation. */
	ccv_nnc_net_inference_f inference;
	ccv_nnc_net_backprop_f backprop;
} ccv_nnc_api_t;

/**
 * Level-0 API
 */
void ccv_nnc_init(void);

/**
 * Level-1 API
 */
CCV_WARN_UNUSED(ccv_nnc_tensor_t*) ccv_nnc_tensor_new(const void* ptr, const ccv_nnc_tensor_param_t params, const int flags);
void ccv_nnc_tensor_free(ccv_nnc_tensor_t* tensor);
CCV_WARN_UNUSED(ccv_nnc_net_t*) ccv_nnc_net_new(const void* ptr, const int type, const ccv_nnc_net_param_t params, const int flags);
void ccv_nnc_net_free(ccv_nnc_net_t* net);
CCV_WARN_UNUSED(int) ccv_nnc_net_hint_verify(const ccv_nnc_net_hint_t hint, const ccv_nnc_net_param_t net, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b);
CCV_WARN_UNUSED(ccv_nnc_net_hint_t) ccv_nnc_net_hint_guess(const ccv_nnc_net_param_t net, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b);
void ccv_nnc_net_inference(const ccv_nnc_net_t* net, const ccv_nnc_tensor_t* a, ccv_nnc_tensor_t* b, const ccv_nnc_tensor_t* w, const ccv_nnc_tensor_t* bias);
void ccv_nnc_net_backprop(const ccv_nnc_net_t* net, const ccv_nnc_tensor_t* a, const ccv_nnc_tensor_t* b, const ccv_nnc_tensor_t* c, ccv_nnc_tensor_t* d, ccv_nnc_tensor_t* w, ccv_nnc_tensor_t* bias);

/**
 * Level-2 API
 */
typedef struct {
} ccv_nnc_solver_t;

#endif
