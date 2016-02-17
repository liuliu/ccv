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
	// These are the list of computation kernels.
	CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD = 0,
	CCV_NNC_COMPUTE_CONVOLUTIONAL_BACKWARD,
	CCV_NNC_COMPUTE_FULL_CONNECT_FORWARD,
	CCV_NNC_COMPUTE_FULL_CONNECT_BACKWARD,
	CCV_NNC_COMPUTE_MAX_POOL_FORWARD,
	CCV_NNC_COMPUTE_MAX_POOL_BACKWARD,
	CCV_NNC_COMPUTE_AVERAGE_POOL_FORWARD,
	CCV_NNC_COMPUTE_AVERAGE_POOL_BACKWARD,
	CCV_NNC_COMPUTE_SOFTMAX_FORWARD,
	CCV_NNC_COMPUTE_SOFTMAX_BACKWARD,
	CCV_NNC_COMPUTE_BATCH_NORM_FORWARD,
	CCV_NNC_COMPUTE_BATCH_NORM_BACKWARD,
	CCV_NNC_COMPUTE_RELU_FORWARD,
	CCV_NNC_COMPUTE_RELU_BACKWARD,
	CCV_NNC_COMPUTE_COUNT,
};

enum {
	CCV_NNC_ACCUMULATE_OUTPUT = 0x01, // enable accumulate outputs
};

typedef struct {
	struct {
		int dim[CCV_NNC_MAX_DIM_ALLOC];
	} size; /**< [size] The window size for the layer. For full connect layer, it is 1 because it is 1x1 convolutional layer with count of filters */
	union {
		struct {
			int count; /**< [convolutional.count] The number of filters for convolutional layer. */
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
} ccv_nnc_net_command_param_t;

typedef struct {
	struct {
		int dim[CCV_NNC_MAX_DIM_ALLOC];
	} stride;
	struct {
		int begin[CCV_NNC_MAX_DIM_ALLOC];
		int end[CCV_NNC_MAX_DIM_ALLOC];
	} border;
} ccv_nnc_net_hint_t;

typedef struct {
	int type;
	int compute;
	int backend;
	ccv_nnc_net_command_param_t info;
} ccv_nnc_net_command_t;

typedef void(*ccv_nnc_net_command_exec_f)(const ccv_nnc_net_command_t command, const ccv_nnc_net_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size);

typedef struct {
	int tensor_formats; /**< [formats] The supported formats for this API implementation. */
	ccv_nnc_net_command_exec_f exec;
} ccv_nnc_command_api_t;

/**
 * Level-0 API
 */
void ccv_nnc_init(void);

/**
 * Level-1 API
 */
// For tensor
CCV_WARN_UNUSED(ccv_nnc_tensor_t*) ccv_nnc_tensor_new(const void* ptr, const ccv_nnc_tensor_param_t params, const int flags);
void ccv_nnc_tensor_free(ccv_nnc_tensor_t* tensor);
// Allocating on stack
CCV_WARN_UNUSED(ccv_nnc_tensor_t) ccv_nnc_tensor(const void* ptr, const ccv_nnc_tensor_param_t params, const int flags);
// For computation node
CCV_WARN_UNUSED(ccv_nnc_net_command_t) ccv_nnc_net_command(const int compute, const ccv_nnc_net_command_param_t params, const int flags);
CCV_WARN_UNUSED(int) ccv_nnc_net_hint_verify(const ccv_nnc_net_hint_t hint, const ccv_nnc_net_command_param_t node, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b);
CCV_WARN_UNUSED(ccv_nnc_net_hint_t) ccv_nnc_net_hint_guess(const ccv_nnc_net_command_param_t node, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_tensor_param_t* outputs, const int output_size);
void ccv_nnc_net_command_exec(const ccv_nnc_net_command_t command, const ccv_nnc_net_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size);

/**
 * Level-2 API
 */

typedef struct ccv_nnc_net_graph_s ccv_nnc_net_graph_t;

/**
 * The biggest design principle of these methods, especially now it becomes more involved on the
 * interface level comparing to libccv's most functions (which takes some matrices, and output
 * some matrices or arrays). Therefore, ownership of these pointers are important for C interface.
 * For each of these functions, ownership of these pointers are especially called out to cause no
 * confusions at all.
 */
// Create a graph containing one node with input and output tensors allocated.
// In this function, no ownership of these passed tensors changed, you are responsible to
// deallocate the tensors when it is done.
CCV_WARN_UNUSED(ccv_nnc_net_graph_t*) ccv_nnc_net_graph_new(const ccv_nnc_net_command_t node, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size);
// Concatenate input graph nodes with an output graph node to create a new graph.
// This method will "consume" the input graphs and output graph, that means, you don't
// need to deallocate these graphs afterwards. You are only responsible for the new
// graph created.
CCV_WARN_UNUSED(ccv_nnc_net_graph_t*) ccv_nnc_net_graph_concat(ccv_nnc_net_graph_t* const* inputs, const int input_size, const ccv_nnc_net_graph_t* output);
void ccv_nnc_net_graph_run(const ccv_nnc_net_graph_t* graph);
// This graph, and its relevant auxiliary objects (opaque to user) are deallocated.
void ccv_nnc_net_graph_free(ccv_nnc_net_graph_t* graph);

/**
 * Level-3 API
 */
typedef struct {
} ccv_nnc_net_graph_solver_t;

#endif
