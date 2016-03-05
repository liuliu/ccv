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

// c99 only, make sure your compiler supports that.
#define TENSOR_LIST_X(...) (ccv_nnc_tensor_t* []){__VA_ARGS__}
#define TENSOR_LIST_COUNT_N(_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32,_33,_34,_35,_36,_37,_38,_39,_40,_41,_42,_43,_44,_45,_46,_47,_48,_49,_50,_51,_52,_53,_54,_55,_56,_57,_58,_59,_60,_61,_62,_63,...) (_63)
#define TENSOR_LIST_COUNT(...) TENSOR_LIST_COUNT_N(__VA_ARGS__,62,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,42,41,40,39,38,37,36,35,34,33,32,31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1)
#define TENSOR_LIST(...) TENSOR_LIST_X(__VA_ARGS__), TENSOR_LIST_COUNT(__VA_ARGS__)

enum {
	// These are the list of computation kernels.
	CCV_NNC_COMPUTE_CUSTOM = 0,
	CCV_NNC_COMPUTE_CONVOLUTIONAL_FORWARD,
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
		void* userdata;
	};
} ccv_nnc_cmd_param_t;

typedef struct {
	struct {
		int dim[CCV_NNC_MAX_DIM_ALLOC];
	} stride;
	struct {
		int begin[CCV_NNC_MAX_DIM_ALLOC];
		int end[CCV_NNC_MAX_DIM_ALLOC];
	} border;
} ccv_nnc_hint_t;

typedef struct ccv_nnc_cmd_s {
	int compute;
	int backend;
	ccv_nnc_cmd_param_t info;
	// This has to be the same as the ccv_nnc_cmd_exec_f type.
	// This is for type CCV_NNC_COMPUTE_CUSTOM
	void(*exec)(const struct ccv_nnc_cmd_s cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size);
} ccv_nnc_cmd_t;

typedef void(*ccv_nnc_cmd_exec_f)(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size);

typedef struct {
	int tensor_formats; /**< [formats] The supported formats for this API implementation. */
	ccv_nnc_cmd_exec_f exec;
} ccv_nnc_cmd_api_t;

/**
 * Level-0 API
 */
void ccv_nnc_init(void);
/**
 * Level-1 API
 */
// For tensor
CCV_WARN_UNUSED(ccv_nnc_tensor_t*) ccv_nnc_tensor_new(const void* ptr, const ccv_nnc_tensor_param_t params, const int flags);
// Allocating on stack
CCV_WARN_UNUSED(ccv_nnc_tensor_t) ccv_nnc_tensor(const void* ptr, const ccv_nnc_tensor_param_t params, const int flags);
void ccv_nnc_tensor_free(ccv_nnc_tensor_t* tensor);
CCV_WARN_UNUSED(ccv_nnc_tensor_view_t*) ccv_nnc_tensor_view_new(const ccv_nnc_tensor_t* tensor, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int dim[CCV_NNC_MAX_DIM_ALLOC]);
// Allocating on stack
CCV_WARN_UNUSED(ccv_nnc_tensor_view_t) ccv_nnc_tensor_view(const ccv_nnc_tensor_t* tensor, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int dim[CCV_NNC_MAX_DIM_ALLOC]);
void ccv_nnc_tensor_view_free(ccv_nnc_tensor_view_t* tensor_view);
// All these functions afterwards should be compatible with both tensor and tensor view unless assertion.
void ccv_nnc_tensor_zero(void* tensor);
int ccv_nnc_tensor_eq(const ccv_nnc_tensor_t* a, const ccv_nnc_tensor_t* b);
// For computation node
CCV_WARN_UNUSED(ccv_nnc_cmd_t) ccv_nnc_cmd(const int compute, ccv_nnc_cmd_exec_f exec, const ccv_nnc_cmd_param_t params, const int flags);
CCV_WARN_UNUSED(int) ccv_nnc_hint_verify(const ccv_nnc_hint_t hint, const ccv_nnc_cmd_param_t node, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b);
CCV_WARN_UNUSED(ccv_nnc_hint_t) ccv_nnc_hint_guess(const ccv_nnc_cmd_param_t node, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_tensor_param_t* outputs, const int output_size);
void ccv_nnc_cmd_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size);

/**
 * Level-2 API
 */

typedef struct ccv_nnc_graph_s ccv_nnc_graph_t;

typedef struct {
	int32_t d;
	const ccv_nnc_graph_t* graph;
} ccv_nnc_graph_exec_t;

// Create an empty graph.
// Note that all graph mutation method are not thread-safe.
// You should only operate the graph in serial fashion.
CCV_WARN_UNUSED(ccv_nnc_graph_t*) ccv_nnc_graph_new(void);
// Create a node with specific command execution, as well as its inputs & outputs.
// Underlying, the graph maintains the backing object for the node, and all you get is
// a on-stack object to index the backing object from the graph.
CCV_WARN_UNUSED(ccv_nnc_graph_exec_t) ccv_nnc_graph_deferred_exec(const ccv_nnc_graph_t* graph, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size);
// Concatenate input graph nodes with an output graph node to create a new graph.
// Return non-zero if cannot concat successfully.
int ccv_nnc_graph_exec_concat(const ccv_nnc_graph_t* graph, const ccv_nnc_graph_exec_t source, const ccv_nnc_graph_exec_t destination);
// Run the graph from source nodes all the way to the destination nodes.
void ccv_nnc_graph_run(const ccv_nnc_graph_t* graph, const int flags, const ccv_nnc_graph_exec_t* sources, const int source_size, const ccv_nnc_graph_exec_t* destinations, const int destination_size);
// This graph, and its relevant auxiliary objects (opaque to user) are deallocated.
void ccv_nnc_graph_free(ccv_nnc_graph_t* graph);

/**
 * Level-3 API
 */
typedef struct {
} ccv_nnc_graph_solver_t;

#endif
