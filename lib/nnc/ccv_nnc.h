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
	CCV_NNC_COMPUTE_DATA_TRANSFER,
	CCV_NNC_COMPUTE_FORMAT_TRANSFORM,
	CCV_NNC_COMPUTE_COUNT,
};

enum {
	CCV_NNC_ACCUMULATE_OUTPUT = 0x01, // Enable accumulate outputs.
	CCV_NNC_ZERO_MEMORY_ALLOC = 0x02, // Don't allocate any extra memory for this operation.
};

enum {
	CCV_NNC_EXEC_SUCCESS = 0,
	CCV_NNC_EXEC_INVALID = -1,
	CCV_NNC_EXEC_NO_KERNEL = -2,
	CCV_NNC_EXEC_OOM = -3,
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

typedef struct ccv_nnc_stream_context_s ccv_nnc_stream_context_t;

typedef struct ccv_nnc_cmd_s {
	int compute;
	int backend;
	int algorithm;
	ccv_nnc_cmd_param_t info;
	// This has to be the same as the ccv_nnc_cmd_exec_f type.
	// This is for type CCV_NNC_COMPUTE_CUSTOM
	int(*exec)(const struct ccv_nnc_cmd_s cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context);
} ccv_nnc_cmd_t;

typedef int(*ccv_nnc_cmd_exec_f)(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context);

typedef int(*ccv_nnc_cmd_autotune_f)(const ccv_nnc_cmd_t cmd, const size_t max_workspace_size, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context);

typedef struct {
	int tensor_formats; /**< [formats] The supported formats for this API implementation. */
	int tensor_memory; /**< [memory] The supported tensor memory type for this API implementation. */
	int algorithms; /**< [algorithms] Number of algorithms variation. */
	ccv_nnc_cmd_exec_f exec;
	ccv_nnc_cmd_autotune_f autotune;
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
// Return high precision time unit.
uint64_t ccv_nnc_cmd_abs_time(void);
CCV_WARN_UNUSED(ccv_nnc_cmd_t) ccv_nnc_cmd(const int compute, ccv_nnc_cmd_exec_f exec, const ccv_nnc_cmd_param_t params, const int flags);
// Guess and verify the hint
CCV_WARN_UNUSED(int) ccv_nnc_hint_verify(const ccv_nnc_hint_t hint, const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_tensor_param_t b);
CCV_WARN_UNUSED(ccv_nnc_hint_t) ccv_nnc_hint_guess(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* inputs, const int input_size, const ccv_nnc_tensor_param_t* outputs, const int output_size);
// Run autotune to find the best kernel and configuration for the given input, returned is the modified
// cmd that contains the updated configuration.
CCV_WARN_UNUSED(ccv_nnc_cmd_t) ccv_nnc_cmd_autotune(const ccv_nnc_cmd_t cmd, const size_t max_workspace_size, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context);
int ccv_nnc_cmd_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* inputs, const int input_size, ccv_nnc_tensor_t** outputs, const int output_size, const ccv_nnc_stream_context_t* stream_context);

// Control flow constructs
// Follow heavily based along CUDA's stream / event idea.
enum {
	CCV_STREAM_CONTEXT_CPU = 0x1,
	CCV_STREAM_CONTEXT_GPU = 0x2,
};
#define CCV_STREAM_GET_CONTEXT(type) ((type) & 0x3)
#define CCV_STREAM_GET_DEVICE(type) ((type) & 0xff00)
#define CCV_STREAM_GET_DEVICE_ID(type) (CCV_STREAM_GET_DEVICE(type) >> 8)
// Flag is a combination of CPU / GPU and DEVICE_ID
CCV_WARN_UNUSED(ccv_nnc_stream_context_t*) ccv_nnc_stream_context_new(int type);
void ccv_nnc_stream_context_wait(const ccv_nnc_stream_context_t* stream);
void ccv_nnc_stream_context_free(ccv_nnc_stream_context_t* stream_context);

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
