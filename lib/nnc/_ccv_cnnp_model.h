/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_cnnp_model_internal_h
#define GUARD_ccv_cnnp_model_internal_h

#include "ccv_nnc.h"

typedef void(*ccv_cnnp_state_initializer_f)(void* const context, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const input, const ccv_nnc_tensor_symbol_t output_symbol);
typedef void(*ccv_cnnp_cmd_updater_f)(void* const context, const ccv_nnc_graph_exec_symbol_t symbol, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint);
/**
 * This is the virtual table of the model.
 */
typedef struct {
	void (*deinit)(ccv_cnnp_model_t* const self); /**< It can be nil. */
	void (*build)(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size); /**< Call this graph to build computation. No need to specify input size or output size, as it is defined along in the model already. */
	void (*init_states)(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context); /**< This is called to init ccv_nnc_tensor_symbol_t with a exec. */
	void (*add_to_trainable)(ccv_cnnp_model_t* const self, ccv_array_t* const trainables); /**< This is called to add ccv_nnc_tensor_symbol_t to as list of trainables. */
	void (*add_to_output)(ccv_cnnp_model_t* const self, ccv_array_t* const outputs); /**< This is called to add ccv_nnc_tensor_symbol_t to as list of outputs for retention. The final outputs are already added. This method is optional for any additional values we want to retain. */
	void (*set_is_test)(ccv_cnnp_model_t* const self, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context); /**< This is called when it is switched between test or training. */
} ccv_cnnp_model_vtab_t;

enum {
	CCV_CNNP_MODEL_GRAPH_FIT_MODE, // This mode computes loss, backprop, and then apply gradients.
	CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE_NO_GRAD, // This mode allows you to only use ccv_cnnp_model_evaluate (others require gradient).
	CCV_CNNP_MODEL_GRAPH_MULTISTAGE_MODE, // This mode allows you to use ccv_cnnp_model_evaluate, ccv_cnnp_model_backward, ccv_cnnp_model_apply_gradients separately.
};

enum {
	CCV_CNNP_COMPILED_DATA_GRADIENT_NONE,
	CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES,
	CCV_CNNP_COMPILED_DATA_GRADIENT_TRAINABLES_AND_INPUTS,
};

enum {
	CCV_CNNP_REWIND_GRAPH_EXEC,
	CCV_CNNP_REWIND_TENSOR,
};

typedef struct {
	int type;
	union {
		ccv_nnc_tensor_symbol_t tensor;
		ccv_nnc_graph_exec_symbol_t graph_exec;
	};
} ccv_cnnp_rewind_symbol_t;

// This contains relevant information after model compilation.
typedef struct {
	int graph_mode;
	int gradient_mode; // Have init gradient graph.
	int is_test;
	int stream_type;
	int parallel_count; // How many parallel devices.
	int memory_compression; // Whether to enable memory compression for training phase.
	size_t workspace_size; // Set the default workspace size.
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_array_t* trainables;
	ccv_array_t* retainables; // Additional symbols need to retain.
	ccv_nnc_tensor_symbol_t* gradients;
	ccv_nnc_tensor_symbol_t* updated_trainables;
	ccv_nnc_graph_exec_symbol_t* update_nodes;
	ccv_nnc_tensor_symbol_map_t* saved_aux;
	ccv_array_t* rewindables;
	struct {
		ccv_nnc_tensor_t** retainables; // Additional need to retained tensors.
		ccv_nnc_tensor_t** trainables;
		ccv_nnc_tensor_t** gradients;
		ccv_nnc_tensor_t** accum_gradients;
	} tensors;
	struct {
		int to_op_size;
		int to_size;
		ccv_nnc_graph_exec_t* to_ops;
		ccv_nnc_graph_exec_symbol_t* tos;
	} evaluate; // Data related to ccv_cnnp_model_evaluate
	struct {
		int count; // Called backward how many times. Starting with 0.
		int from_op_size;
		ccv_nnc_graph_exec_t* from_ops; // These are the ops in the main graph.
		int to_size;
		ccv_nnc_graph_exec_symbol_t* tos;
		ccv_nnc_graph_t* accum; // The graph to accumulate gradients.
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
		ccv_nnc_tensor_symbol_t* gradients; // The new gradients.
		ccv_nnc_tensor_symbol_t* accum_gradients; // The old accumulate gradients.
		ccv_nnc_tensor_symbol_t* updated_accum_gradients; // The new accumulate gradients.
	} backward;
	struct {
		ccv_nnc_graph_t* graph;
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	} apply_gradients;
	struct {
		ccv_nnc_cmd_t minimizer;
		ccv_cnnp_model_minimizer_set_f setter;
		const void*context;
	} minimize;
	ccv_nnc_cmd_t loss;
	ccv_nnc_tensor_symbol_t fits[1];
} ccv_cnnp_compiled_data_t;

struct ccv_cnnp_model_s {
	const ccv_cnnp_model_vtab_t* isa;
	int input_size;
	int output_size;
	ccv_array_t* io; // The opaque io that can be nil.
	ccv_nnc_symbolic_graph_t* graph;
	ccv_nnc_tensor_symbol_t* inputs; // Unlike outputs, which is not dynamically allocated, inputs is dynamically allocated, and may be 0.
	ccv_nnc_tensor_symbol_t* outputs;
	ccv_cnnp_compiled_data_t* compiled_data;
};

static inline void ccv_cnnp_model_build(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	if (outputs && output_size)
	{
		assert(output_size == self->output_size);
		self->isa->build(self, graph, inputs, input_size, outputs, output_size);
		memcpy(self->outputs, outputs, sizeof(ccv_nnc_tensor_symbol_t) * output_size);
	} else
		self->isa->build(self, graph, inputs, input_size, self->outputs, self->output_size);
}

static inline void ccv_cnnp_model_init_states(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	if (self->isa->init_states)
		self->isa->init_states(self, graph, initializer, context);
}

static inline void ccv_cnnp_model_set_is_test(ccv_cnnp_model_t* const self, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context)
{
	if (self->isa->set_is_test)
		self->isa->set_is_test(self, is_test, updater, context);
}

static inline void ccv_cnnp_model_add_to_trainable(ccv_cnnp_model_t* const self, ccv_array_t* const trainables)
{
	if (self->isa->add_to_trainable)
		self->isa->add_to_trainable(self, trainables);
}

static inline void ccv_cnnp_model_add_to_output(ccv_cnnp_model_t* const self, ccv_array_t* const outputs)
{
	if (self->isa->add_to_output)
		self->isa->add_to_output(self, outputs);
}

void ccv_cnnp_model_tensors_init(const ccv_nnc_symbolic_graph_t* const graph, ccv_cnnp_compiled_data_t* const compiled_data);

#endif
