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

typedef void(*ccv_cnnp_state_initializer_f)(void* const context, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, const ccv_nnc_tensor_symbol_t symbol);
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
	CCV_CNNP_MODEL_GRAPH_FIT_MODE,
	CCV_CNNP_MODEL_GRAPH_EVALUATE_MODE,
};

// This contains relevant information after model compilation.
typedef struct {
	int graph_mode;
	int dest_to_eval_size;
	int dest_to_eval_exec_size; // This is different from dest_to_eval_size because some of the execs will be optimized away.
	int gradient_init; // Have init gradient graph.
	int is_test;
	int stream_type;
	int parallel_count; // How many parallel devices.
	size_t workspace_size; // Set the default workspace size.
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_nnc_graph_exec_symbol_t* dest_to_evals;
	ccv_nnc_graph_exec_t* dest_to_eval_execs; // Destinations to end evaluation.
	ccv_array_t* trainables;
	ccv_array_t* retains; // Additional symbols need to retain.
	ccv_nnc_tensor_symbol_t* updated_trainables;
	ccv_nnc_graph_exec_symbol_t* update_execs;
	ccv_nnc_tensor_t** retain_tensors; // Additional need to retained tensors.
	ccv_nnc_tensor_t** trainable_tensors;
	ccv_nnc_tensor_symbol_map_t* saved_aux;
	ccv_nnc_cmd_t minimizer;
	ccv_nnc_cmd_t loss;
	ccv_nnc_tensor_symbol_t fits[1];
} ccv_cnnp_compiled_data_t;

struct ccv_cnnp_model_s {
	const ccv_cnnp_model_vtab_t* isa;
	int input_size;
	int output_size;
	ccv_cnnp_model_io_t io; // The opaque io that can be nil.
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
