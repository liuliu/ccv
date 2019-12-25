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

typedef void(*ccv_cnnp_cmd_updater_f)(void* const context, const ccv_nnc_graph_exec_symbol_t symbol, const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint);
typedef void(*ccv_cnnp_add_to_array_f)(void* const context, const ccv_nnc_tensor_symbol_t symbol);
/**
 * This is the virtual table of the model.
 */
typedef struct {
	void (*deinit)(ccv_cnnp_model_t* const self); /**< It can be nil. */
	void (*build)(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size); /**< Call this graph to build computation. No need to specify input size or output size, as it is defined along in the model already. */
	void (*init_states)(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context); /**< This is called to init ccv_nnc_tensor_symbol_t with a exec. */
	void (*add_to_trainable)(ccv_cnnp_model_t* const self, const ccv_cnnp_add_to_array_f add_to_array, void* const trainables); /**< This is called to add ccv_nnc_tensor_symbol_t to as list of trainables. */
	void (*add_to_output)(ccv_cnnp_model_t* const self, const ccv_cnnp_add_to_array_f add_to_array, void* const outputs); /**< This is called to add ccv_nnc_tensor_symbol_t to as list of outputs for retention. The final outputs are already added. This method is optional for any additional values we want to retain. */
	ccv_cnnp_model_t* (*copy)(const ccv_cnnp_model_t* const self); /**< This is called to make a deep copy of itself. */
	void (*set_is_test)(ccv_cnnp_model_t* const self, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context); /**< This is called when it is switched between test or training. */
	void (*add_to_trainable_indices)(ccv_cnnp_model_t* const self, const int index, ccv_array_t* const trainable_indices); /**< This is called when we try to get trainable indices out of a given model */
} ccv_cnnp_model_vtab_t;

struct ccv_cnnp_model_io_s {
	int visit; // Temporary bits stored in the ccv_cnnp_model_io_t object, whoever uses it should clean it up.
	ccv_cnnp_model_t* model; // Reference back to the model who holds it. This is required because the model is the one whole holds the io.
	ccv_array_t* incomings; // Array of ccv_cnnp_model_io_t. The order is important because it impacts the order of symbols.
	ccv_array_t* outgoings; // Array of ccv_cnnp_model_io_t.
	ccv_nnc_tensor_symbol_t* outputs; // This is different from the outputs from a model. A model could be reused, causing the outputs on that model to be the most recent one. This keeps the outputs of each.
};

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
	ccv_nnc_graph_t* graph;
	ccv_nnc_tensor_arena_t* tensor_arena;
	ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	ccv_array_t* trainables;
	ccv_array_t* retainables; // Additional symbols need to retain.
	ccv_nnc_tensor_symbol_t* gradients;
	ccv_nnc_tensor_symbol_t* outgrads;
	ccv_nnc_tensor_symbol_t* updated_trainables;
	ccv_nnc_graph_exec_symbol_t* update_nodes;
	ccv_nnc_tensor_symbol_map_t* saved_aux;
	ccv_array_t* rewindables;
	struct {
		int size;
		uint32_t* v;
	} tensors_init;
	struct {
		ccv_nnc_tensor_t** retainables; // Additional need to retained tensors.
		ccv_nnc_tensor_t** trainables;
		ccv_nnc_tensor_t** gradients;
		ccv_nnc_tensor_t** accum_gradients;
	} tensors;
	struct {
		ccv_array_t* trainables;
		ccv_array_t* retainables;
	} ids;
	struct {
		int to_op_size;
		int to_size;
		ccv_nnc_graph_exec_t* to_ops;
		ccv_nnc_graph_exec_symbol_t* tos;
		ccv_nnc_graph_static_schedule_t* schedule; // The partial schedule for running evaluate step.
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
		ccv_nnc_graph_static_schedule_t* schedule; // The partial schedule for running backward step.
	} backward;
	struct {
		ccv_nnc_graph_t* graph;
		ccv_nnc_tensor_arena_t* tensor_arena;
		ccv_nnc_graph_exec_arena_t* graph_exec_arena;
	} apply_gradients;
	struct {
		ccv_nnc_cmd_t minimizer;
		ccv_array_t* trainable_spans;
		int max_saved_aux_size;
	} minimize;
	ccv_nnc_cmd_t loss;
	ccv_nnc_tensor_symbol_t* f;
	ccv_nnc_tensor_symbol_t fits[1];
} ccv_cnnp_compiled_data_t;

struct ccv_cnnp_model_s {
	const ccv_cnnp_model_vtab_t* isa;
	int input_size;
	int output_size;
	ccv_array_t* io; // The opaque io that can be nil.
	ccv_array_t* trainable_indices; // The indexes for trainables in the final model.
	ccv_nnc_symbolic_graph_t* graph;
	ccv_nnc_tensor_symbol_t* inputs; // Unlike outputs, which is not dynamically allocated, inputs is dynamically allocated, and may be 0.
	ccv_nnc_tensor_symbol_t* outputs;
	char* name;
	ccv_cnnp_compiled_data_t* compiled_data;
	int parallel_count; // How many parallel devices.
	int memory_compression; // Whether to enable memory compression for training phase.
	size_t workspace_size; // Set the default workspace size.
};

enum {
	CCV_CNNP_MODEL_SEQUENCE,
	CCV_CNNP_MODEL_NAME,
};

typedef struct {
	int type;
	union {
		const char* name;
		int sequence;
	};
} ccv_cnnp_model_name_t;

typedef struct {
	int it;
	ccv_cnnp_model_t* model;
	ccv_array_t* sequences;
} ccv_cnnp_model_sequence_t;

static inline void ccv_cnnp_model_push(ccv_cnnp_model_t* const self, void* const context)
{
	ccv_cnnp_model_sequence_t* const model_sequence = (ccv_cnnp_model_sequence_t*)context;
	// Reset to 0.
	if (!model_sequence->sequences)
		model_sequence->sequences = ccv_array_new(sizeof(ccv_cnnp_model_name_t), 1, 0);
	ccv_cnnp_model_name_t name = {
		.type = CCV_CNNP_MODEL_SEQUENCE,
		.sequence = 0,
	};
	if (self->name)
	{
		name.type = CCV_CNNP_MODEL_NAME;
		name.name = self->name;
	}
	ccv_array_push(model_sequence->sequences, &name);
	model_sequence->it = 0;
	model_sequence->model = self;
}

static inline void ccv_cnnp_model_pop(const ccv_cnnp_model_t* const self, void* const context)
{
	ccv_cnnp_model_sequence_t* const model_sequence = (ccv_cnnp_model_sequence_t*)context;
	--model_sequence->sequences->rnum;
	assert(model_sequence->sequences->rnum >= 0);
	if (model_sequence->sequences->rnum > 0)
	{
		ccv_cnnp_model_name_t* const name = (ccv_cnnp_model_name_t*)ccv_array_get(model_sequence->sequences, model_sequence->sequences->rnum - 1);
		if (name->type == CCV_CNNP_MODEL_SEQUENCE)
			++name->sequence;
	}
}

static inline void ccv_cnnp_model_copy_name(ccv_cnnp_model_t* const self, const char* const name)
{
	if (name)
	{
		const size_t len = strnlen(name, 63);
		const size_t n = len + 1;
		self->name = (char*)ccmalloc(n);
		// Don't use strndup because this way I can have custom allocator (for ccmalloc).
		memcpy(self->name, name, n);
		self->name[len] = 0;
	}
}

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

static inline void ccv_cnnp_model_add_to_trainable(ccv_cnnp_model_t* const self, const ccv_cnnp_add_to_array_f add_to_array, void* const trainables)
{
	if (self->isa->add_to_trainable)
	{
		ccv_cnnp_model_push(self, trainables);
		self->isa->add_to_trainable(self, add_to_array, trainables);
		ccv_cnnp_model_pop(self, trainables);
	}
}

static inline void ccv_cnnp_model_add_to_trainable_indices(ccv_cnnp_model_t* const self, const int index, ccv_array_t* const trainable_indices)
{
	if (self->isa->add_to_trainable_indices)
		self->isa->add_to_trainable_indices(self, index, trainable_indices);
	else {
		int i;
		if (!self->trainable_indices)
			return;
		if (index == -1)
			for (i = 0; i < self->trainable_indices->rnum; i++)
				ccv_array_push(trainable_indices, ccv_array_get(self->trainable_indices, i));
		else if (index < self->trainable_indices->rnum)
			ccv_array_push(trainable_indices, ccv_array_get(self->trainable_indices, index));
	}
}

static inline void ccv_cnnp_model_add_to_output(ccv_cnnp_model_t* const self, const ccv_cnnp_add_to_array_f add_to_array, void* const outputs)
{
	if (self->isa->add_to_output)
	{
		ccv_cnnp_model_push(self, outputs);
		self->isa->add_to_output(self, add_to_array, outputs);
		ccv_cnnp_model_pop(self, outputs);
	}
}

void ccv_cnnp_model_tensors_init(const ccv_cnnp_model_t* const model, ccv_cnnp_compiled_data_t* const compiled_data);

#endif
