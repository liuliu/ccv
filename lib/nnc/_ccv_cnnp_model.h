#include "ccv_nnc.h"

typedef struct {
	void (*deinit)(ccv_cnnp_model_t* const self); // It can be nil.
	void (*build)(ccv_cnnp_model_t* const self, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size); // Call this graph to build computation. No need to specify input size or output size, as it is defined along in the model already.
	void (*add_to_trainable)(ccv_cnnp_model_t* const self, ccv_array_t* const trainables); // This is called to add ccv_nnc_tensor_symbol_t to as list of trainables.
} ccv_cnnp_model_vtab_t;

struct ccv_cnnp_model_s {
	const ccv_cnnp_model_vtab_t* isa;
	int input_size;
	int output_size;
	int input_dim[CCV_NNC_MAX_DIM_ALLOC]; // The input_dim of the model (it may not be applicable if it is a functional model).
	ccv_cnnp_model_io_t io; // The opaque io that can be nil.
	ccv_nnc_symbolic_graph_t* graph;
	ccv_nnc_tensor_symbol_t* inputs; // Unlike outputs, which is not dynamically allocated, inputs is dynamically allocated, and may be 0.
	ccv_nnc_tensor_symbol_t* outputs;
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

static inline void ccv_cnnp_model_add_to_trainable(ccv_cnnp_model_t* const self, ccv_array_t* const trainables)
{
	if (self->isa->add_to_trainable)
		self->isa->add_to_trainable(self, trainables);
}
