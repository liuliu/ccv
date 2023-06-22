#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_model.h"
#include "3rdparty/khash/khash.h"

// MARK - Baisc Layers

static const ccv_cnnp_model_vtab_t ccv_cnnp_input_isa;

#define CCV_CNNP_IS_MODEL_INPUT(x) ((x)->isa == &ccv_cnnp_input_isa)

#define CCV_CNNP_IS_MODEL_PARAMETER(x) ((x)->param_ref != 0 || (x)->param_sel != 0)

typedef struct {
	ccv_cnnp_model_t super;
	int sequence_size;
	ccv_cnnp_model_t* sequence[1];
} ccv_cnnp_sequential_model_t;

static void _ccv_cnnp_sequential_model_deinit(ccv_cnnp_model_t* const super)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i, j;
	for (i = 0; i < self->sequence_size; i++)
	{
		ccv_cnnp_model_t* const model = self->sequence[i];
		if (!model)
			continue;
		ccv_cnnp_model_free(model);
		for (j = i + 1; j < self->sequence_size; j++)
			if (self->sequence[j] == model)
				self->sequence[j] = 0;
	}
}

static void _ccv_cnnp_sequential_model_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	ccv_cnnp_model_t* const sub_model = self->sequence[0];
	// Go through each sub model to build the graph.
	ccv_nnc_tensor_symbol_t input;
	sub_model->data = self->super.data;
	ccv_cnnp_model_build(sub_model, graph, inputs, input_size, &input, 1);
	sub_model->data = 0;
	int i;
	for (i = 1; i < self->sequence_size; i++)
	{
		ccv_nnc_tensor_symbol_t output;
		ccv_cnnp_model_t* const sub_model = self->sequence[i];
		// Go through each sub model to build the graph.
		sub_model->data = self->super.data;
		ccv_cnnp_model_build(sub_model, graph, &input, 1, &output, 1);
		sub_model->data = 0;
		input = output;
	}
	outputs[0] = input;
}

static void _ccv_cnnp_sequential_model_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_init_states(self->sequence[i], graph, initializer, context);
}

static void _ccv_cnnp_sequential_model_set_is_test(ccv_cnnp_model_t* const super, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_set_is_test(self->sequence[i], is_test, updater, context);
}

static ccv_cnnp_model_t* _ccv_cnnp_sequential_model_copy(const ccv_cnnp_model_t* const super, void* const context);

static void _ccv_cnnp_sequential_model_add_to_parameter_indices(ccv_cnnp_model_t* const super, const int index, ccv_array_t* const parameter_indices)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_add_to_parameter_indices(self->sequence[i], index, parameter_indices);
}

static void _ccv_cnnp_sequential_model_notify(const ccv_cnnp_model_t* const super, const int tag, void* const payload)
{
	ccv_cnnp_sequential_model_t* const self = (ccv_cnnp_sequential_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
		ccv_cnnp_model_notify(self->sequence[i], tag, payload);
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_sequential_model_isa = {
	.deinit = _ccv_cnnp_sequential_model_deinit,
	.build = _ccv_cnnp_sequential_model_build,
	.init_states = _ccv_cnnp_sequential_model_init_states,
	.copy = _ccv_cnnp_sequential_model_copy,
	.set_is_test = _ccv_cnnp_sequential_model_set_is_test,
	.add_to_parameter_indices = _ccv_cnnp_sequential_model_add_to_parameter_indices,
	.notify = _ccv_cnnp_sequential_model_notify,
};

KHASH_MAP_INIT_INT64(model, ccv_cnnp_model_t*)

static ccv_cnnp_model_t* _ccv_cnnp_sequential_model_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_sequential_model_t* const self = (const ccv_cnnp_sequential_model_t*)super;
	ccv_cnnp_sequential_model_t* const sequential_model = (ccv_cnnp_sequential_model_t*)cccalloc(1, sizeof(ccv_cnnp_sequential_model_t) + sizeof(ccv_cnnp_model_t*) * (self->sequence_size - 1) + sizeof(ccv_nnc_tensor_symbol_t));
	sequential_model->super.isa = &ccv_cnnp_sequential_model_isa;
	sequential_model->super.input_size = 1;
	sequential_model->super.outputs = (ccv_nnc_tensor_symbol_t*)(sequential_model->sequence + self->sequence_size);
	sequential_model->super.output_size = 1;
	ccv_cnnp_model_copy_name(&sequential_model->super, self->super.name);
	sequential_model->sequence_size = self->sequence_size;
	int i;
	khash_t(model)* model_map = context ? (khash_t(model)*)context : kh_init(model);
	for (i = 0; i < self->sequence_size; i++)
	{
		ccv_cnnp_model_t* const sub_model = self->sequence[i];
		int ret;
		khiter_t k = kh_put(model, model_map, (uint64_t)(uintptr_t)sub_model, &ret);
		ccv_cnnp_model_t* model_copy;
		if (ret != 0)
			model_copy = kh_val(model_map, k) = _ccv_cnnp_model_copy(sub_model, model_map);
		else
			model_copy = kh_val(model_map, k);
		sequential_model->sequence[i] = model_copy;
	}
	if (!context)
		kh_destroy(model, model_map);
	return (ccv_cnnp_model_t*)sequential_model;
}

ccv_cnnp_model_t* ccv_cnnp_sequential_new(ccv_cnnp_model_t* const* const models, const int model_size, const int is_trainable, const char* const name)
{
	assert(model_size > 0);
	ccv_cnnp_sequential_model_t* const sequential_model = (ccv_cnnp_sequential_model_t*)cccalloc(1, sizeof(ccv_cnnp_sequential_model_t) + sizeof(ccv_cnnp_model_t*) * (model_size - 1) + sizeof(ccv_nnc_tensor_symbol_t));
	sequential_model->super.isa = &ccv_cnnp_sequential_model_isa;
	sequential_model->super.input_size = models[0]->input_size;
	sequential_model->super.outputs = (ccv_nnc_tensor_symbol_t*)(sequential_model->sequence + model_size);
	sequential_model->super.output_size = 1;
	sequential_model->super.is_trainable = is_trainable;
	ccv_cnnp_model_copy_name(&sequential_model->super, name);
	sequential_model->sequence_size = model_size;
	memcpy(sequential_model->sequence, models, sizeof(ccv_cnnp_model_t*) * model_size);
	return (ccv_cnnp_model_t*)sequential_model;
}

typedef struct {
	ccv_cnnp_model_t super;
	// The model's outputs, it is different from super.output_size, as latter is for actual tensor symbols.
	int model_output_size;
	// The name is similar to sequential model, but it is just topological sorted models.
	int sequence_size;
	int* model_outputs; // Which model, as in sequences, have some outputs.
	ccv_cnnp_model_io_t sequence[1];
} ccv_cnnp_functional_model_t;

static void _ccv_cnnp_functional_model_deinit(ccv_cnnp_model_t* const super)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i, j = 0, k;
	for (i = 0; i < self->sequence_size; i++)
	{
		ccv_cnnp_model_t* const model = self->sequence[i]->model;
		if (!model)
			continue;
		self->sequence[j++] = (ccv_cnnp_model_io_t)model;
		// Go through all their IO to remove itself as model.
		assert(model->io);
		for (k = 0; k < model->io->rnum; k++)
		{
			ccv_cnnp_model_io_t model_io = *(ccv_cnnp_model_io_t*)ccv_array_get(model->io, k);
			model_io->model = 0;
		}
	}
	for (i = 0; i < j; i++)
		ccv_cnnp_model_free((ccv_cnnp_model_t*)self->sequence[i]);
}

KHASH_MAP_INIT_INT64(io_node, ccv_array_t*)

static void _ccv_cnnp_functional_model_build_node_new(void* context, const ccv_nnc_graph_exec_symbol_t symbol, const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const char* const name)
{
		ccv_array_t* const nodes = (ccv_array_t*)context;
		ccv_array_push(nodes, &symbol);
}

static void _ccv_cnnp_functional_model_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	assert(self->super.input_size == input_size);
	assert(self->super.output_size == output_size);
	int i, j, k;
	for (i = 0; i < self->super.input_size; i++)
		self->sequence[i]->outputs[0] = self->sequence[i]->model->outputs[0] = inputs[i]; // Assigning the output symbol of input layer to be the input symbol.
	ccv_array_t* input_symbols = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 1, 0);
	ccv_array_t* parameter_indices = 0;
	khash_t(io_node)* io_node_map = kh_init(io_node);
	for (i = self->super.input_size; i < self->sequence_size; i++)
	{
		ccv_cnnp_model_t* const sub_model = self->sequence[i]->model;
		ccv_array_clear(input_symbols);
		const ccv_array_t* const incomings = self->sequence[i]->incomings;
		if (incomings)
			for (j = 0; j < incomings->rnum; j++)
			{
				const ccv_cnnp_model_io_t input = *(ccv_cnnp_model_io_t*)ccv_array_get(incomings, j);
				if (CCV_CNNP_IS_MODEL_PARAMETER(input))
				{
					if (!parameter_indices)
						parameter_indices = ccv_array_new(sizeof(int), 0, 0);
					else
						ccv_array_clear(parameter_indices);
					const int param_sel = input->param_sel > 0 ? input->param_sel - 1 : input->param_sel;
					assert(input->param_sel != 0);
					ccv_cnnp_model_add_to_parameter_indices(input->model, param_sel, parameter_indices);
					assert(parameter_indices->rnum > 0);
					const int param_ref = input->param_ref > 0 ? input->param_ref - 1 : input->param_ref;
					assert(input->param_ref != 0);
					if (param_ref >= 0)
					{
						assert(param_ref < parameter_indices->rnum);
						const ccv_nnc_tensor_symbol_t parameter = ccv_cnnp_parameter_from_indice(super, *(int*)ccv_array_get(parameter_indices, param_ref));
						ccv_array_push(input_symbols, &parameter);
					} else // Otherwise, all of them.
						for (k = 0; k < parameter_indices->rnum; k++)
						{
							const ccv_nnc_tensor_symbol_t parameter = ccv_cnnp_parameter_from_indice(super, *(int*)ccv_array_get(parameter_indices, k));
							ccv_array_push(input_symbols, &parameter);
						}
				} else {
					for (k = 0; k < input->model->output_size; k++)
						ccv_array_push(input_symbols, &input->outputs[k]);
				}
			}
		// Go through each sub model to build the graph.
		ccv_array_t* nodes;
		const ccv_array_t* const dependencies = self->sequence[i]->dependencies;
		if ((dependencies && dependencies->rnum > 0) || self->sequence[i]->dependents > 0)
		{
			int ret;
			khiter_t k = kh_put(io_node, io_node_map, (uint64_t)(uintptr_t)self->sequence[i], &ret);
			if (ret != 0)
				nodes = kh_val(io_node_map, k) = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 1, 0);
			else
				nodes = kh_val(io_node_map, k);
			ccv_nnc_graph_exec_symbol_new_hook(graph, _ccv_cnnp_functional_model_build_node_new, nodes);
		}
		sub_model->data = self->super.data;
		ccv_cnnp_model_build(sub_model, graph, (ccv_nnc_tensor_symbol_t*)ccv_array_get(input_symbols, 0), input_symbols->rnum, self->sequence[i]->outputs, sub_model->output_size);
		if ((dependencies && dependencies->rnum > 0) || self->sequence[i]->dependents > 0)
		{
			ccv_nnc_graph_exec_symbol_new_hook(graph, 0, 0);
			if (dependencies)
				for (j = 0; j < dependencies->rnum; j++)
				{
					const ccv_cnnp_model_io_t dependency = *(ccv_cnnp_model_io_t*)ccv_array_get(dependencies, j);
					khiter_t k = kh_get(io_node, io_node_map, (uint64_t)(uintptr_t)dependency);
					if (k == kh_end(io_node_map))
						continue;
					const ccv_array_t* const dependency_nodes = kh_val(io_node_map, k);
					int x, y;
					for (y = 0; y < dependency_nodes->rnum; y++)
						for (x = 0; x < nodes->rnum; x++)
							ccv_nnc_graph_exec_symbol_concat(graph, *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(dependency_nodes, y), *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(nodes, x));
				}
		}
		sub_model->data = 0;
	}
	khiter_t it;
	for (it = kh_begin(io_node_map); it != kh_end(io_node_map); ++it)
	{
		if (!kh_exist(io_node_map, it))
			continue;
		ccv_array_t* const nodes = kh_val(io_node_map, it);
		ccv_array_free(nodes);
	}
	kh_destroy(io_node, io_node_map);
	ccv_array_free(input_symbols);
	if (parameter_indices)
		ccv_array_free(parameter_indices);
	for (i = 0, k = 0; k < self->model_output_size; k++)
	{
		ccv_cnnp_model_t* const sub_model = self->sequence[self->model_outputs[k]]->model;
		for (j = 0; j < sub_model->output_size; j++)
			outputs[i + j] = self->sequence[self->model_outputs[k]]->outputs[j];
		i += sub_model->output_size;
	}
	assert(i == output_size);
}

static void _ccv_cnnp_functional_model_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i;
	for (i = self->super.input_size; i < self->sequence_size; i++)
		ccv_cnnp_model_init_states(self->sequence[i]->model, graph, initializer, context);
}

static void _ccv_cnnp_functional_model_set_is_test(ccv_cnnp_model_t* const super, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i;
	for (i = self->super.input_size; i < self->sequence_size; i++)
		ccv_cnnp_model_set_is_test(self->sequence[i]->model, is_test, updater, context);
}

static void _ccv_cnnp_functional_model_add_to_parameter_indices(ccv_cnnp_model_t* const super, const int index, ccv_array_t* const parameter_indices)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i;
	for (i = self->super.input_size; i < self->sequence_size; i++)
		ccv_cnnp_model_add_to_parameter_indices(self->sequence[i]->model, index, parameter_indices);
}

static void _ccv_cnnp_functional_model_notify(const ccv_cnnp_model_t* const super, const int tag, void* const payload)
{
	ccv_cnnp_functional_model_t* const self = (ccv_cnnp_functional_model_t*)super;
	int i;
	for (i = 0; i < self->sequence_size; i++)
	{
		const ccv_cnnp_model_t* const model = self->sequence[i]->model;
		ccv_cnnp_model_notify(model, tag, payload);
	}
}

static ccv_cnnp_model_t* _ccv_cnnp_functional_model_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_functional_model_isa = {
	.deinit = _ccv_cnnp_functional_model_deinit,
	.build = _ccv_cnnp_functional_model_build,
	.init_states = _ccv_cnnp_functional_model_init_states,
	.copy = _ccv_cnnp_functional_model_copy,
	.set_is_test = _ccv_cnnp_functional_model_set_is_test,
	.add_to_parameter_indices = _ccv_cnnp_functional_model_add_to_parameter_indices,
	.notify = _ccv_cnnp_functional_model_notify,
};

KHASH_MAP_INIT_INT64(model_io, ccv_cnnp_model_io_t)

static ccv_cnnp_model_t* _ccv_cnnp_functional_model_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_functional_model_t* const self = (const ccv_cnnp_functional_model_t*)super;
	ccv_cnnp_functional_model_t* const functional_model = (ccv_cnnp_functional_model_t*)cccalloc(1, sizeof(ccv_cnnp_functional_model_t) + sizeof(ccv_cnnp_model_t*) * (self->sequence_size - 1) + sizeof(ccv_nnc_tensor_symbol_t) * self->super.output_size + sizeof(int) * self->model_output_size);
	functional_model->super.isa = &ccv_cnnp_functional_model_isa;
	functional_model->super.outputs = (ccv_nnc_tensor_symbol_t*)(functional_model->sequence + self->sequence_size);
	functional_model->super.output_size = self->super.output_size;
	functional_model->super.input_size = self->super.input_size;
	ccv_cnnp_model_copy_name(&functional_model->super, self->super.name);
	functional_model->sequence_size = self->sequence_size;
	functional_model->model_output_size = self->model_output_size;
	functional_model->model_outputs = (int*)(functional_model->super.outputs + functional_model->super.output_size);
	memcpy(functional_model->model_outputs, self->model_outputs, sizeof(int) * self->model_output_size);
	// Now the difficult part, copy over the model_io.
	khash_t(model_io)* model_io_map = kh_init(model_io);
	khash_t(model)* model_map = context ? (khash_t(model)*)context : kh_init(model);
	int i, j;
	for (i = 0; i < self->sequence_size; i++)
	{
		const ccv_cnnp_model_t* const sub_model = self->sequence[i]->model;
		int ret;
		khiter_t k = kh_put(model, model_map, (uint64_t)(uintptr_t)sub_model, &ret);
		ccv_cnnp_model_t* model_copy;
		if (ret != 0)
			model_copy = kh_val(model_map, k) = _ccv_cnnp_model_copy(sub_model, model_map);
		else
			model_copy = kh_val(model_map, k);
		ccv_cnnp_model_io_t model_io = functional_model->sequence[i] = ccmalloc(sizeof(struct ccv_cnnp_model_io_s) + sizeof(ccv_nnc_tensor_symbol_t) * sub_model->output_size);
		model_io->param_ref = 0;
		model_io->param_sel = 0;
		model_io->visit = 0;
		model_io->model = model_copy;
		model_io->dependencies = 0;
		model_io->dependents = 0;
		model_io->incomings = 0;
		model_io->outgoings = 0;
		model_io->outputs = (ccv_nnc_tensor_symbol_t*)(model_io + 1);
		if (!model_copy->io)
			model_copy->io = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
		ccv_array_push(model_copy->io, &model_io);
		k = kh_put(model_io, model_io_map, (uint64_t)(uintptr_t)self->sequence[i], &ret);
		kh_val(model_io_map, k) = functional_model->sequence[i];
	}
	for (i = self->super.input_size; i < self->sequence_size; i++)
	{
		if (self->sequence[i]->incomings)
			for (j = 0; j < self->sequence[i]->incomings->rnum; j++)
			{
				const ccv_cnnp_model_io_t input = *(ccv_cnnp_model_io_t*)ccv_array_get(self->sequence[i]->incomings, j);
				if (CCV_CNNP_IS_MODEL_PARAMETER(input)) // I am pretty sure this is not in the model_io_map.
				{
					int ret;
					khiter_t k = kh_put(model_io, model_io_map, (uint64_t)(uintptr_t)input, &ret);
					if (ret != 0)
					{
						// The model may not exist on the map due to wrapping (it is inside another sequential or functional model).
						khiter_t m = kh_get(model, model_map, (uint64_t)(uintptr_t)input->model);
						assert(m != kh_end(model_map));
						ccv_cnnp_model_t* const model_copy = kh_val(model_map, m);
						ccv_cnnp_model_io_t model_io = ccmalloc(sizeof(struct ccv_cnnp_model_io_s));
						model_io->param_ref = input->param_ref;
						model_io->param_sel = input->param_sel;
						model_io->visit = 0;
						model_io->model = model_copy;
						model_io->incomings = 0;
						model_io->dependencies = 0;
						model_io->dependents = 0;
						model_io->outgoings = 0;
						model_io->outputs = 0;
						if (!model_copy->io)
							model_copy->io = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
						ccv_array_push(model_copy->io, &model_io);
						kh_val(model_io_map, k) = model_io;
						if (input->outgoings)
						{
							model_io->outgoings = ccv_array_new(sizeof(ccv_cnnp_model_io_t), input->outgoings->rnum, 0);
							int x;
							for (x = 0; x < input->outgoings->rnum; x++)
							{
								khiter_t k = kh_get(model_io, model_io_map, (uint64_t)(uintptr_t)(*(ccv_cnnp_model_io_t*)ccv_array_get(input->outgoings, x)));
								assert(k != kh_end(model_io_map));
								ccv_cnnp_model_io_t outgoing_io = kh_val(model_io_map, k);
								ccv_array_push(model_io->outgoings, &outgoing_io);
							}
						}
					}
				}
			}
	}
	if (!context)
		kh_destroy(model, model_map);
	for (i = 0; i < self->sequence_size; i++)
	{
		const ccv_cnnp_model_io_t model_io = self->sequence[i];
		ccv_cnnp_model_io_t model_io_copy = functional_model->sequence[i];
		model_io_copy->param_ref = model_io->param_ref;
		model_io_copy->param_sel = model_io->param_sel;
		if (model_io->incomings)
		{
			model_io_copy->incomings = ccv_array_new(sizeof(ccv_cnnp_model_io_t), model_io->incomings->rnum, 0);
			for (j = 0; j < model_io->incomings->rnum; j++)
			{
				khiter_t k = kh_get(model_io, model_io_map, (uint64_t)(uintptr_t)(*(ccv_cnnp_model_io_t*)ccv_array_get(model_io->incomings, j)));
				assert(k != kh_end(model_io_map));
				ccv_cnnp_model_io_t input_io = kh_val(model_io_map, k);
				ccv_array_push(model_io_copy->incomings, &input_io);
			}
		}
		if (model_io->dependencies)
		{
			model_io_copy->dependencies = ccv_array_new(sizeof(ccv_cnnp_model_io_t), model_io->dependencies->rnum, 0);
			for (j = 0; j < model_io->dependencies->rnum; j++)
			{
				khiter_t k = kh_get(model_io, model_io_map, (uint64_t)(uintptr_t)(*(ccv_cnnp_model_io_t*)ccv_array_get(model_io->dependencies, j)));
				assert(k != kh_end(model_io_map));
				ccv_cnnp_model_io_t input_io = kh_val(model_io_map, k);
				ccv_array_push(model_io_copy->dependencies, &input_io);
			}
		}
		model_io_copy->dependents = model_io->dependents;
		if (model_io->outgoings)
		{
			model_io_copy->outgoings = ccv_array_new(sizeof(ccv_cnnp_model_io_t), model_io->outgoings->rnum, 0);
			for (j = 0; j < model_io->outgoings->rnum; j++)
			{
				khiter_t k = kh_get(model_io, model_io_map, (uint64_t)(uintptr_t)(*(ccv_cnnp_model_io_t*)ccv_array_get(model_io->outgoings, j)));
				assert(k != kh_end(model_io_map));
				ccv_cnnp_model_io_t outgoing_io = kh_val(model_io_map, k);
				ccv_array_push(model_io_copy->outgoings, &outgoing_io);
			}
		}
	}
	kh_destroy(model_io, model_io_map);
	return (ccv_cnnp_model_t*)functional_model;
}

ccv_cnnp_model_t* ccv_cnnp_model_new(const ccv_cnnp_model_io_t* const inputs, const int input_size, const ccv_cnnp_model_io_t* const outputs, const int output_size, const int is_trainable, const char* const name)
{
	assert(output_size > 0);
	// Do topological sort.
	ccv_array_t* const reverse_top = ccv_array_new(sizeof(ccv_cnnp_model_io_t), output_size, 0);
	int i, j, k;
	// Go through output one by one, reverse traversal them, to detect potential overlap (overlap means, for example,
	// outputs[1] is an incoming node for outputs[0]. Thus, if we reverse them, we may have outputs[0] build before outputs[1],
	// hence, having issues.
	for (i = 0; i < output_size; i++)
		outputs[i]->visit = 2;
	for (i = output_size - 1; i >= 0; i--)
	{
		if (outputs[i]->visit == 3) // If we need to remove it, no need to visit.
			continue;
		assert(outputs[i]->visit == 2);
		ccv_array_clear(reverse_top);
		ccv_array_push(reverse_top, &outputs[i]);
		for (j = 0; j < reverse_top->rnum; j++)
		{
			const ccv_cnnp_model_io_t output = *(ccv_cnnp_model_io_t*)ccv_array_get(reverse_top, j);
			assert(!CCV_CNNP_IS_MODEL_INPUT(output->model));
			// If it is input, push it here.
			if (output->incomings && !CCV_CNNP_IS_MODEL_PARAMETER(output))
				for (k = 0; k < output->incomings->rnum; k++)
				{
					const ccv_cnnp_model_io_t input = *(ccv_cnnp_model_io_t*)ccv_array_get(output->incomings, k);
					// If it is an input or parameter, skip.
					if (CCV_CNNP_IS_MODEL_INPUT(input->model) || CCV_CNNP_IS_MODEL_PARAMETER(input))
						continue;
					if (input->visit == 1 || input->visit == 3) // Visited, skip.
						continue;
					// If this is an output, we need to remove it from the output array. Otherwise mark it as visited.
					input->visit = input->visit == 2 ? 3 : 1;
					ccv_array_push(reverse_top, &input);
				}
			// Similar for dependencies.
			if (output->dependencies && !CCV_CNNP_IS_MODEL_PARAMETER(output))
				for (k = 0; k < output->dependencies->rnum; k++)
				{
					const ccv_cnnp_model_io_t dependency = *(ccv_cnnp_model_io_t*)ccv_array_get(output->dependencies, k);
					// If it is an input or parameter, skip.
					if (CCV_CNNP_IS_MODEL_INPUT(dependency->model) || CCV_CNNP_IS_MODEL_PARAMETER(dependency))
						continue;
					if (dependency->visit == 1 || dependency->visit == 3) // Visited, skip.
						continue;
					// If this is an output, we need to remove it from the output array. Otherwise mark it as visited.
					dependency->visit = dependency->visit == 2 ? 3 : 1;
					ccv_array_push(reverse_top, &dependency);
				}
		}
		for (j = 1; j < reverse_top->rnum; j++)
		{
			const ccv_cnnp_model_io_t output = *(ccv_cnnp_model_io_t*)ccv_array_get(reverse_top, j);
			if (output->visit == 1) // Clean the visit back.
				output->visit = 0;
		}
	}
	ccv_array_clear(reverse_top);
	for (i = 0; i < output_size; i++) // We will assign sequence in reverse order, thus, reverse the reverse top when copying the outputs.
	{
		if (outputs[output_size - 1 - i]->visit == 2)
			ccv_array_push(reverse_top, &outputs[output_size - 1 - i]);
		assert(outputs[output_size - 1 - i]->visit == 2 || outputs[output_size - 1 - i]->visit == 3);
		outputs[output_size - 1 - i]->visit = 0; // Clean up all visits.
	}
	// Go from the output, until we meet inputs.
	uint64_t input_bitmask[((input_size - 1) >> 6) + 1];
	memset(input_bitmask, 0, sizeof(uint64_t) * (((input_size - 1) >> 6) + 1));
	int tensor_output_size = 0; // io can be mapped to multiple tensor outputs, therefore, need to compute the exact tensor output size.
	for (i = 0; i < output_size; i++)
		tensor_output_size += outputs[i]->model->output_size;
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_cnnp_model_io_t output = *(ccv_cnnp_model_io_t*)ccv_array_get(reverse_top, i);
		assert(!CCV_CNNP_IS_MODEL_INPUT(output->model));
		// If it is input, push it here.
		if (output->incomings && !CCV_CNNP_IS_MODEL_PARAMETER(output))
			for (j = 0; j < output->incomings->rnum; j++)
			{
				const ccv_cnnp_model_io_t input = *(ccv_cnnp_model_io_t*)ccv_array_get(output->incomings, j);
				++input->visit; // Mark it as visited.
				if (input->visit != input->outgoings->rnum + input->dependents) // Not all dependencies visited.
					continue;
				if (!CCV_CNNP_IS_MODEL_INPUT(input->model) && !CCV_CNNP_IS_MODEL_PARAMETER(input))
					ccv_array_push(reverse_top, &input);
				else if (CCV_CNNP_IS_MODEL_INPUT(input->model)) {
					for (k = 0; k < input_size; k++)
						if (input == inputs[k])
							break;
					assert(k < input_size);
					input_bitmask[k >> 6] |= ((uint64_t)1 << (k & 63));
				}
			}
		if (output->dependencies && !CCV_CNNP_IS_MODEL_PARAMETER(output))
			for (j = 0; j < output->dependencies->rnum; j++)
			{
				const ccv_cnnp_model_io_t dependency = *(ccv_cnnp_model_io_t*)ccv_array_get(output->dependencies, j);
				++dependency->visit; // Mark it as visited.
				if (dependency->visit != dependency->outgoings->rnum + dependency->dependents) // Not all dependencies visited.
					continue;
				if (!CCV_CNNP_IS_MODEL_INPUT(dependency->model) && !CCV_CNNP_IS_MODEL_PARAMETER(dependency))
					ccv_array_push(reverse_top, &dependency);
				else if (CCV_CNNP_IS_MODEL_INPUT(dependency->model)) {
					for (k = 0; k < input_size; k++)
						if (dependency == inputs[k])
							break;
					assert(k < input_size);
					input_bitmask[k >> 6] |= ((uint64_t)1 << (k & 63));
				}
			}
	}
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_cnnp_model_io_t output = *(ccv_cnnp_model_io_t*)ccv_array_get(reverse_top, i);
		output->visit = 0; // Clean the visit back.
	}
	for (i = 0; i < input_size; i++)
		inputs[i]->visit = 0; // Clean the visit back.
	for (i = 0; i < input_size; i++)
		{ assert((input_bitmask[i >> 6] & ((uint64_t)1 << (i & 63)))); } // Assuming they all match.
	const int sequence_size = reverse_top->rnum + input_size;
	ccv_cnnp_functional_model_t* const functional_model = (ccv_cnnp_functional_model_t*)cccalloc(1, sizeof(ccv_cnnp_functional_model_t) + sizeof(ccv_cnnp_model_t*) * (sequence_size - 1) + sizeof(ccv_nnc_tensor_symbol_t) * tensor_output_size + sizeof(int) * output_size);
	functional_model->super.isa = &ccv_cnnp_functional_model_isa;
	functional_model->super.outputs = (ccv_nnc_tensor_symbol_t*)(functional_model->sequence + sequence_size);
	functional_model->super.output_size = tensor_output_size;
	functional_model->super.input_size = input_size;
	functional_model->super.is_trainable = is_trainable;
	functional_model->model_output_size = output_size;
	functional_model->model_outputs = (int*)(functional_model->super.outputs + tensor_output_size);
	ccv_cnnp_model_copy_name(&functional_model->super, name);
	functional_model->sequence_size = sequence_size;
	memcpy(functional_model->sequence, inputs, sizeof(ccv_cnnp_model_io_t) * input_size);
	for (i = 0; i < reverse_top->rnum; i++)
		functional_model->sequence[input_size + i] = *(ccv_cnnp_model_io_t*)ccv_array_get(reverse_top, reverse_top->rnum - 1 - i);
	for (i = 0; i < output_size; i++)
	{
		for (j = sequence_size - 1; j >= input_size; j--)
			if (functional_model->sequence[j] == outputs[i])
			{
				functional_model->model_outputs[i] = j;
				break;
			}
	}
	ccv_array_free(reverse_top);
	return (ccv_cnnp_model_t*)functional_model;
}

static ccv_cnnp_model_t* _ccv_cnnp_input_copy(const ccv_cnnp_model_t* const self, void* const context)
{
	ccv_cnnp_model_t* const input = (ccv_cnnp_model_t*)cccalloc(1, sizeof(ccv_cnnp_model_t) + sizeof(ccv_nnc_tensor_symbol_t));
	input->isa = &ccv_cnnp_input_isa;
	input->outputs = (ccv_nnc_tensor_symbol_t*)(input + 1);
	input->output_size = 1;
	return input;
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_input_isa = {
	.copy = _ccv_cnnp_input_copy,
};

ccv_cnnp_model_io_t ccv_cnnp_input(void)
{
	ccv_cnnp_model_t* const input = (ccv_cnnp_model_t*)cccalloc(1, sizeof(ccv_cnnp_model_t) + sizeof(ccv_nnc_tensor_symbol_t));
	input->isa = &ccv_cnnp_input_isa;
	input->io = ccv_array_new(sizeof(ccv_cnnp_model_io_t), 1, 0);
	ccv_cnnp_model_io_t input_io = ccmalloc(sizeof(struct ccv_cnnp_model_io_s) + sizeof(ccv_nnc_tensor_symbol_t));
	input_io->param_ref = 0;
	input_io->param_sel = 0;
	input_io->visit = 0;
	input_io->incomings = 0;
	input_io->dependencies = 0;
	input_io->dependents = 0;
	input_io->outgoings = 0;
	input_io->model = input;
	input_io->outputs = (ccv_nnc_tensor_symbol_t*)(input_io + 1);
	ccv_array_push(input->io, &input_io);
	input->outputs = (ccv_nnc_tensor_symbol_t*)(input + 1);
	input->output_size = 1;
	return input_io;
}

// MARK - Dynamic Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_cnnp_model_dynamic_f func;
	void* context;
	ccv_cnnp_model_t* model;
} ccv_cnnp_dynamic_model_t;

static void _ccv_cnnp_dynamic_model_deinit(ccv_cnnp_model_t* const super)
{
	ccv_cnnp_dynamic_model_t* const self = (ccv_cnnp_dynamic_model_t*)super;
	if (self->model)
		ccv_cnnp_model_free(self->model);
}

static void _ccv_cnnp_dynamic_model_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	ccv_cnnp_dynamic_model_t* const self = (ccv_cnnp_dynamic_model_t*)super;
	if (!self->model)
	{
		ccv_nnc_tensor_param_t input_params[input_size];
		int i;
		for (i = 0; i < input_size; i++)
			input_params[i] = ccv_nnc_tensor_symbol_params(graph, inputs[i]);
		self->model = self->func(input_params, input_size, self->context);
		// Update to use the settings of the compiled model.
		self->super.input_size = self->model->input_size;
		self->super.outputs = self->model->outputs;
		self->super.output_size = self->model->output_size;
	}
	self->model->data = self->super.data;
	ccv_cnnp_model_build(self->model, graph, inputs, input_size, outputs, output_size);
	self->model->data = 0;
}

static void _ccv_cnnp_dynamic_model_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_dynamic_model_t* const self = (ccv_cnnp_dynamic_model_t*)super;
	assert(self->model);
	ccv_cnnp_model_init_states(self->model, graph, initializer, context);
}

static void _ccv_cnnp_dynamic_model_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable)
{
	ccv_cnnp_dynamic_model_t* const self = (ccv_cnnp_dynamic_model_t*)super;
	assert(self->model);
	ccv_cnnp_model_add_to_parameter(self->model, add_to_array, parameters, is_trainable);
}

static void _ccv_cnnp_dynamic_model_add_to_output(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const outputs)
{
	ccv_cnnp_dynamic_model_t* const self = (ccv_cnnp_dynamic_model_t*)super;
	assert(self->model);
	ccv_cnnp_model_add_to_output(self->model, add_to_array, outputs);
}

static void _ccv_cnnp_dynamic_model_set_is_test(ccv_cnnp_model_t* const super, const int is_test, const ccv_cnnp_cmd_updater_f updater, void* const context)
{
	ccv_cnnp_dynamic_model_t* const self = (ccv_cnnp_dynamic_model_t*)super;
	assert(self->model);
	ccv_cnnp_model_set_is_test(self->model, is_test, updater, context);
}

static ccv_cnnp_model_t* _ccv_cnnp_dynamic_model_copy(const ccv_cnnp_model_t* const super, void* const context);

static void _ccv_cnnp_dynamic_model_add_to_parameter_indices(ccv_cnnp_model_t* const super, const int index, ccv_array_t* const parameter_indices)
{
	ccv_cnnp_dynamic_model_t* const self = (ccv_cnnp_dynamic_model_t*)super;
	assert(self->model);
	ccv_cnnp_model_add_to_parameter_indices(self->model, index, parameter_indices);
}

static void _ccv_cnnp_dynamic_model_notify(const ccv_cnnp_model_t* const super, const int tag, void* const payload)
{
	ccv_cnnp_dynamic_model_t* const self = (ccv_cnnp_dynamic_model_t*)super;
	if (self->model)
		ccv_cnnp_model_notify(self->model, tag, payload);
}

static const ccv_cnnp_model_vtab_t ccv_cnnp_dynamic_model_isa = {
	.deinit = _ccv_cnnp_dynamic_model_deinit,
	.build = _ccv_cnnp_dynamic_model_build,
	.init_states = _ccv_cnnp_dynamic_model_init_states,
	.add_to_parameter = _ccv_cnnp_dynamic_model_add_to_parameter,
	.add_to_output = _ccv_cnnp_dynamic_model_add_to_output,
	.copy = _ccv_cnnp_dynamic_model_copy,
	.set_is_test = _ccv_cnnp_dynamic_model_set_is_test,
	.add_to_parameter_indices = _ccv_cnnp_dynamic_model_add_to_parameter_indices,
	.notify = _ccv_cnnp_dynamic_model_notify,
};

ccv_cnnp_model_t* ccv_cnnp_dynamic_new(ccv_cnnp_model_dynamic_f func, void* const context, const char* const name)
{
	ccv_cnnp_dynamic_model_t* const dynamic_model = (ccv_cnnp_dynamic_model_t*)cccalloc(1, sizeof(ccv_cnnp_dynamic_model_t));
	dynamic_model->super.isa = &ccv_cnnp_dynamic_model_isa;
	dynamic_model->func = func;
	dynamic_model->context = context;
	ccv_cnnp_model_copy_name(&dynamic_model->super, name);
	return (ccv_cnnp_model_t*)dynamic_model;
}

static ccv_cnnp_model_t* _ccv_cnnp_dynamic_model_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_dynamic_model_t* const self = (const ccv_cnnp_dynamic_model_t*)super;
	return ccv_cnnp_dynamic_new(self->func, self->context, self->super.name);
}

// MARK - Command Layer

typedef struct {
	ccv_cnnp_model_t super;
	ccv_nnc_cmd_t cmd;
	ccv_nnc_hint_t hint;
	ccv_nnc_tensor_symbol_t* input_symbols; // This is only valid for INIT_SHARED_TENSOR / INIT_SHARED_TENSOR_AS_TRAINABLE
	ccv_nnc_tensor_symbol_t* output_symbols; // This is just for the output symbol (in case we need to have no tensor symbol).
	ccv_cnnp_cmd_exec_io_t* inputs;
	int flags;
	int input_size;
	int* outputs;
	int output_size;
} ccv_cnnp_model_cmd_exec_t;

static void _ccv_cnnp_cmd_exec_build(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	ccv_cnnp_model_cmd_exec_t* const self = (ccv_cnnp_model_cmd_exec_t*)super;
	ccv_nnc_tensor_param_t input_params[ccv_max(1, self->input_size)];
	int i, j;
	for (i = 0, j = 0; i < self->input_size; i++)
		if (self->inputs[i].type == CCV_CNNP_IO)
		{
			self->input_symbols[i] = inputs[j++];
			input_params[i] = ccv_nnc_tensor_symbol_params(graph, self->input_symbols[i]);
		} else if (self->inputs[i].type == CCV_CNNP_NO_TENSOR) {
			self->input_symbols[i] = NO_TENSOR_SYMBOL;
		} else if (!self->input_symbols[i].graph) {
			// Otherwise, we only create this symbol if it doesn't exist.
			const ccv_nnc_tensor_param_t params = self->inputs[i].init_state.info;
			input_params[i] = params;
			self->input_symbols[i] = ccv_nnc_tensor_symbol_new(graph, params, 0);
		}
	// We cannot simply mark the outputs as auto, because the subsequent build call may require this output to have params setup.
	// Infer the parameters here.
	ccv_nnc_tensor_param_t output_params[ccv_max(1, self->output_size)];
	ccv_nnc_hint_tensor_auto(self->cmd, input_params, self->input_size, self->hint, output_params, self->output_size);
	for (i = 0, j = 0; i < self->output_size; i++)
		if (self->outputs[i] == CCV_CNNP_IO)
			self->output_symbols[i] = outputs[j++] = ccv_nnc_tensor_symbol_new(graph, output_params[i], 0);
		else if (self->outputs[i] == CCV_CNNP_TENSOR_NOT_OUTPUT)
			self->output_symbols[i] = ccv_nnc_tensor_symbol_new(graph, output_params[i], 0);
		else
			self->output_symbols[i] = NO_TENSOR_SYMBOL;
	ccv_nnc_graph_exec_symbol_new(graph, self->cmd, self->input_symbols, self->input_size, self->output_symbols, self->output_size, 0);
}

static void _ccv_cnnp_cmd_exec_init_states(ccv_cnnp_model_t* const super, ccv_nnc_symbolic_graph_t* const graph, const ccv_cnnp_state_initializer_f initializer, void* const context)
{
	ccv_cnnp_model_cmd_exec_t* const self = (ccv_cnnp_model_cmd_exec_t*)super;
	int i;
	for (i = 0; i < self->input_size; i++)
		if (self->inputs[i].type == CCV_CNNP_INIT_SHARED_TENSOR || self->inputs[i].type == CCV_CNNP_INIT_SHARED_TENSOR_AS_TRAINABLE)
			self->inputs[i].init_state.init(self->input_symbols[i], initializer, context, self->inputs[i].init_state.context);
}

static void _ccv_cnnp_cmd_exec_add_to_output(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const outputs)
{
	ccv_cnnp_model_cmd_exec_t* const self = (ccv_cnnp_model_cmd_exec_t*)super;
	int i;
	for (i = 0; i < self->input_size; i++)
		if (self->inputs[i].type == CCV_CNNP_INIT_SHARED_TENSOR)
			add_to_array(outputs, self->input_symbols[i], 0); // Push this as retainable because it need to be init.
}

static void _ccv_cnnp_cmd_exec_add_to_parameter(ccv_cnnp_model_t* const super, const ccv_cnnp_add_to_array_f add_to_array, void* const parameters, const int is_trainable)
{
	ccv_cnnp_model_cmd_exec_t* const self = (ccv_cnnp_model_cmd_exec_t*)super;
	int i;
	for (i = 0; i < self->input_size; i++)
		if (self->inputs[i].type == CCV_CNNP_INIT_SHARED_TENSOR_AS_TRAINABLE)
			add_to_array(parameters, self->input_symbols[i], is_trainable); // Push this as parameter.
}

static void _ccv_cnnp_cmd_exec_deinit(ccv_cnnp_model_t* const super)
{
	ccv_cnnp_model_cmd_exec_t* const self = (ccv_cnnp_model_cmd_exec_t*)super;
	int i, j;
	for (i = 0; i < self->input_size; i++)
		if ((self->inputs[i].type == CCV_CNNP_INIT_SHARED_TENSOR || self->inputs[i].type == CCV_CNNP_INIT_SHARED_TENSOR_AS_TRAINABLE) &&
			self->inputs[i].init_state.context)
		{
			void* const context = self->inputs[i].init_state.context;
			if (self->inputs[i].init_state.deinit)
				self->inputs[i].init_state.deinit(context);
			self->inputs[i].init_state.init = 0;
			self->inputs[i].init_state.deinit = 0;
			self->inputs[i].init_state.context = 0;
			for (j = i + 1; j < self->input_size; j++)
				if (self->inputs[j].init_state.context == context)
				{
					self->inputs[j].init_state.init = 0;
					self->inputs[j].init_state.deinit = 0;
					self->inputs[j].init_state.context = 0;
				}
		}
}

static ccv_cnnp_model_t* _ccv_cnnp_cmd_exec_copy(const ccv_cnnp_model_t* const super, void* const context);

static const ccv_cnnp_model_vtab_t ccv_cnnp_cmd_exec_isa = {
	.build = _ccv_cnnp_cmd_exec_build,
	.init_states = _ccv_cnnp_cmd_exec_init_states,
	.add_to_parameter = _ccv_cnnp_cmd_exec_add_to_parameter,
	.add_to_output = _ccv_cnnp_cmd_exec_add_to_output,
	.deinit = _ccv_cnnp_cmd_exec_deinit,
	.copy = _ccv_cnnp_cmd_exec_copy,
};

static ccv_cnnp_model_t* _ccv_cnnp_cmd_exec(const ccv_nnc_cmd_t cmd, int copy_io, const ccv_nnc_hint_t hint, const int flags, const ccv_cnnp_cmd_exec_io_t* const inputs, const int input_size, const int* const outputs, const int output_size, const int is_trainable, const char* const name)
{
	assert(input_size >= 0);
	assert(output_size > 0);
	int i;
	int io_input_size = 0;
	for (i = 0; i < input_size; i++)
		if (inputs[i].type == CCV_CNNP_IO)
			++io_input_size;
		else {
			assert(inputs[i].type == CCV_CNNP_INIT_SHARED_TENSOR || inputs[i].type == CCV_CNNP_INIT_SHARED_TENSOR_AS_TRAINABLE);
			assert(inputs[i].init_state.init);
		}
	int io_output_size = 0;
	for (i = 0; i < output_size; i++)
		if (outputs[i] == CCV_CNNP_IO)
			++io_output_size;
		else {
			assert(outputs[i] == CCV_CNNP_TENSOR_NOT_OUTPUT || outputs[i] == CCV_CNNP_NO_TENSOR);
		}
	assert(io_output_size > 0);
	ccv_cnnp_model_cmd_exec_t* const model_cmd_exec = (ccv_cnnp_model_cmd_exec_t*)cccalloc(1, sizeof(ccv_cnnp_model_cmd_exec_t) + sizeof(ccv_nnc_tensor_symbol_t) * (io_output_size + input_size + output_size) + sizeof(ccv_cnnp_cmd_exec_io_t) * input_size + sizeof(int) * output_size);
	model_cmd_exec->super.isa = &ccv_cnnp_cmd_exec_isa;
	model_cmd_exec->super.input_size = io_input_size;
	model_cmd_exec->super.outputs = (ccv_nnc_tensor_symbol_t*)(model_cmd_exec + 1);
	model_cmd_exec->super.output_size = io_output_size;
	model_cmd_exec->super.is_trainable = is_trainable;
	ccv_cnnp_model_copy_name(&model_cmd_exec->super, name);
	model_cmd_exec->cmd = cmd;
	model_cmd_exec->hint = hint;
	model_cmd_exec->flags = flags;
	model_cmd_exec->input_size = input_size;
	model_cmd_exec->input_symbols = model_cmd_exec->super.outputs + io_output_size;
	model_cmd_exec->output_symbols = model_cmd_exec->input_symbols + input_size;
	model_cmd_exec->inputs = (ccv_cnnp_cmd_exec_io_t*)(model_cmd_exec->output_symbols + output_size);
	if (input_size > 0)
	{
		memcpy(model_cmd_exec->inputs, inputs, sizeof(ccv_cnnp_cmd_exec_io_t) * input_size);
		if (copy_io)
			for (i = 0; i < input_size; i++)
				if (inputs[i].type != CCV_CNNP_IO && inputs[i].init_state.copy)
					model_cmd_exec->inputs[i].init_state.context = inputs[i].init_state.copy(inputs[i].init_state.context);
	}
	model_cmd_exec->output_size = output_size;
	model_cmd_exec->outputs = (int*)(model_cmd_exec->inputs + input_size);
	if (output_size > 0)
		memcpy(model_cmd_exec->outputs, outputs, sizeof(int) * output_size);
	return (ccv_cnnp_model_t*)model_cmd_exec;
}

ccv_cnnp_model_t* ccv_cnnp_cmd_exec(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, const ccv_cnnp_cmd_exec_io_t* const inputs, const int input_size, const int* const outputs, const int output_size, const int is_trainable, const char* const name)
{
	return _ccv_cnnp_cmd_exec(cmd, 0, hint, flags, inputs, input_size, outputs, output_size, is_trainable, name);
}

static ccv_cnnp_model_t* _ccv_cnnp_cmd_exec_copy(const ccv_cnnp_model_t* const super, void* const context)
{
	const ccv_cnnp_model_cmd_exec_t* const self = (const ccv_cnnp_model_cmd_exec_t*)super;
	return _ccv_cnnp_cmd_exec(self->cmd, 1, self->hint, self->flags, self->inputs, self->input_size, self->outputs, self->output_size, self->super.is_trainable, self->super.name);
}

static void _ccv_cnnp_cmd_exec_io_copy(const ccv_nnc_tensor_symbol_t tensor_symbol, const ccv_cnnp_state_initializer_f initializer, void* const initializer_context, void* const context)
{
	initializer(initializer_context, CMD_DATA_TRANSFER_FORWARD(), ccv_nnc_no_hint, 0, (ccv_nnc_tensor_t*)context, tensor_symbol);
}

ccv_cnnp_cmd_exec_io_init_state_t ccv_cnnp_cmd_exec_io_copy(const ccv_nnc_tensor_t* const tensor)
{
	return (ccv_cnnp_cmd_exec_io_init_state_t){
		.info = tensor->info,
		.context = (void *)tensor,
		.init = _ccv_cnnp_cmd_exec_io_copy,
	};
}

typedef struct {
	ccv_nnc_cmd_t cmd;
	ccv_nnc_hint_t hint;
	int flags;
} ccv_cnnp_cmd_exec_io_set_by_t;

static void _ccv_cnnp_cmd_exec_io_set_by(const ccv_nnc_tensor_symbol_t tensor_symbol, const ccv_cnnp_state_initializer_f initializer, void* const initializer_context, void* const context)
{
	const ccv_cnnp_cmd_exec_io_set_by_t* const set_by = (ccv_cnnp_cmd_exec_io_set_by_t*)context;
	initializer(initializer_context, set_by->cmd, set_by->hint, set_by->flags, 0, tensor_symbol);
}

static void* _ccv_cnnp_cmd_exec_io_set_by_copy(void* const context)
{
	ccv_cnnp_cmd_exec_io_set_by_t* const set_by = (ccv_cnnp_cmd_exec_io_set_by_t*)ccmalloc(sizeof(ccv_cnnp_cmd_exec_io_set_by_t));
	memcpy(set_by, context, sizeof(ccv_cnnp_cmd_exec_io_set_by_t));
	return set_by;
}

ccv_cnnp_cmd_exec_io_init_state_t ccv_cnnp_cmd_exec_io_set_by(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, const ccv_nnc_tensor_param_t params)
{
	ccv_cnnp_cmd_exec_io_set_by_t* const set_by = (ccv_cnnp_cmd_exec_io_set_by_t*)ccmalloc(sizeof(ccv_cnnp_cmd_exec_io_set_by_t));
	set_by->cmd = cmd;
	set_by->hint = hint;
	set_by->flags = flags;
	return (ccv_cnnp_cmd_exec_io_init_state_t){
		.info = params,
		.context = set_by,
		.init = _ccv_cnnp_cmd_exec_io_set_by,
		.copy = _ccv_cnnp_cmd_exec_io_set_by_copy,
		.deinit = ccfree,
	};
}
