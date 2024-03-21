#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_cnnp_model.h"
// This can be removed once we organized ccv_cnnp_apply_gradient_checkpoints better.
#include "_ccv_nnc_symbolic_graph.h"

typedef struct {
	ccv_array_t* outgoings;
} ccv_nnc_graph_exec_symbol_reverse_t;

typedef struct {
	ccv_array_t* tensor_symbols;
	void* old_tensor_symbol_new_hook_context;
	ccv_nnc_tensor_symbol_new_hook_f old_tensor_symbol_new_hook;
	void* old_tensor_symbol_alias_new_hook_context;
	ccv_nnc_tensor_symbol_alias_new_hook_f old_tensor_symbol_alias_new_hook;
	ccv_array_t* graph_exec_symbols;
	ccv_nnc_graph_exec_symbol_new_hook_f old_graph_exec_symbol_new_hook;
	void* old_graph_exec_symbol_new_hook_context;
} ccv_cnnp_gradient_checkpoint_build_t;

static void _ccv_cnnp_gradient_checkpoint_tensor_symbol_new_hook(void* context, const ccv_nnc_tensor_symbol_t symbol, const ccv_nnc_tensor_param_t info, const char* const name)
{
	ccv_cnnp_gradient_checkpoint_build_t* const build_context = (ccv_cnnp_gradient_checkpoint_build_t*)context;
	ccv_array_push(build_context->tensor_symbols, &symbol);
	if (build_context->old_tensor_symbol_new_hook)
		build_context->old_tensor_symbol_new_hook(build_context->old_tensor_symbol_new_hook_context, symbol, info, name);
}

static void _ccv_cnnp_gradient_checkpoint_tensor_symbol_alias_new_hook(void* context, const ccv_nnc_tensor_symbol_t symbol, const ccv_nnc_tensor_symbol_t from_symbol, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info, const char* const name)
{
	ccv_cnnp_gradient_checkpoint_build_t* const build_context = (ccv_cnnp_gradient_checkpoint_build_t*)context;
	ccv_array_push(build_context->tensor_symbols, &symbol);
	if (build_context->old_tensor_symbol_alias_new_hook)
		build_context->old_tensor_symbol_alias_new_hook(build_context->old_tensor_symbol_alias_new_hook_context, symbol, from_symbol, ofs, inc, info, name);
}

static void _ccv_cnnp_model_gradient_checkpoint_graph_exec_symbol_new_hook(void* context, const ccv_nnc_graph_exec_symbol_t symbol, const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const char* const name)
{
	ccv_cnnp_gradient_checkpoint_build_t* const build = (ccv_cnnp_gradient_checkpoint_build_t*)context;
	ccv_array_push(build->graph_exec_symbols, &symbol);
	if (build->old_graph_exec_symbol_new_hook)
		build->old_graph_exec_symbol_new_hook(build->old_graph_exec_symbol_new_hook_context, symbol, cmd, inputs, input_size, outputs, output_size, name);
}

KHASH_MAP_INIT_INT(ccv_cnnp_tensor_symbol_map, int)

void ccv_cnnp_model_apply_gradient_checkpoints(ccv_cnnp_compiled_data_t* const compiled_data, ccv_nnc_symbolic_graph_t* const graph)
{
	ccv_array_t* const gradient_checkpoints = compiled_data->gradient_checkpoints;
	if (!gradient_checkpoints || gradient_checkpoints->rnum == 0) // No saved gradient checkpoints, this is an easy way out.
		return;
	// Otherwise, for each gradient checkpoint, there are 3 steps:
	// 1. Find currently, what execs exists from inputs to outputs.
	// 2. Find execs that generates the outputs, and their corresponding backward execs.
	// 3. Find all backward execs flow from outputs back to inputs.
	// 4. Generate new ops by calling build again with old inputs, record all new tensors / execs.
	// 5. Replace inputs in backward execs with the new tensors.
	// 6. Hook the execs takes inputs with edge from parents of backward execs in step 2.
	// 7. Delete newly generated execs that has no use (i.e. its outputs are not used by backward pass).
	// 8. Mark all new execs with DISABLE_OPT to avoid common sub-expression elimination pass.
	int i, j, k, l;
	ccv_array_t* input_execs = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 0, 0);
	ccv_array_t* output_execs = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 0, 0);
	ccv_array_t* input_gradient_execs = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 0, 0);
	ccv_array_t* output_gradient_execs = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 0, 0);
	ccv_array_t* visited_backward_execs = ccv_array_new(sizeof(int), 0, 0);
	ccv_array_t* replaced_backward_execs = ccv_array_new(sizeof(int), 0, 0);
	const int exec_rnum = graph->exec_symbol_info->rnum;
	ccv_nnc_graph_exec_symbol_reverse_t* const reversed_nodes = cccalloc(exec_rnum, sizeof(ccv_nnc_graph_exec_symbol_reverse_t));
	for (i = 0; i < exec_rnum; i++)
	{
		const int* tos = 0;
		int to_size = 0;
		ccv_nnc_graph_exec_symbol_to(graph, (ccv_nnc_graph_exec_symbol_t){
			.graph = graph,
			.d = i
		}, &tos, &to_size);
		if (tos)
			for (j = 0; j < to_size; j++)
			{
				if (!reversed_nodes[tos[j]].outgoings)
					reversed_nodes[tos[j]].outgoings = ccv_array_new(sizeof(int), 1, 0);
				ccv_array_add_unique_int(reversed_nodes[tos[j]].outgoings, i);
			}
	}
	uint32_t* const maskbit = cccalloc((exec_rnum + 31) >> 5, sizeof(uint32_t));
	// Temporary for build_data.
	ccv_array_t* const parameters = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
	ccv_array_t* const parameter_ids = ccv_array_new(sizeof(char*), 0, 0);
	ccv_array_t* const parameter_trainables = ccv_array_new(sizeof(int), 0, 0);
	ccv_array_t* const internals = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0);
	ccv_array_t* const internal_ids = ccv_array_new(sizeof(char*), 0, 0);
	ccv_array_t* const buf = ccv_array_new(sizeof(int), 0, 0);
	int max_output_size = 0;
	for (i = 0; i < gradient_checkpoints->rnum; i++)
	{
		ccv_cnnp_model_gradient_checkpoint_t* const checkpoint = (ccv_cnnp_model_gradient_checkpoint_t*)ccv_array_get(gradient_checkpoints, i);
		max_output_size = ccv_max(checkpoint->output_size, max_output_size);
	}
	ccv_nnc_tensor_symbol_t* max_outputs = ccmalloc(sizeof(ccv_nnc_tensor_symbol_t) * max_output_size);
	ccv_array_t* newly_used_outputs = ccv_array_new(sizeof(int), 0, 0);
	for (i = 0; i < gradient_checkpoints->rnum; i++)
	{
		ccv_cnnp_model_gradient_checkpoint_t* const checkpoint = (ccv_cnnp_model_gradient_checkpoint_t*)ccv_array_get(gradient_checkpoints, i);
		ccv_array_clear(input_execs);
		ccv_array_clear(output_execs);
		ccv_nnc_graph_exec_symbol_info_t* exec_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0);
		for (j = 0; j < exec_rnum; j++)
		{
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_info[j].flags))
				continue;
			const int* inputs = exec_info[j].inputs;
			int input_size = exec_info[j].input_size;
			const int* outputs = exec_info[j].outputs;
			int output_size = exec_info[j].output_size;
			if (input_size == 0 && output_size == 0)
				continue;
			// Only go through forward pass.
			if (ccv_nnc_cmd_is_backward(exec_info[j].cmd))
				continue;
			const ccv_nnc_graph_exec_symbol_t symbol = {
				.graph = graph,
				.d = j
			};
			int flag = 0;
			for (k = 0; inputs && k < input_size && !flag; k++)
				if (inputs[k] >= 0)
				for (l = 0; l < checkpoint->input_size && !flag; l++)
					if (checkpoint->inputs[l].d >= 0 && inputs[k] == checkpoint->inputs[l].d)
						flag = 1;
			if (flag)
				ccv_array_push(input_execs, &symbol);
			flag = 0;
			for (k = 0; outputs && k < output_size && !flag; k++)
				if (outputs[k] >= 0)
					for (l = 0; l < checkpoint->output_size && !flag; l++)
						if (checkpoint->outputs[l].d >= 0 && outputs[k] == checkpoint->outputs[l].d)
							flag = 1;
			if (flag)
				ccv_array_push(output_execs, &symbol);
		}
		if (input_execs->rnum <= 0 || output_execs->rnum <= 0)
			continue;
		// Fill in blanks (i.e. the backward ops that are not showing in above, but should be included to avoid excluding necessary ones). This is done by flowing gradients from outputs back all the way to inputs.
		ccv_array_clear(input_gradient_execs);
		ccv_array_clear(output_gradient_execs);
		for (j = 0; j < input_execs->rnum; j++)
		{
			const int d = ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(input_execs, j))->d;
			for (k = 0; k < exec_info[d].input_size; k++)
				if (exec_info[d].inputs[k] >= 0)
				{
					const ccv_nnc_tensor_symbol_t gradient_symbol = ccv_nnc_tensor_symbol_for_backward(graph, (ccv_nnc_tensor_symbol_t){
						.graph = graph,
						.d = exec_info[d].inputs[k]
					});
					if (gradient_symbol.d < 0)
						continue;
					const ccv_nnc_graph_exec_symbol_t backward = ccv_nnc_graph_exec_symbol_for_backward(graph, gradient_symbol);
					if (backward.d < 0)
						continue;
					if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_info[backward.d].flags))
						continue;
					int flag = 0;
					for (l = 0; !flag && l < output_gradient_execs->rnum; l++)
						if (((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(output_gradient_execs, l))->d == backward.d)
							flag = 1;
					if (!flag)
						ccv_array_push(output_gradient_execs, &backward);
				}
			if (exec_info[d].outgoings && exec_info[d].outgoings->rnum > 0)
				for (k = 0; k < exec_info[d].outgoings->rnum; k++)
				{
					const int to_d = *(int*)ccv_array_get(exec_info[d].outgoings, k);
					if (!ccv_nnc_cmd_is_backward(exec_info[to_d].cmd))
						continue;
					int flag = 0;
					for (l = 0; !flag && l < output_gradient_execs->rnum; l++)
						if (((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(output_gradient_execs, l))->d == to_d)
							flag = 1;
					if (!flag)
					{
						const ccv_nnc_graph_exec_symbol_t backward = {
							.graph = graph,
							.d = to_d
						};
						ccv_array_push(output_gradient_execs, &backward);
					}
				}
		}
		// For output_gradient_execs, we can be opportunistic and use the wrt symbols (if exists) to find relevant bits.
		// For input_gradient_execs, there is no other way but to loop over all outgoings, find the ones are direct link as backward execs.
		for (j = 0; j < output_execs->rnum; j++)
		{
			const int d = ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(output_execs, j))->d;
			if (exec_info[d].outgoings && exec_info[d].outgoings->rnum > 0)
				for (k = 0; k < exec_info[d].outgoings->rnum; k++)
				{
					const int to_d = *(int*)ccv_array_get(exec_info[d].outgoings, k);
					if (!ccv_nnc_cmd_is_backward(exec_info[to_d].cmd))
						continue;
					int flag = 0;
					for (l = 0; !flag && l < input_gradient_execs->rnum; l++)
						if (((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(input_gradient_execs, l))->d == to_d)
							flag = 1;
					if (!flag)
					{
						const ccv_nnc_graph_exec_symbol_t backward = {
							.graph = graph,
							.d = to_d
						};
						ccv_array_push(input_gradient_execs, &backward);
					}
				}
		}
		// Note that we have to use up-to-date ones because the exec_info might have outgoings that is up-to-date.
		ccv_nnc_graph_visit_t* const visit = ccv_nnc_graph_visit_new(graph, exec_info, graph->exec_symbol_info->rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(input_gradient_execs, 0), input_gradient_execs->rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(output_gradient_execs, 0), output_gradient_execs->rnum, 1);
		ccv_nnc_graph_visit_for(visit, exec_info, node, idx) {
			if (idx < exec_rnum && !CCV_NNC_GRAPH_EXEC_IS_DEAD(node->flags))
				maskbit[idx >> 5] |= (1u << (idx & 0x1f));
		} ccv_nnc_graph_visit_endfor
		ccv_array_clear(visited_backward_execs);
		// Add more backward pass to the list. Note that we don't add everything, particularly there are new nodes created through gradient checkpointing are ignored.
#define visitor(node, idx, _) \
		if (idx < exec_rnum && !CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_info[idx].flags) && maskbit[idx >> 5] & (1u << (idx & 0x1f))) \
			ccv_array_add_unique_int(visited_backward_execs, idx);
		CCV_NNC_GRAPH_VISIT(graph, reversed_nodes, exec_rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(output_gradient_execs, 0), output_gradient_execs->rnum, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(input_gradient_execs, 0), input_gradient_execs->rnum, 0, visitor);
		for (j = 0; j < input_gradient_execs->rnum; j++)
			ccv_array_add_unique_int(visited_backward_execs, ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(input_gradient_execs, j))->d);
#undef visitor
		ccv_cnnp_gradient_checkpoint_build_t build = {
			.tensor_symbols = ccv_array_new(sizeof(ccv_nnc_tensor_symbol_t), 0, 0),
			.graph_exec_symbols = ccv_array_new(sizeof(ccv_nnc_graph_exec_symbol_t), 0, 0),
		};
		build.old_tensor_symbol_new_hook_context = ccv_nnc_tensor_symbol_new_hook(graph, _ccv_cnnp_gradient_checkpoint_tensor_symbol_new_hook, &build, &build.old_tensor_symbol_new_hook);
		build.old_tensor_symbol_alias_new_hook_context = ccv_nnc_tensor_symbol_alias_new_hook(graph, _ccv_cnnp_gradient_checkpoint_tensor_symbol_alias_new_hook, &build, &build.old_tensor_symbol_alias_new_hook);
		build.old_graph_exec_symbol_new_hook_context = ccv_nnc_graph_exec_symbol_new_hook(graph, _ccv_cnnp_model_gradient_checkpoint_graph_exec_symbol_new_hook, &build, &build.old_graph_exec_symbol_new_hook);
		ccv_array_clear(parameters);
		ccv_array_clear(parameter_ids);
		ccv_array_clear(parameter_trainables);
		ccv_array_clear(internals);
		ccv_array_clear(internal_ids);
		ccv_cnnp_model_sequence_t model_sequence = {
			.bank = kh_init(ccv_cnnp_model_name_bank)
		};
		ccv_cnnp_model_add_to_array_context_t add_to_parameter_context = {
			.sequence = &model_sequence,
			.prefix = 't',
			.symbols = parameters,
			.ids = parameter_ids,
			.trainables = parameter_trainables,
		};
		ccv_cnnp_model_add_to_array_context_t add_to_output_context = {
			.sequence = &model_sequence,
			.prefix = 'r',
			.symbols = internals,
			.ids = internal_ids,
			.trainables = 0,
		};
		ccv_cnnp_model_build_data_t build_data = {
			.is_trainable = checkpoint->is_trainable,
			.model_sequence = &model_sequence,
			.add_to_array = ccv_cnnp_model_add_to_array,
			.parameters = parameters,
			.context = {
				.add_to_parameter = &add_to_parameter_context,
				.add_to_output = &add_to_output_context,
			},
			.is_gradient_checkpointing = 1, // Mark this as true so we don't allocate gradient_checkpoints array or override the hooks.
			.gradient_checkpoints = 0,
		};
		checkpoint->model->data = &build_data;
		checkpoint->build(checkpoint->model, graph, checkpoint->inputs, checkpoint->input_size, max_outputs, checkpoint->output_size);
		checkpoint->model->data = 0;
		kh_destroy(ccv_cnnp_model_name_bank, model_sequence.bank);
		if (model_sequence.sequences)
			ccv_array_free(model_sequence.sequences);
		ccv_nnc_tensor_symbol_new_hook(graph, build.old_tensor_symbol_new_hook, build.old_tensor_symbol_new_hook_context, 0);
		ccv_nnc_tensor_symbol_alias_new_hook(graph, build.old_tensor_symbol_alias_new_hook, build.old_tensor_symbol_alias_new_hook_context, 0);
		ccv_nnc_graph_exec_symbol_autogen(graph, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(build.graph_exec_symbols, 0), build.graph_exec_symbols->rnum, 0);
		for (j = 0; j < parameter_ids->rnum; j++)
			ccfree(*(char**)ccv_array_get(parameter_ids, j));
		for (j = 0; j < internal_ids->rnum; j++)
			ccfree(*(char**)ccv_array_get(internal_ids, j));
		// Note that there is no graph optimization applied here.
		exec_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0);
		// Reuse existing one.
		ccv_array_t* const newly_input_execs = input_execs;
		ccv_array_t* const newly_output_execs = output_execs;
		ccv_array_clear(newly_input_execs);
		ccv_array_clear(newly_output_execs);
		for (j = 0; j < build.graph_exec_symbols->rnum; j++)
		{
			const int idx = ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(build.graph_exec_symbols, j))->d;
			if (idx < 0)
				continue;
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_info[idx].flags))
				continue;
			const ccv_nnc_graph_exec_symbol_t symbol = {
				.graph = graph,
				.d = idx
			};
			const int* inputs = exec_info[idx].inputs;
			int input_size = exec_info[idx].input_size;
			// Only go through forward pass.
			assert(!ccv_nnc_cmd_is_backward(exec_info[idx].cmd));
			int flag = 0;
			for (k = 0; inputs && k < input_size && !flag; k++)
				if (inputs[k] >= 0)
				for (l = 0; l < checkpoint->input_size && !flag; l++)
					if (checkpoint->inputs[l].d >= 0 && inputs[k] == checkpoint->inputs[l].d)
						flag = 1;
			if (flag)
				ccv_array_push(newly_input_execs, &symbol);
			flag = 0;
			const int* outputs = exec_info[idx].outputs;
			int output_size = exec_info[idx].output_size;
			for (k = 0; inputs && k < output_size && !flag; k++)
				if (outputs[k] >= 0)
				for (l = 0; l < checkpoint->output_size && !flag; l++)
					if (max_outputs[l].d >= 0 && outputs[k] == max_outputs[l].d)
						flag = 1;
			if (flag)
				ccv_array_push(newly_output_execs, &symbol);
		}
		for (j = 0; j < checkpoint->input_size; j++)
			if (checkpoint->inputs[j].d >= 0)
				ccv_array_push(parameters, checkpoint->inputs + j);
		ccv_nnc_symbolic_graph_simplify(graph,
			SYMBOLIC_GRAPH_PASSES(CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION,
				CCV_NNC_SIMPLIFY_DATA_TRANSFER_OPT,
				CCV_NNC_SIMPLIFY_OPS_FUSION),
			ccv_array_get(parameters, 0), parameters->rnum,
			max_outputs, checkpoint->output_size,
			ccv_array_get(newly_input_execs, 0), newly_input_execs->rnum, ccv_array_get(newly_output_execs, 0), newly_output_execs->rnum);
		ccv_nnc_graph_exec_symbol_new_hook(graph, build.old_graph_exec_symbol_new_hook, build.old_graph_exec_symbol_new_hook_context, 0);
		// Need to autogen and redo source / destination.
		ccv_nnc_graph_exec_symbol_autogen(graph, (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(build.graph_exec_symbols, 0), build.graph_exec_symbols->rnum, 0);
		exec_info = (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0);
		ccv_array_clear(newly_input_execs);
		for (j = 0; j < build.graph_exec_symbols->rnum; j++)
		{
			const int idx = ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(build.graph_exec_symbols, j))->d;
			if (idx < 0)
				continue;
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_info[idx].flags))
				continue;
			const ccv_nnc_graph_exec_symbol_t symbol = {
				.graph = graph,
				.d = idx
			};
			const int* inputs = exec_info[idx].inputs;
			int input_size = exec_info[idx].input_size;
			// Only go through forward pass.
			assert(!ccv_nnc_cmd_is_backward(exec_info[idx].cmd));
			int flag = 0;
			for (k = 0; inputs && k < input_size && !flag; k++)
				if (inputs[k] >= 0)
				for (l = 0; l < checkpoint->input_size && !flag; l++)
					if (checkpoint->inputs[l].d >= 0 && inputs[k] == checkpoint->inputs[l].d)
						flag = 1;
			if (flag)
				ccv_array_push(newly_input_execs, &symbol);
		}
		// Build a map between old tensor symbols and new tensor symbols.
		khash_t(ccv_cnnp_tensor_symbol_map)* symbol_map = kh_init(ccv_cnnp_tensor_symbol_map);
		assert(build.tensor_symbols->rnum <= checkpoint->tensor_symbols->rnum);
		// Build a map to potentially map from old input to new input. 
		for (j = 0, k = 0; j < build.tensor_symbols->rnum && k < checkpoint->tensor_symbols->rnum;)
		{
			const int from_d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(checkpoint->tensor_symbols, k))->d;
			assert(from_d >= 0);
			const int to_d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(build.tensor_symbols, j))->d;
			assert(to_d >= 0);
			int from_flag = 0;
			int to_flag = 0;
			for (l = 0; (!from_flag || !to_flag) && l < parameters->rnum; l++)
			{
				const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(parameters, l))->d;
				if (d == from_d)
					from_flag = 1;
				if (d == to_d)
					to_flag = 1;
			}
			if (!from_flag || !to_flag)
				for (l = 0; (!from_flag || !to_flag) && l < internals->rnum; l++)
				{
					const int d = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(internals, l))->d;
					if (d == from_d)
						from_flag = 1;
					if (d == to_d)
						to_flag = 1;
				}
			if (from_flag)
				++k;
			if (to_flag)
				++j;
			if (from_flag || to_flag)
				continue;
			++k;
			++j;
			// Skip if from_d is outputs.
			for (l = 0; l < !from_flag && checkpoint->output_size; l++)
				if (checkpoint->outputs[l].d == from_d)
					from_flag = 1;
			if (from_flag)
				continue;
			int ret = 0;
			khiter_t h = kh_put(ccv_cnnp_tensor_symbol_map, symbol_map, from_d, &ret);
			kh_val(symbol_map, h) = to_d;
		}
		// Now go over all backward passes to replace inputs with the ones from symbol map. Record these that are used.
		ccv_array_clear(newly_used_outputs);
		ccv_array_clear(replaced_backward_execs);
		for (j = 0; j < visited_backward_execs->rnum; j++)
		{
			const int idx = *(int*)ccv_array_get(visited_backward_execs, j);
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_info[idx].flags))
				continue;
			assert(idx >= 0);
			assert(idx < exec_rnum);
			if (!ccv_nnc_cmd_is_backward(exec_info[idx].cmd))
				continue;
			for (k = 0; k < exec_info[idx].input_size; k++)
				if (exec_info[idx].inputs[k] >= 0)
				{
					const khiter_t h = kh_get(ccv_cnnp_tensor_symbol_map, symbol_map, exec_info[idx].inputs[k]);
					if (h != kh_end(symbol_map)) // Replacing it.
					{
						const int newly_created_output = kh_val(symbol_map, h);
						exec_info[idx].inputs[k] = newly_created_output;
						ccv_array_add_unique_int(newly_used_outputs, newly_created_output);
						ccv_array_add_unique_int(replaced_backward_execs, idx);
					}
				}
		}
		for (j = 0; j < build.graph_exec_symbols->rnum; j++)
		{
			ccv_nnc_graph_exec_symbol_t* const symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(build.graph_exec_symbols, j);
			if (symbol->d < 0)
				continue;
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_info[symbol->d].flags))
				continue;
			int x, y;
			for (k = 0; k < replaced_backward_execs->rnum; k++)
			{
				const int idx = *(int*)ccv_array_get(replaced_backward_execs, k);
				assert(idx >= 0);
				assert(idx < exec_rnum);
				assert(ccv_nnc_cmd_is_backward(exec_info[idx].cmd));
				int flag = 0;
				for (x = 0; !flag && x < exec_info[idx].input_size; x++)
					for (y = 0; !flag && y < exec_info[symbol->d].output_size; y++)
						if (exec_info[idx].inputs[x] == exec_info[symbol->d].outputs[y])
							flag = 1;
				if (flag)
					ccv_nnc_graph_exec_symbol_concat(graph, *symbol, (ccv_nnc_graph_exec_symbol_t){
						.graph = graph,
						.d = idx
					});
			}
		}
		// Find parents to visited_backward_execs, and use that as the starting point of all newly added graph_exec_symbols. Use the visited backward execs as the source, use all its parents as destination, go through with graph visit.
		ccv_sparse_matrix_t* const exec_dep = ccv_sparse_matrix_new(graph->exec_symbol_info->rnum, graph->exec_symbol_info->rnum, CCV_8U | CCV_C1, CCV_SPARSE_ROW_MAJOR, 0);
#define for_block(x, val) \
		do { \
			if (((uint8_t*)val)[0] != 0) \
				ccv_array_push(buf, &x); \
		} while (0)
		const uint8_t one = 1;
		// Now go from outputs to inputs, unmark visited ones.
		ccv_nnc_graph_visit_for(visit, exec_info, node, idx) {
			if (idx < exec_rnum && !CCV_NNC_GRAPH_EXEC_IS_DEAD(node->flags) && maskbit[idx >> 5] & (1u << (idx & 0x1f)))
			{
				ccv_array_clear(buf);
				ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, idx);
				if (vector)
					CCV_SPARSE_VECTOR_FOREACH(exec_dep, vector, for_block);
				if (node->outgoings && node->outgoings->rnum > 0)
				{
					ccv_array_t* const outgoings = node->outgoings;
					for (k = 0; k < outgoings->rnum; k++)
					{
						const int outgoing_d = *(int*)ccv_array_get(outgoings, k);
						if (outgoing_d >= exec_rnum)
							continue;
						int l;
						// We cannot avoid the ones that visited, because these may not contain all the deps.
						ccv_set_sparse_matrix_cell(exec_dep, outgoing_d, idx, &one);
						for (l = 0; l < buf->rnum; l++)
							ccv_set_sparse_matrix_cell(exec_dep, outgoing_d, *(int*)ccv_array_get(buf, l), &one);
					}
				}
			}
		} ccv_nnc_graph_visit_endfor
		// Now go from outputs to inputs, unmark visited ones.
		ccv_nnc_graph_visit_for(visit, exec_info, node, idx) {
			if (idx < exec_rnum)
				maskbit[idx >> 5] &= ~(1u << (idx & 0x1f));
		} ccv_nnc_graph_visit_endfor
		ccv_nnc_graph_visit_free(visit);
#undef for_block
		// Go through visited backward execs, remove the ones that has no dependency on any replaced backward execs.
		for (j = 0; j < visited_backward_execs->rnum;)
		{
			const int idx = *(int*)ccv_array_get(visited_backward_execs, j);
			if (ccv_array_contain_int(replaced_backward_execs, idx))
			{
				++j;
				continue;
			}
			ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, idx);
			int flag = 0;
#define for_block(x, val) \
			do { \
				if (((uint8_t*)val)[0] != 0) \
					if (ccv_array_contain_int(replaced_backward_execs, x)) \
						flag = 1; \
			} while (0)
			if (vector)
				CCV_SPARSE_VECTOR_FOREACH(exec_dep, vector, for_block);
#undef for_block
			if (!flag)
			{
				if (j < visited_backward_execs->rnum - 1)
					*(int*)ccv_array_get(visited_backward_execs, j) = *(int*)ccv_array_get(visited_backward_execs, visited_backward_execs->rnum - 1);
				--visited_backward_execs->rnum;
				continue;
			}
			++j;
		}
		// Now go through all replaced_backward_execs to find the ones has no dependencies in visited_backward_execs.
		for (j = 0; j < replaced_backward_execs->rnum; j++)
		{
			const int idx = *(int*)ccv_array_get(replaced_backward_execs, j);
			ccv_sparse_matrix_vector_t* vector = ccv_get_sparse_matrix_vector(exec_dep, idx);
			int flag = 0;
#define for_block(x, val) \
			do { \
				if (((uint8_t*)val)[0] != 0) \
					if (ccv_array_contain_int(visited_backward_execs, x)) \
						flag = 1; \
			} while (0)
			if (vector)
				CCV_SPARSE_VECTOR_FOREACH(exec_dep, vector, for_block);
#undef for_block
			// If this one has no parents that is within the visited_backward_execs, it is a good place for us to add all its parents as dependency for input_execs.
			if (!flag)
			{
				assert(idx < exec_rnum);
				ccv_array_t* const outgoings = reversed_nodes[idx].outgoings;
				assert(outgoings);
				for (k = 0; k < outgoings->rnum; k++)
				{
					const int d = *(int*)ccv_array_get(outgoings, k);
					for (l = 0; l < newly_input_execs->rnum; l++)
					{
						ccv_nnc_graph_exec_symbol_concat(graph, (ccv_nnc_graph_exec_symbol_t){
							.graph = graph,
							.d = d
						}, *(ccv_nnc_graph_exec_symbol_t*)ccv_array_get(newly_input_execs, l));
					}
				}
			}
		}
		ccv_matrix_free(exec_dep);
		// Go through all exec, free ones that doesn't have output used.
		// Reuse this array because it is not useful any more.
		ccv_array_t* forward_pass_inputs = visited_backward_execs;
		int any_deleted;
		do {
			// Build a map of still active inputs.
			ccv_array_clear(forward_pass_inputs);
			for (j = 0; j < build.graph_exec_symbols->rnum; j++)
			{
				ccv_nnc_graph_exec_symbol_t* const symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(build.graph_exec_symbols, j);
				if (symbol->d < 0)
					continue;
				if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_info[symbol->d].flags))
					continue;
				int* const inputs = exec_info[symbol->d].inputs;
				const int input_size = exec_info[symbol->d].input_size;
				for (k = 0; k < input_size; k++)
					ccv_array_add_unique_int(forward_pass_inputs, inputs[k]);
			}
			any_deleted = 0;
			for (j = 0; j < build.graph_exec_symbols->rnum; j++)
			{
				ccv_nnc_graph_exec_symbol_t* const symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(build.graph_exec_symbols, j);
				if (symbol->d < 0)
					continue;
				if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_info[symbol->d].flags))
					continue;
				int* const outputs = exec_info[symbol->d].outputs;
				const int output_size = exec_info[symbol->d].output_size;
				int flag = 0;
				for (k = 0; !flag && k < output_size; k++)
					flag = ccv_array_contain_int(newly_used_outputs, outputs[k]) || ccv_array_contain_int(forward_pass_inputs, outputs[k]);
				if (flag)
					continue;
				ccv_nnc_graph_exec_symbol_free(graph, *symbol);
				symbol->d = -1;
				symbol->graph = 0;
				any_deleted = 1;
			}
		} while (any_deleted);
		ccv_array_clear(forward_pass_inputs);
		for (j = 0; j < build.graph_exec_symbols->rnum; j++)
		{
			ccv_nnc_graph_exec_symbol_t* const symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(build.graph_exec_symbols, j);
			if (symbol->d < 0)
				continue;
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_info[symbol->d].flags))
				continue;
			int* const inputs = exec_info[symbol->d].inputs;
			const int input_size = exec_info[symbol->d].input_size;
			for (k = 0; k < input_size; k++)
				ccv_array_add_unique_int(forward_pass_inputs, inputs[k]);
			int* const outputs = exec_info[symbol->d].outputs;
			const int output_size = exec_info[symbol->d].output_size;
			for (k = 0; k < output_size; k++)
				ccv_array_add_unique_int(forward_pass_inputs, outputs[k]);
		}
		// Free unused tensor symbols.
		for (j = 0; j < build.tensor_symbols->rnum; j++)
		{
			const ccv_nnc_tensor_symbol_t* symbol = ((ccv_nnc_tensor_symbol_t*)ccv_array_get(build.tensor_symbols, j));
			if (ccv_array_contain_int(newly_used_outputs, symbol->d) || ccv_array_contain_int(forward_pass_inputs, symbol->d))
				continue;
			ccv_nnc_tensor_symbol_free(graph, *symbol);
		}
		for (j = 0; j < build.graph_exec_symbols->rnum; j++)
		{
			ccv_nnc_graph_exec_symbol_t* const symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(build.graph_exec_symbols, j);
			if (symbol->d < 0)
				continue;
			if (CCV_NNC_GRAPH_EXEC_IS_DEAD(exec_info[symbol->d].flags))
				continue;
			ccv_nnc_graph_exec_symbol_set_flags(graph, *symbol, CCV_NNC_GRAPH_EXEC_DISABLE_OPT);
		}
		// Free these newly created execs and tensor symbols.
		ccv_array_free(build.tensor_symbols);
		ccv_array_free(build.graph_exec_symbols);
		kh_destroy(ccv_cnnp_tensor_symbol_map, symbol_map);
	}
	ccfree(max_outputs);
	ccv_array_free(buf);
	ccv_array_free(newly_used_outputs);
	ccv_array_free(parameters);
	ccv_array_free(parameter_ids);
	ccv_array_free(parameter_trainables);
	ccv_array_free(internals);
	ccv_array_free(internal_ids);
	ccfree(maskbit);
	ccv_array_free(input_gradient_execs);
	ccv_array_free(output_gradient_execs);
	ccv_array_free(input_execs);
	ccv_array_free(output_execs);
	ccv_array_free(replaced_backward_execs);
	ccv_array_free(visited_backward_execs);
	for (i = 0; i < exec_rnum; i++)
		if (reversed_nodes[i].outgoings)
			ccv_array_free(reversed_nodes[i].outgoings);
	ccfree(reversed_nodes);
}
