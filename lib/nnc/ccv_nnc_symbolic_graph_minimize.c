#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"

/**
 * Level-3.5 API
 */

int ccv_nnc_minimizer_saved_aux_size(const ccv_nnc_cmd_t minimizer)
{
	int i, aux_size = -1;
	uint64_t input_bitmask = 0x1;
	uint64_t output_bitmask = 0x0;
	for (i = 0; i < 62 && aux_size < 0; i++)
	{
		input_bitmask |= ((uint64_t)1 << (i + 1));
		output_bitmask |= ((uint64_t)1 << i);
		if (ccv_nnc_cmd_bitmask(minimizer, i + 2, i + 1, &input_bitmask, 1, &output_bitmask, 1))
			aux_size = i;
	}
	return aux_size;
}

void ccv_nnc_symbolic_graph_minimize(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_cmd_t minimizer, const ccv_nnc_tensor_symbol_t* const losses, const int loss_size, const ccv_nnc_tensor_symbol_t* const parameters, const int parameter_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size, ccv_nnc_tensor_symbol_t* const updated_parameters, ccv_nnc_tensor_symbol_map_t* const saved_aux, ccv_nnc_graph_exec_symbol_t* const graph_exec_symbols)
{
	assert(parameter_size > 0);
	assert(loss_size > 0);
	// First, compute gradient.
	ccv_nnc_symbolic_graph_backward(graph, losses, loss_size, parameters, parameter_size, sources, source_size, destinations, destination_size);
	int i, j;
	// At most the minimizer accepts 62 additional parameters.
	const int aux_size = ccv_nnc_minimizer_saved_aux_size(minimizer);
	assert(aux_size >= 0);
	ccv_nnc_tensor_symbol_t inputs[aux_size + 2];
	ccv_nnc_tensor_symbol_t outputs[aux_size + 1];
	for (i = 0; i < parameter_size; i++)
	{
		const ccv_nnc_tensor_symbol_t gradient = ccv_nnc_tensor_symbol_for_backward(graph, parameters[i]);
		const ccv_nnc_graph_exec_symbol_t graph_exec = ccv_nnc_graph_exec_symbol_for_backward(graph, gradient);
		inputs[0] = gradient;
		inputs[1] = parameters[i];
		const ccv_nnc_tensor_param_t info = ccv_nnc_tensor_symbol_params(graph, inputs[1]);
		outputs[0] = updated_parameters[i] = ccv_nnc_tensor_symbol_new(graph, info, 0);
		for (j = 0; j < aux_size; j++)
		{
			inputs[2 + j] = saved_aux[i * aux_size + j].source = ccv_nnc_tensor_symbol_new(graph, info, 0);
			outputs[1 + j] = saved_aux[i * aux_size + j].destination = ccv_nnc_tensor_symbol_new(graph, info, 0);
		}
		const ccv_nnc_graph_exec_symbol_t minimize = ccv_nnc_graph_exec_symbol_new(graph, minimizer, inputs, aux_size + 2, outputs, aux_size + 1, 0);
		ccv_nnc_graph_exec_symbol_concat(graph, graph_exec, minimize);
		graph_exec_symbols[i] = minimize;
	}
}
