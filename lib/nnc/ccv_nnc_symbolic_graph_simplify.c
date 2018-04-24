#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_symbolic_graph.h"
#include "3rdparty/siphash/siphash24.h"

/**
 * Level-3.5 API
 */

static uint8_t key_siphash[16] = "graphcsekvlibnnc";

typedef struct {
	ccv_nnc_symbolic_graph_t* graph;
	ccv_nnc_graph_visit_t* visit;
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info;
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info;
} ccv_nnc_symbolic_graph_simplify_t;

ccv_nnc_symbolic_graph_simplify_t* _ccv_nnc_symbolic_graph_simplify_new(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size)
{
	ccv_nnc_symbolic_graph_simplify_t* const simplify = (ccv_nnc_symbolic_graph_simplify_t*)ccmalloc(sizeof(ccv_nnc_symbolic_graph_simplify_t));
	simplify->graph = graph;
	simplify->visit = ccv_nnc_graph_visit_new(graph, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, 0);
	simplify->tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_info_t) * graph->tensor_symbol_info->rnum);
	simplify->exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_info_t) * graph->exec_symbol_info->rnum);
	ccv_nnc_symbolic_graph_symbol_infer(graph, simplify->visit, sources, source_size, destinations, destination_size, 0, 0, simplify->tensor_symbol_info, simplify->exec_symbol_info);
	return simplify;
}

void _ccv_nnc_symbolic_graph_simplify_free(ccv_nnc_symbolic_graph_simplify_t* const simplify)
{
	ccv_nnc_graph_visit_free(simplify->visit);
	ccfree(simplify->tensor_symbol_info);
	ccfree(simplify->exec_symbol_info);
	ccfree(simplify);
}

typedef struct {
	int d;
	int ifbit;
	uint64_t hash;
} ccv_nnc_cse_hash_t;

static int _ccv_nnc_cse_hash_find(ccv_nnc_cse_hash_t* const hash_map, const uint64_t hash, const int map_size)
{
	int i, j;
	i = hash % map_size;
	for (j = 0; ; j++, i++)
	{
		if (i >= map_size)
			i = 0;
		if (j > hash_map[i].ifbit)
			return -1;
		if (hash_map[i].hash == hash)
			return hash_map[i].d;
	}
}

// This is a simple common sub-expression elimination implementation, particularly, we only replace the later computed output
// with the identical earlier computed output, and let the "elimination" part to the graph pruning.
static void _ccv_nnc_symbolic_graph_common_subexpression_elimination(ccv_nnc_symbolic_graph_simplify_t* const simplify)
{
	// tensor_hash starts with 0s, and it is either marked with the tensor index + 1, or the hash of the computations.
	uint64_t* const tensor_hash = (uint64_t*)cccalloc(simplify->graph->tensor_symbol_info->rnum, sizeof(uint64_t));
	int i;
	ccv_nnc_graph_visit_for(simplify->visit, simplify->exec_symbol_info, node, idx) {
		uint64_t hashout, hashin[3] = {};
		siphash((uint8_t*)&hashin[0], (const uint8_t*)&node->cmd.info, sizeof(node->cmd.info), key_siphash);
		siphash((uint8_t*)&hashin[1], (const uint8_t*)&node->hint, sizeof(node->hint), key_siphash);
		hashin[2] = node->cmd.cmd; // Now actually hash the cmd name.
		siphash((uint8_t*)&hashout, (const uint8_t*)hashin, sizeof(hashin), key_siphash);
		// First, hash the cmd and the hints with the cmd.
		// Note on alias, we cannot really generate proper hash for alias (yet). Thus, just treat alias as normal tensors.
		for (i = 0; i < node->input_size; i++)
		for (i = 0; i < node->input_size; i++)
		{
			// If no hash for the input, use the index + 1 as the hash.
			if (node->inputs[i] >= 0 && tensor_hash[node->inputs[i]] == 0)
				tensor_hash[node->inputs[i]] = node->inputs[i] + 1;
			if (node->inputs[i] >= 0)
			{
				// Hash using the tensor hash.
				hashin[0] = hashout;
				hashin[1] = i; // Encode the positional information.
				hashin[2] = tensor_hash[node->inputs[i]];
			} else {
				// Hash using the input integer (could be special integer).
				hashin[0] = hashout;
				hashin[1] = i; // Encode the positional information.
				hashin[2] = node->inputs[i];
			}
			siphash((uint8_t*)&hashout, (const uint8_t*)hashin, sizeof(hashin), key_siphash);
		}
		for (i = 0; i < node->output_size; i++)
			if (node->outputs[i] >= 0)
			{
				// Assigning once, especially now we don't consider aliases.
				assert(tensor_hash[node->outputs[i]] == 0);
				hashin[0] = hashout;
				hashin[1] = i; // Positional information.
				// Generate hash for the output.
				siphash((uint8_t*)&tensor_hash[node->outputs[i]], (const uint8_t*)hashin, sizeof(uint64_t) * 2, key_siphash);
			}
	} ccv_nnc_graph_visit_endfor
	// Allocate twice as much space, for the simple open address hash map.
	const int map_size = (simplify->graph->tensor_symbol_info->rnum * 3 + 1) / 2;
	ccv_nnc_cse_hash_t* const hash_map = (ccv_nnc_cse_hash_t*)cccalloc(sizeof(ccv_nnc_cse_hash_t), map_size);
	// Now, all tensors are hashed, identify tensors with the same hash code, replace the ones that accessed later.
	ccv_nnc_graph_visit_for(simplify->visit, simplify->exec_symbol_info, node, idx) {
		for (i = 0; i < node->input_size; i++)
			if (node->inputs[i] >= 0)
			{
				const int d = _ccv_nnc_cse_hash_find(hash_map, tensor_hash[node->inputs[i]], map_size);
				node->inputs[i] = d; // It can be replaced.
			}
		for (i = 0; i < node->output_size; i++)
			if (node->outputs[i] >= 0)
			{
				// Insert.
			}
	} ccv_nnc_graph_visit_endfor
	ccfree(tensor_hash);
	ccfree(hash_map);
}

static void _ccv_nnc_symbolic_graph_pruning(ccv_nnc_symbolic_graph_simplify_t* const simplify, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
}

void ccv_nnc_symbolic_graph_simplify(ccv_nnc_symbolic_graph_t* const graph, const int* const passes, const int pass_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size)
{
	ccv_nnc_symbolic_graph_simplify_t* const simplify = _ccv_nnc_symbolic_graph_simplify_new(graph, sources, source_size, destinations, destination_size);
	int i;
	for (i = 0; i < pass_size; i++)
		switch (passes[i])
		{
			case CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION:
				_ccv_nnc_symbolic_graph_common_subexpression_elimination(simplify);
				break;
			case CCV_NNC_SIMPLIFY_GRAPH_PRUNING:
				_ccv_nnc_symbolic_graph_pruning(simplify, outputs, output_size);
				break;
		}
	_ccv_nnc_symbolic_graph_simplify_free(simplify);
}
