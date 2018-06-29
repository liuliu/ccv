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
	int tensor_symbol_info_size;
	int exec_symbol_info_size;
	ccv_nnc_symbolic_graph_t* graph;
	ccv_nnc_graph_visit_t* visit;
	ccv_nnc_tensor_symbol_info_t* tensor_symbol_info;
	ccv_nnc_graph_exec_symbol_info_t* exec_symbol_info;
	uint32_t* exec_dead; // Mark a exec is dead and need to be cleared, each bit represent a exec.
	uint32_t* tensor_dead; // Mark a tensor is dead and need to be cleared, each bit represent a tensor.
	int* output_execs; // Mapping from tensor to the exec that generates this tensor.
} ccv_nnc_symbolic_graph_simplify_t;

static void _ccv_nnc_symbolic_graph_simplify_update_output_execs(ccv_nnc_symbolic_graph_simplify_t* const simplify)
{
	int i;
	for (i = 0; i < simplify->tensor_symbol_info_size; i++)
		simplify->output_execs[i] = -1;
	ccv_nnc_graph_visit_for(simplify->visit, simplify->exec_symbol_info, node, idx) {
		if (simplify->exec_dead[idx >> 5] & (1u << (idx & 0x1f)))
			continue;
		for (i = 0; i < node->output_size; i++)
			if (node->outputs[i] >= 0)
				simplify->output_execs[node->outputs[i]] = idx; // A tensor can only be written once.
	} ccv_nnc_graph_visit_endfor
}

static ccv_nnc_symbolic_graph_simplify_t* _ccv_nnc_symbolic_graph_simplify_new(ccv_nnc_symbolic_graph_t* const graph, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size)
{
	ccv_nnc_symbolic_graph_simplify_t* const simplify = (ccv_nnc_symbolic_graph_simplify_t*)ccmalloc(sizeof(ccv_nnc_symbolic_graph_simplify_t));
	simplify->graph = graph;
	simplify->visit = ccv_nnc_graph_visit_new(graph, (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(graph->exec_symbol_info, 0), graph->exec_symbol_info->rnum, sources, source_size, destinations, destination_size, 0);
	simplify->tensor_symbol_info = (ccv_nnc_tensor_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_tensor_symbol_info_t) * graph->tensor_symbol_info->rnum);
	simplify->exec_symbol_info = (ccv_nnc_graph_exec_symbol_info_t*)ccmalloc(sizeof(ccv_nnc_graph_exec_symbol_info_t) * graph->exec_symbol_info->rnum);
	ccv_nnc_symbolic_graph_symbol_infer(graph, simplify->visit, sources, source_size, destinations, destination_size, 0, 0, simplify->tensor_symbol_info, simplify->exec_symbol_info);
	simplify->tensor_symbol_info_size = graph->tensor_symbol_info->rnum;
	simplify->exec_symbol_info_size = graph->exec_symbol_info->rnum;
	simplify->exec_dead = cccalloc(((simplify->exec_symbol_info_size + 31) >> 5) + ((simplify->tensor_symbol_info_size + 31) >> 5), sizeof(uint32_t));
	simplify->tensor_dead = simplify->exec_dead + ((simplify->exec_symbol_info_size + 31) >> 5);
	simplify->output_execs = (int*)ccmalloc(sizeof(int) * simplify->tensor_symbol_info_size);
	return simplify;
}

static void _ccv_nnc_symbolic_graph_simplify_apply(ccv_nnc_symbolic_graph_simplify_t* const simplify)
{
	int i, j;
	for (i = 0; i < simplify->exec_symbol_info_size; i++)
		if (simplify->exec_dead[i >> 5] & (1u << (i & 0x1f)))
			ccv_nnc_graph_exec_symbol_free(simplify->graph, (ccv_nnc_graph_exec_symbol_t){
				.d = i,
				.graph = simplify->graph,
			});
		else // If it is not marked as dead, go through to unmark tensor 
			for (j = 0; j < simplify->exec_symbol_info[i].output_size; j++)
			{
				const int d = simplify->exec_symbol_info[i].outputs[j];
				if (d >= 0)
					simplify->tensor_dead[d >> 5] &= ~(1u << (d & 0x1f));
			}
	for (i = 0; i < simplify->tensor_symbol_info_size; i++)
		if (simplify->tensor_dead[i >> 5] & (1u << (i & 0x1f)))
			ccv_nnc_tensor_symbol_free(simplify->graph, (ccv_nnc_tensor_symbol_t){
				.d = i,
				.graph = simplify->graph,
			});
	// autogen the sources / destinations.
	ccv_nnc_graph_exec_symbol_autogen(simplify->graph, 0, 0, CCV_NNC_AUTOGEN_SOURCES_AND_DESTINATIONS);
}

static void _ccv_nnc_symbolic_graph_simplify_free(ccv_nnc_symbolic_graph_simplify_t* const simplify)
{
	ccv_nnc_graph_visit_free(simplify->visit);
	ccfree(simplify->tensor_symbol_info);
	ccfree(simplify->exec_symbol_info);
	ccfree(simplify->exec_dead);
	ccfree(simplify->output_execs);
	ccfree(simplify);
}

typedef struct {
	int d;
	int ifbit;
	uint64_t hash;
} ccv_nnc_cse_hash_t;

static int _ccv_nnc_cse_hash_find(ccv_nnc_cse_hash_t* const hash_map, const uint64_t hash, const int map_size)
{
	assert(hash > 0);
	int i, j;
	i = (hash - 1) % map_size;
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

static void _ccv_nnc_cse_hash_add(ccv_nnc_cse_hash_t* const hash_map, uint64_t hash, int d, const int map_size)
{
	assert(hash > 0);
	int i, j;
	i = (hash - 1) % map_size;
	for (j = 0; ; j++, i++)
	{
		if (i >= map_size)
			i = 0;
		if (hash_map[i].hash == hash) // Already exists, do nothing.
			return;
		if (hash_map[i].hash == 0)
		{
			// Insert.
			hash_map[i].d = d;
			hash_map[i].ifbit = j;
			hash_map[i].hash = hash;
			return;
		}
		if (j > hash_map[i].ifbit)
		{
			const ccv_nnc_cse_hash_t old_hash = hash_map[i];
			// Swap, and continue, until find an empty slot.
			hash_map[i].d = d;
			hash_map[i].ifbit = j;
			hash_map[i].hash = hash;
			d = old_hash.d;
			j = old_hash.ifbit;
			hash = old_hash.hash;
		}
	}
}

static int _ccv_nnc_symbolic_graph_update_refs(ccv_nnc_symbolic_graph_simplify_t* const simplify, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const int* const refs, const int output_exec_ref_dead)
{
	int i, j;
	// Go over refs, if a tensor is an alias, mark it reference to the new one.
	for (i = 0; i < simplify->tensor_symbol_info_size; i++)
		if (refs[i] >= 0)
			// Mark this tensor as dead.
			simplify->tensor_dead[i >> 5] |= (1u << (i & 0x1f));
		else if (simplify->tensor_symbol_info[i].alias_ref && refs[simplify->tensor_symbol_info[i].alias_ref - 1] >= 0) {
			const int alias_ref = simplify->tensor_symbol_info[i].alias_ref - 1;
			simplify->tensor_symbol_info[i].alias_ref = refs[alias_ref] + 1;
			((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(simplify->graph->tensor_symbol_info, i))->alias_ref = refs[alias_ref] + 1;
		}
	for (i = 0; i < output_size; i++)
		// If the output is an alias, that's fine, because if the alias is re-binded, we are good.
		simplify->tensor_dead[outputs[i].d >> 5] &= ~(1u << (outputs[i].d & 0x1f)); // Undead for output tensor symbols.
	// Merge s_refs if the tensor is dead.
	for (i = 0; i < simplify->tensor_symbol_info_size; i++)
		if (refs[i] >= 0 && (simplify->tensor_dead[i >> 5] & (1u << (i & 0x1f))))
		{
			const int ref = refs[i];
			if (simplify->tensor_symbol_info[i].s_ref && simplify->tensor_symbol_info[i].s_ref->rnum)
			{
				if (!simplify->tensor_symbol_info[ref].s_ref) // If there is no s_ref, simple, just assign the pointer and set the old one to nil.
				{
					simplify->tensor_symbol_info[ref].s_ref = simplify->tensor_symbol_info[i].s_ref;
					simplify->tensor_symbol_info[i].s_ref = 0;
					((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(simplify->graph->tensor_symbol_info, i))->s_ref = 0;
				} else {
					ccv_array_t* const ref_s_ref = simplify->tensor_symbol_info[ref].s_ref;
					ccv_array_t* const i_s_ref = simplify->tensor_symbol_info[i].s_ref;
					const int ref_s_ref_rnum = ref_s_ref->rnum;
					int flag = 0;
					// Detect conflict, if there is, undead.
					for (j = 0; !flag && j < ccv_min(ref_s_ref_rnum, i_s_ref->rnum); j++)
					{
						const int ref_s_ref_k = *(int*)ccv_array_get(ref_s_ref, j);
						const int i_s_ref_k = *(int*)ccv_array_get(i_s_ref, j);
						// If for the same sub-graph, they have different tensors linked, we cannot merge these two.
						flag = (ref_s_ref_k > 0 && i_s_ref_k > 0 && ref_s_ref_k != i_s_ref_k);
					}
					if (flag)
					{
						simplify->tensor_dead[i >> 5] &= ~(1u << (i & 0x1f)); // Undead
						continue;
					}
					if (ref_s_ref_rnum < i_s_ref->rnum)
					{
						ccv_array_resize(ref_s_ref, i_s_ref->rnum);
						memcpy(ccv_array_get(ref_s_ref, ref_s_ref_rnum), ccv_array_get(i_s_ref, ref_s_ref_rnum), sizeof(int) * (i_s_ref->rnum - ref_s_ref_rnum));
					}
					for (j = 0; j < ccv_min(ref_s_ref_rnum, i_s_ref->rnum); j++)
					{
						const int ref_s_ref_k = *(int*)ccv_array_get(ref_s_ref, j);
						const int i_s_ref_k = *(int*)ccv_array_get(i_s_ref, j);
						assert(ref_s_ref_k == 0 || i_s_ref_k == 0);
						if (i_s_ref_k)
							*(int*)ccv_array_get(ref_s_ref, j) = i_s_ref_k;
					}
					ccv_array_free(simplify->tensor_symbol_info[i].s_ref);
					simplify->tensor_symbol_info[i].s_ref = 0;
					((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(simplify->graph->tensor_symbol_info, i))->s_ref = 0;
					for (j = 0; j < ref_s_ref->rnum; j++)
					{
						const int ref_k = *(int*)ccv_array_get(ref_s_ref, j) - 1;
						if (ref_k >= 0)
						{
							ccv_nnc_symbolic_graph_t* const sub_graph = *(ccv_nnc_symbolic_graph_t**)ccv_array_get(simplify->graph->sub_graphs, j);
							assert(sub_graph);
							// Update its p_ref.
							((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(sub_graph->tensor_symbol_info, ref_k))->p_ref = ref + 1;
						}
					}
				}
				assert(simplify->tensor_symbol_info[i].s_ref == ((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(simplify->graph->tensor_symbol_info, i))->s_ref);
				assert(simplify->tensor_symbol_info[ref].s_ref == ((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(simplify->graph->tensor_symbol_info, ref))->s_ref);
			}
		}
	// Going through refs that we are updating, going through its p_ref to make sure both are updated.
	for (i = 0; i < simplify->tensor_symbol_info_size; i++)
		if (refs[i] >= 0 && (simplify->tensor_dead[i >> 5] & (1u << (i & 0x1f))) && simplify->tensor_symbol_info[i].p_ref)
		{
			const int ref = refs[i];
			const int p_ref = simplify->tensor_symbol_info[i].p_ref - 1;
			assert(p_ref >= 0);
			assert(simplify->graph->p);
			ccv_array_t* const s_ref = ((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(simplify->graph->p->tensor_symbol_info, p_ref))->s_ref;
			const int s_idx = simplify->graph->p_idx - 1;
			assert(s_idx >= 0);
			assert(s_ref && s_ref->rnum > s_idx);
			*(int*)ccv_array_get(s_ref, s_idx) = ref + 1; // Update so it references to the new s_ref.
			assert(!simplify->tensor_symbol_info[ref].p_ref);
			simplify->tensor_symbol_info[ref].p_ref = p_ref + 1;
			((ccv_nnc_tensor_symbol_info_t*)ccv_array_get(simplify->graph->tensor_symbol_info, ref))->p_ref = p_ref + 1;
		}
	// Now go over exec to mark them as dead because we don't need these to generate refs.
	if (output_exec_ref_dead)
		for (i = 0; i < simplify->tensor_symbol_info_size; i++)
			if (refs[i] >= 0 && (simplify->tensor_dead[i >> 5] & (1u << (i & 0x1f))))
			{
				const int output_exec = simplify->output_execs[i];
				assert(output_exec >= 0);
				const ccv_nnc_graph_exec_symbol_info_t* const symbol_info = simplify->exec_symbol_info + output_exec;
				int flag = 0;
				for (j = 0; !flag && j < symbol_info->output_size; j++)
				{
					const int d = symbol_info->outputs[j];
					if (d >= 0)
						flag = (!(simplify->tensor_dead[d >> 5] & (1u << (d & 0x1f)))); // If some of the output is not dead, we cannot proceed.
				}
				if (!flag) // If all outputs are dead, mark the exec as dead.
					simplify->exec_dead[output_exec >> 5] |= (1u << (output_exec & 0x1f));
			}
	int updated_refs = 0;
	// Go over replace inputs / outputs.
	ccv_nnc_graph_visit_for(simplify->visit, simplify->exec_symbol_info, node, idx) {
		for (i = 0; i < node->input_size; i++)
		{
			const int d = node->inputs[i];
			if (d >= 0 && refs[d] >= 0 && (simplify->tensor_dead[d >> 5] & (1u << (d & 0x1f))))
			{
					node->inputs[i] = refs[d]; // It can be replaced.
					updated_refs = 1;
			}
		}
		for (i = 0; i < node->output_size; i++)
		{
			const int d = node->outputs[i];
			if (d >= 0 && refs[d] >= 0 && (simplify->tensor_dead[d >> 5] & (1u << (d & 0x1f))))
			{
					node->outputs[i] = refs[d]; // It can be replaced.
					updated_refs = 1;
			}
		}
		assert(((ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(simplify->graph->exec_symbol_info, idx))->inputs == node->inputs);
		assert(((ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(simplify->graph->exec_symbol_info, idx))->outputs == node->outputs);
	} ccv_nnc_graph_visit_endfor
	const ccv_nnc_graph_exec_symbol_info_t* const p_node_info = simplify->graph->p ? (ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(simplify->graph->p->exec_symbol_info, simplify->graph->exec_idx - 1) : 0;
	if (p_node_info && (p_node_info->flags & CCV_NNC_GRAPH_EXEC_P_WHILE))
		// Go over the while inputs as well.
		for (i = 0; i < p_node_info->p_while.input_size; i++)
		{
			const int d = p_node_info->p_while.inputs[i];
			if (d >= 0 && refs[d] >= 0 && (simplify->tensor_dead[d >> 5] & (1u << (d & 0x1f))))
			{
				p_node_info->p_while.inputs[i] = refs[d];
				updated_refs = 1;
			}
		}
	return updated_refs;
}

// This is a simple common sub-expression elimination implementation, particularly, we only replace the later computed output
// with the identical earlier computed output, and let the "elimination" part to the graph pruning.
static void _ccv_nnc_symbolic_graph_common_subexpression_elimination(ccv_nnc_symbolic_graph_simplify_t* const simplify, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	_ccv_nnc_symbolic_graph_simplify_update_output_execs(simplify);
	// tensor_hash starts with 0s, and it is either marked with the tensor index + 1, or the hash of the computations.
	uint64_t* const tensor_hash = (uint64_t*)cccalloc(simplify->tensor_symbol_info_size, sizeof(uint64_t));
	int i;
	ccv_nnc_graph_visit_for(simplify->visit, simplify->exec_symbol_info, node, idx) {
		// We cannot support graph / custom command (we cannot model them properly).
		if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD ||
			node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD ||
			node->cmd.cmd == CCV_NNC_CUSTOM_FORWARD ||
			node->cmd.cmd == CCV_NNC_CUSTOM_BACKWARD)
			continue;
		// If already marked as dead, skip.
		if (simplify->exec_dead[idx >> 5] & (1u << (idx & 0x1f)))
			continue;
		uint64_t hashout, hashin[3] = {};
		siphash((uint8_t*)&hashin[0], (const uint8_t*)&node->cmd.info, sizeof(node->cmd.info), key_siphash);
		siphash((uint8_t*)&hashin[1], (const uint8_t*)&node->hint, sizeof(node->hint), key_siphash);
		hashin[2] = node->cmd.cmd; // Now actually hash the cmd name.
		siphash((uint8_t*)&hashout, (const uint8_t*)hashin, sizeof(hashin), key_siphash);
		// First, hash the cmd and the hints with the cmd.
		// Note on alias, we cannot really generate proper hash for alias (yet). Thus, just treat alias as normal tensors.
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
				siphash((uint8_t*)&hashin[2], (const uint8_t*)&simplify->tensor_symbol_info[node->outputs[i]].info,
						sizeof(simplify->tensor_symbol_info[node->outputs[i]].info), key_siphash);
				// Generate hash for the output.
				siphash((uint8_t*)&tensor_hash[node->outputs[i]], (const uint8_t*)hashin, sizeof(hashin), key_siphash);
			}
	} ccv_nnc_graph_visit_endfor
	// Allocate 3 / 2 as much space, for the simple robin-hood open address hash map.
	const int map_size = (simplify->tensor_symbol_info_size * 3 + 1) / 2;
	int* const refs = (int*)ccmalloc(sizeof(int) * simplify->tensor_symbol_info_size);
	for (i = 0; i < simplify->tensor_symbol_info_size; i++)
		refs[i] = -1;
	ccv_nnc_cse_hash_t* const hash_map = (ccv_nnc_cse_hash_t*)cccalloc(map_size, sizeof(ccv_nnc_cse_hash_t));
	// Now, all tensors are hashed, identify tensors with the same hash code, replace the ones that accessed later.
	ccv_nnc_graph_visit_for(simplify->visit, simplify->exec_symbol_info, node, idx) {
		// If already marked as dead, skip.
		if (simplify->exec_dead[idx >> 5] & (1u << (idx & 0x1f)))
			continue;
		for (i = 0; i < node->input_size; i++)
			if (node->inputs[i] >= 0)
			{
				const int d = node->inputs[i];
				assert(tensor_hash[d]);
				const int new_d = _ccv_nnc_cse_hash_find(hash_map, tensor_hash[d], map_size);
				if (new_d >= 0 && new_d != d)
				{
					// Check whether this can be replaced.
					assert(refs[d] == -1 || refs[d] == new_d);
					assert(!simplify->tensor_symbol_info[d].assign_ref);
					assert(!simplify->tensor_symbol_info[d].r_assign_ref);
					assert(!simplify->tensor_symbol_info[d].bypass_ref);
					assert(!simplify->tensor_symbol_info[new_d].assign_ref);
					assert(!simplify->tensor_symbol_info[new_d].r_assign_ref);
					assert(!simplify->tensor_symbol_info[new_d].bypass_ref);
					// Ignore if there is a peer_ref (again, peer_ref has side effect that is deeper (using tape))
					if (simplify->tensor_symbol_info[d].peer_ref)
						continue;
					// If both have p_ref, we cannot merge.
					if (simplify->tensor_symbol_info[d].p_ref && simplify->tensor_symbol_info[new_d].p_ref)
						continue;
					// Merge s_refs from ref[d] later.
					if (refs[d] != new_d)
						refs[d] = new_d;
					assert(simplify->output_execs[new_d] >= 0);
					// Establish new dependency.
					ccv_nnc_graph_exec_symbol_concat(simplify->graph, (ccv_nnc_graph_exec_symbol_t){
						.d = simplify->output_execs[new_d],
						.graph = simplify->graph,
					}, (ccv_nnc_graph_exec_symbol_t){
						.d = idx,
						.graph = simplify->graph,
					});
				}
			}
		// We can reuse the input, but we cannot do that for output of these commands.
		if (node->cmd.cmd == CCV_NNC_GRAPH_FORWARD ||
			node->cmd.cmd == CCV_NNC_GRAPH_BACKWARD ||
			node->cmd.cmd == CCV_NNC_CUSTOM_FORWARD ||
			node->cmd.cmd == CCV_NNC_CUSTOM_BACKWARD)
			continue;
		for (i = 0; i < node->output_size; i++)
			if (node->outputs[i] >= 0) // This tensor can be reused by others.
				_ccv_nnc_cse_hash_add(hash_map, tensor_hash[node->outputs[i]], node->outputs[i], map_size);
	} ccv_nnc_graph_visit_endfor
	_ccv_nnc_symbolic_graph_update_refs(simplify, outputs, output_size, refs, 1 /* For these exec that generates refs, we don't need them any more. */);
	ccfree(tensor_hash);
	ccfree(hash_map);
	ccfree(refs);
}

static void _ccv_nnc_symbolic_graph_data_transfer_opt(ccv_nnc_symbolic_graph_simplify_t* const simplify, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	_ccv_nnc_symbolic_graph_simplify_update_output_execs(simplify);
	uint32_t* const exec_dead = simplify->exec_dead;
	const ccv_nnc_tensor_symbol_info_t* const tensor_symbol_info = simplify->tensor_symbol_info;
	int i, j;
	uint32_t* const has_alias = ccmalloc(sizeof(uint32_t) * ((simplify->tensor_symbol_info_size + 31) >> 5));
	int* const refs = (int*)ccmalloc(sizeof(int) * simplify->tensor_symbol_info_size);
	int updated_refs;
	do {
		memset(has_alias, 0, sizeof(uint32_t) * ((simplify->tensor_symbol_info_size + 31) >> 5));
		// Go through until no updates is possible. This won't result an infinite loop because every time,
		// a tensor is eliminated.
		for (i = 0; i < simplify->tensor_symbol_info_size; i++)
		{
			refs[i] = -1;
			if (tensor_symbol_info[i].alias_ref)
			{
				const int alias_ref = tensor_symbol_info[i].alias_ref - 1;
				has_alias[alias_ref >> 5] |= (1u << (alias_ref & 0x1f));
			}
		}
		ccv_nnc_graph_visit_for(simplify->visit, simplify->exec_symbol_info, node, idx) {
			// If already marked as dead, skip.
			if (exec_dead[idx >> 5] & (1u << (idx & 0x1f)))
				continue;
			if (node->cmd.cmd != CCV_NNC_DATA_TRANSFER_FORWARD &&
				node->cmd.cmd != CCV_NNC_DATA_TRANSFER_BACKWARD)
				continue;
			for (i = 0; i < node->output_size; i++) // For data transfer, we only respect output size.
				if (node->inputs[i] >= 0 && node->outputs[i] >= 0)
				{
					assert(node->inputs[i] < simplify->tensor_symbol_info_size);
					assert(node->outputs[i] < simplify->tensor_symbol_info_size);
					const ccv_nnc_tensor_symbol_info_t* const input = tensor_symbol_info + node->inputs[i];
					const ccv_nnc_tensor_symbol_info_t* const output = tensor_symbol_info + node->outputs[i];
					assert(input->info.datatype == output->info.datatype);
					assert(input->info.format == output->info.format);
					// If they are not on the same device (even for NUMA), skip.
					if (input->info.type != output->info.type)
						continue;
					// If both are alias, we cannot consolidate this.
					if (input->alias_ref && output->alias_ref)
						continue;
					// If input is alias, and output has alias reference to it, output cannot be the same as input.
					if (input->alias_ref && (has_alias[node->outputs[i] >> 5] & (1u << (node->outputs[i] & 0x1f))))
						continue;
					// If output is alias, and input has alias reference to it, input cannot be the same as output.
					if (output->alias_ref && (has_alias[node->inputs[i] >> 5] & (1u << (node->inputs[i] & 0x1f))))
						continue;
					// If either are carry overs (for while), we cannot do anything.
					if (input->assign_ref || output->assign_ref ||
						input->r_assign_ref || output->r_assign_ref)
						continue;
					// If either are bypasses (for case..of), we cannot do anything.
					if (input->bypass_ref || output->bypass_ref ||
						input->r_bypass_ref || output->r_bypass_ref)
						continue;
					int flag = 0;
					for (j = 0; !flag && j < output_size; j++)
						flag = (outputs[j].d == node->inputs[i] || outputs[j].d == node->outputs[i]);
					// Either input or output cannot be in the outputs.
					if (flag)
						continue;
					// If the type is the same, check which one is the alias.
					// We always prefer alias.
					if (output->alias_ref)
						refs[node->inputs[i]] = node->outputs[i];
					else // if (input->alias_ref), else
						refs[node->outputs[i]] = node->inputs[i];
				}
		} ccv_nnc_graph_visit_endfor
		// Make sure refs reference to the end.
		for (i = 0; i < simplify->tensor_symbol_info_size; i++)
			if (refs[i] >= 0)
			{
				int ref = refs[i];
				while (refs[ref] >= 0)
					ref = refs[i];
				refs[i] = ref;
			}
		updated_refs = _ccv_nnc_symbolic_graph_update_refs(simplify, outputs, output_size, refs, 0 /* We still need these exec that generates the refs. */);
	} while (updated_refs);
	ccfree(refs);
	ccfree(has_alias);
	// Now, all references updated, remove data transfers that sources and destinations are the same.
	ccv_nnc_graph_visit_for(simplify->visit, simplify->exec_symbol_info, node, idx) {
		// If already marked as dead, skip.
		if (exec_dead[idx >> 5] & (1u << (idx & 0x1f)))
			continue;
		if (node->cmd.cmd != CCV_NNC_DATA_TRANSFER_FORWARD &&
			node->cmd.cmd != CCV_NNC_DATA_TRANSFER_BACKWARD)
			continue;
		for (i = 0; i < node->output_size; i++) // For data transfer, we only respect output size.
			if (node->inputs[i] == node->outputs[i])
			{
				if (i + 1 < node->output_size)
				{
					node->inputs[i] = node->inputs[i + 1];
					node->outputs[i] = node->outputs[i + 1];
				}
				--node->output_size;
				--i;
			}
		node->input_size = node->output_size;
		((ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(simplify->graph->exec_symbol_info, idx))->input_size = node->input_size;
		((ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(simplify->graph->exec_symbol_info, idx))->output_size = node->output_size;
		// Remove this data transfer node if it has no outputs.
		if (node->output_size == 0)
			exec_dead[idx >> 5] |= (1u << (idx & 0x1f));
	} ccv_nnc_graph_visit_endfor
}

static void _ccv_nnc_symbolic_graph_pruning_undead_exec(ccv_nnc_symbolic_graph_simplify_t* const simplify, const int exec_idx, uint32_t* const tensor_visited, ccv_array_t* const next)
{
	assert(exec_idx >= 0);
	uint32_t* const exec_dead = simplify->exec_dead;
	uint32_t* const tensor_dead = simplify->tensor_dead;
	exec_dead[exec_idx >> 5] &= ~(1u << (exec_idx & 0x1f)); // Undead the exec.
	ccv_nnc_graph_exec_symbol_info_t* const exec_symbol_info = simplify->exec_symbol_info + exec_idx;
	int i;
	if (exec_symbol_info->cmd.cmd == CCV_NNC_GRAPH_FORWARD ||
		exec_symbol_info->cmd.cmd == CCV_NNC_GRAPH_BACKWARD ||
		exec_symbol_info->cmd.cmd == CCV_NNC_CUSTOM_FORWARD ||
		exec_symbol_info->cmd.cmd == CCV_NNC_CUSTOM_BACKWARD)
	{
		// All of its inputs / outputs need to be undead for these commands.
		for (i = 0; i < exec_symbol_info->input_size; i++)
		{
			const int d = exec_symbol_info->inputs[i];
			if (d >= 0 && !(tensor_visited[d >> 5] & (1u << (d & 0x1f))))
			{
				ccv_array_push(next, &d); // Push to the next round to be undead.
				tensor_visited[d >> 5] |= (1u << (d & 0x1f));
			}
		}
		for (i = 0; i < exec_symbol_info->output_size; i++)
		{
			const int d = exec_symbol_info->outputs[i];
			if (d >= 0 && !(tensor_visited[d >> 5] & (1u << (d & 0x1f))))
			{
				ccv_array_push(next, &d); // Push to the next round to be undead.
				tensor_visited[d >> 5] |= (1u << (d & 0x1f));
			}
		}
		return;
	}
	// Go through the input / output, to make sure that all of them can be available.
	const int input_bitmask_size = (exec_symbol_info->input_size + 63) >> 6;
	const int output_bitmask_size = (exec_symbol_info->output_size + 63) >> 6;
	uint64_t input_bitmasks[ccv_max(1, input_bitmask_size)];
	for (i = 0; i < input_bitmask_size; i++)
		input_bitmasks[i] = 0;
	uint64_t output_bitmasks[ccv_max(1, output_bitmask_size)];
	for (i = 0; i < output_bitmask_size; i++)
		output_bitmasks[i] = 0;
	for (i = 0; i < exec_symbol_info->input_size; i++)
		if (exec_symbol_info->inputs[i] >= 0)
			input_bitmasks[i >> 6] |= ((uint64_t)1 << (i & 63));
	for (i = 0; i < exec_symbol_info->output_size; i++)
		if (exec_symbol_info->outputs[i] >= 0)
			output_bitmasks[i >> 6] |= ((uint64_t)1 << (i & 63));
	// First, mark everything with bitmasks, and verify it works.
	assert(ccv_nnc_cmd_bitmask(exec_symbol_info->cmd, exec_symbol_info->input_size, exec_symbol_info->output_size, input_bitmasks, input_bitmask_size, output_bitmasks, output_bitmask_size));
	int flag;
	do {
		flag = 0;
		// Try to eliminate one at a time. Go over output first.
		for (i = 0; i < exec_symbol_info->output_size; i++)
		{
			const int d = exec_symbol_info->outputs[i];
			// If this tensor currently is marked as dead, try to see whether it works when we don't have this tensor at all.
			if (d >= 0 && (tensor_dead[d >> 5] & (1u << (d & 0x1f))) &&
				(output_bitmasks[i >> 6] & ((uint64_t)1 << (i & 63))))
			{
				output_bitmasks[i >> 6] &= ~((uint64_t)1 << (i & 63));
				if (ccv_nnc_cmd_bitmask(exec_symbol_info->cmd, exec_symbol_info->input_size, exec_symbol_info->output_size, input_bitmasks, input_bitmask_size, output_bitmasks, output_bitmask_size))
					flag = 1;
				else // Reset the bitmask.
					output_bitmasks[i >> 6] |= ((uint64_t)1 << (i & 63));
			}
		}
		// For inputs, no matter if it s dead or not, we try to limit our input to the smallest number.
		for (i = 0; i < exec_symbol_info->input_size; i++)
		{
			const int d = exec_symbol_info->inputs[i];
			// If this tensor currently is marked as dead, try to see whether it works when we don't have this tensor at all.
			if (d >= 0 && (input_bitmasks[i >> 6] & ((uint64_t)1 << (i & 63))))
			{
				input_bitmasks[i >> 6] &= ~((uint64_t)1 << (i & 63));
				if (ccv_nnc_cmd_bitmask(exec_symbol_info->cmd, exec_symbol_info->input_size, exec_symbol_info->output_size, input_bitmasks, input_bitmask_size, output_bitmasks, output_bitmask_size))
					flag = 1;
				else // Reset the bitmask.
					input_bitmasks[i >> 6] |= ((uint64_t)1 << (i & 63));
			}
		}
	} while (flag);
	// Now we know which one to keep, which one to undead.
	for (i = 0; i < exec_symbol_info->input_size; i++)
		if (input_bitmasks[i >> 6] & ((uint64_t)1 << (i & 63)))
		{
			const int d = exec_symbol_info->inputs[i];
			if (d >= 0 && !(tensor_visited[d >> 5] & (1u << (d & 0x1f))))
			{
				ccv_array_push(next, &d); // Push to the next round to be undead.
				tensor_visited[d >> 5] |= (1u << (d & 0x1f));
			}
		} else {
			// Clean up the inputs.
			exec_symbol_info->inputs[i] = CCV_NNC_NO_TENSOR_SYMBOL;
			assert(((ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(simplify->graph->exec_symbol_info, exec_idx))->inputs == exec_symbol_info->inputs);
		}
	for (i = 0; i < exec_symbol_info->output_size; i++)
		if (output_bitmasks[i >> 6] & ((uint64_t)1 << (i & 63)))
		{
			const int d = exec_symbol_info->outputs[i];
			if (d >= 0 && !(tensor_visited[d >> 5] & (1u << (d & 0x1f))))
			{
				ccv_array_push(next, &d); // Push to the next round to be undead.
				tensor_visited[d >> 5] |= (1u << (d & 0x1f));
			}
		} else {
			// Clean up the outputs.
			exec_symbol_info->outputs[i] = CCV_NNC_NO_TENSOR_SYMBOL;
			assert(((ccv_nnc_graph_exec_symbol_info_t*)ccv_array_get(simplify->graph->exec_symbol_info, exec_idx))->outputs == exec_symbol_info->outputs);
		}
}

static void _ccv_nnc_symbolic_graph_pruning(ccv_nnc_symbolic_graph_simplify_t* const simplify, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size)
{
	uint32_t* const tensor_visited = (uint32_t*)cccalloc(sizeof(uint32_t), (simplify->tensor_symbol_info_size + 31) >> 5);
	ccv_array_t* const preserve[2] = {
		ccv_array_new(sizeof(int), output_size, 0),
		ccv_array_new(sizeof(int), 0, 0),
	};
	int i, j;
	ccv_array_t** const r_alias_refs = (ccv_array_t**)cccalloc(sizeof(ccv_array_t*), simplify->tensor_symbol_info_size);
	for (i = 0; i < simplify->tensor_symbol_info_size; i++)
		if (simplify->tensor_symbol_info[i].alias_ref)
		{
			const int alias_ref = simplify->tensor_symbol_info[i].alias_ref - 1;
			assert(alias_ref < simplify->tensor_symbol_info_size);
			if (!r_alias_refs[alias_ref])
				r_alias_refs[alias_ref] = ccv_array_new(sizeof(int), 1, 0);
			ccv_array_push(r_alias_refs[alias_ref], &i);
		}
	uint32_t* const exec_dead = simplify->exec_dead;
	uint32_t* const tensor_dead = simplify->tensor_dead;
	int* const output_execs = simplify->output_execs;
	_ccv_nnc_symbolic_graph_simplify_update_output_execs(simplify);
	// Mark everything visited as dead.
	ccv_nnc_graph_visit_for(simplify->visit, simplify->exec_symbol_info, node, idx) {
		exec_dead[idx >> 5] |= (1u << (idx & 0x1f));
		for (i = 0; i < node->input_size; i++)
		{
			const int d = node->inputs[i];
			if (d >= 0)
				tensor_dead[d >> 5] |= (1u << (d & 0x1f));
		}
		for (i = 0; i < node->output_size; i++)
		{
			const int d = node->outputs[i];
			if (d >= 0)
				tensor_dead[d >> 5] |= (1u << (d & 0x1f));
		}
	} ccv_nnc_graph_visit_endfor
	// If the tensor symbol is used by other exec that is not visited, unmark it.
	for (i = 0; i < simplify->exec_symbol_info_size; i++)
	{
		if (exec_dead[i >> 5] & (1u << (i & 0x1f)))
			continue;
		const ccv_nnc_graph_exec_symbol_info_t* const node = simplify->exec_symbol_info + i;
		for (j = 0; j < node->input_size; j++)
		{
			const int d = node->inputs[j];
			// Undead it.
			if (d >= 0)
				tensor_dead[d >> 5] &= ~(1u << (d & 0x1f));
		}
		for (j = 0; j < node->output_size; j++)
		{
			const int d = node->outputs[j];
			// Undead it.
			if (d >= 0)
				tensor_dead[d >> 5] &= ~(1u << (d & 0x1f));
		}
	}
	for (i = 0; i < output_size; i++)
		ccv_array_push(preserve[0], &outputs[i].d);
	int p = 0, q = 1;
	// BFS to mark execs / tensors as not dead.
	while (preserve[p]->rnum > 0)
	{
		ccv_array_clear(preserve[q]);
		// First, undead all relevant tensors.
		for (i = 0; i < preserve[p]->rnum; i++)
		{
			const int d = *(int*)ccv_array_get(preserve[p], i);
			// Undead the outputs.
			tensor_dead[d >> 5] &= ~(1u << (d & 0x1f));
			int alias_ref = d;
			if (simplify->tensor_symbol_info[d].alias_ref)
			{
				alias_ref = simplify->tensor_symbol_info[d].alias_ref - 1;
				tensor_dead[alias_ref >> 5] &= ~(1u << (alias_ref & 0x1f));
				assert(r_alias_refs[alias_ref]);
			}
			if (r_alias_refs[alias_ref])
				for (j = 0; j < r_alias_refs[alias_ref]->rnum; j++)
				{
					const int b = *(int*)ccv_array_get(r_alias_refs[alias_ref], j);
					if (output_execs[b] >= 0) // Only revive if it is written alias.
						tensor_dead[b >> 5] &= ~(1u << (b & 0x1f));
				}
		}
		for (i = 0; i < preserve[p]->rnum; i++)
		{
			const int d = *(int*)ccv_array_get(preserve[p], i);
			const int output_exec = output_execs[d];
			// Undead the exec.
			if (output_exec >= 0)
				_ccv_nnc_symbolic_graph_pruning_undead_exec(simplify, output_exec, tensor_visited, preserve[q]);
			int alias_ref = d;
			if (simplify->tensor_symbol_info[d].alias_ref)
			{
				alias_ref = simplify->tensor_symbol_info[d].alias_ref - 1;
				const int output_exec = output_execs[alias_ref];
				if (output_exec >= 0)
					_ccv_nnc_symbolic_graph_pruning_undead_exec(simplify, output_exec, tensor_visited, preserve[q]);
			}
			if (r_alias_refs[alias_ref])
				for (j = 0; j < r_alias_refs[alias_ref]->rnum; j++)
				{
					const int b = *(int*)ccv_array_get(r_alias_refs[alias_ref], j);
					const int output_exec = output_execs[b];
					if (output_exec >= 0)
						_ccv_nnc_symbolic_graph_pruning_undead_exec(simplify, output_exec, tensor_visited, preserve[q]);
				}
		}
		CCV_SWAP(p, q, i);
	}
	ccfree(tensor_visited);
	ccv_array_free(preserve[0]);
	ccv_array_free(preserve[1]);
	for (i = 0; i < simplify->tensor_symbol_info_size; i++)
		if (r_alias_refs[i])
			ccv_array_free(r_alias_refs[i]);
	ccfree(r_alias_refs);
}

void ccv_nnc_symbolic_graph_simplify(ccv_nnc_symbolic_graph_t* const graph, const int* const passes, const int pass_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const ccv_nnc_graph_exec_symbol_t* const sources, const int source_size, const ccv_nnc_graph_exec_symbol_t* const destinations, const int destination_size)
{
	ccv_nnc_symbolic_graph_simplify_t* const simplify = _ccv_nnc_symbolic_graph_simplify_new(graph, sources, source_size, destinations, destination_size);
	int i;
	for (i = 0; i < pass_size; i++)
		switch (passes[i])
		{
			case CCV_NNC_SIMPLIFY_COMMON_SUBEXPRESSION_ELIMINATION:
				_ccv_nnc_symbolic_graph_common_subexpression_elimination(simplify, outputs, output_size);
				break;
			case CCV_NNC_SIMPLIFY_DATA_TRANSFER_OPT:
				_ccv_nnc_symbolic_graph_data_transfer_opt(simplify, outputs, output_size);
				break;
			case CCV_NNC_SIMPLIFY_GRAPH_PRUNING:
				_ccv_nnc_symbolic_graph_pruning(simplify, outputs, output_size);
				break;
		}
	_ccv_nnc_symbolic_graph_simplify_apply(simplify);
	_ccv_nnc_symbolic_graph_simplify_free(simplify);
}
