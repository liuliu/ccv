/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_dynamic_graph_internal_h
#define GUARD_ccv_nnc_dynamic_graph_internal_h

#include "ccv_nnc.h"
#include "ccv_nnc_internal.h"

#define CCV_NNC_IS_EXTERN_TENSOR_VIEW(tv) ((uintptr_t)(tv) & 1)
#define CCV_NNC_TENSOR_VIEW(tv) ((ccv_nnc_tensor_view_t*)((uintptr_t)(tv) & ~(uintptr_t)1))

enum {
	CCV_NNC_TENSOR_VARIABLE,
	CCV_NNC_TENSOR_CONSTANT,
};

struct ccv_nnc_tensor_variable_s {
	int type;
	int index;
	int alias_ref;
	ccv_nnc_tensor_param_t info;
	ccv_nnc_tensor_symbol_t symbol;
	ccv_nnc_tensor_view_t* tensor_view;
	int ofs[CCV_NNC_MAX_DIM_ALLOC];
	int inc[CCV_NNC_MAX_DIM_ALLOC];
};

enum {
	CCV_NNC_TENSOR_NO_VARIABLE = -1,
	CCV_NNC_TENSOR_NO_VARIABLE_BUT_USED = -2,
};

typedef struct { // Extra information kept per tensor symbol along with symbolic graph.
	int type;
	int index; // The index back into the tensor variable. -1 meant no associated tensor vairable.
	ccv_array_t* sources; // array of graph_exec_symbol, use this tensor symbol as output.
	ccv_array_t* destinations; // array of graph_exec_symbol, use this tensor symbol as input.
	ccv_nnc_tensor_view_t* tensor_view; // Transfer ownership of the tensor view to here.
} ccv_nnc_tensor_variable_graph_bind_t;

struct ccv_nnc_dynamic_graph_s {
	int reuse_var; // -1 if no var can be reused. Otherwise first locate the reuse var without increase array size.
	ccv_array_t* vars; // Array keeps track of all allocated tensor variable.
	ccv_array_t* binds; // Array keeps track of extra information for a tensor symbol.
	ccv_nnc_symbolic_graph_t* tape; // Symbolic graph to keep track of computation.
	ccv_array_t* ws; // array of integers as workspace
};

ccv_nnc_tensor_variable_t ccv_nnc_tensor_variable_exchange_new(ccv_nnc_dynamic_graph_t* const graph, ccv_nnc_tensor_variable_t tensor_variable);

static inline void ccv_nnc_insert_if_prior_to_any(const ccv_nnc_symbolic_graph_t* const graph, const int d, ccv_array_t* const sources, uint32_t* const visited, int* const buf0, int* const buf1)
{
	if (visited[(d >> 5)] & (1u << (d & 31)))
		return;
	visited[(d >> 5)] |= (1u << (d & 31));
	buf0[0] = d;
	int* buf[2] = {
		buf0, buf1
	};
	int buf_size[2] = {
		1, 0
	};
	int p = 0, q = 1;
	int i, j, k;
	int flag = 0;
	while (buf_size[p] > 0)
	{
		buf_size[q] = 0;
		for (i = 0; i < buf_size[p]; i++)
		{
			const int* outgoings; int outgoing_size;
			ccv_nnc_graph_exec_symbol_to(graph, (ccv_nnc_graph_exec_symbol_t){
				.d = buf[p][i],
				.graph = graph
			}, &outgoings, &outgoing_size);
			for (j = 0; j < outgoing_size; j++)
			{
				const int outgoing_idx = outgoings[j];
				for (k = 0; k < sources->rnum; k++)
				{
					ccv_nnc_graph_exec_symbol_t* const source_symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, k);
					// If this outgoing idx is one of the source, replace it with d, or delete it.
					if (source_symbol->d == outgoing_idx)
					{
						if (!flag)
						{
							source_symbol->d = d;
							flag = 1;
						} else {
							// Delete this from the list.
							if (k < sources->rnum - 1)
								source_symbol->d = ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(sources, sources->rnum - 1))->d;
							--sources->rnum;
						}
						break;
					}
				}
				if (visited[(outgoing_idx >> 5)] & (1u << (outgoing_idx & 31)))
					continue;
				visited[(outgoing_idx >> 5)] |= (1u << (outgoing_idx & 31));
				buf[q][buf_size[q]] = outgoing_idx;
				++buf_size[q];
			}
		}
		CCV_SWAP(p, q, i);
	}
	// If this node is not visited, and we cannot find anything in the sources to replace, this is a new top node.
	if (!flag)
	{
		const ccv_nnc_graph_exec_symbol_t source_symbol = {
			.d = d,
			.graph = graph
		};
		ccv_array_push(sources, &source_symbol);
	}
}

static inline void ccv_nnc_remove_if_prior_to_any(const ccv_nnc_symbolic_graph_t* const graph, const int d, ccv_array_t* const destinations, uint32_t* const visited, int* const buf0, int* const buf1)
{
	int i, j, k;
	// If it is already visited, this is the later one, we are good.
	if (visited[(d >> 5)] & (1u << (d & 31)))
		return;
	visited[(d >> 5)] |= (1u << (d & 31));
	buf0[0] = d;
	int* buf[2] = {
		buf0, buf1
	};
	int buf_size[2] = {
		1, 0
	};
	int p = 0, q = 1;
	int flag = 0;
	while (!flag && buf_size[p] > 0)
	{
		buf_size[q] = 0;
		for (i = 0; !flag && i < buf_size[p]; i++)
		{
			const int* outgoings; int outgoing_size;
			ccv_nnc_graph_exec_symbol_to(graph, (ccv_nnc_graph_exec_symbol_t){
				.d = buf[p][i],
				.graph = graph
			}, &outgoings, &outgoing_size);
			for (j = 0; j < outgoing_size; j++)
			{
				const int outgoing_idx = outgoings[j];
				// If this node happens to be visited, do nothing.
				if (visited[(outgoing_idx >> 5)] & (1u << (outgoing_idx & 31)))
					continue;
				for (k = 0; !flag && k < destinations->rnum; k++)
				{
					ccv_nnc_graph_exec_symbol_t* const destination_symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, k);
					// If this outgoing idx is one of the destination, delete current node.
					flag = (destination_symbol->d == outgoing_idx);
				}
				visited[(outgoing_idx >> 5)] |= (1u << (outgoing_idx & 31));
				buf[q][buf_size[q]] = outgoing_idx;
				++buf_size[q];
			}
		}
		CCV_SWAP(p, q, i);
	}
	if (flag)
		for (i = 0; i < destinations->rnum; i++)
		{
			ccv_nnc_graph_exec_symbol_t* const destination_symbol = (ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, i);
			// If this outgoing idx is one of the destination, delete current node.
			if (destination_symbol->d == d)
			{
				// Delete this from the list.
				if (i < destinations->rnum - 1)
					destination_symbol->d = ((ccv_nnc_graph_exec_symbol_t*)ccv_array_get(destinations, destinations->rnum - 1))->d;
				--destinations->rnum;
				break;
			}
		}
}

enum {
	CCV_NNC_SYMBOL_TENSOR,
	CCV_NNC_SYMBOL_TENSOR_ALIAS,
	CCV_NNC_SYMBOL_GRAPH_EXEC,
};

typedef struct {
	int type;
	int d;
} ccv_nnc_tape_symbol_t;

static inline void ccv_nnc_dynamic_graph_push_backward_tensor_symbol(void* context, const ccv_nnc_tensor_symbol_t symbol, const ccv_nnc_tensor_param_t info, const char* const name)
{
	ccv_array_t* const stack = (ccv_array_t*)context;
	ccv_nnc_tape_symbol_t tape_symbol = {
		.d = symbol.d,
		.type = CCV_NNC_SYMBOL_TENSOR,
	};
	ccv_array_push(stack, &tape_symbol);
}

static inline void ccv_nnc_dynamic_graph_push_backward_tensor_symbol_alias(void* context, const ccv_nnc_tensor_symbol_t symbol, const ccv_nnc_tensor_symbol_t from_symbol, const int ofs[CCV_NNC_MAX_DIM_ALLOC], const int inc[CCV_NNC_MAX_DIM_ALLOC], const ccv_nnc_tensor_param_t info, const char* const name)
{
	ccv_array_t* const stack = (ccv_array_t*)context;
	ccv_nnc_tape_symbol_t tape_symbol = {
		.d = symbol.d,
		.type = CCV_NNC_SYMBOL_TENSOR_ALIAS,
	};
	ccv_array_push(stack, &tape_symbol);
}

static inline void ccv_nnc_dynamic_graph_push_backward_graph_exec_symbol(void* context, const ccv_nnc_graph_exec_symbol_t symbol, const ccv_nnc_cmd_t cmd, const ccv_nnc_tensor_symbol_t* const inputs, const int input_size, const ccv_nnc_tensor_symbol_t* const outputs, const int output_size, const char* const name)
{
	ccv_array_t* const stack = (ccv_array_t*)context;
	ccv_nnc_tape_symbol_t tape_symbol = {
		.d = symbol.d,
		.type = CCV_NNC_SYMBOL_GRAPH_EXEC,
	};
	ccv_array_push(stack, &tape_symbol);
}

#endif
