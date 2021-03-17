#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_micro.h"

static int _ccv_nnc_same_index_term(const ccv_nnc_micro_loop_index_term_t a_index, const ccv_nnc_micro_loop_index_term_t b_index, const int* const groups)
{
	if (a_index.type != b_index.type)
		return 0;
	const int type = a_index.type;
	switch (type)
	{
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL:
			return a_index.immediate_value == b_index.immediate_value;
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID:
			assert(a_index.id.type != CCV_NNC_MICRO_LOOP_ID);
			assert(b_index.id.type != CCV_NNC_MICRO_LOOP_ID);
			if (groups)
			{
				if (a_index.id.type != b_index.id.type)
					return 0;
				if (a_index.id.d != b_index.id.d)
					return 0;
				switch (a_index.id.type)
				{
					case CCV_NNC_MICRO_TENSOR_ID:
					case CCV_NNC_MICRO_AXIS_SIZE_ID: {
						int a_root = groups[a_index.id.id];
						while (groups[a_root] != a_root)
							a_root = groups[a_root];
						int b_root = groups[b_index.id.id];
						while (groups[b_root] != b_root)
							b_root = groups[b_root];
						return a_root == b_root;
					}
				}
				return a_index.id.id == b_index.id.id;
			} else
				return (a_index.id.type == b_index.id.type && a_index.id.d == b_index.id.d && a_index.id.id == b_index.id.id);
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_BINARY: {
			return a_index.binary->op == b_index.binary->op && _ccv_nnc_same_index_term(a_index.binary->left, b_index.binary->left, groups) && _ccv_nnc_same_index_term(a_index.binary->right, b_index.binary->right, groups);
		}
	}
	return 0;
}

static int _ccv_nnc_same_shape(const ccv_nnc_micro_loop_index_term_t* const a_shape, const ccv_nnc_micro_loop_index_term_t* const b_shape, const int dimensions)
{
	int i;
	for (i = 0; i < dimensions; i++)
		if (!_ccv_nnc_same_index_term(a_shape[i], b_shape[i], 0))
			return 0;
	return 1;
}

static int _ccv_nnc_same_loop(const ccv_nnc_micro_loop_block_t* const left_block, const ccv_nnc_micro_loop_block_t* const right_block, const int* const groups)
{
	int left_loop_idx[left_block->loop_count];
	int right_loop_idx[right_block->loop_count];
	int i, j;
	enum {
		ONE = -2,
		UNASSIGNED = -1,
	};
	for (i = 0; i < left_block->loop_count; i++)
		if (left_block->loops[i].start_index.type == CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL && left_block->loops[i].start_index.immediate_value == 0 &&
			left_block->loops[i].end_index.type == CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL && left_block->loops[i].end_index.immediate_value == 1)
			left_loop_idx[i] = ONE;
		else
			left_loop_idx[i] = UNASSIGNED;
	for (i = 0; i < right_block->loop_count; i++)
		if (right_block->loops[i].start_index.type == CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL && right_block->loops[i].start_index.immediate_value == 0 &&
			right_block->loops[i].end_index.type == CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL && right_block->loops[i].end_index.immediate_value == 1)
			right_loop_idx[i] = ONE;
		else
			right_loop_idx[i] = UNASSIGNED;
	for (i = 0; i < left_block->loop_count; i++) // Find corresponding loop on the right.
	{
		if (left_loop_idx[i] != UNASSIGNED)
			continue;
		int flag = UNASSIGNED;
		for (j = 0; j < right_block->loop_count && flag == UNASSIGNED; j++)
		{
			if (right_loop_idx[j] != UNASSIGNED)
				continue;
			if (_ccv_nnc_same_index_term(left_block->loops[i].start_index, right_block->loops[j].start_index, groups) && 
				_ccv_nnc_same_index_term(left_block->loops[i].end_index, right_block->loops[j].end_index, groups))
				flag = j;
		}
		if (flag != UNASSIGNED)
		{
			left_loop_idx[i] = flag;
			right_loop_idx[flag] = i;
		}
	}
	// If still have unmatched, they don't share the same loop.
	for (i = 0; i < left_block->loop_count; i++)
		if (left_loop_idx[i] == UNASSIGNED)
			return 0;
	for (i = 0; i < right_block->loop_count; i++)
		if (right_loop_idx[i] == UNASSIGNED)
			return 0;
	return 1;
}

void ccv_nnc_micro_combine_simplify(ccv_nnc_micro_combine_t* const combine)
{
	// Nothing to simplify for.
	if (combine->function_count < 1)
		return;
	// Only one block, nothing to simplify for.
	if (combine->function_count == 1 && combine->functions[0].block_count == 1)
		return;
	// Union-find to group all variables with the same shape.
	ccv_nnc_micro_tensor_t* const vars = combine->vars;
	const int var_count = combine->var_count;
	int* const groups = (int*)ccmalloc(sizeof(int) * var_count);
	int i, j;
	for (i = 0; i < var_count; i++)
		groups[i] = i;
	// If no shape, they should match these input.
	for (i = var_count - 1; i >= 0; i--)
		if (vars[i].input >= 0 && !vars[i].shape)
		{
			int root = vars[i].input;
			while (groups[root] != root)
				root = groups[root];
			groups[i] = root;
		}
	for (i = var_count - 1; i > 0; i--)
	{
		// If this is input (no other tensor as the input), we skip.
		if (vars[i].input < 0)
			continue;
		int root = i;
		while (groups[root] != root)
			root = groups[root];
		// If the sibling exists and we haven't visited yet, mark them has the same group as us.
		if (vars[i].sibling >= 0 && vars[i].sibling < i && groups[vars[i].sibling] < 0)
			groups[vars[i].sibling] = root;
	}
	for (i = var_count - 1; i > 0; i--)
	{
		// Now matching the shape.
		if (vars[i].input < 0 || !vars[i].shape)
			continue;
		int root = i;
		while (groups[root] != root)
			root = groups[root];
		for (j = i - 1; j >= 0; j--)
			if (vars[j].shape && vars[j].dimensions == vars[i].dimensions &&
				_ccv_nnc_same_shape(vars[j].shape, vars[i].shape, vars[i].dimensions))
				groups[j] = root;
	}
	// First, flat out all functions into blocks.
	ccv_array_t* const blocks = ccv_array_new(sizeof(ccv_nnc_micro_loop_block_t), 0, 0);
	ccv_nnc_micro_function_t* const functions = combine->functions;
	const int function_count = combine->function_count;
	for (i = 0; i < function_count; i++)
	{
		const int block_count = functions[i].block_count;
		ccv_nnc_micro_loop_block_t* const function_blocks = block_count == 1 ? &functions[i].one_block : functions[i].blocks;
		for (j = 0; j < block_count; j++)
			ccv_array_push(blocks, &function_blocks[j]);
	}
	// Merge loops from blocks.
	for (i = 0; i < blocks->rnum - 1; i++)
	{
		ccv_nnc_micro_loop_block_t* const left_block = (ccv_nnc_micro_loop_block_t*)ccv_array_get(blocks, i);
		if (left_block->loop_count == 0)
			continue;
		for (j = i + 1; j < blocks->rnum; j++)
		{
			ccv_nnc_micro_loop_block_t* const right_block = (ccv_nnc_micro_loop_block_t*)ccv_array_get(blocks, j);
			if (right_block->loop_count == 0)
				continue;
			if (_ccv_nnc_same_loop(left_block, right_block, groups))
			{
			}
		}
	}
	// Reallocate function to be 1.
	for (i = 0; i < function_count; i++)
		if (functions[i].block_count > 1)
			ccfree(functions[i].blocks);
	combine->functions = (ccv_nnc_micro_function_t*)ccrealloc(combine->functions, sizeof(ccv_nnc_micro_function_t));
	combine->functions[0].block_count = blocks->rnum;
	if (blocks->rnum > 1)
	{
		combine->functions[0].blocks = (ccv_nnc_micro_loop_block_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_block_t) * blocks->rnum);
		memcpy(combine->functions[0].blocks, ccv_array_get(blocks, 0), sizeof(ccv_nnc_micro_loop_block_t) * blocks->rnum);
	} else
		combine->functions[0].one_block = *(ccv_nnc_micro_loop_block_t*)ccv_array_get(blocks, 0);
	combine->function_count = 1;
	free(groups);
	ccv_array_free(blocks);
}
