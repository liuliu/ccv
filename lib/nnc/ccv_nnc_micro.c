#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_micro.h"
#include "3rdparty/khash/khash.h"

// MARK - Level-1 API

KHASH_MAP_INIT_STR(ccv_nnc_micro_bind_scalar, uint32_t)

static uint32_t _scalars_lookup(const void* const context, const char* const name)
{
	const khash_t(ccv_nnc_micro_bind_scalar)* const bind_scalars = (const khash_t(ccv_nnc_micro_bind_scalar)*)context;
	khiter_t k = kh_get(ccv_nnc_micro_bind_scalar, bind_scalars, name);
	assert(k != kh_end(bind_scalars));
	return kh_val(bind_scalars, k);
}

KHASH_SET_INIT_INT64(ccv_nnc_ids)

CCV_WARN_UNUSED(ccv_nnc_micro_combine_t*) ccv_nnc_micro_combine_new(const ccv_nnc_micro_io_t* const inputs, const int input_size, const char* const* const parameters, const int parameter_size, const ccv_nnc_micro_io_t* const outputs, const int output_size, const ccv_nnc_micro_io_t* const grad_inputs, const int grad_input_size, const ccv_nnc_micro_io_t* const grad_outputs, const ccv_nnc_micro_io_t* const grad_output_size)
{
	assert(output_size > 0);
	assert(input_size > 0);
	int i, j, k;
	// First, do reverse topological sort (from output and then reverse the order).
	// We can do this simple thing because there is no overlaps of the outputs, thus, no cases where
	// output[0] is the input for output[1]. Otherwise we need to detect this, see ccv_cnnp_model_new
	// for more details on why.
	for (i = 0; i < output_size - 1; i++)
		for (j = i + 1; j < output_size; j++)
		{ assert(outputs[i] != outputs[j]); }
	uint64_t input_bitmask[((input_size - 1) >> 6) + 1];
	memset(input_bitmask, 0, sizeof(uint64_t) * (((input_size - 1) >> 6) + 1));
	ccv_array_t* const reverse_top = ccv_array_new(sizeof(ccv_nnc_micro_io_t), output_size + input_size, 0);
	ccv_array_resize(reverse_top, output_size);
	memcpy(ccv_array_get(reverse_top, 0), outputs, sizeof(ccv_nnc_micro_io_t) * output_size);
	khash_t(ccv_nnc_ids)* const ids = kh_init(ccv_nnc_ids);
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, i);
		for (j = 0; j < output->input_size; j++)
			if (!CCV_NNC_IS_MICRO_IO_INPUT(output->inputs[j]))
			{
				int ret;
				kh_put(ccv_nnc_ids, ids, (int64_t)(intptr_t)output->inputs[j], &ret);
				if (ret != 0)
					ccv_array_push(reverse_top, &output->inputs[j]);
			} else {
				// This is an input, it must be represented in inputs, try to find it.
				for (k = 0; k < input_size; k++)
					if (inputs[k] == output->inputs[j])
						break;
				assert(k < input_size); // Cannot find the inputs, error!
				input_bitmask[k >> 6] |= ((uint64_t)1 << (k & 63));
			}
	}
	kh_destroy(ccv_nnc_ids, ids);
	for (i = 0; i < input_size; i++)
		{ assert((input_bitmask[i >> 6] & ((uint64_t)1 << (i & 63)))); } // Assuming they all match.
	// Second, binding parameters (bounded scalars).
	khash_t(ccv_nnc_micro_bind_scalar)* const bind_scalars = kh_init(ccv_nnc_micro_bind_scalar);
	for (i = 0; i < parameter_size; i++)
	{
		int ret;
		khiter_t k = kh_put(ccv_nnc_micro_bind_scalar, bind_scalars, parameters[i], &ret);
		assert(ret != 0);
		kh_val(bind_scalars, k) = i;
	}
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, reverse_top->rnum - 1 - i);
		ccv_nnc_micro_bind_scalars(output, _scalars_lookup, bind_scalars);
	}
	kh_destroy(ccv_nnc_micro_bind_scalar, bind_scalars);
	const int var_count = reverse_top->rnum + input_size;
	// Applying numbering for the inputs. Note that our variables are numbered in reverse topological order.
	for (i = 0; i < input_size; i++)
		ccv_nnc_micro_numbering(inputs[i], i + reverse_top->rnum, var_count);
	// Applying numbering for the outputs.
	for (i = reverse_top->rnum - 1; i >= 0; i--)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, i);
		ccv_nnc_micro_numbering(output, i, var_count);
	}
	// Third, lower each ccv_nnc_micro_io_t (except the input) op into nested loops such that we can
	// apply optimizations later.
	const int function_count = reverse_top->rnum;
	ccv_nnc_micro_function_t* const functions = (ccv_nnc_micro_function_t*)ccmalloc(sizeof(ccv_nnc_micro_function_t) * function_count);
	for (i = 0; i < function_count; i++)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, function_count - 1 - i);
		functions[i] = ccv_nnc_micro_emit(output);
	}
	ccv_nnc_micro_tensor_t* const vars = (ccv_nnc_micro_tensor_t*)cccalloc(reverse_top->rnum + input_size, sizeof(ccv_nnc_micro_tensor_t));
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, i);
		vars[i] = ccv_nnc_micro_return_shape(output);
	}
	for (i = 0; i < input_size; i++)
	{
		vars[i + reverse_top->rnum].dimensions = inputs[i]->dimensions;
		vars[i + reverse_top->rnum].input = -1;
	}
	ccv_nnc_micro_combine_t* const combine = (ccv_nnc_micro_combine_t*)ccmalloc(sizeof(ccv_nnc_micro_combine_t));
	combine->input_size = input_size;
	combine->output_size = output_size;
	combine->parameter_size = parameter_size;
	combine->forward.var_count = var_count;
	combine->forward.vars = vars;
	combine->forward.function_count = function_count;
	combine->forward.functions = functions;
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, i);
		ccfree(output);
	}
	ccv_array_free(reverse_top);
	for (i = 0; i < input_size; i++)
		ccfree(inputs[i]);
	ccv_nnc_micro_program_simplify(&combine->forward, input_size, output_size);
	return combine;
}

void ccv_nnc_micro_loop_index_free(ccv_nnc_micro_loop_index_term_t* const term)
{
	if (term->type == CCV_NNC_MICRO_LOOP_INDEX_TYPE_BINARY)
	{
		ccv_nnc_micro_loop_index_free(&term->binary->left);
		ccv_nnc_micro_loop_index_free(&term->binary->right);
		ccfree(term->binary);
	}
}

static void _ccv_nnc_micro_loop_variable_free(ccv_nnc_micro_loop_variable_t* const var)
{
	int i;
	for (i = 0; i < var->index_count; i++)
		ccv_nnc_micro_loop_index_free(&var->index[i]);
}

static void _ccv_nnc_micro_loop_expression_free(ccv_nnc_micro_loop_expression_t* const expr)
{
	switch (expr->type)
	{
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR: {
			_ccv_nnc_micro_loop_variable_free(&expr->variable);
			break;
		}
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_UNARY: {
			_ccv_nnc_micro_loop_expression_free(expr->unary.x);
			ccfree(expr->unary.x);
		}
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_BINARY: {
			_ccv_nnc_micro_loop_expression_free(expr->binary.left);
			ccfree(expr->binary.left);
			_ccv_nnc_micro_loop_expression_free(expr->binary.right);
			ccfree(expr->binary.right);
		}
	}
}

void ccv_nnc_micro_loops_free(ccv_nnc_micro_loop_t* const loops, const int loop_count)
{
	int i, j;
	for (i = 0; i < loop_count; i++)
	{
		for (j = 0; j < loops[i].statement_count; j++)
		{
			ccv_nnc_micro_loop_statement_t statement = loops[i].statements[j];
			switch (statement.type)
			{
				case CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_COMPOUND_ASSIGNMENT: {
					_ccv_nnc_micro_loop_expression_free(&statement.compound_assignment.rvalue);
					break;
				}
				case CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_ASSIGNMENT: {
					_ccv_nnc_micro_loop_variable_free(&statement.assignment.lvalue);
					_ccv_nnc_micro_loop_expression_free(&statement.assignment.rvalue);
					break;
				}
			}
		}
		if (loops[i].statements)
			ccfree(loops[i].statements);
		if (loops[i].carrieds)
			ccfree(loops[i].carrieds);
	}
}

void ccv_nnc_micro_combine_free(ccv_nnc_micro_combine_t* const combine)
{
	int i, j;
	const int var_count = combine->forward.var_count;
	for (i = 0; i < var_count; i++)
	{
		if (combine->forward.vars[i].shape)
		{
			for (j = 0; j < combine->forward.vars[i].dimensions; j++)
				ccv_nnc_micro_loop_index_free(&combine->forward.vars[i].shape[j]);
			ccfree(combine->forward.vars[i].shape);
		}
	}
	ccfree(combine->forward.vars);
	const int function_count = combine->forward.function_count;
	for (i = 0; i < function_count; i++)
	{
		const int block_count = combine->forward.functions[i].block_count;
		ccv_nnc_micro_loop_block_t* const blocks = (block_count == 1) ? &combine->forward.functions[i].one_block : combine->forward.functions[i].blocks;
		for (j = 0; j < block_count; j++)
		{
			ccv_nnc_micro_loop_block_t block = blocks[j];
			ccv_nnc_micro_loops_free(block.loops, block.loop_count);
			ccfree(block.loops);
		}
		if (block_count > 1)
			ccfree(combine->forward.functions[i].blocks);
	}
	ccfree(combine->forward.functions);
	ccfree(combine);
}

char* ccv_nnc_micro_combine_c(ccv_nnc_micro_combine_t* const combine)
{
	return 0;
}
