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

CCV_WARN_UNUSED(ccv_nnc_micro_combine_t*) ccv_nnc_micro_combine_new(const ccv_nnc_micro_io_t* const inputs, const int input_size, const char* const* const parameters, const int parameter_size, const ccv_nnc_micro_io_t* const outputs, const int output_size, const ccv_nnc_micro_io_t* const ingrads, const int ingrad_size, const ccv_nnc_micro_io_t* const outgrads, const int outgrad_size)
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
		ccv_nnc_micro_numbering(inputs[i], i, var_count);
	ccv_array_t* const equal_assertions = ccv_array_new(sizeof(ccv_nnc_micro_id_equal_assertion_t), 0, 0);
	// Applying numbering for the outputs and collect equal assertions.
	for (i = reverse_top->rnum - 1; i >= 0; i--)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, reverse_top->rnum - 1 - i);
		ccv_nnc_micro_numbering(output, i + input_size, var_count);
		ccv_nnc_micro_equal_assertions(output, equal_assertions);
	}
	for (i = 0; i < ingrad_size; i++)
		ccv_nnc_micro_numbering(ingrads[i], -1, var_count);
	for (i = 0; i < outgrad_size; i++)
		ccv_nnc_micro_numbering(outgrads[i], -1, var_count);
	// Fill in shapes for variables.
	ccv_nnc_micro_tensor_t* const vars = (ccv_nnc_micro_tensor_t*)cccalloc(var_count * 2, sizeof(ccv_nnc_micro_tensor_t));
	for (i = 0; i < input_size; i++)
	{
		vars[i].dimensions = inputs[i]->dimensions;
		vars[i].input = -1;
	}
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, reverse_top->rnum - 1 - i);
		vars[i + input_size] = ccv_nnc_micro_return_shape(output);
	}
	for (i = var_count; i < 2 * var_count; i++)
	{
		vars[i].dimensions = vars[2 * var_count - 1 - i].dimensions;
		vars[i].input = 2 * var_count - 1 - i;
	}
	// Lower each ccv_nnc_micro_io_t (except the input) op into nested loops such that we can
	// apply optimizations later.
	int function_count = reverse_top->rnum;
	ccv_nnc_micro_function_t* functions = (ccv_nnc_micro_function_t*)ccmalloc(sizeof(ccv_nnc_micro_function_t) * function_count);
	for (i = 0; i < function_count; i++)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, function_count - 1 - i);
		functions[i] = ccv_nnc_micro_emit(output);
	}
	ccv_nnc_micro_combine_t* const combine = (ccv_nnc_micro_combine_t*)ccmalloc(sizeof(ccv_nnc_micro_combine_t));
	combine->parameter_size = parameter_size;
	combine->forward.input_size = input_size;
	combine->forward.inputs = (int*)ccmalloc(sizeof(int) * (input_size + output_size));
	for (i = 0; i < input_size; i++)
		combine->forward.inputs[i] = inputs[i]->id;
	combine->forward.output_size = output_size;
	combine->forward.outputs = combine->forward.inputs + input_size;
	for (i = 0; i < output_size; i++)
		combine->forward.outputs[i] = outputs[i]->id;
	combine->forward.var_count = var_count;
	// We copied forward.vars so backward.vars and forward.vars can maintain separate states.
	// However, shape and related allocations are shared because these are not going to be mutated.
	combine->forward.vars = (ccv_nnc_micro_tensor_t*)ccmalloc(sizeof(ccv_nnc_micro_tensor_t) * var_count);
	memcpy(combine->forward.vars, vars, sizeof(ccv_nnc_micro_tensor_t) * var_count);
	combine->forward.function_count = function_count;
	combine->forward.functions = functions;
	ccv_nnc_micro_program_simplify(&combine->forward, inputs, input_size, outputs, output_size, equal_assertions);
	function_count = reverse_top->rnum * 2;
	functions = (ccv_nnc_micro_function_t*)ccmalloc(sizeof(ccv_nnc_micro_function_t) * function_count);
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, reverse_top->rnum - 1 - i);
		functions[i] = ccv_nnc_micro_emit(output);
	}
	for (i = reverse_top->rnum; i < function_count; i++)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, i - reverse_top->rnum);
		functions[i] = ccv_nnc_micro_emit_grad(output, var_count);
	}
	combine->backward.input_size = ingrad_size;
	combine->backward.inputs = ingrad_size + outgrad_size > 0 ? (int*)ccmalloc(sizeof(int) * (ingrad_size + outgrad_size)) : 0;
	for (i = 0; i < ingrad_size; i++)
		combine->backward.inputs[i] = ingrads[i]->id;
	combine->backward.output_size = outgrad_size;
	combine->backward.outputs = outgrad_size > 0 ? combine->backward.inputs + ingrad_size : 0;
	for (i = 0; i < outgrad_size; i++)
		combine->backward.outputs[i] = outgrads[i]->id;
	combine->backward.var_count = var_count * 2;
	combine->backward.vars = vars;
	combine->backward.function_count = function_count;
	combine->backward.functions = functions;
	ccv_nnc_micro_program_simplify(&combine->backward, ingrads, ingrad_size, outgrads, outgrad_size, equal_assertions);
	ccv_array_free(equal_assertions);
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, i);
		ccv_nnc_micro_deinit(output);
		ccfree(output);
	}
	ccv_array_free(reverse_top);
	// It may overlap with inputs, in that case, skip.
	for (i = 0; i < ingrad_size; i++)
	{
		int flag = 0;
		for (j = 0; !flag && j < input_size; j++)
			flag = (inputs[j] == ingrads[i]);
		if (!flag)
		{
			ccv_nnc_micro_deinit(ingrads[i]);
			ccfree(ingrads[i]);
		}
	}
	for (i = 0; i < input_size; i++)
	{
		ccv_nnc_micro_deinit(inputs[i]);
		ccfree(inputs[i]);
	}
	for (i = 0; i < outgrad_size; i++) // Should be no overlap on outgrads.
	{
		ccv_nnc_micro_deinit(outgrads[i]);
		ccfree(outgrads[i]);
	}
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

void ccv_nnc_micro_loop_variable_free(ccv_nnc_micro_loop_variable_t* const var)
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
			ccv_nnc_micro_loop_variable_free(&expr->variable);
			break;
		}
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_UNARY: {
			_ccv_nnc_micro_loop_expression_free(expr->unary.x);
			ccfree(expr->unary.x);
			break;
		}
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_BINARY: {
			_ccv_nnc_micro_loop_expression_free(expr->binary.left);
			ccfree(expr->binary.left);
			_ccv_nnc_micro_loop_expression_free(expr->binary.right);
			ccfree(expr->binary.right);
			break;
		}
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_TERNAY: {
			_ccv_nnc_micro_loop_expression_free(expr->ternary.pivot);
			ccfree(expr->ternary.pivot);
			_ccv_nnc_micro_loop_expression_free(expr->ternary.left);
			ccfree(expr->ternary.left);
			_ccv_nnc_micro_loop_expression_free(expr->ternary.right);
			ccfree(expr->ternary.right);
			break;
		}
	}
}

void ccv_nnc_micro_loop_statement_lvalue_free(ccv_nnc_micro_loop_statement_t* const statement)
{
	switch (statement->type)
	{
		case CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_COMPOUND_ASSIGNMENT: {
			if (statement->compound_assignment.lvalue.type == CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR)
				ccv_nnc_micro_loop_variable_free(&statement->compound_assignment.lvalue.variable);
			break;
		}
		case CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_ASSIGNMENT: {
			ccv_nnc_micro_loop_variable_free(&statement->assignment.lvalue);
			break;
		}
	}
}

void ccv_nnc_micro_loop_statement_free(ccv_nnc_micro_loop_statement_t* const statement)
{
	switch (statement->type)
	{
		case CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_COMPOUND_ASSIGNMENT: {
			if (statement->compound_assignment.lvalue.type == CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR)
				ccv_nnc_micro_loop_variable_free(&statement->compound_assignment.lvalue.variable);
			_ccv_nnc_micro_loop_expression_free(&statement->compound_assignment.rvalue);
			break;
		}
		case CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_ASSIGNMENT: {
			ccv_nnc_micro_loop_variable_free(&statement->assignment.lvalue);
			_ccv_nnc_micro_loop_expression_free(&statement->assignment.rvalue);
			break;
		}
	}
}

void ccv_nnc_micro_loops_free(ccv_nnc_micro_loop_t* const loops, const int loop_count)
{
	int i, j;
	for (i = 0; i < loop_count; i++)
	{
		for (j = 0; j < loops[i].statement_count; j++)
			ccv_nnc_micro_loop_statement_free(&loops[i].statements[j]);
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
		if (combine->forward.vars[i].shape)
		{
			for (j = 0; j < combine->forward.vars[i].dimensions; j++)
				ccv_nnc_micro_loop_index_free(&combine->forward.vars[i].shape[j]);
			ccfree(combine->forward.vars[i].shape);
		}
	ccfree(combine->forward.vars);
	ccfree(combine->backward.vars);
	int function_count = combine->forward.function_count;
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
	ccfree(combine->forward.inputs);
	// Backward and forward share the same vars.
	function_count = combine->backward.function_count;
	for (i = 0; i < function_count; i++)
	{
		const int block_count = combine->backward.functions[i].block_count;
		ccv_nnc_micro_loop_block_t* const blocks = (block_count == 1) ? &combine->backward.functions[i].one_block : combine->backward.functions[i].blocks;
		for (j = 0; j < block_count; j++)
		{
			ccv_nnc_micro_loop_block_t block = blocks[j];
			ccv_nnc_micro_loops_free(block.loops, block.loop_count);
			ccfree(block.loops);
		}
		if (block_count > 1)
			ccfree(combine->backward.functions[i].blocks);
	}
	ccfree(combine->backward.functions);
	if (combine->backward.inputs)
		ccfree(combine->backward.inputs);
	ccfree(combine);
}

char* ccv_nnc_micro_combine_c(ccv_nnc_micro_combine_t* const combine)
{
	return 0;
}
