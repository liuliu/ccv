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

CCV_WARN_UNUSED(ccv_nnc_micro_combine_t*) ccv_nnc_micro_combine_new(const ccv_nnc_micro_io_t* const inputs, const int input_size, const char* const* const parameters, const int parameter_size, const ccv_nnc_micro_io_t* const outputs, const int output_size)
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
	// Applying numbering for the inputs. Note that our variables are numbered in reverse topological order.
	for (i = 0; i < input_size; i++)
		ccv_nnc_micro_numbering(inputs[i], i + reverse_top->rnum);
	// Applying numbering for the outputs.
	for (i = reverse_top->rnum - 1; i >= 0; i--)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, i);
		ccv_nnc_micro_numbering(output, i);
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
	combine->var_count = reverse_top->rnum + input_size;
	combine->vars = vars;
	combine->function_count = function_count;
	combine->functions = functions;
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, i);
		ccfree(output);
	}
	ccv_array_free(reverse_top);
	for (i = 0; i < input_size; i++)
		ccfree(inputs[i]);
	ccv_nnc_micro_combine_simplify(combine, output_size);
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

void ccv_nnc_micro_combine_free(ccv_nnc_micro_combine_t* const combine)
{
	int i, j, k, l;
	const int var_count = combine->var_count;
	for (i = 0; i < var_count; i++)
	{
		if (combine->vars[i].shape)
		{
			for (j = 0; j < combine->vars[i].dimensions; j++)
				ccv_nnc_micro_loop_index_free(&combine->vars[i].shape[j]);
			ccfree(combine->vars[i].shape);
		}
	}
	ccfree(combine->vars);
	const int function_count = combine->function_count;
	for (i = 0; i < function_count; i++)
	{
		const int block_count = combine->functions[i].block_count;
		ccv_nnc_micro_loop_block_t* const blocks = (block_count == 1) ? &combine->functions[i].one_block : combine->functions[i].blocks;
		for (j = 0; j < block_count; j++)
		{
			ccv_nnc_micro_loop_block_t block = blocks[j];
			for (k = 0; k < block.loop_count; k++)
			{
				for (l = 0; l < block.loops[k].statement_count; l++)
				{
					ccv_nnc_micro_loop_statement_t statement = block.loops[k].statements[l];
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
				if (block.loops[k].statements)
					ccfree(block.loops[k].statements);
				if (block.loops[k].carrieds)
					ccfree(block.loops[k].carrieds);
			}
			ccfree(block.loops);
		}
		if (block_count > 1)
			ccfree(combine->functions[i].blocks);
	}
	ccfree(combine->functions);
	ccfree(combine);
}

static int _ccv_nnc_micro_index_interpret(const ccv_nnc_micro_loop_index_term_t index, const int* const loop_counter, const int* const shapes, const ccv_nnc_micro_scalar_t* const values, const int parameter_size)
{
	switch (index.type)
	{
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL:
			return index.immediate_value;
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID:
			switch (index.id.type)
			{
				case CCV_NNC_MICRO_AXIS_SIZE_ID:
					return shapes[index.id.id * CCV_NNC_MAX_DIM_ALLOC + index.id.d];
				case CCV_NNC_MICRO_LOOP_ID:
					return loop_counter[index.id.id];
				case CCV_NNC_MICRO_SCALAR_ID:
					switch (values[index.id.id].type)
					{
						case CCV_8U:
							return values[index.id.id].u8;
						case CCV_32S:
							return values[index.id.id].i32;
						case CCV_64S:
							return (int)values[index.id.id].i64;
					}
					break;
			}
			break;
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_BINARY: {
			const int left = _ccv_nnc_micro_index_interpret(index.binary->left, loop_counter, shapes, values, parameter_size);
			const int right = _ccv_nnc_micro_index_interpret(index.binary->right, loop_counter, shapes, values, parameter_size);
			switch (index.binary->op)
			{
				case CCV_NNC_MICRO_BINARY_OP_PLUS:
					return left + right;
				case CCV_NNC_MICRO_BINARY_OP_MINUS:
					return left - right;
				case CCV_NNC_MICRO_BINARY_OP_MUL:
					return left * right;
				case CCV_NNC_MICRO_BINARY_OP_DIV:
					return left / right;
				case CCV_NNC_MICRO_BINARY_OP_MAX:
					return ccv_max(left, right);
				case CCV_NNC_MICRO_BINARY_OP_MIN:
					return ccv_min(left, right);
			}
			break;
		}
	}
	return 0;
}

static float _ccv_nnc_micro_expression_interpret(const ccv_nnc_micro_loop_expression_t* const expression, const int* const loop_counter, const ccv_nnc_micro_scalar_t* const carrieds, const int carried_count, float* const* const vars_mem, const int* const shapes, const ccv_nnc_micro_scalar_t* const values, const int parameter_size)
{
	int i;
	switch (expression->type)
	{
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_ID: {
			assert(expression->id.type == CCV_NNC_MICRO_LOOP_CARRIED_ID);
			return carrieds[expression->id.id].f32;
		}
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR: {
			const ccv_nnc_micro_loop_variable_t variable = expression->variable;
			assert(variable.id.type == CCV_NNC_MICRO_TENSOR_ID);
			float* ptr = vars_mem[variable.id.id];
			size_t size = 1;
			for (i = variable.index_count - 1; i >= 0; i--)
			{
				const int index = _ccv_nnc_micro_index_interpret(variable.index[i], loop_counter, shapes, values, parameter_size);
				ptr += index * size;
				size *= shapes[variable.id.id * CCV_NNC_MAX_DIM_ALLOC + i];
			}
			return ptr[0];
		}
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_UNARY: {
			const float left = _ccv_nnc_micro_expression_interpret(expression->unary.x, loop_counter, carrieds, carried_count, vars_mem, shapes, values, parameter_size);
			switch (expression->unary.unary_op)
			{
				case CCV_NNC_MICRO_UNARY_OP_EXP:
					return exp(left);
				case CCV_NNC_MICRO_UNARY_OP_LOG:
					return log(left);
			}
			break;
		}
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_BINARY: {
			const float left = _ccv_nnc_micro_expression_interpret(expression->binary.left, loop_counter, carrieds, carried_count, vars_mem, shapes, values, parameter_size);
			const float right = _ccv_nnc_micro_expression_interpret(expression->binary.right, loop_counter, carrieds, carried_count, vars_mem, shapes, values, parameter_size);
			switch (expression->binary.binary_op)
			{
				case CCV_NNC_MICRO_BINARY_OP_PLUS:
					return left + right;
				case CCV_NNC_MICRO_BINARY_OP_MINUS:
					return left - right;
				case CCV_NNC_MICRO_BINARY_OP_MUL:
					return left * right;
				case CCV_NNC_MICRO_BINARY_OP_DIV:
					return left / right;
				case CCV_NNC_MICRO_BINARY_OP_MAX:
					return ccv_max(left, right);
				case CCV_NNC_MICRO_BINARY_OP_MIN:
					return ccv_min(left, right);
			}
			break;
		}
	}
	return 0;
}

static void _ccv_nnc_micro_statement_interpret(const ccv_nnc_micro_loop_statement_t statement, const int* const loop_counter, ccv_nnc_micro_scalar_t* const carrieds, const int carried_count, float* const* const vars_mem, const int* const shapes, const ccv_nnc_micro_scalar_t* const values, const int parameter_size)
{
	int i;
	switch (statement.type)
	{
		case CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_ASSIGNMENT: {
			assert(statement.assignment.lvalue.id.type == CCV_NNC_MICRO_TENSOR_ID);
			const ccv_nnc_micro_loop_variable_t variable = statement.assignment.lvalue;
			float* ptr = vars_mem[variable.id.id];
			size_t size = 1;
			for (i = variable.index_count - 1; i >= 0; i--)
			{
				const int index = _ccv_nnc_micro_index_interpret(variable.index[i], loop_counter, shapes, values, parameter_size);
				ptr += index * size;
				size *= shapes[variable.id.id * CCV_NNC_MAX_DIM_ALLOC + i];
			}
			ptr[0] = _ccv_nnc_micro_expression_interpret(&statement.assignment.rvalue, loop_counter, carrieds, carried_count, vars_mem, shapes, values, parameter_size);
			break;
		}
		case CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_COMPOUND_ASSIGNMENT: {
			const float rvalue = _ccv_nnc_micro_expression_interpret(&statement.compound_assignment.rvalue, loop_counter, carrieds, carried_count, vars_mem, shapes, values, parameter_size);
			switch (statement.compound_assignment.lvalue.type)
			{
				case CCV_NNC_MICRO_LOOP_EXPR_TYPE_ID: {
					assert(statement.compound_assignment.lvalue.id.type == CCV_NNC_MICRO_LOOP_CARRIED_ID);
					switch (statement.compound_assignment.lvalue.id.d)
					{
						case CCV_NNC_MICRO_REDUCE_OP_MAX:
							carrieds[statement.compound_assignment.lvalue.id.id].f32 = ccv_max(carrieds[statement.compound_assignment.lvalue.id.id].f32, rvalue);
							break;
						case CCV_NNC_MICRO_REDUCE_OP_MIN:
							carrieds[statement.compound_assignment.lvalue.id.id].f32 = ccv_min(carrieds[statement.compound_assignment.lvalue.id.id].f32, rvalue);
							break;
						case CCV_NNC_MICRO_REDUCE_OP_ARGMAX:
							assert(0);
							break;
						case CCV_NNC_MICRO_REDUCE_OP_ARGMIN:
							assert(0);
							break;
						case CCV_NNC_MICRO_REDUCE_OP_MEAN:
							carrieds[statement.compound_assignment.lvalue.id.id].f32 += rvalue;
							break;
						case CCV_NNC_MICRO_REDUCE_OP_SUM:
							carrieds[statement.compound_assignment.lvalue.id.id].f32 += rvalue;
							break;
						case CCV_NNC_MICRO_REDUCE_OP_PROD:
							carrieds[statement.compound_assignment.lvalue.id.id].f32 *= rvalue;
							break;
					}
					break;
				}
				case CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR: {
					assert(statement.compound_assignment.lvalue.id.type == CCV_NNC_MICRO_TENSOR_ID);
					const ccv_nnc_micro_loop_variable_t variable = statement.compound_assignment.lvalue.variable;
					float* ptr = vars_mem[variable.id.id];
					size_t size = 1;
					for (i = variable.index_count - 1; i >= 0; i--)
					{
						const int index = _ccv_nnc_micro_index_interpret(variable.index[i], loop_counter, shapes, values, parameter_size);
						ptr += index * size;
						size *= shapes[variable.id.id * CCV_NNC_MAX_DIM_ALLOC + i];
					}
					ptr[0] += rvalue;
					break;
				}
			}
			break;
		}
	}
}

static void _ccv_nnc_micro_loop_interpret(const ccv_nnc_micro_loop_t* const loops, const int loop_count, const int index, int* const loop_counter, ccv_nnc_micro_scalar_t* const carrieds, const int carried_count, float* const* const vars_mem, const int* const shapes, const ccv_nnc_micro_scalar_t* const values, const int parameter_size)
{
	if (index >= loop_count)
		return;
	const int start_index = _ccv_nnc_micro_index_interpret(loops[index].start_index, loop_counter, shapes, values, parameter_size);
	const int end_index = _ccv_nnc_micro_index_interpret(loops[index].end_index, loop_counter, shapes, values, parameter_size);
	int i, j;
	const ccv_nnc_micro_loop_statement_t* const statements = loops[index].statements;
	const int statement_count = loops[index].statement_count;
	const ccv_nnc_micro_loop_carried_t* const carried_refs = loops[index].carrieds;
	const int carried_ref_count = loops[index].carried_count;
	for (i = start_index; i < end_index; i++)
	{
		loop_counter[loops[index].id.id] = i;
		for (j = 0; j < carried_ref_count; j++)
		{
			assert(carried_refs[j].id.type == CCV_NNC_MICRO_LOOP_CARRIED_ID);
			assert(carried_refs[j].id.id < carried_count);
			switch (carried_refs[j].id.d)
			{
				case CCV_NNC_MICRO_REDUCE_OP_MAX:
					carrieds[carried_refs[j].id.id].f32 = -FLT_MAX;
					break;
				case CCV_NNC_MICRO_REDUCE_OP_MIN:
					carrieds[carried_refs[j].id.id].f32 = FLT_MAX;
					break;
				case CCV_NNC_MICRO_REDUCE_OP_ARGMAX:
					carrieds[carried_refs[j].id.id].i32 = -1;
					break;
				case CCV_NNC_MICRO_REDUCE_OP_ARGMIN:
					carrieds[carried_refs[j].id.id].i32 = -1;
					break;
				case CCV_NNC_MICRO_REDUCE_OP_MEAN:
					carrieds[carried_refs[j].id.id].f32 = 0;
					break;
				case CCV_NNC_MICRO_REDUCE_OP_SUM:
					carrieds[carried_refs[j].id.id].f32 = 0;
					break;
				case CCV_NNC_MICRO_REDUCE_OP_PROD:
					carrieds[carried_refs[j].id.id].f32 = 1;
					break;
			}
		}
		_ccv_nnc_micro_loop_interpret(loops, loop_count, index + 1, loop_counter, carrieds, carried_count, vars_mem, shapes, values, parameter_size);
		for (j = 0; j < statement_count; j++)
			_ccv_nnc_micro_statement_interpret(statements[j], loop_counter, carrieds, carried_count, vars_mem, shapes, values, parameter_size);
	}
}

void ccv_nnc_micro_combine_interpret(ccv_nnc_micro_combine_t* const combine, const uint32_t cmd, ccv_nnc_tensor_t* const* const inputs, const int input_size, const ccv_nnc_micro_scalar_t* const values, const int parameter_size, ccv_nnc_tensor_t* const* const outputs, const int output_size)
{
	// We haven't optimized for emit_grad at the moment yet.
	assert(cmd == CCV_NNC_CUSTOM_FORWARD);
	int i, j;
	const int var_count = combine->var_count;
	assert(input_size == combine->input_size);
	assert(output_size == combine->output_size);
	assert(parameter_size == combine->parameter_size);
	int* const shapes = (int*)cccalloc(var_count, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	ccv_nnc_micro_tensor_t* const vars = combine->vars;
	for (i = 0; i < input_size; i++)
		memcpy(shapes + (var_count - input_size + i) * CCV_NNC_MAX_DIM_ALLOC, &inputs[i]->info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	int loop_counter[CCV_NNC_MAX_DIM_ALLOC];
	for (i = var_count - input_size - 1; i >= 0; i--)
	{
		if (vars[i].shape)
		{
			for (j = 0; j < vars[i].dimensions; j++)
				shapes[i * CCV_NNC_MAX_DIM_ALLOC + j] = _ccv_nnc_micro_index_interpret(vars[i].shape[j], loop_counter, shapes, values, parameter_size);
		} else
			memcpy(shapes + i * CCV_NNC_MAX_DIM_ALLOC, shapes + vars[i].input * CCV_NNC_MAX_DIM_ALLOC, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	}
	size_t total_size = 0;
	for (i = output_size; i < var_count - input_size; i++)
	{
		if (vars[i].no_alloc) // This is skipped.
			continue;
		// allocating memory for these.
		size_t size = 1;
		for (j = 0; j < vars[i].dimensions; j++)
			size *= shapes[i * CCV_NNC_MAX_DIM_ALLOC + j];
		total_size += size;
	}
	float** const vars_mem = (float**)ccmalloc(sizeof(float*) * var_count + sizeof(float) * total_size);
	float* ptr = (float*)(vars_mem + var_count);
	// Assuming these are not tensor_view_t.
	for (i = 0; i < output_size; i++)
	{
		assert(!CCV_IS_TENSOR_VIEW(outputs[i]));
		vars_mem[i] = outputs[i]->data.f32;
	}
	for (i = output_size; i < var_count - input_size; i++)
	{
		if (vars[i].no_alloc) // This is skipped.
		{
			vars_mem[i] = 0;
			continue;
		}
		// allocating memory for these.
		size_t size = 1;
		for (j = 0; j < vars[i].dimensions; j++)
			size *= shapes[i * CCV_NNC_MAX_DIM_ALLOC + j];
		vars_mem[i] = ptr;
		ptr += size;
	}
	for (i = var_count - input_size; i < var_count; i++)
	{
		assert(!CCV_IS_TENSOR_VIEW(inputs[i - (var_count - input_size)]));
		vars_mem[i] = inputs[i - (var_count - input_size)]->data.f32;
	}
	ccv_nnc_micro_function_t* const functions = combine->functions;
	const int function_count = combine->function_count;
	int max_carried_count = 0;
	for (i = 0; i < function_count; i++)
	{
		const int block_count = functions[i].block_count;
		ccv_nnc_micro_loop_block_t* const blocks = block_count == 1 ? &functions[i].one_block : functions[i].blocks;
		for (j = 0; j < block_count; j++)
			max_carried_count = ccv_max(max_carried_count, blocks[j].carried_count);
	}
	ccv_nnc_micro_scalar_t* const carrieds = max_carried_count > 0 ? (ccv_nnc_micro_scalar_t*)ccmalloc(sizeof(ccv_nnc_micro_scalar_t) * max_carried_count) : 0;
	for (i = 0; i < function_count; i++)
	{
		const int block_count = functions[i].block_count;
		ccv_nnc_micro_loop_block_t* const blocks = block_count == 1 ? &functions[i].one_block : functions[i].blocks;
		for (j = 0; j < block_count; j++)
			_ccv_nnc_micro_loop_interpret(blocks[j].loops, blocks[j].loop_count, 0, loop_counter, carrieds, blocks[j].carried_count, vars_mem, shapes, values, parameter_size);
	}
	if (carrieds)
		ccfree(carrieds);
	ccfree(vars_mem);
	ccfree(shapes);
}

char* ccv_nnc_micro_combine_c(ccv_nnc_micro_combine_t* const combine)
{
	return 0;
}
