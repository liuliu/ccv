#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_micro.h"

// MARK - Level-1 API

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

static float _ccv_nnc_micro_expression_interpret(const ccv_nnc_micro_loop_expression_t* const expression, const int* const loop_counter, const ccv_nnc_micro_scalar_t* const carrieds, const int carried_count, float* const* const vars_mem, const int* const shapes, const ccv_nnc_micro_scalar_t* const values, const int parameter_size, int* const out_of_bound_ref)
{
	int i;
	switch (expression->type)
	{
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_ID: {
			assert(expression->id.type == CCV_NNC_MICRO_LOOP_CARRIED_ID);
			return carrieds[expression->id.id].f32;
		}
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAL: {
			return expression->immediate_value.f32;
		}
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR: {
			const ccv_nnc_micro_loop_variable_t variable = expression->variable;
			assert(variable.id.type == CCV_NNC_MICRO_TENSOR_ID);
			float* ptr = vars_mem[variable.id.id];
			size_t size = 1;
			int out_of_bound = 0;
			for (i = variable.index_count - 1; !out_of_bound && i >= 0; i--)
			{
				const int index = _ccv_nnc_micro_index_interpret(variable.index[i], loop_counter, shapes, values, parameter_size);
				if (!variable.no_check_bound[i] &&
					(index < 0 || index >= shapes[variable.id.id * CCV_NNC_MAX_DIM_ALLOC + i]))
					out_of_bound = 1;
				ptr += index * size;
				size *= shapes[variable.id.id * CCV_NNC_MAX_DIM_ALLOC + i];
			}
			if (out_of_bound)
			{
				*out_of_bound_ref = 1;
				return 0;
			}
			return ptr[0];
		}
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_UNARY: {
			const float left = _ccv_nnc_micro_expression_interpret(expression->unary.x, loop_counter, carrieds, carried_count, vars_mem, shapes, values, parameter_size, out_of_bound_ref);
			if (*out_of_bound_ref)
				return 0;
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
			const float left = _ccv_nnc_micro_expression_interpret(expression->binary.left, loop_counter, carrieds, carried_count, vars_mem, shapes, values, parameter_size, out_of_bound_ref);
			if (*out_of_bound_ref)
				return 0;
			const float right = _ccv_nnc_micro_expression_interpret(expression->binary.right, loop_counter, carrieds, carried_count, vars_mem, shapes, values, parameter_size, out_of_bound_ref);
			if (*out_of_bound_ref)
				return 0;
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
			int out_of_bound = 0;
			for (i = variable.index_count - 1; !out_of_bound && i >= 0; i--)
			{
				const int index = _ccv_nnc_micro_index_interpret(variable.index[i], loop_counter, shapes, values, parameter_size);
				if (!variable.no_check_bound[i] &&
					(index < 0 || index >= shapes[variable.id.id * CCV_NNC_MAX_DIM_ALLOC + i]))
					out_of_bound = 1;
				ptr += index * size;
				size *= shapes[variable.id.id * CCV_NNC_MAX_DIM_ALLOC + i];
			}
			if (out_of_bound)
				return;
			const float val = _ccv_nnc_micro_expression_interpret(&statement.assignment.rvalue, loop_counter, carrieds, carried_count, vars_mem, shapes, values, parameter_size, &out_of_bound);
			if (out_of_bound)
				return;
			ptr[0] = val;
			break;
		}
		case CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_COMPOUND_ASSIGNMENT: {
			int out_of_bound = 0;
			const float rvalue = _ccv_nnc_micro_expression_interpret(&statement.compound_assignment.rvalue, loop_counter, carrieds, carried_count, vars_mem, shapes, values, parameter_size, &out_of_bound);
			if (out_of_bound)
				return;
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
					for (i = variable.index_count - 1; !out_of_bound && i >= 0; i--)
					{
						const int index = _ccv_nnc_micro_index_interpret(variable.index[i], loop_counter, shapes, values, parameter_size);
						if (!variable.no_check_bound[i] &&
							(index < 0 || index >= shapes[variable.id.id * CCV_NNC_MAX_DIM_ALLOC + i]))
							out_of_bound = 1;
						ptr += index * size;
						size *= shapes[variable.id.id * CCV_NNC_MAX_DIM_ALLOC + i];
					}
					if (out_of_bound)
						return;
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
	assert(cmd == CCV_NNC_CUSTOM_FORWARD || cmd == CCV_NNC_CUSTOM_BACKWARD);
	int i, j;
	const ccv_nnc_micro_program_t* const program = cmd == CCV_NNC_CUSTOM_FORWARD ? &combine->forward : &combine->backward;
	const int var_count = program->var_count;
	assert(input_size == program->input_size);
	assert(output_size == program->output_size);
	assert(parameter_size == combine->parameter_size);
	int* const shapes = (int*)cccalloc(var_count, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	ccv_nnc_micro_tensor_t* const vars = program->vars;
	for (i = 0; i < input_size; i++)
		memcpy(shapes + program->inputs[i] * CCV_NNC_MAX_DIM_ALLOC, &inputs[i]->info.dim, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	int loop_counter[CCV_NNC_MAX_DIM_ALLOC];
	for (i = 0; i < var_count; i++)
	{
		int flag = 0;
		for (j = 0; !flag && j < input_size; j++)
			flag = (program->inputs[j] == i);
		if (flag)
			continue;
		if (vars[i].shape)
		{
			for (j = 0; j < vars[i].dimensions; j++)
				shapes[i * CCV_NNC_MAX_DIM_ALLOC + j] = _ccv_nnc_micro_index_interpret(vars[i].shape[j], loop_counter, shapes, values, parameter_size);
		} else
			memcpy(shapes + i * CCV_NNC_MAX_DIM_ALLOC, shapes + vars[i].input * CCV_NNC_MAX_DIM_ALLOC, sizeof(int) * CCV_NNC_MAX_DIM_ALLOC);
	}
	size_t total_size = 0;
	for (i = 0; i < var_count; i++)
	{
		int flag = 0;
		for (j = 0; !flag && j < input_size; j++)
			flag = (program->inputs[j] == i);
		for (j = 0; !flag && j < output_size; j++)
			flag = (program->outputs[j] == i);
		if (flag)
			continue;
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
		vars_mem[program->outputs[i]] = outputs[i]->data.f32;
	}
	for (i = 0; i < var_count; i++)
	{
		int flag = 0;
		for (j = 0; !flag && j < input_size; j++)
			flag = (program->inputs[j] == i);
		for (j = 0; !flag && j < output_size; j++)
			flag = (program->outputs[j] == i);
		if (flag)
			continue;
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
	for (i = 0; i < input_size; i++)
	{
		assert(!CCV_IS_TENSOR_VIEW(inputs[i]));
		vars_mem[program->inputs[i]] = inputs[i]->data.f32;
	}
	ccv_nnc_micro_function_t* const functions = program->functions;
	const int function_count = program->function_count;
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
