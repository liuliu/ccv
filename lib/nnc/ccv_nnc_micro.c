#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_micro.h"
#include "3rdparty/khash/khash.h"

// MARK - Level-1 API

static __thread uint16_t micro_io_id = 0;

ccv_nnc_micro_io_t ccv_nnc_micro_input(const int dimensions)
{
	assert(dimensions <= CCV_NNC_MAX_DIM_ALLOC);
	return (ccv_nnc_micro_io_t){
		.type = CCV_NNC_MICRO_INPUT,
		.id = ++micro_io_id,
		.dimensions = dimensions,
	};
}

// A simple recursive descent parser. Omitted tokenisation step.
static int _accept(const char** const pos, int* const remain_size, const char* symbol, int size)
{
	if (*remain_size < size)
		return 0;
	if (memcmp(*pos, symbol, size) == 0)
	{
		*remain_size -= size;
		*pos += size;
		return 1;
	}
	return 0;
}

static int _expect(const char** const pos, int* const remain_size, const char* symbol, int size)
{
	if (_accept(pos, remain_size, symbol, size))
		return 1;
	assert(0 && "unexpected symbol");
	return 0;
}

static int _constant(const char** const pos, int* const remain_size, int* id)
{
	int size = 0;
	*id = 0;
	while (*remain_size - size > 0 && pos[0][size] >= '0' && pos[0][size] <= '9')
	{
		*id *= 10;
		*id += (pos[0][size] - '0');
		++size;
	}
	*remain_size -= size;
	*pos += size;
	return size > 0;
}

static int _index(const char** const pos, int* const remain_size, int* id)
{
	if (!(*remain_size > 0 && pos[0][0] == 'i'))
		return 0;
	int size = 1;
	*id = 0;
	while (*remain_size - size > 0 && pos[0][size] >= '0' && pos[0][size] <= '9')
	{
		*id *= 10;
		*id += (pos[0][size] - '0');
		++size;
	}
	if (size > 1)
	{
		*remain_size -= size;
		*pos += size;
		return 1;
	}
	return 0;
}

static int _dim(const char** const pos, int* const remain_size, int* id)
{
	if (!(*remain_size > 0 && pos[0][0] == 'd'))
		return 0;
	int size = 1;
	*id = 0;
	while (*remain_size - size > 0 && pos[0][size] >= '0' && pos[0][size] <= '9')
	{
		*id *= 10;
		*id += (pos[0][size] - '0');
		++size;
	}
	if (size > 1)
	{
		*remain_size -= size;
		*pos += size;
		return 1;
	}
	return 0;
}

static int _var(const char** const pos, int* const remain_size, char** name)
{
	if (!(*remain_size > 0 && pos[0][0] == '$'))
		return 0;
	int size = 1;
	while (*remain_size - size > 0 &&
			((pos[0][size] >= '0' && pos[0][size] <= '9') ||
			 (pos[0][size] >= 'a' && pos[0][size] <= 'z') ||
			 (pos[0][size] >= 'A' && pos[0][size] <= 'Z') ||
			 pos[0][size] == '_'))
		++size;
	if (size > 1)
	{
		*name = ccmalloc(size + 1);
		memcpy(*name, pos, size);
		name[0][size] = 0;
		*remain_size -= size;
		*pos += size;
		return 1;
	}
	return 0;
}

static CCV_WARN_UNUSED(ccv_nnc_micro_loop_index_term_t) _expression(const int xid, const char** const pos, int* const remain_size);

static ccv_nnc_micro_loop_index_term_t _factor(const int xid, const char** const pos, int* const remain_size)
{
	ccv_nnc_micro_loop_index_term_t term;
	while (_accept(pos, remain_size, " ", 1)) {}
	int id;
	char* name;
	if (_constant(pos, remain_size, &id)) {
		term.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL;
		term.immediate_value = id;
	} else if (_index(pos, remain_size, &id)) {
		term.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
		term.id.type = CCV_NNC_MICRO_LOOP_ID;
		term.id.id = id;
	} else if (_dim(pos, remain_size, &id)) {
		term.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
		term.id.type = CCV_NNC_MICRO_AXIS_SIZE_ID;
		term.id.d = id;
		term.id.id = xid;
	} else if (_var(pos, remain_size, &name)) {
		term.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_UNBOUND_VAR;
		term.name = name;
	} else if (_accept(pos, remain_size, "(", 1)) {
		term = _expression(xid, pos, remain_size);
		_expect(pos, remain_size, ")", 1);
	} else {
		assert(0 && "factor: syntax error");
	}
	while (_accept(pos, remain_size, " ", 1)) {}
	return term;
}

static ccv_nnc_micro_loop_index_term_t _term(const int xid, const char** const pos, int* const remain_size)
{
	while (_accept(pos, remain_size, " ", 1)) {}
	ccv_nnc_micro_loop_index_term_t term = _factor(xid, pos, remain_size);
	while (*remain_size > 0 && (pos[0][0] == '*' || pos[0][0] == '/'))
	{
		const int op = pos[0][0] == '*' ? CCV_NNC_MICRO_BINARY_OP_MUL : CCV_NNC_MICRO_BINARY_OP_DIV;
		*remain_size -= 1;
		*pos += 1;
		const ccv_nnc_micro_loop_index_term_t left = term;
		const ccv_nnc_micro_loop_index_term_t right = _factor(xid, pos, remain_size);
		term.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_EXPR;
		term.expression = (ccv_nnc_micro_loop_index_expression_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_index_expression_t));
		term.expression->op = op;
		term.expression->left = left;
		term.expression->right = right;
	}
	while (_accept(pos, remain_size, " ", 1)) {}
	return term;
}

static ccv_nnc_micro_loop_index_term_t _expression(const int xid, const char** const pos, int* const remain_size)
{
	while (_accept(pos, remain_size, " ", 1)) {}
	int prefix_op = -1;
	if (*remain_size > 0 && (pos[0][0] == '+' || pos[0][0] == '-'))
	{
		prefix_op = pos[0][0] == '+' ? CCV_NNC_MICRO_BINARY_OP_PLUS : CCV_NNC_MICRO_BINARY_OP_MINUS;
		*remain_size -= 1;
		*pos += 1;
	}
	ccv_nnc_micro_loop_index_term_t node = _term(xid, pos, remain_size);
	while (*remain_size > 0 && (pos[0][0] == '+' || pos[0][0] == '-'))
	{
		const int op = pos[0][0] == '+' ? CCV_NNC_MICRO_BINARY_OP_PLUS : CCV_NNC_MICRO_BINARY_OP_MINUS;
		*remain_size -= 1;
		*pos += 1;
		const ccv_nnc_micro_loop_index_term_t left = node;
		const ccv_nnc_micro_loop_index_term_t right = _term(xid, pos, remain_size);
		node.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_EXPR;
		node.expression = (ccv_nnc_micro_loop_index_expression_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_index_expression_t));
		node.expression->op = op;
		node.expression->left = left;
		node.expression->right = right;
	}
	while (_accept(pos, remain_size, " ", 1)) {}
	if (prefix_op >= 0)
	{
		ccv_nnc_micro_loop_index_expression_t* const expr = (ccv_nnc_micro_loop_index_expression_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_index_expression_t));
		expr->op = prefix_op;
		expr->left = node;
		expr->right.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_NONE;
		node.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_EXPR;
		node.expression = expr;
	}
	return node;
}

static void _no_index(const ccv_nnc_micro_loop_index_term_t term)
{
	switch (term.type) {
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID:
			// Can only be axis size id. No loop index.
			assert(term.id.type == CCV_NNC_MICRO_AXIS_SIZE_ID);
			break;
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_EXPR:
			_no_index(term.expression->left);
			_no_index(term.expression->right);
			break;
	}
}

typedef struct {
	int reindex_count;
	ccv_nnc_micro_io_t x;
	ccv_nnc_micro_loop_index_term_t* shape;
	ccv_nnc_micro_loop_index_term_t* reindex;
} ccv_nnc_micro_reindex_t;

ccv_nnc_micro_io_t ccv_nnc_micro_reindex(const char* const* const shape, const char* const* const reindex, const int reindex_count, const ccv_nnc_micro_io_t x)
{
	assert(reindex_count <= CCV_NNC_MAX_DIM_ALLOC);
	int i;
	ccv_nnc_micro_reindex_t* const data = (ccv_nnc_micro_reindex_t*)ccmalloc(sizeof(ccv_nnc_micro_reindex_t) + sizeof(ccv_nnc_micro_loop_index_term_t) * reindex_count * 2);
	data->reindex_count = reindex_count;
	data->x = x;
	data->shape = (ccv_nnc_micro_loop_index_term_t*)(data + 1);
	data->reindex = data->shape + reindex_count;
	// Parse shape into expressions and validate the grammar. Do this upfront so we don't fail on parsing
	// later, which can be confusing.
	// CFG:
	// VAR -> $[a-zA-Z0-9]+
	// DIM -> d[0-9]+
	// INDEX -> i[0-9]+
	// CONST -> [0-9]+
	// FACTOR -> VAR | DIM | CONST | INDEX
	// TERM -> FACTOR { ("*" | "/") FACTOR }
	// EXPRESSION -> ["+" | "-"] TERM { ("+" | "-") TERM }
	// Also, we choose to reuse the index expression structure even some information (such as id of tensors
	// and the binding variables) not available. In this way, there is no need to reallocate index expression
	// later, we just need to simply "patch" it in ccv_nnc_micro_combine_t.
	for (i = 0; i < reindex_count; i++)
	{
		int remain_size = strlen(shape[i]);
		const char* pos = shape[i];
		ccv_nnc_micro_loop_index_term_t term = _expression(x.id, &pos, &remain_size);
		_no_index(term); // Make sure this is not index, no loop index.
		data->shape[i] = term;
	}
	// Parse reindex.
	for (i = 0; i < reindex_count; i++)
	{
		int remain_size = strlen(reindex[i]);
		const char* pos = reindex[i];
		data->reindex[i] = _expression(x.id, &pos, &remain_size);
	}
	return (ccv_nnc_micro_io_t){
		.type = CCV_NNC_MICRO_REINDEX,
		.id = ++micro_io_id,
		.dimensions = reindex_count,
		.data = data,
	};
}

typedef struct {
	uint32_t op;
	ccv_nnc_micro_io_t left;
	ccv_nnc_micro_io_t right;
} ccv_nnc_micro_binary_t;

ccv_nnc_micro_io_t ccv_nnc_micro_binary(const uint32_t op, const ccv_nnc_micro_io_t x, const ccv_nnc_micro_io_t y)
{
	ccv_nnc_micro_binary_t* const data = (ccv_nnc_micro_binary_t*)ccmalloc(sizeof(ccv_nnc_micro_binary_t));
	data->op = op;
	data->left = x;
	data->right = y;
	assert(x.dimensions == y.dimensions);
	return (ccv_nnc_micro_io_t){
		.type = CCV_NNC_MICRO_BINARY,
		.id = ++micro_io_id,
		.dimensions = x.dimensions,
		.data = data,
	};
}

typedef struct {
	uint32_t op;
	int axis_count;
	ccv_nnc_micro_io_t x;
	int axis[1];
} ccv_nnc_micro_reduce_t;

ccv_nnc_micro_io_t ccv_nnc_micro_reduce(const uint32_t op, const int* const axis, const int axis_count, const ccv_nnc_micro_io_t x)
{
	assert(axis_count > 0);
	ccv_nnc_micro_reduce_t* const data = (ccv_nnc_micro_reduce_t*)ccmalloc(sizeof(ccv_nnc_micro_reduce_t) + sizeof(int) * (axis_count - 1));
	data->op = op;
	data->x = x;
	data->axis_count = axis_count;
	assert(axis_count <= x.dimensions);
	int i;
	for (i = 0; i < axis_count; i++)
	{ assert(axis[i] <= x.dimensions); }
	memcpy(data->axis, axis, sizeof(int) * axis_count);
	return (ccv_nnc_micro_io_t){
		.type = CCV_NNC_MICRO_REDUCE,
		.id = ++micro_io_id,
		.dimensions = x.dimensions,
		.data = data,
	};
}

typedef struct {
	int axis;
	ccv_nnc_micro_io_t x;
	ccv_nnc_micro_io_t index;
} ccv_nnc_micro_select_t;

ccv_nnc_micro_io_t ccv_nnc_micro_select(const int axis, const ccv_nnc_micro_io_t x, const ccv_nnc_micro_io_t index)
{
	ccv_nnc_micro_select_t* const data = (ccv_nnc_micro_select_t*)ccmalloc(sizeof(ccv_nnc_micro_select_t));
	data->x = x;
	data->index = index;
	data->axis = axis;
	assert(axis <= CCV_NNC_MAX_DIM_ALLOC);
	return (ccv_nnc_micro_io_t){
		.type = CCV_NNC_MICRO_SELECT,
		.id = ++micro_io_id,
		.dimensions = x.dimensions,
		.data = data,
	};
}

KHASH_MAP_INIT_STR(ccv_nnc_micro_var, uint32_t)

static void _ccv_nnc_id_vars_in_term(ccv_nnc_micro_loop_index_term_t* const term, khash_t(ccv_nnc_micro_var)* const vars)
{
	switch (term->type)
	{
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_EXPR:
			_ccv_nnc_id_vars_in_term(&term->expression->left, vars);
			_ccv_nnc_id_vars_in_term(&term->expression->right, vars);
			break;
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_UNBOUND_VAR: {
			khiter_t k = kh_get(ccv_nnc_micro_var, vars, term->name);
			assert(k != kh_end(vars));
			ccfree(term->name);
			term->id.id = kh_val(vars, k);
			term->id.d = 0;
			term->id.type = CCV_NNC_MICRO_SCALAR_ID;
			term->type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
			break;
		}
	}
}

static void _ccv_nnc_id_vars(const ccv_nnc_micro_io_t output, khash_t(ccv_nnc_micro_var)* const vars)
{
	switch (output.type)
	{
		case CCV_NNC_MICRO_BINARY: {
			ccv_nnc_micro_binary_t* const binary = (ccv_nnc_micro_binary_t*)output.data;
			_ccv_nnc_id_vars(binary->left, vars);
			_ccv_nnc_id_vars(binary->right, vars);
			break;
		}
		case CCV_NNC_MICRO_SELECT: {
			ccv_nnc_micro_select_t* const select = (ccv_nnc_micro_select_t*)output.data;
			_ccv_nnc_id_vars(select->x, vars);
			_ccv_nnc_id_vars(select->index, vars);
			break;
		}
		case CCV_NNC_MICRO_REDUCE: {
			ccv_nnc_micro_reduce_t* const reduce = (ccv_nnc_micro_reduce_t*)output.data;
			_ccv_nnc_id_vars(reduce->x, vars);
			break;
		}
		case CCV_NNC_MICRO_REINDEX: {
			ccv_nnc_micro_reindex_t* const reindex = (ccv_nnc_micro_reindex_t*)output.data;
			int i;
			for (i = 0; i < reindex->reindex_count; i++)
			{
				_ccv_nnc_id_vars_in_term(reindex->shape + i, vars);
				_ccv_nnc_id_vars_in_term(reindex->reindex + i, vars);
			}
			_ccv_nnc_id_vars(reindex->x, vars);
			break;
		}
	}
}

CCV_WARN_UNUSED(ccv_nnc_micro_nested_loop_t) _ccv_nnc_micro_binary_loop(const ccv_nnc_micro_io_t output)
{
	assert(output.type == CCV_NNC_MICRO_BINARY);
	ccv_nnc_micro_binary_t* const binary = (ccv_nnc_micro_binary_t*)output.data;
	const int loop_count = output.dimensions;
	assert(binary->left.dimensions == loop_count);
	assert(binary->right.dimensions == loop_count);
	assert(loop_count <= CCV_NNC_MAX_DIM_ALLOC);
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	int i;
	for (i = 0; i < loop_count; i++)
	{
		// Start is always 0.
		loops[i].start_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL;
		loops[i].start_index.immediate_value = 0;
		// End is the size of this axis.
		loops[i].end_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
		loops[i].end_index.id.type = CCV_NNC_MICRO_AXIS_SIZE_ID;
		loops[i].end_index.id.id = output.id;
		loops[i].end_index.id.d = i;
		loops[i].carry_over_count = 0;
		loops[i].carry_overs = 0;
		loops[i].block_count = 0;
		loops[i].blocks = 0;
		loops[i].id.type = CCV_NNC_MICRO_LOOP_ID;
		loops[i].id.d = 0;
		loops[i].id.id = i;
	}
	ccv_nnc_micro_loop_block_t block;
	block.type = CCV_NNC_MICRO_LOOP_BLOCK_TYPE_ASSIGNMENT;
	block.assignment.lvalue.id.type = CCV_NNC_MICRO_TENSOR_ID;
	block.assignment.lvalue.id.d = 0;
	block.assignment.lvalue.id.id = output.id;
	block.assignment.lvalue.index_count = loop_count;
	block.assignment.rvalue.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_BINARY;
	block.assignment.rvalue.expression.op = binary->op;
	block.assignment.rvalue.expression.left.type = CCV_NNC_MICRO_LOOP_BINARY_TYPE_VAR;
	block.assignment.rvalue.expression.left.variable.id.type = CCV_NNC_MICRO_TENSOR_ID;
	block.assignment.rvalue.expression.left.variable.id.id = binary->left.id;
	block.assignment.rvalue.expression.left.variable.index_count = loop_count;
	block.assignment.rvalue.expression.right.type = CCV_NNC_MICRO_LOOP_BINARY_TYPE_VAR;
	block.assignment.rvalue.expression.right.variable.id.type = CCV_NNC_MICRO_TENSOR_ID;
	block.assignment.rvalue.expression.right.variable.id.id = binary->right.id;
	block.assignment.rvalue.expression.right.variable.index_count = loop_count;
	for (i = 0; i < loop_count; i++)
	{
		block.assignment.lvalue.index[i].type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
		block.assignment.lvalue.index[i].id = loops[i].id;
		block.assignment.rvalue.expression.left.variable.index[i].type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
		block.assignment.rvalue.expression.left.variable.index[i].id = loops[i].id;
		block.assignment.rvalue.expression.right.variable.index[i].type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
		block.assignment.rvalue.expression.right.variable.index[i].id = loops[i].id;
	}
	loops[loop_count - 1].block_count = 1;
	loops[loop_count - 1].blocks = (ccv_nnc_micro_loop_block_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_block_t));
	loops[loop_count - 1].blocks[0] = block;
	return (ccv_nnc_micro_nested_loop_t){
		.loop_count = loop_count,
		.loops = loops,
	};
}

CCV_WARN_UNUSED(ccv_nnc_micro_nested_loop_t) _ccv_nnc_micro_select_loop(const ccv_nnc_micro_io_t output)
{
	assert(output.type == CCV_NNC_MICRO_SELECT);
	ccv_nnc_micro_select_t* const select = (ccv_nnc_micro_select_t*)output.data;
	const int loop_count = output.dimensions;
	assert(select->x.dimensions == loop_count);
	assert(select->index.dimensions == loop_count);
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	int i;
	for (i = 0; i < loop_count; i++)
	{
		// Start is always 0.
		loops[i].start_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL;
		loops[i].start_index.immediate_value = 0;
		// End is the size of this axis.
		if (i == select->axis)
		{
			loops[i].end_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL;
			loops[i].end_index.immediate_value = 1;
		} else {
			loops[i].end_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
			loops[i].end_index.id.type = CCV_NNC_MICRO_AXIS_SIZE_ID;
			loops[i].end_index.id.id = output.id;
			loops[i].end_index.id.d = i;
		}
		loops[i].carry_over_count = 0;
		loops[i].carry_overs = 0;
		loops[i].block_count = 0;
		loops[i].blocks = 0;
		loops[i].id.type = CCV_NNC_MICRO_LOOP_ID;
		loops[i].id.d = 0;
		loops[i].id.id = i;
	}
	ccv_nnc_micro_loop_block_t block;
	block.type = CCV_NNC_MICRO_LOOP_BLOCK_TYPE_ASSIGNMENT;
	block.assignment.lvalue.id.type = CCV_NNC_MICRO_TENSOR_ID;
	block.assignment.lvalue.id.d = 0;
	block.assignment.lvalue.id.id = output.id;
	block.assignment.lvalue.index_count = loop_count;
	block.assignment.rvalue.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR;
	block.assignment.rvalue.variable.id.type = CCV_NNC_MICRO_TENSOR_ID;
	block.assignment.rvalue.variable.id.d = 0;
	block.assignment.rvalue.variable.id.id = select->x.id;
	block.assignment.rvalue.variable.index_count = loop_count;
	for (i = 0; i < loop_count; i++)
	{
		block.assignment.lvalue.index[i].type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
		block.assignment.lvalue.index[i].id = loops[i].id;
		block.assignment.rvalue.variable.index[i].type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
		if (i == select->axis)
		{
			// Use the value of index tensor as the index.
			block.assignment.rvalue.variable.index[i].id.type = CCV_NNC_MICRO_TENSOR_ID;
			block.assignment.rvalue.variable.index[i].id.d = 0;
			block.assignment.rvalue.variable.index[i].id.id = select->index.id;
		} else
			block.assignment.rvalue.variable.index[i].id = loops[i].id;
	}
	loops[loop_count - 1].block_count = 1;
	loops[loop_count - 1].blocks = (ccv_nnc_micro_loop_block_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_block_t));
	loops[loop_count - 1].blocks[0] = block;
	return (ccv_nnc_micro_nested_loop_t){
		.loop_count = loop_count,
		.loops = loops,
	};
}

CCV_WARN_UNUSED(ccv_nnc_micro_nested_loop_t) _ccv_nnc_micro_reduce_loop(const ccv_nnc_micro_io_t output)
{
	assert(output.type == CCV_NNC_MICRO_REDUCE);
	ccv_nnc_micro_reduce_t* const reduce = (ccv_nnc_micro_reduce_t*)output.data;
	const int loop_count = output.dimensions;
	assert(reduce->x.dimensions == loop_count);
	// If axis_count == loop_count, we need extra loop to reduce.
	int has_extra_loop = (reduce->axis_count == loop_count);
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * (loop_count + has_extra_loop));
	int i, j, k;
	int reduce_axis[loop_count];
	memset(reduce_axis, 0, sizeof(int) * loop_count);
	for (i = 0; i < reduce->axis_count; i++)
		reduce_axis[reduce->axis[i]] = 1;
	j = 0;
	// If loop_count == reduce_axis_count, we have extra loop for carry_overs and block.
	if (has_extra_loop)
	{
		// Start is always 0.
		loops[0].start_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL;
		loops[0].start_index.immediate_value = 0;
		// End is the size of this axis.
		loops[0].end_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL;
		loops[0].end_index.immediate_value = 1;
		loops[0].carry_over_count = 0;
		loops[0].carry_overs = 0;
		loops[0].block_count = 0;
		loops[0].blocks = 0;
		loops[0].id.type = CCV_NNC_MICRO_LOOP_ID;
		loops[0].id.d = 0;
		loops[0].id.id = 0;
		k = 1;
	} else
		k = loop_count - reduce->axis_count;
	for (i = 0; i < loop_count; i++)
	{
		if (reduce_axis[i])
		{
			// Start is always 0.
			loops[k].start_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL;
			loops[k].start_index.immediate_value = 0;
			// End is the size of this axis.
			loops[k].end_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
			loops[k].end_index.id.type = CCV_NNC_MICRO_AXIS_SIZE_ID;
			loops[k].end_index.id.id = output.id;
			loops[k].end_index.id.d = i;
			loops[k].carry_over_count = 0;
			loops[k].carry_overs = 0;
			loops[k].block_count = 0;
			loops[k].blocks = 0;
			loops[k].id.type = CCV_NNC_MICRO_LOOP_ID;
			loops[k].id.d = 0;
			loops[k].id.id = i + has_extra_loop;
			++k;
		} else {
			// Start is always 0.
			loops[j].start_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL;
			loops[j].start_index.immediate_value = 0;
			// End is the size of this axis.
			loops[j].end_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
			loops[j].end_index.id.type = CCV_NNC_MICRO_AXIS_SIZE_ID;
			loops[j].end_index.id.id = output.id;
			loops[j].end_index.id.d = i;
			loops[j].carry_over_count = 0;
			loops[j].carry_overs = 0;
			loops[j].block_count = 0;
			loops[j].blocks = 0;
			loops[j].id.type = CCV_NNC_MICRO_LOOP_ID;
			loops[j].id.d = 0;
			loops[j].id.id = i + has_extra_loop;
			++j;
		}
	}
	const int carry_over_loop_idx = has_extra_loop ? 0 : loop_count - reduce->axis_count - 1;
	loops[carry_over_loop_idx].carry_over_count = 1;
	loops[carry_over_loop_idx].carry_overs = (ccv_nnc_micro_loop_carry_over_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_carry_over_t));
	loops[carry_over_loop_idx].carry_overs[0].reduce_op = reduce->op;
	loops[carry_over_loop_idx].carry_overs[0].id.type = CCV_NNC_MICRO_CARRY_OVER_ID;
	loops[carry_over_loop_idx].carry_overs[0].id.d = 0;
	loops[carry_over_loop_idx].carry_overs[0].id.id = ++micro_io_id;
	ccv_nnc_micro_loop_block_t block;
	block.type = CCV_NNC_MICRO_LOOP_BLOCK_TYPE_COMPOUND;
	block.compound.id = loops[carry_over_loop_idx].carry_overs[0].id;
	block.compound.rvalue.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR;
	block.compound.rvalue.variable.id.type = CCV_NNC_MICRO_TENSOR_ID;
	block.compound.rvalue.variable.id.d = 0;
	block.compound.rvalue.variable.id.id = reduce->x.id;
	block.compound.rvalue.variable.index_count = loop_count;
	j = 0;
	// If loop_count == reduce_axis_count, we have extra loop for carry_overs and block.
	k = has_extra_loop ? 1 : loop_count - reduce->axis_count;
	for (i = 0; i < loop_count; i++)
	{
		block.compound.rvalue.variable.index[i].type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
		if (reduce_axis[i])
		{
			block.compound.rvalue.variable.index[i].id = loops[k].id;
			++k;
		} else {
			block.compound.rvalue.variable.index[i].id = loops[j].id;
			++j;
		}
	}
	loops[carry_over_loop_idx + reduce->axis_count].block_count = 1;
	loops[carry_over_loop_idx + reduce->axis_count].blocks = (ccv_nnc_micro_loop_block_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_block_t));
	loops[carry_over_loop_idx + reduce->axis_count].blocks[0] = block;
	block.type = CCV_NNC_MICRO_LOOP_BLOCK_TYPE_ASSIGNMENT;
	block.assignment.lvalue.id.type = CCV_NNC_MICRO_TENSOR_ID;
	block.assignment.lvalue.id.d = 0;
	block.assignment.lvalue.id.id = output.id;
	block.assignment.lvalue.index_count = loop_count;
	j = 0;
	for (i = 0; i < loop_count; i++)
	{
		if (reduce_axis[i])
		{
			block.assignment.lvalue.index[i].type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL;
			block.assignment.lvalue.index[i].immediate_value = 0;
		} else {
			block.assignment.lvalue.index[i].type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
			block.assignment.lvalue.index[i].id = loops[j].id;
			++j;
		}
	}
	block.assignment.rvalue.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_ID;
	block.assignment.rvalue.id = loops[carry_over_loop_idx].carry_overs[0].id;
	loops[carry_over_loop_idx].block_count = 1;
	loops[carry_over_loop_idx].blocks = (ccv_nnc_micro_loop_block_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_block_t));
	loops[carry_over_loop_idx].blocks[0] = block;
	return (ccv_nnc_micro_nested_loop_t){
		.loop_count = loop_count + has_extra_loop,
		.loops = loops,
	};
}

CCV_WARN_UNUSED(ccv_nnc_micro_nested_loop_t) _ccv_nnc_micro_reindex_loop(const ccv_nnc_micro_io_t output)
{
	assert(output.type == CCV_NNC_MICRO_REINDEX);
	ccv_nnc_micro_reindex_t* const reindex = (ccv_nnc_micro_reindex_t*)output.data;
	const int loop_count = output.dimensions;
	assert(loop_count <= CCV_NNC_MAX_DIM_ALLOC);
	assert(loop_count == reindex->reindex_count);
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	int i;
	for (i = 0; i < loop_count; i++)
	{
		// Start is always 0.
		loops[i].start_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL;
		loops[i].start_index.immediate_value = 0;
		// End is the size of this axis.
		loops[i].end_index.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
		loops[i].end_index.id.type = CCV_NNC_MICRO_AXIS_SIZE_ID;
		loops[i].end_index.id.id = output.id;
		loops[i].end_index.id.d = i;
		loops[i].carry_over_count = 0;
		loops[i].carry_overs = 0;
		loops[i].block_count = 0;
		loops[i].blocks = 0;
		loops[i].id.type = CCV_NNC_MICRO_LOOP_ID;
		loops[i].id.d = 0;
		loops[i].id.id = i;
	}
	ccv_nnc_micro_loop_block_t block;
	block.type = CCV_NNC_MICRO_LOOP_BLOCK_TYPE_ASSIGNMENT;
	block.assignment.lvalue.id.type = CCV_NNC_MICRO_TENSOR_ID;
	block.assignment.lvalue.id.d = 0;
	block.assignment.lvalue.id.id = output.id;
	block.assignment.lvalue.index_count = loop_count;
	block.assignment.rvalue.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR;
	block.assignment.rvalue.variable.id.type = CCV_NNC_MICRO_TENSOR_ID;
	block.assignment.rvalue.variable.id.d = 0;
	block.assignment.rvalue.variable.id.id = reindex->x.id;
	block.assignment.rvalue.variable.index_count = loop_count;
	for (i = 0; i < loop_count; i++)
	{
		block.assignment.lvalue.index[i].type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
		block.assignment.lvalue.index[i].id = loops[i].id;
		block.assignment.rvalue.variable.index[i] = reindex->reindex[i];
	}
	loops[loop_count - 1].block_count = 1;
	loops[loop_count - 1].blocks = (ccv_nnc_micro_loop_block_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_block_t));
	loops[loop_count - 1].blocks[0] = block;
	return (ccv_nnc_micro_nested_loop_t){
		.loop_count = loop_count,
		.loops = loops,
	};
}

KHASH_SET_INIT_INT(ccv_nnc_ids)

static void _ccv_nnc_micro_loop_index_free(ccv_nnc_micro_loop_index_term_t* const term)
{
	if (term->type == CCV_NNC_MICRO_LOOP_INDEX_TYPE_EXPR)
	{
		_ccv_nnc_micro_loop_index_free(&term->expression->left);
		_ccv_nnc_micro_loop_index_free(&term->expression->right);
		ccfree(term->expression);
	}
}

CCV_WARN_UNUSED(ccv_nnc_micro_combine_t*) ccv_nnc_micro_combine_new(const ccv_nnc_micro_io_t* const inputs, const int input_size, const char* const* const parameters, const int parameter_size, const ccv_nnc_micro_io_t* const outputs, const int output_size)
{
	assert(output_size > 0);
	assert(input_size > 0);
	int i, j;
	// First, id each tensors and binding parameters (bounded var).
	khash_t(ccv_nnc_micro_var)* const vars = kh_init(ccv_nnc_micro_var);
	for (i = 0; i < parameter_size; i++)
	{
		int ret;
		khiter_t k = kh_put(ccv_nnc_micro_var, vars, parameters[i], &ret);
		assert(ret != 0);
		kh_val(vars, k) = ++micro_io_id;
	}
	for (i = 0; i < output_size; i++)
		_ccv_nnc_id_vars(outputs[i], vars);
	kh_destroy(ccv_nnc_micro_var, vars);
	// Second, do reverse topological sort (from output and then reverse the order).
	// We can do this simple thing because there is no overlaps of the outputs, thus, no cases where
	// output[0] is the input for output[1]. Otherwise we need to detect this, see ccv_cnnp_model_new
	// for more details on why.
	for (i = 0; i < output_size - 1; i++)
		for (j = i + 1; j < output_size; j++)
		{ assert(outputs[i].id != outputs[j].id); }
	uint64_t input_bitmask[((input_size - 1) >> 6) + 1];
	memset(input_bitmask, 0, sizeof(uint64_t) * (((input_size - 1) >> 6) + 1));
	ccv_array_t* const reverse_top = ccv_array_new(sizeof(ccv_nnc_micro_io_t), output_size + input_size, 0);
	ccv_array_resize(reverse_top, output_size);
	memcpy(ccv_array_get(reverse_top, 0), outputs, sizeof(ccv_nnc_micro_io_t) * output_size);
	khash_t(ccv_nnc_ids)* const ids = kh_init(ccv_nnc_ids);
	for (i = 0; i < reverse_top->rnum; i++)
	{
		ccv_nnc_micro_io_t* const output = (ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, i);
		int ret;
		switch (output->type)
		{
			case CCV_NNC_MICRO_BINARY: {
				ccv_nnc_micro_binary_t* const binary = (ccv_nnc_micro_binary_t*)output->data;
				if (binary->left.type != CCV_NNC_MICRO_INPUT)
				{
					kh_put(ccv_nnc_ids, ids, binary->left.id, &ret);
					if (ret != 0)
						ccv_array_push(reverse_top, &binary->left);
				} else {
					// This is an input, it must be represented in inputs, try to find it.
					for (j = 0; j < input_size; j++)
						if (inputs[j].id == binary->left.id)
							break;
					assert(j < input_size); // Cannot find the inputs, error!
					input_bitmask[j >> 6] |= ((uint64_t)1 << (j & 63));
				}
				if (binary->right.type != CCV_NNC_MICRO_INPUT)
				{
					kh_put(ccv_nnc_ids, ids, binary->right.id, &ret);
					if (ret != 0)
						ccv_array_push(reverse_top, &binary->right);
				} else {
					// This is an input, it must be represented in inputs, try to find it.
					for (j = 0; j < input_size; j++)
						if (inputs[j].id == binary->right.id)
							break;
					assert(j < input_size); // Cannot find the inputs, error!
					input_bitmask[j >> 6] |= ((uint64_t)1 << (j & 63));
				}
				break;
			}
			case CCV_NNC_MICRO_SELECT: {
				ccv_nnc_micro_select_t* const select = (ccv_nnc_micro_select_t*)output->data;
				if (select->x.type != CCV_NNC_MICRO_INPUT)
				{
					kh_put(ccv_nnc_ids, ids, select->x.id, &ret);
					if (ret != 0)
						ccv_array_push(reverse_top, &select->x);
				} else {
					// This is an input, it must be represented in inputs, try to find it.
					for (j = 0; j < input_size; j++)
						if (inputs[j].id == select->x.id)
							break;
					assert(j < input_size); // Cannot find the inputs, error!
					input_bitmask[j >> 6] |= ((uint64_t)1 << (j & 63));
				}
				if (select->index.type != CCV_NNC_MICRO_INPUT)
				{
					kh_put(ccv_nnc_ids, ids, select->index.id, &ret);
					if (ret != 0)
						ccv_array_push(reverse_top, &select->index);
				} else {
					// This is an input, it must be represented in inputs, try to find it.
					for (j = 0; j < input_size; j++)
						if (inputs[j].id == select->index.id)
							break;
					assert(j < input_size); // Cannot find the inputs, error!
					input_bitmask[j >> 6] |= ((uint64_t)1 << (j & 63));
				}
				break;
			}
			case CCV_NNC_MICRO_REDUCE: {
				ccv_nnc_micro_reduce_t* const reduce = (ccv_nnc_micro_reduce_t*)output->data;
				if (reduce->x.type != CCV_NNC_MICRO_INPUT)
				{
					kh_put(ccv_nnc_ids, ids, reduce->x.id, &ret);
					if (ret != 0)
						ccv_array_push(reverse_top, &reduce->x);
				} else {
					// This is an input, it must be represented in inputs, try to find it.
					for (j = 0; j < input_size; j++)
						if (inputs[j].id == reduce->x.id)
							break;
					assert(j < input_size); // Cannot find the inputs, error!
					input_bitmask[j >> 6] |= ((uint64_t)1 << (j & 63));
				}
				break;
			}
			case CCV_NNC_MICRO_REINDEX: {
				ccv_nnc_micro_reindex_t* const reindex = (ccv_nnc_micro_reindex_t*)output->data;
				if (reindex->x.type != CCV_NNC_MICRO_INPUT)
				{
					kh_put(ccv_nnc_ids, ids, reindex->x.id, &ret);
					if (ret != 0)
						ccv_array_push(reverse_top, &reindex->x);
				} else {
					// This is an input, it must be represented in inputs, try to find it.
					for (j = 0; j < input_size; j++)
						if (inputs[j].id == reindex->x.id)
							break;
					assert(j < input_size); // Cannot find the inputs, error!
					input_bitmask[j >> 6] |= ((uint64_t)1 << (j & 63));
				}
				break;
			}
			case CCV_NNC_MICRO_INPUT: {
				assert(0 && "No input inserted.");
			}
		}
	}
	kh_destroy(ccv_nnc_ids, ids);
	for (i = 0; i < input_size; i++)
		{ assert((input_bitmask[i >> 6] & ((uint64_t)1 << (i & 63)))); } // Assuming they all match.
	// Third, lower each ccv_nnc_micro_io_t (except the input) op into nested loops such that we can
	// apply optimizations later.
	const int loop_count = reverse_top->rnum;
	ccv_nnc_micro_nested_loop_t* const loops = (ccv_nnc_micro_nested_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_nested_loop_t) * loop_count);
	for (i = 0; i < loop_count; i++)
	{
		const ccv_nnc_micro_io_t output = *(ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, loop_count - 1 - i);
		switch (output.type)
		{
			case CCV_NNC_MICRO_BINARY: {
				loops[i] = _ccv_nnc_micro_binary_loop(output);
				break;
			}
			case CCV_NNC_MICRO_SELECT: {
				loops[i] = _ccv_nnc_micro_select_loop(output);
				break;
			}
			case CCV_NNC_MICRO_REDUCE: {
				loops[i] = _ccv_nnc_micro_reduce_loop(output);
				break;
			}
			case CCV_NNC_MICRO_REINDEX: {
				loops[i] = _ccv_nnc_micro_reindex_loop(output);
				break;
			}
		}
	}
	ccv_nnc_micro_combine_t* const combine = (ccv_nnc_micro_combine_t*)ccmalloc(sizeof(ccv_nnc_micro_combine_t));
	combine->loop_count = loop_count;
	for (i = 0; i < reverse_top->rnum; i++)
	{
		const ccv_nnc_micro_io_t* const output = (ccv_nnc_micro_io_t*)ccv_array_get(reverse_top, i);
		if (output->type == CCV_NNC_MICRO_REINDEX)
		{
			ccv_nnc_micro_reindex_t* const reindex = (ccv_nnc_micro_reindex_t*)output->data;
			for (j = 0; j < reindex->reindex_count; j++)
				_ccv_nnc_micro_loop_index_free(&reindex->shape[j]);
		}
		if (output->data)
			ccfree(output->data);
	}
	ccv_array_free(reverse_top);
	combine->loops = loops;
	return combine;
}

static void _ccv_nnc_micro_block_variable_free(ccv_nnc_micro_loop_variable_t* const var)
{
	int i;
	for (i = 0; i < var->index_count; i++)
		_ccv_nnc_micro_loop_index_free(&var->index[i]);
}

static void _ccv_nnc_micro_block_binary_expression_free(ccv_nnc_micro_loop_binary_expression_t* const binary)
{
	switch (binary->left.type) {
		case CCV_NNC_MICRO_LOOP_BINARY_TYPE_EXPR: {
			_ccv_nnc_micro_block_binary_expression_free(binary->left.expression);
			ccfree(binary->left.expression);
			break;
		}
		case CCV_NNC_MICRO_LOOP_BINARY_TYPE_VAR: {
			_ccv_nnc_micro_block_variable_free(&binary->left.variable);
			break;
		}
	}
	switch (binary->right.type) {
		case CCV_NNC_MICRO_LOOP_BINARY_TYPE_EXPR: {
			_ccv_nnc_micro_block_binary_expression_free(binary->right.expression);
			ccfree(binary->right.expression);
			break;
		}
		case CCV_NNC_MICRO_LOOP_BINARY_TYPE_VAR: {
			_ccv_nnc_micro_block_variable_free(&binary->right.variable);
			break;
		}
	}
}

static void _ccv_nnc_micro_block_expression_free(ccv_nnc_micro_loop_expression_t* const expr)
{
	switch (expr->type) {
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR: {
			_ccv_nnc_micro_block_variable_free(&expr->variable);
			break;
		}
		case CCV_NNC_MICRO_LOOP_EXPR_TYPE_BINARY: {
			_ccv_nnc_micro_block_binary_expression_free(&expr->expression);
		}
	}
}

void ccv_nnc_micro_combine_free(ccv_nnc_micro_combine_t* const combine)
{
	const int loop_count = combine->loop_count;
	int i, j, k;
	for (i = 0; i < loop_count; i++)
	{
		const int loop_count = combine->loops[i].loop_count;
		for (j = 0; j < loop_count; j++)
		{
			for (k = 0; k < combine->loops[i].loops[j].block_count; k++)
			{
				ccv_nnc_micro_loop_block_t block = combine->loops[i].loops[j].blocks[k];
				switch (block.type) {
					case CCV_NNC_MICRO_LOOP_BLOCK_TYPE_COMPOUND: {
						_ccv_nnc_micro_block_expression_free(&block.compound.rvalue);
						break;
					}
					case CCV_NNC_MICRO_LOOP_BLOCK_TYPE_ASSIGNMENT: {
						_ccv_nnc_micro_block_variable_free(&block.assignment.lvalue);
						_ccv_nnc_micro_block_expression_free(&block.assignment.rvalue);
						break;
					}
				}
			}
			if (combine->loops[i].loops[j].blocks)
				ccfree(combine->loops[i].loops[j].blocks);
			if (combine->loops[i].loops[j].carry_overs)
				ccfree(combine->loops[i].loops[j].carry_overs);
		}
		ccfree(combine->loops[i].loops);
	}
	ccfree(combine->loops);
	ccfree(combine);
}

char* ccv_nnc_micro_combine_c(ccv_nnc_micro_combine_t* const combine)
{
	return 0;
}
