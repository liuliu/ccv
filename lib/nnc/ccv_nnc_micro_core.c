#include "ccv_nnc.h"
#include "ccv_nnc_easy.h"
#include "ccv_nnc_internal.h"
#include "ccv_internal.h"
#include "_ccv_nnc_micro.h"
#include "3rdparty/khash/khash.h"

// MARK - Level-1 API

const ccv_nnc_micro_io_vtab_t ccv_nnc_micro_io_input_isa = {};

#define GRAD(_id) (2 * (var_count) - 1 - (_id))

ccv_nnc_micro_io_t ccv_nnc_micro_input(const int dimensions)
{
	assert(dimensions <= CCV_NNC_MAX_DIM_ALLOC);
	ccv_nnc_micro_io_t input = cccalloc(1, sizeof(struct ccv_nnc_micro_io_s));
	input->isa = &ccv_nnc_micro_io_input_isa;
	input->dimensions = dimensions;
	input->id = 0;
	return input;
}
struct ccv_nnc_micro_io_grad_s {
	struct ccv_nnc_micro_io_s super;
	ccv_nnc_micro_io_t x;
};

static void _ccv_nnc_micro_grad_numbering(const ccv_nnc_micro_io_t super, const int id, const int var_count)
{
	struct ccv_nnc_micro_io_grad_s* const self = (struct ccv_nnc_micro_io_grad_s*)super;
	const int sid = self->x->id;
	self->super.id = GRAD(sid);
}

const ccv_nnc_micro_io_vtab_t ccv_nnc_micro_io_grad_isa = {
	.numbering = _ccv_nnc_micro_grad_numbering
};

ccv_nnc_micro_io_t ccv_nnc_micro_grad(const ccv_nnc_micro_io_t x)
{
	struct ccv_nnc_micro_io_grad_s* const grad = cccalloc(1, sizeof(struct ccv_nnc_micro_io_grad_s));
	grad->super.isa = &ccv_nnc_micro_io_grad_isa;
	grad->super.dimensions = x->dimensions;
	grad->super.id = 0;
	grad->x = x;
	return (ccv_nnc_micro_io_t)grad;
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
		memcpy(*name, *pos, size);
		name[0][size] = 0;
		*remain_size -= size;
		*pos += size;
		return 1;
	}
	return 0;
}

static CCV_WARN_UNUSED(ccv_nnc_micro_loop_index_term_t) _expression(const char** const pos, int* const remain_size);

static ccv_nnc_micro_loop_index_term_t _factor(const char** const pos, int* const remain_size)
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
		term.id.id = -1;
	} else if (_var(pos, remain_size, &name)) {
		term.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_UNBOUND_SCALAR;
		term.name = name;
	} else if (_accept(pos, remain_size, "(", 1)) {
		term = _expression(pos, remain_size);
		_expect(pos, remain_size, ")", 1);
	} else {
		assert(0 && "factor: syntax error");
	}
	while (_accept(pos, remain_size, " ", 1)) {}
	return term;
}

static ccv_nnc_micro_loop_index_term_t _term(const char** const pos, int* const remain_size)
{
	while (_accept(pos, remain_size, " ", 1)) {}
	ccv_nnc_micro_loop_index_term_t term = _factor(pos, remain_size);
	while (*remain_size > 0 && (pos[0][0] == '*' || pos[0][0] == '/'))
	{
		const int op = pos[0][0] == '*' ? CCV_NNC_MICRO_BINARY_OP_MUL : CCV_NNC_MICRO_BINARY_OP_DIV;
		*remain_size -= 1;
		*pos += 1;
		const ccv_nnc_micro_loop_index_term_t left = term;
		const ccv_nnc_micro_loop_index_term_t right = _factor(pos, remain_size);
		term.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_BINARY;
		term.binary = (ccv_nnc_micro_loop_index_binary_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_index_binary_t));
		term.binary->op = op;
		term.binary->left = left;
		term.binary->right = right;
	}
	while (_accept(pos, remain_size, " ", 1)) {}
	return term;
}

static ccv_nnc_micro_loop_index_term_t _expression(const char** const pos, int* const remain_size)
{
	while (_accept(pos, remain_size, " ", 1)) {}
	int prefix_op = -1;
	if (*remain_size > 0 && (pos[0][0] == '+' || pos[0][0] == '-'))
	{
		prefix_op = pos[0][0] == '+' ? CCV_NNC_MICRO_BINARY_OP_PLUS : CCV_NNC_MICRO_BINARY_OP_MINUS;
		*remain_size -= 1;
		*pos += 1;
	}
	ccv_nnc_micro_loop_index_term_t node = _term(pos, remain_size);
	while (*remain_size > 0 && (pos[0][0] == '+' || pos[0][0] == '-'))
	{
		const int op = pos[0][0] == '+' ? CCV_NNC_MICRO_BINARY_OP_PLUS : CCV_NNC_MICRO_BINARY_OP_MINUS;
		*remain_size -= 1;
		*pos += 1;
		const ccv_nnc_micro_loop_index_term_t left = node;
		const ccv_nnc_micro_loop_index_term_t right = _term(pos, remain_size);
		node.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_BINARY;
		node.binary = (ccv_nnc_micro_loop_index_binary_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_index_binary_t));
		node.binary->op = op;
		node.binary->left = left;
		node.binary->right = right;
	}
	while (_accept(pos, remain_size, " ", 1)) {}
	if (prefix_op >= 0)
	{
		ccv_nnc_micro_loop_index_binary_t* const expr = (ccv_nnc_micro_loop_index_binary_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_index_binary_t));
		expr->op = prefix_op;
		expr->left = node;
		expr->right.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_NONE;
		node.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_BINARY;
		node.binary = expr;
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
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_BINARY:
			_no_index(term.binary->left);
			_no_index(term.binary->right);
			break;
	}
}

static void _sid_to_axis_size_term(ccv_nnc_micro_loop_index_term_t* const term, const int sid)
{
	switch (term->type) {
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID:
			// Can only be axis size id. No loop index.
			if (term->id.type == CCV_NNC_MICRO_AXIS_SIZE_ID)
				term->id.id = sid;
			break;
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_BINARY:
			_sid_to_axis_size_term(&term->binary->left, sid);
			_sid_to_axis_size_term(&term->binary->right, sid);
			break;
	}
}

struct ccv_nnc_micro_io_reindex_s {
	struct ccv_nnc_micro_io_s super;
	ccv_nnc_micro_io_t s;
	ccv_nnc_micro_io_t x;
	ccv_nnc_micro_loop_index_term_t* shape;
	ccv_nnc_micro_loop_index_term_t* reindex;
};

static void _ccv_nnc_micro_reindex_numbering(const ccv_nnc_micro_io_t super, const int id, const int var_count)
{
	struct ccv_nnc_micro_io_reindex_s* const self = (struct ccv_nnc_micro_io_reindex_s*)super;
	const int sid = self->s->id;
	int i;
	for (i = 0; i < self->super.dimensions; i++)
		_sid_to_axis_size_term(&self->shape[i], sid);
	for (i = 0; i < self->x->dimensions; i++)
		_sid_to_axis_size_term(&self->reindex[i], sid);
	self->super.id = id;
}

static void _ccv_nnc_bind_scalars_in_term(ccv_nnc_micro_loop_index_term_t* const term, ccv_nnc_micro_scalar_lookup_f lookup, const void* const context)
{
	switch (term->type)
	{
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_BINARY:
			_ccv_nnc_bind_scalars_in_term(&term->binary->left, lookup, context);
			_ccv_nnc_bind_scalars_in_term(&term->binary->right, lookup, context);
			break;
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_UNBOUND_SCALAR: {
			char* const name = term->name;
			term->id.id = lookup(context, name);
			ccfree(name);
			term->id.d = 0;
			term->id.type = CCV_NNC_MICRO_SCALAR_ID;
			term->type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID;
			break;
		}
	}
}

static void _ccv_nnc_micro_reindex_bind_scalars(const ccv_nnc_micro_io_t super, ccv_nnc_micro_scalar_lookup_f lookup, const void* const context)
{
	struct ccv_nnc_micro_io_reindex_s* const self = (struct ccv_nnc_micro_io_reindex_s*)super;
	int i;
	for (i = 0; i < self->super.dimensions; i++)
		_ccv_nnc_bind_scalars_in_term(&self->shape[i], lookup, context);
	for (i = 0; i < self->x->dimensions; i++)
		_ccv_nnc_bind_scalars_in_term(&self->reindex[i], lookup, context);
}

ccv_nnc_micro_loop_index_term_t ccv_nnc_micro_loop_index_deep_copy(const ccv_nnc_micro_loop_index_term_t* const term)
{
	switch (term->type)
	{
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_BINARY: {
			ccv_nnc_micro_loop_index_term_t copy = *term;
			copy.binary = (ccv_nnc_micro_loop_index_binary_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_index_binary_t));
			*copy.binary = *term->binary;
			copy.binary->left = ccv_nnc_micro_loop_index_deep_copy(&term->binary->left);
			copy.binary->right = ccv_nnc_micro_loop_index_deep_copy(&term->binary->right);
			return copy;
		}
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_NONE:
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID:
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL:
		case CCV_NNC_MICRO_LOOP_INDEX_TYPE_UNBOUND_SCALAR:
			return *term;
	}
	return *term;
}

static CCV_WARN_UNUSED(ccv_nnc_micro_function_t) _ccv_nnc_micro_reindex_emit(const ccv_nnc_micro_io_t super)
{
	struct ccv_nnc_micro_io_reindex_s* const self = (struct ccv_nnc_micro_io_reindex_s*)super;
	const int loop_count = self->super.dimensions;
	assert(loop_count <= CCV_NNC_MAX_DIM_ALLOC);
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	int i;
	for (i = 0; i < loop_count; i++)
		loops[i] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(self->super.id, i), i);
	const ccv_nnc_micro_loop_statement_t statement = ccv_nnc_micro_loop_assignment(
		ccv_nnc_micro_loop_variable_of_tensor(self->super.id, loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)),
		ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->x->id, self->x->dimensions, self->reindex))
	);
	for (i = 0; i < self->x->dimensions; i++)
		self->reindex[i] = ccv_nnc_micro_loop_index_deep_copy(&self->reindex[i]);
	loops[loop_count - 1].statement_count = 1;
	loops[loop_count - 1].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	loops[loop_count - 1].statements[0] = statement;
	return (ccv_nnc_micro_function_t){
		.block_count = 1,
		.one_block = {
			.loop_count = loop_count,
			.loops = loops
		}
	};
}

static CCV_WARN_UNUSED(ccv_nnc_micro_function_t) _ccv_nnc_micro_reindex_emit_grad(const ccv_nnc_micro_io_t super, const int var_count)
{
	// The grad is var_count + original id.
	struct ccv_nnc_micro_io_reindex_s* const self = (struct ccv_nnc_micro_io_reindex_s*)super;
	const int reset_loop_count = self->x->dimensions;
	ccv_nnc_micro_loop_t* const reset_loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * reset_loop_count);
	// This loop reset grad to 0.
	int i;
	for (i = 0; i < reset_loop_count; i++)
		reset_loops[i] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(GRAD(self->x->id), i), i);
	const ccv_nnc_micro_loop_statement_t reset_statement = ccv_nnc_micro_loop_assignment(
		ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->x->id), reset_loop_count, ccv_nnc_micro_index_of_loops(reset_loops, reset_loop_count)),
		ccv_nnc_micro_loop_expression_of_value(0)
	);
	reset_loops[reset_loop_count - 1].statement_count = 1;
	reset_loops[reset_loop_count - 1].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	reset_loops[reset_loop_count - 1].statements[0] = reset_statement;
	const int loop_count = self->super.dimensions;
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	for (i = 0; i < loop_count; i++)
		loops[i] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(GRAD(self->super.id), i), i);
	const ccv_nnc_micro_loop_statement_t statement = ccv_nnc_micro_loop_compound_assignment_of_tensor(
		ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->x->id), self->x->dimensions, self->reindex),
		ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)))
	);
	for (i = 0; i < self->x->dimensions; i++)
		self->reindex[i] = ccv_nnc_micro_loop_index_deep_copy(&self->reindex[i]);
	loops[loop_count - 1].statement_count = 1;
	loops[loop_count - 1].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	loops[loop_count - 1].statements[0] = statement;
	ccv_nnc_micro_loop_block_t* const blocks = (ccv_nnc_micro_loop_block_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_block_t) * 2);
	blocks[0] = (ccv_nnc_micro_loop_block_t){
		.loop_count = reset_loop_count,
		.loops = reset_loops
	};
	blocks[1] = (ccv_nnc_micro_loop_block_t){
		.loop_count = loop_count,
		.loops = loops
	};
	return (ccv_nnc_micro_function_t){
		.block_count = 2,
		.blocks = blocks
	};
}

static ccv_nnc_micro_tensor_t _ccv_nnc_micro_reindex_return_shape(const ccv_nnc_micro_io_t super)
{
	struct ccv_nnc_micro_io_reindex_s* const self = (struct ccv_nnc_micro_io_reindex_s*)super;
	ccv_nnc_micro_tensor_t var = {};
	var.dimensions = self->super.dimensions;
	var.sibling = -1;
	var.input = self->x->id;
	var.shape = (ccv_nnc_micro_loop_index_term_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_index_term_t) * self->super.dimensions);
	memcpy(var.shape, self->shape, sizeof(ccv_nnc_micro_loop_index_term_t) * self->super.dimensions);
	return var;
}

static void _ccv_nnc_micro_reindex_deinit(const ccv_nnc_micro_io_t super)
{
	struct ccv_nnc_micro_io_reindex_s* const self = (struct ccv_nnc_micro_io_reindex_s*)super;
	int i;
	for (i = 0; i < self->x->dimensions; i++)
		ccv_nnc_micro_loop_index_free(&self->reindex[i]);
}

static const ccv_nnc_micro_io_vtab_t ccv_nnc_micro_io_reindex_isa = {
	.numbering = _ccv_nnc_micro_reindex_numbering,
	.bind_scalars = _ccv_nnc_micro_reindex_bind_scalars,
	.emit = _ccv_nnc_micro_reindex_emit,
	.emit_grad = _ccv_nnc_micro_reindex_emit_grad,
	.return_shape = _ccv_nnc_micro_reindex_return_shape,
	.deinit = _ccv_nnc_micro_reindex_deinit
};

ccv_nnc_micro_io_t ccv_nnc_micro_reindex(const char* const* const shape, const int shape_count, const ccv_nnc_micro_io_t s, const char* const* const reindex, const int reindex_count, const ccv_nnc_micro_io_t x)
{
	assert(shape_count <= CCV_NNC_MAX_DIM_ALLOC);
	assert(reindex_count <= CCV_NNC_MAX_DIM_ALLOC);
	assert(reindex_count == x->dimensions);
	int i;
	struct ccv_nnc_micro_io_reindex_s* const self = (struct ccv_nnc_micro_io_reindex_s*)cccalloc(1, sizeof(struct ccv_nnc_micro_io_reindex_s) + sizeof(ccv_nnc_micro_loop_index_term_t) * (shape_count + reindex_count));
	self->super.isa = &ccv_nnc_micro_io_reindex_isa;
	self->super.dimensions = shape_count;
	self->super.id = 0;
	self->super.inputs = &self->s;
	self->super.input_size = 2;
	self->s = s;
	self->x = x;
	self->shape = (ccv_nnc_micro_loop_index_term_t*)(self + 1);
	self->reindex = self->shape + shape_count;
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
	for (i = 0; i < shape_count; i++)
	{
		int remain_size = strlen(shape[i]);
		const char* pos = shape[i];
		ccv_nnc_micro_loop_index_term_t term = _expression(&pos, &remain_size);
		_no_index(term); // Make sure this is not index, no loop index.
		self->shape[i] = term;
	}
	// Parse reindex.
	for (i = 0; i < reindex_count; i++)
	{
		int remain_size = strlen(reindex[i]);
		const char* pos = reindex[i];
		self->reindex[i] = _expression(&pos, &remain_size);
	}
	return (ccv_nnc_micro_io_t)self;
}

struct ccv_nnc_micro_io_unary_s {
	struct ccv_nnc_micro_io_s super;
	uint32_t unary_op;
	ccv_nnc_micro_io_t x;
};

static CCV_WARN_UNUSED(ccv_nnc_micro_function_t) _ccv_nnc_micro_unary_emit(const ccv_nnc_micro_io_t super)
{
	struct ccv_nnc_micro_io_unary_s* const self = (struct ccv_nnc_micro_io_unary_s*)super;
	const int loop_count = self->super.dimensions;
	assert(self->x->dimensions == loop_count);
	assert(loop_count <= CCV_NNC_MAX_DIM_ALLOC);
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	int i;
	for (i = 0; i < loop_count; i++)
		loops[i] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(self->super.id, i), i);
	const ccv_nnc_micro_loop_statement_t statement = ccv_nnc_micro_loop_assignment(
		ccv_nnc_micro_loop_variable_of_tensor(self->super.id, loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)),
		ccv_nnc_micro_loop_expression_of_unary(
			self->unary_op,
			ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->x->id, loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)))
		)
	);
	loops[loop_count - 1].statement_count = 1;
	loops[loop_count - 1].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	loops[loop_count - 1].statements[0] = statement;
	return (ccv_nnc_micro_function_t){
		.block_count = 1,
		.one_block = {
			.loop_count = loop_count,
			.loops = loops
		}
	};
}

static CCV_WARN_UNUSED(ccv_nnc_micro_function_t) _ccv_nnc_micro_unary_emit_grad(const ccv_nnc_micro_io_t super, const int var_count)
{
	struct ccv_nnc_micro_io_unary_s* const self = (struct ccv_nnc_micro_io_unary_s*)super;
	const int loop_count = self->super.dimensions;
	assert(self->x->dimensions == loop_count);
	assert(loop_count <= CCV_NNC_MAX_DIM_ALLOC);
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	int i;
	for (i = 0; i < loop_count; i++)
		loops[i] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(GRAD(self->super.id), i), i);
	ccv_nnc_micro_loop_statement_t statement;
	switch (self->unary_op)
	{
		case CCV_NNC_MICRO_UNARY_OP_NEG:
			statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->x->id), loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_unary(
					CCV_NNC_MICRO_UNARY_OP_NEG,
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)))
				)
			);
			break;
		case CCV_NNC_MICRO_UNARY_OP_EXP:
			statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->x->id), loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_binary(
					CCV_NNC_MICRO_BINARY_OP_MUL,
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->super.id, loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count))),
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)))
				)
			);
			break;
		case CCV_NNC_MICRO_UNARY_OP_LOG:
			statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->x->id), loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_binary(
					CCV_NNC_MICRO_BINARY_OP_DIV,
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count))),
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->x->id, loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)))
				)
			);
			break;
	}
	loops[loop_count - 1].statement_count = 1;
	loops[loop_count - 1].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	loops[loop_count - 1].statements[0] = statement;
	return (ccv_nnc_micro_function_t){
		.block_count = 1,
		.one_block = {
			.loop_count = loop_count,
			.loops = loops
		}
	};
}

static ccv_nnc_micro_tensor_t _ccv_nnc_micro_unary_return_shape(const ccv_nnc_micro_io_t super)
{
	struct ccv_nnc_micro_io_unary_s* const self = (struct ccv_nnc_micro_io_unary_s*)super;
	ccv_nnc_micro_tensor_t var = {};
	var.dimensions = self->super.dimensions;
	var.input = self->x->id;
	var.sibling = -1;
	return var;
}

static const ccv_nnc_micro_io_vtab_t ccv_nnc_micro_io_unary_isa = {
	.emit = _ccv_nnc_micro_unary_emit,
	.emit_grad = _ccv_nnc_micro_unary_emit_grad,
	.return_shape = _ccv_nnc_micro_unary_return_shape
};

ccv_nnc_micro_io_t ccv_nnc_micro_unary(const uint32_t op, const ccv_nnc_micro_io_t x)
{
	struct ccv_nnc_micro_io_unary_s* const self = (struct ccv_nnc_micro_io_unary_s*)cccalloc(1, sizeof(struct ccv_nnc_micro_io_unary_s));
	self->super.isa = &ccv_nnc_micro_io_unary_isa;
	self->super.dimensions = x->dimensions;
	self->super.id = 0;
	self->super.inputs = &self->x;
	self->super.input_size = 1;
	self->unary_op = op;
	self->x = x;
	return (ccv_nnc_micro_io_t)self;
}

struct ccv_nnc_micro_io_binary_s {
	struct ccv_nnc_micro_io_s super;
	uint32_t binary_op;
	ccv_nnc_micro_io_t left;
	ccv_nnc_micro_io_t right;
};

static CCV_WARN_UNUSED(ccv_nnc_micro_function_t) _ccv_nnc_micro_binary_emit(const ccv_nnc_micro_io_t super)
{
	struct ccv_nnc_micro_io_binary_s* const self = (struct ccv_nnc_micro_io_binary_s*)super;
	const int loop_count = self->super.dimensions;
	assert(self->left->dimensions == loop_count);
	assert(self->right->dimensions == loop_count);
	assert(loop_count <= CCV_NNC_MAX_DIM_ALLOC);
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	int i;
	for (i = 0; i < loop_count; i++)
		loops[i] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(self->super.id, i), i);
	const ccv_nnc_micro_loop_statement_t statement = ccv_nnc_micro_loop_assignment(
		ccv_nnc_micro_loop_variable_of_tensor(self->super.id, loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)),
		ccv_nnc_micro_loop_expression_of_binary(
			self->binary_op,
			ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->left->id, loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count))),
			ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->right->id, loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)))
		)
	);
	loops[loop_count - 1].statement_count = 1;
	loops[loop_count - 1].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	loops[loop_count - 1].statements[0] = statement;
	return (ccv_nnc_micro_function_t){
		.block_count = 1,
		.one_block = {
			.loop_count = loop_count,
			.loops = loops
		}
	};
}

static CCV_WARN_UNUSED(ccv_nnc_micro_function_t) _ccv_nnc_micro_binary_emit_grad(const ccv_nnc_micro_io_t super, const int var_count)
{
	struct ccv_nnc_micro_io_binary_s* const self = (struct ccv_nnc_micro_io_binary_s*)super;
	const int loop_count = self->super.dimensions;
	assert(self->left->dimensions == loop_count);
	assert(self->right->dimensions == loop_count);
	assert(loop_count <= CCV_NNC_MAX_DIM_ALLOC);
	int i;
	ccv_nnc_micro_loop_t* const left_loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	for (i = 0; i < loop_count; i++)
		left_loops[i] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(GRAD(self->super.id), i), i);
	ccv_nnc_micro_loop_t* const right_loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	for (i = 0; i < loop_count; i++)
		right_loops[i] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(GRAD(self->super.id), i), i);
	ccv_nnc_micro_loop_statement_t left_statement;
	ccv_nnc_micro_loop_statement_t right_statement;
	switch (self->binary_op)
	{
		case CCV_NNC_MICRO_BINARY_OP_DIV:
			left_statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->left->id), loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_binary(
					CCV_NNC_MICRO_BINARY_OP_DIV,
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count))),
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->right->id, loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)))
				)
			);
			right_statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->right->id), loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_binary(
					CCV_NNC_MICRO_BINARY_OP_MUL,
					ccv_nnc_micro_loop_expression_of_binary(
						CCV_NNC_MICRO_BINARY_OP_DIV,
						ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->super.id, loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count))),
						ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->right->id, loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)))
					),
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)))
				)
			);
			break;
		case CCV_NNC_MICRO_BINARY_OP_MUL:
			left_statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->left->id), loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_binary(
					CCV_NNC_MICRO_BINARY_OP_MUL,
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count))),
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->right->id, loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)))
				)
			);
			right_statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->right->id), loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_binary(
					CCV_NNC_MICRO_BINARY_OP_MUL,
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->left->id, loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count))),
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)))
				)
			);
			break;
		case CCV_NNC_MICRO_BINARY_OP_PLUS:
			left_statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->left->id), loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)))
			);
			right_statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->right->id), loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)))
			);
			break;
		case CCV_NNC_MICRO_BINARY_OP_MINUS:
			left_statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->left->id), loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)))
			);
			right_statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->right->id), loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_unary(
					CCV_NNC_MICRO_UNARY_OP_NEG,
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)))
				)
			);
			break;
		case CCV_NNC_MICRO_BINARY_OP_MIN:
			left_statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->left->id), loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_ternary(
					ccv_nnc_micro_loop_expression_of_binary(CCV_NNC_MICRO_BINARY_OP_LESS_THAN,
						ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->right->id, loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count))),
						ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->left->id, loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)))
					),
					ccv_nnc_micro_loop_expression_of_value(0),
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)))
				)
			);
			right_statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->right->id), loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_ternary(
					ccv_nnc_micro_loop_expression_of_binary(CCV_NNC_MICRO_BINARY_OP_LESS_THAN,
						ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->left->id, loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count))),
						ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->right->id, loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)))
					),
					ccv_nnc_micro_loop_expression_of_value(0),
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)))
				)
			);
			break;
		case CCV_NNC_MICRO_BINARY_OP_MAX:
			left_statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->left->id), loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_ternary(
					ccv_nnc_micro_loop_expression_of_binary(CCV_NNC_MICRO_BINARY_OP_LESS_THAN,
						ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->left->id, loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count))),
						ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->right->id, loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)))
					),
					ccv_nnc_micro_loop_expression_of_value(0),
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(left_loops, loop_count)))
				)
			);
			right_statement = ccv_nnc_micro_loop_assignment(
				ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->right->id), loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)),
				ccv_nnc_micro_loop_expression_of_ternary(
					ccv_nnc_micro_loop_expression_of_binary(CCV_NNC_MICRO_BINARY_OP_LESS_THAN,
						ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->right->id, loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count))),
						ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->left->id, loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)))
					),
					ccv_nnc_micro_loop_expression_of_value(0),
					ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(right_loops, loop_count)))
				)
			);
			break;
	}
	left_loops[loop_count - 1].statement_count = 1;
	left_loops[loop_count - 1].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	left_loops[loop_count - 1].statements[0] = left_statement;
	right_loops[loop_count - 1].statement_count = 1;
	right_loops[loop_count - 1].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	right_loops[loop_count - 1].statements[0] = right_statement;
	ccv_nnc_micro_loop_block_t* const blocks = (ccv_nnc_micro_loop_block_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_block_t) * 2);
	blocks[0] = (ccv_nnc_micro_loop_block_t){
		.loop_count = loop_count,
		.loops = left_loops
	};
	blocks[1] = (ccv_nnc_micro_loop_block_t){
		.loop_count = loop_count,
		.loops = right_loops
	};
	return (ccv_nnc_micro_function_t){
		.block_count = 2,
		.blocks = blocks
	};
}

static ccv_nnc_micro_tensor_t _ccv_nnc_micro_binary_return_shape(const ccv_nnc_micro_io_t super)
{
	struct ccv_nnc_micro_io_binary_s* const self = (struct ccv_nnc_micro_io_binary_s*)super;
	ccv_nnc_micro_tensor_t var = {};
	var.dimensions = self->super.dimensions;
	var.input = self->left->id;
	var.sibling = self->right->id;
	return var;
}

static const ccv_nnc_micro_io_vtab_t ccv_nnc_micro_io_binary_isa = {
	.emit = _ccv_nnc_micro_binary_emit,
	.emit_grad = _ccv_nnc_micro_binary_emit_grad,
	.return_shape = _ccv_nnc_micro_binary_return_shape
};

ccv_nnc_micro_io_t ccv_nnc_micro_binary(const uint32_t op, const ccv_nnc_micro_io_t x, const ccv_nnc_micro_io_t y)
{
	struct ccv_nnc_micro_io_binary_s* const self = (struct ccv_nnc_micro_io_binary_s*)cccalloc(1, sizeof(struct ccv_nnc_micro_io_binary_s));
	self->super.isa = &ccv_nnc_micro_io_binary_isa;
	self->super.dimensions = x->dimensions;
	self->super.id = 0;
	self->super.inputs = &self->left;
	self->super.input_size = 2;
	self->binary_op = op;
	self->left = x;
	self->right = y;
	assert(x->dimensions == y->dimensions);
	return (ccv_nnc_micro_io_t)self;
}

struct ccv_nnc_micro_io_reduce_s {
	struct ccv_nnc_micro_io_s super;
	uint32_t reduce_op;
	int axis_count;
	ccv_nnc_micro_io_t x;
	int axis[1];
};

static CCV_WARN_UNUSED(ccv_nnc_micro_function_t) _ccv_nnc_micro_reduce_emit(const ccv_nnc_micro_io_t super)
{
	struct ccv_nnc_micro_io_reduce_s* const self = (struct ccv_nnc_micro_io_reduce_s*)super;
	const int loop_count = self->super.dimensions;
	assert(self->x->dimensions == loop_count);
	// If axis_count == loop_count, we need extra loop to reduce.
	int has_extra_loop = (self->axis_count == loop_count);
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * (loop_count + has_extra_loop));
	int i, j, k;
	int8_t reduce_axis[loop_count];
	memset(reduce_axis, 0, sizeof(int8_t) * loop_count);
	for (i = 0; i < self->axis_count; i++)
		reduce_axis[self->axis[i]] = 1;
	j = 0;
	// If loop_count == reduce_axis_count, we have extra loop for carried variables and blocks.
	if (has_extra_loop)
	{
		loops[0] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_value(1), 0);
		k = 1;
	} else
		k = loop_count - self->axis_count;
	for (i = 0; i < loop_count; i++)
		if (reduce_axis[i])
		{
			loops[k] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(self->x->id, i), i + has_extra_loop);
			++k;
		} else {
			loops[j] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(self->x->id, i), i + has_extra_loop);
			++j;
		}
	const int carried_loop_idx = has_extra_loop ? 0 : loop_count - self->axis_count - 1;
	loops[carried_loop_idx].carried_count = 1;
	loops[carried_loop_idx].carrieds = (ccv_nnc_micro_loop_carried_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_carried_t));
	loops[carried_loop_idx].carrieds[0] = ccv_nnc_micro_loop_carried(self->reduce_op, 0);
	j = 0;
	k = has_extra_loop ? 1 : loop_count - self->axis_count;
	// If loop_count == reduce_axis_count, we have extra loop for carrieds and block.
	ccv_nnc_micro_loop_index_term_t index[CCV_NNC_MAX_DIM_ALLOC];
	for (i = 0; i < loop_count; i++)
		if (reduce_axis[i])
		{
			index[i] = ccv_nnc_micro_index_of_id(loops[k].id);
			++k;
		} else {
			index[i] = ccv_nnc_micro_index_of_id(loops[j].id);
			++j;
		}
	ccv_nnc_micro_loop_statement_t statement = ccv_nnc_micro_loop_compound_assignment_of_id(
		loops[carried_loop_idx].carrieds[0].id,
		ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->x->id, loop_count, index))
	);
	loops[carried_loop_idx + self->axis_count].statement_count = 1;
	loops[carried_loop_idx + self->axis_count].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	loops[carried_loop_idx + self->axis_count].statements[0] = statement;
	j = 0;
	for (i = 0; i < loop_count; i++)
		if (reduce_axis[i])
			index[i] = ccv_nnc_micro_index_of_value(0);
		else {
			index[i] = ccv_nnc_micro_index_of_id(loops[j].id);
			++j;
		}
	statement = ccv_nnc_micro_loop_assignment(
		ccv_nnc_micro_loop_variable_of_tensor(self->super.id, loop_count, index),
		ccv_nnc_micro_loop_expression_of_id(loops[carried_loop_idx].carrieds[0].id)
	);
	loops[carried_loop_idx].statement_count = 1;
	loops[carried_loop_idx].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	loops[carried_loop_idx].statements[0] = statement;
	return (ccv_nnc_micro_function_t){
		.block_count = 1,
		.one_block = {
			.carried_count = 1,
			.loop_count = loop_count + has_extra_loop,
			.loops = loops
		}
	};
}

static CCV_WARN_UNUSED(ccv_nnc_micro_function_t) _ccv_nnc_micro_reduce_emit_grad(const ccv_nnc_micro_io_t super, const int var_count)
{
	struct ccv_nnc_micro_io_reduce_s* const self = (struct ccv_nnc_micro_io_reduce_s*)super;
	assert(self->reduce_op == CCV_NNC_MICRO_REDUCE_OP_SUM); // I haven't figure out how to do mean without add additional opcode.
	const int loop_count = self->super.dimensions;
	assert(self->x->dimensions == loop_count);
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	int i, j, k;
	int8_t reduce_axis[loop_count];
	memset(reduce_axis, 0, sizeof(int8_t) * loop_count);
	for (i = 0; i < self->axis_count; i++)
		reduce_axis[self->axis[i]] = 1;
	j = 0;
	k = loop_count - self->axis_count;
	for (i = 0; i < loop_count; i++)
		if (reduce_axis[i])
		{
			loops[k] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(GRAD(self->x->id), i), i);
			++k;
		} else {
			loops[j] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(GRAD(self->x->id), i), i);
			++j;
		}
	j = 0;
	k = loop_count - self->axis_count;
	// If loop_count == reduce_axis_count, we have extra loop for carrieds and block.
	ccv_nnc_micro_loop_index_term_t index[CCV_NNC_MAX_DIM_ALLOC];
	for (i = 0; i < loop_count; i++)
		if (reduce_axis[i])
		{
			index[i] = ccv_nnc_micro_index_of_id(loops[k].id);
			++k;
		} else {
			index[i] = ccv_nnc_micro_index_of_id(loops[j].id);
			++j;
		}
	j = 0;
	ccv_nnc_micro_loop_index_term_t reduced_index[CCV_NNC_MAX_DIM_ALLOC];
	for (i = 0; i < loop_count; i++)
		if (reduce_axis[i])
			reduced_index[i] = ccv_nnc_micro_index_of_value(0);
		else {
			reduced_index[i] = ccv_nnc_micro_index_of_id(loops[j].id);
			++j;
		}
	ccv_nnc_micro_loop_statement_t statement = ccv_nnc_micro_loop_assignment(
		ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->x->id), loop_count, index),
		ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, reduced_index))
	);
	loops[loop_count - 1].statement_count = 1;
	loops[loop_count - 1].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	loops[loop_count - 1].statements[0] = statement;
	return (ccv_nnc_micro_function_t){
		.block_count = 1,
		.one_block = {
			.carried_count = 1,
			.loop_count = loop_count,
			.loops = loops
		}
	};
}

static ccv_nnc_micro_tensor_t _ccv_nnc_micro_reduce_return_shape(const ccv_nnc_micro_io_t super)
{
	struct ccv_nnc_micro_io_reduce_s* const self = (struct ccv_nnc_micro_io_reduce_s*)super;
	ccv_nnc_micro_tensor_t var = {};
	var.dimensions = self->super.dimensions;
	var.input = self->x->id;
	var.sibling = -1;
	var.shape = (ccv_nnc_micro_loop_index_term_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_index_term_t) * self->super.dimensions);
	int i;
	for (i = 0; i < self->super.dimensions; i++)
		var.shape[i] = ccv_nnc_micro_index_of_axis_size(self->x->id, i);
	for (i = 0; i < self->axis_count; i++)
		var.shape[self->axis[i]] = ccv_nnc_micro_index_of_value(1);
	return var;
}

static const ccv_nnc_micro_io_vtab_t ccv_nnc_micro_io_reduce_isa = {
	.emit = _ccv_nnc_micro_reduce_emit,
	.emit_grad = _ccv_nnc_micro_reduce_emit_grad,
	.return_shape = _ccv_nnc_micro_reduce_return_shape
};

ccv_nnc_micro_io_t ccv_nnc_micro_reduce(const uint8_t op, const int* const axis, const int axis_count, const ccv_nnc_micro_io_t x)
{
	struct ccv_nnc_micro_io_reduce_s* const self = (struct ccv_nnc_micro_io_reduce_s*)cccalloc(1, sizeof(struct ccv_nnc_micro_io_reduce_s) + sizeof(int) * (axis_count - 1));
	self->super.isa = &ccv_nnc_micro_io_reduce_isa;
	self->super.dimensions = x->dimensions;
	self->super.id = 0;
	self->super.inputs = &self->x;
	self->super.input_size = 1;
	self->reduce_op = op;
	self->x = x;
	self->axis_count = axis_count;
	assert(axis_count <= x->dimensions);
	int i;
	for (i = 0; i < axis_count; i++)
	{ assert(axis[i] < x->dimensions); }
	memcpy(self->axis, axis, sizeof(int) * axis_count);
	return (ccv_nnc_micro_io_t)self;
}

struct ccv_nnc_micro_io_select_s {
	struct ccv_nnc_micro_io_s super;
	int axis;
	ccv_nnc_micro_io_t x;
	ccv_nnc_micro_io_t index;
};

static CCV_WARN_UNUSED(ccv_nnc_micro_function_t) _ccv_nnc_micro_select_emit(const ccv_nnc_micro_io_t super)
{
	struct ccv_nnc_micro_io_select_s* const self = (struct ccv_nnc_micro_io_select_s*)super;
	const int loop_count = self->super.dimensions;
	assert(self->x->dimensions == loop_count);
	assert(self->index->dimensions == loop_count);
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	int i;
	for (i = 0; i < loop_count; i++)
	{
		if (i == self->axis)
			loops[i] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_value(1), i);
		else
			loops[i] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(self->super.id, i), i);
	}
	ccv_nnc_micro_loop_index_term_t index[CCV_NNC_MAX_DIM_ALLOC];
	for (i = 0; i < loop_count; i++)
	{
		if (i == self->axis)
			index[i] = ccv_nnc_micro_index_of_id(ccv_nnc_micro_id_of_tensor(self->index->id));
		else
			index[i] = ccv_nnc_micro_index_of_id(loops[i].id);
	}
	const ccv_nnc_micro_loop_statement_t statement = ccv_nnc_micro_loop_assignment(
		ccv_nnc_micro_loop_variable_of_tensor(self->super.id, loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)),
		ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(self->x->id, loop_count, index))
	);
	loops[loop_count - 1].statement_count = 1;
	loops[loop_count - 1].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	loops[loop_count - 1].statements[0] = statement;
	return (ccv_nnc_micro_function_t){
		.block_count = 1,
		.one_block = {
			.loop_count = loop_count,
			.loops = loops
		}
	};
}

static CCV_WARN_UNUSED(ccv_nnc_micro_function_t) _ccv_nnc_micro_select_emit_grad(const ccv_nnc_micro_io_t super, const int var_count)
{
	struct ccv_nnc_micro_io_select_s* const self = (struct ccv_nnc_micro_io_select_s*)super;
	const int loop_count = self->super.dimensions;
	assert(self->x->dimensions == loop_count);
	assert(self->index->dimensions == loop_count);
	ccv_nnc_micro_loop_t* const reset_loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	int i;
	for (i = 0; i < loop_count; i++)
		reset_loops[i] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(GRAD(self->x->id), i), i);
	const ccv_nnc_micro_loop_statement_t reset_statement = ccv_nnc_micro_loop_assignment(
		ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->x->id), loop_count, ccv_nnc_micro_index_of_loops(reset_loops, loop_count)),
		ccv_nnc_micro_loop_expression_of_value(0)
	);
	reset_loops[loop_count - 1].statement_count = 1;
	reset_loops[loop_count - 1].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	reset_loops[loop_count - 1].statements[0] = reset_statement;
	ccv_nnc_micro_loop_t* const loops = (ccv_nnc_micro_loop_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_t) * loop_count);
	for (i = 0; i < loop_count; i++)
		loops[i] = ccv_nnc_micro_for_in(ccv_nnc_micro_index_of_value(0), ccv_nnc_micro_index_of_axis_size(GRAD(self->x->id), i), i);
	ccv_nnc_micro_loop_index_term_t index[CCV_NNC_MAX_DIM_ALLOC];
	for (i = 0; i < loop_count; i++)
	{
		if (i == self->axis)
			index[i] = ccv_nnc_micro_index_of_id(ccv_nnc_micro_id_of_tensor(self->index->id));
		else
			index[i] = ccv_nnc_micro_index_of_id(loops[i].id);
	}
	// This is only for x, nothing for index.
	const ccv_nnc_micro_loop_statement_t statement = ccv_nnc_micro_loop_compound_assignment_of_tensor(
		ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->x->id), loop_count, index),
		ccv_nnc_micro_loop_expression_of_variable(ccv_nnc_micro_loop_variable_of_tensor(GRAD(self->super.id), loop_count, ccv_nnc_micro_index_of_loops(loops, loop_count)))
	);
	loops[loop_count - 1].statement_count = 1;
	loops[loop_count - 1].statements = (ccv_nnc_micro_loop_statement_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_statement_t));
	loops[loop_count - 1].statements[0] = statement;
	ccv_nnc_micro_loop_block_t* const blocks = (ccv_nnc_micro_loop_block_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_block_t) * 2);
	blocks[0] = (ccv_nnc_micro_loop_block_t){
		.loop_count = loop_count,
		.loops = reset_loops
	};
	blocks[1] = (ccv_nnc_micro_loop_block_t){
		.loop_count = loop_count,
		.loops = loops
	};
	return (ccv_nnc_micro_function_t){
		.block_count = 2,
		.blocks = blocks
	};
}

static ccv_nnc_micro_tensor_t _ccv_nnc_micro_select_return_shape(const ccv_nnc_micro_io_t super)
{
	struct ccv_nnc_micro_io_select_s* const self = (struct ccv_nnc_micro_io_select_s*)super;
	ccv_nnc_micro_tensor_t var = {};
	var.dimensions = self->super.dimensions;
	var.input = self->x->id;
	var.sibling = -1;
	var.shape = (ccv_nnc_micro_loop_index_term_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_index_term_t) * self->super.dimensions);
	int i;
	for (i = 0; i < self->super.dimensions; i++)
	{
		if (i != self->axis)
			var.shape[i] = ccv_nnc_micro_index_of_axis_size(self->x->id, i);
		else
			var.shape[i] = ccv_nnc_micro_index_of_value(1);
	}
	return var;
}

static const ccv_nnc_micro_io_vtab_t ccv_nnc_micro_io_select_isa = {
	.emit = _ccv_nnc_micro_select_emit,
	.emit_grad = _ccv_nnc_micro_select_emit_grad,
	.return_shape = _ccv_nnc_micro_select_return_shape
};

ccv_nnc_micro_io_t ccv_nnc_micro_select(const int axis, const ccv_nnc_micro_io_t x, const ccv_nnc_micro_io_t index)
{
	struct ccv_nnc_micro_io_select_s* const self = (struct ccv_nnc_micro_io_select_s*)cccalloc(1, sizeof(struct ccv_nnc_micro_io_select_s));
	self->super.isa = &ccv_nnc_micro_io_select_isa;
	self->super.dimensions = x->dimensions;
	self->super.id = 0;
	self->super.inputs = &self->x;
	self->super.input_size = 2;
	self->x = x;
	self->index = index;
	self->axis = axis;
	assert(axis <= CCV_NNC_MAX_DIM_ALLOC);
	return (ccv_nnc_micro_io_t)self;
}
