/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_micro_internal_h
#define GUARD_ccv_nnc_micro_internal_h

#include "ccv_nnc.h"

enum {
	CCV_NNC_MICRO_LOOP_ID,
	CCV_NNC_MICRO_LOOP_CARRIED_ID,
	CCV_NNC_MICRO_AXIS_SIZE_ID,
	CCV_NNC_MICRO_TENSOR_ID,
	CCV_NNC_MICRO_SCALAR_ID,
};

typedef struct {
	uint8_t type;
	uint8_t d; // Only used for axis_size, identify which axis for a tensor.
	uint16_t id;
} ccv_nnc_micro_id_t;

enum {
	CCV_NNC_MICRO_LOOP_INDEX_TYPE_NONE,
	CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID,
	CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL,
	CCV_NNC_MICRO_LOOP_INDEX_TYPE_BINARY,
	CCV_NNC_MICRO_LOOP_INDEX_TYPE_UNBOUND_SCALAR, // Unbounded scalar variable, shouldn't be there after fully-evaluated.
};

typedef struct ccv_nnc_micro_loop_index_binary_s ccv_nnc_micro_loop_index_binary_t;

typedef struct {
	int type;
	union {
		char* name; // binding variable name.
		ccv_nnc_micro_id_t id;
		int immediate_value;
		ccv_nnc_micro_loop_index_binary_t* binary;
	};
} ccv_nnc_micro_loop_index_term_t;

struct ccv_nnc_micro_loop_index_binary_s {
	int op;
	ccv_nnc_micro_loop_index_term_t left;
	ccv_nnc_micro_loop_index_term_t right;
};

typedef struct {
	ccv_nnc_micro_id_t id;
	int index_count;
	int no_check_bound[CCV_NNC_MAX_DIM_ALLOC];
	ccv_nnc_micro_loop_index_term_t index[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_micro_loop_variable_t;

enum {
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_ID,
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAL,
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR,
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_UNARY,
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_BINARY,
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_TERNAY,
};

typedef struct ccv_nnc_micro_loop_expression_s ccv_nnc_micro_loop_expression_t;

typedef struct {
	int unary_op;
	ccv_nnc_micro_loop_expression_t* x;
} ccv_nnc_micro_loop_unary_t;

typedef struct {
	int binary_op;
	ccv_nnc_micro_loop_expression_t* left;
	ccv_nnc_micro_loop_expression_t* right;
} ccv_nnc_micro_loop_binary_t;

typedef struct {
	ccv_nnc_micro_loop_expression_t* pivot; // If it is 0, choose left, otherwise choose right.
	ccv_nnc_micro_loop_expression_t* left;
	ccv_nnc_micro_loop_expression_t* right;
} ccv_nnc_micro_loop_ternary_t;

struct ccv_nnc_micro_loop_expression_s  {
	int type;
	union {
		ccv_nnc_micro_id_t id; // If this is a compound assignment, the id to that.
		ccv_nnc_micro_scalar_t immediate_value;
		ccv_nnc_micro_loop_variable_t variable;
		ccv_nnc_micro_loop_unary_t unary;
		ccv_nnc_micro_loop_binary_t binary;
		ccv_nnc_micro_loop_ternary_t ternary;
	};
};

typedef struct {
	ccv_nnc_micro_loop_variable_t lvalue;
	ccv_nnc_micro_loop_expression_t rvalue;
} ccv_nnc_micro_loop_assignment_t;


typedef struct  {
	int type;
	union {
		ccv_nnc_micro_id_t id; // If this is a compound assignment, the id to that.
		ccv_nnc_micro_loop_variable_t variable; // This only implies PLUS at the moment.
	};
} ccv_nnc_micro_loop_compound_assignment_lvalue_t;

typedef struct {
	ccv_nnc_micro_loop_compound_assignment_lvalue_t lvalue; // It shouldn't be unary or binary, only id or variable.
	ccv_nnc_micro_loop_expression_t rvalue;
} ccv_nnc_micro_loop_compound_assignment_t;

enum {
	CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_ASSIGNMENT,
	CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_COMPOUND_ASSIGNMENT,
};

// A generic statement within a loop.
// For our purpose, there will be two types of generic statement:
// an assignment statement (for tensors).
// an compound assignment statement (for loop carry overs).
typedef struct {
	int type;
	union {
		ccv_nnc_micro_loop_assignment_t assignment;
		ccv_nnc_micro_loop_compound_assignment_t compound_assignment;
	};
} ccv_nnc_micro_loop_statement_t;

typedef struct {
	ccv_nnc_micro_id_t id;
} ccv_nnc_micro_loop_carried_t; // The accumulating register.

// A loop is identified with a loop counter id, some blocks inside this loop, some carry overs within
// this loop and can be used outside of this loop.
// If this loop has another loop nested (represented as the next one in the ccv_nnc_micro_nested_loop_t)
// all blocks are executed after the nested loop finished.
typedef struct {
	ccv_nnc_micro_id_t id; // Loop counter's id, this will be used for indexing.
	int carried_count;
	int statement_count;
	ccv_nnc_micro_loop_index_term_t start_index;
	ccv_nnc_micro_loop_index_term_t end_index;
	ccv_nnc_micro_loop_carried_t* carrieds;
	ccv_nnc_micro_loop_statement_t* statements;
} ccv_nnc_micro_loop_t;

// A loop block contains many loops within each other.
typedef struct {
	int carried_count;
	int loop_count;
	ccv_nnc_micro_loop_t* loops;
} ccv_nnc_micro_loop_block_t;

typedef struct {
	int input; // The one it derives its shape from. If shape is nullptr, it has the same shape as input. -1 means it is an input.
	int sibling; // The sibling that has the same shape.
	int dimensions;
	int no_alloc; // No need to allocate this tensor.
	ccv_nnc_micro_loop_index_term_t* shape;
} ccv_nnc_micro_tensor_t;

// A function contains a list of loop blocks that will be executed serially.
// It also contains references to its dependencies so a function knows its inputs / outputs.
typedef struct {
	int block_count;
	union {
		ccv_nnc_micro_loop_block_t* blocks; // Heap-allocated blocks.
		ccv_nnc_micro_loop_block_t one_block; // In-place block to optimize memory allocation for one block cases.
	};
} ccv_nnc_micro_function_t;

typedef struct {
	int input_size; // Size of inputs.
	int output_size; // Size of outputs.
	// Combined ops only have global vars, there is no local vars. All vars are tensors.
	int var_count;
	// loops are our constructs of IR ops really. It is hierarchical.
	int function_count;
	int* inputs;
	int* outputs;
	ccv_nnc_micro_tensor_t* vars;
	ccv_nnc_micro_function_t* functions;
} ccv_nnc_micro_program_t;

// A combined op is constructed with many nested loops. These loops may have data dependencies
// between each other, but they are ordered in topological order to make sure one is finished
// after the another.
struct ccv_nnc_micro_combine_s {
	int parameter_size; // Size of parameters.
	ccv_nnc_micro_program_t forward;
	ccv_nnc_micro_program_t backward;
};

typedef uint32_t(*ccv_nnc_micro_scalar_lookup_f)(const void* const context, const char* const name);

/**
 * This is the virtual table for micro op.
 */
struct ccv_nnc_micro_io_vtab_s {
	void (*bind_scalars)(const ccv_nnc_micro_io_t self, ccv_nnc_micro_scalar_lookup_f lookup, const void* const context); /**< Bind scalar name to a scoped id. */
	void (*numbering)(const ccv_nnc_micro_io_t self, const int id, const int var_count); /**< Assign id to the output of this micro op. */
	ccv_nnc_micro_function_t (*emit)(const ccv_nnc_micro_io_t self); /**< Emit instructions for this micro op. */
	ccv_nnc_micro_function_t (*emit_grad)(const ccv_nnc_micro_io_t self, const int var_count); /**< Emit backward instructions for this micro op. */
	ccv_nnc_micro_tensor_t (*return_shape)(const ccv_nnc_micro_io_t self); /**< The shape of the returned tensor. */
	void (*deinit)(const ccv_nnc_micro_io_t self); /**< Deinit this micro io. */
};

extern const ccv_nnc_micro_io_vtab_t ccv_nnc_micro_io_input_isa;
extern const ccv_nnc_micro_io_vtab_t ccv_nnc_micro_io_grad_isa;

#define CCV_NNC_IS_MICRO_IO_INPUT(x) ((x)->isa == &ccv_nnc_micro_io_input_isa)
#define CCV_NNC_IS_MICRO_IO_GRAD(x) ((x)->isa == &ccv_nnc_micro_io_grad_isa)

static inline void ccv_nnc_micro_numbering(const ccv_nnc_micro_io_t self, const int id, const int var_count)
{
	const ccv_nnc_micro_io_vtab_t* const isa = self->isa;
	// If we numbering with negative id, we really just enumerate the grad.
	if (id < 0 && !CCV_NNC_IS_MICRO_IO_GRAD(self))
		return;
	if (isa->numbering)
		isa->numbering(self, id, var_count);
	else
		self->id = id;
}

static inline void ccv_nnc_micro_bind_scalars(const ccv_nnc_micro_io_t self, ccv_nnc_micro_scalar_lookup_f lookup, const void* const context)
{
	const ccv_nnc_micro_io_vtab_t* const isa = self->isa;
	if (isa->bind_scalars)
		isa->bind_scalars(self, lookup, context);
}

static inline void ccv_nnc_micro_deinit(const ccv_nnc_micro_io_t self)
{
	const ccv_nnc_micro_io_vtab_t* const isa = self->isa;
	if (isa->deinit)
		isa->deinit(self);
}

static inline CCV_WARN_UNUSED(ccv_nnc_micro_function_t) ccv_nnc_micro_emit(const ccv_nnc_micro_io_t self)
{
	const ccv_nnc_micro_io_vtab_t* const isa = self->isa;
	return isa->emit(self);
}

static inline CCV_WARN_UNUSED(ccv_nnc_micro_function_t) ccv_nnc_micro_emit_grad(const ccv_nnc_micro_io_t self, const int var_count)
{
	const ccv_nnc_micro_io_vtab_t* const isa = self->isa;
	return isa->emit_grad(self, var_count);
}

static inline CCV_WARN_UNUSED(ccv_nnc_micro_tensor_t) ccv_nnc_micro_return_shape(const ccv_nnc_micro_io_t self)
{
	const ccv_nnc_micro_io_vtab_t* const isa = self->isa;
	return isa->return_shape(self);
}

/**
 * Helpers to construct the micro IR.
 */

static inline ccv_nnc_micro_id_t ccv_nnc_micro_id_of_tensor(const int id)
{
	return (ccv_nnc_micro_id_t){
		.type = CCV_NNC_MICRO_TENSOR_ID,
		.id = id,
		.d = 0
	};
}

static inline ccv_nnc_micro_loop_index_term_t ccv_nnc_micro_index_of_value(const int value)
{
	return (ccv_nnc_micro_loop_index_term_t){
		.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL,
		.immediate_value = value
	};
}

static inline ccv_nnc_micro_loop_index_term_t ccv_nnc_micro_index_of_id(const ccv_nnc_micro_id_t id)
{
	return (ccv_nnc_micro_loop_index_term_t){
		.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID,
		.id = id
	};
}

static inline ccv_nnc_micro_loop_index_term_t ccv_nnc_micro_index_of_axis_size(const int id, const int level)
{
	return (ccv_nnc_micro_loop_index_term_t){
		.type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID,
		.id = {
			.type = CCV_NNC_MICRO_AXIS_SIZE_ID,
			.id = id,
			.d = level
		}
	};
}

static inline ccv_nnc_micro_loop_t ccv_nnc_micro_for_in(const ccv_nnc_micro_loop_index_term_t start_index, const ccv_nnc_micro_loop_index_term_t end_index, const int level)
{
	return (ccv_nnc_micro_loop_t){
		.start_index = start_index,
		.end_index = end_index,
		.carried_count = 0,
		.carrieds = 0,
		.statement_count = 0,
		.statements = 0,
		.id = {
			.type = CCV_NNC_MICRO_LOOP_ID,
			.d = 0,
			.id = level,
		}
	};
}

// This is a macro because C cannot return array type.
#define ccv_nnc_micro_index_of_loops(_loops, _loop_count) \
	(ccv_nnc_micro_loop_index_term_t [CCV_NNC_MAX_DIM_ALLOC]){ \
		{ .type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID, .id = _loop_count > 0 ? _loops[0].id : (ccv_nnc_micro_id_t){} }, \
		{ .type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID, .id = _loop_count > 1 ? _loops[1].id : (ccv_nnc_micro_id_t){} }, \
		{ .type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID, .id = _loop_count > 2 ? _loops[2].id : (ccv_nnc_micro_id_t){} }, \
		{ .type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID, .id = _loop_count > 3 ? _loops[3].id : (ccv_nnc_micro_id_t){} }, \
		{ .type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID, .id = _loop_count > 4 ? _loops[4].id : (ccv_nnc_micro_id_t){} }, \
		{ .type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID, .id = _loop_count > 5 ? _loops[5].id : (ccv_nnc_micro_id_t){} }, \
		{ .type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID, .id = _loop_count > 6 ? _loops[6].id : (ccv_nnc_micro_id_t){} }, \
		{ .type = CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID, .id = _loop_count > 7 ? _loops[7].id : (ccv_nnc_micro_id_t){} } \
	}

static inline ccv_nnc_micro_loop_variable_t ccv_nnc_micro_loop_variable_of_tensor(const int id, const int index_count, const ccv_nnc_micro_loop_index_term_t* const index)
{
	ccv_nnc_micro_loop_variable_t variable = {
		.id = ccv_nnc_micro_id_of_tensor(id),
		.index_count = index_count
	};
	int i;
	for (i = 0; i < index_count; i++)
		variable.index[i] = index[i];
	return variable;
}

static inline ccv_nnc_micro_loop_expression_t ccv_nnc_micro_loop_expression_of_variable(const ccv_nnc_micro_loop_variable_t variable)
{
	return (ccv_nnc_micro_loop_expression_t){
		.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR,
		.variable = variable
	};
}

static inline ccv_nnc_micro_loop_expression_t ccv_nnc_micro_loop_expression_of_value(const float value)
{
	return (ccv_nnc_micro_loop_expression_t){
		.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAL,
		.immediate_value = {
			.type = CCV_32F,
			.f32 = value
		}
	};
}

static inline ccv_nnc_micro_loop_expression_t ccv_nnc_micro_loop_expression_of_id(const ccv_nnc_micro_id_t id)
{
	return (ccv_nnc_micro_loop_expression_t){
		.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_ID,
		.id = id
	};
}

static inline ccv_nnc_micro_loop_statement_t ccv_nnc_micro_loop_assignment(const ccv_nnc_micro_loop_variable_t lvalue, const ccv_nnc_micro_loop_expression_t rvalue)
{
	return (ccv_nnc_micro_loop_statement_t){
		.type = CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_ASSIGNMENT,
		.assignment = {
			.lvalue = lvalue,
			.rvalue = rvalue
		}
	};
}

static inline ccv_nnc_micro_loop_expression_t ccv_nnc_micro_loop_expression_of_unary(const int unary_op, const ccv_nnc_micro_loop_expression_t x)
{
	ccv_nnc_micro_loop_expression_t expression = {
		.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_BINARY
	};
	expression.unary.unary_op = unary_op;
	expression.unary.x = (ccv_nnc_micro_loop_expression_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_expression_t));
	*expression.unary.x = x;
	return expression;
}

static inline ccv_nnc_micro_loop_expression_t ccv_nnc_micro_loop_expression_of_binary(const int binary_op, const ccv_nnc_micro_loop_expression_t left, const ccv_nnc_micro_loop_expression_t right)
{
	ccv_nnc_micro_loop_expression_t expression = {
		.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_BINARY
	};
	expression.binary.binary_op = binary_op;
	expression.binary.left = (ccv_nnc_micro_loop_expression_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_expression_t));
	*expression.binary.left = left;
	expression.binary.right = (ccv_nnc_micro_loop_expression_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_expression_t));
	*expression.binary.right = right;
	return expression;
}

static inline ccv_nnc_micro_loop_expression_t ccv_nnc_micro_loop_expression_of_ternary(const ccv_nnc_micro_loop_expression_t pivot, const ccv_nnc_micro_loop_expression_t left, const ccv_nnc_micro_loop_expression_t right)
{
	ccv_nnc_micro_loop_expression_t expression = {
		.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_TERNAY
	};
	expression.ternary.pivot = (ccv_nnc_micro_loop_expression_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_expression_t));
	*expression.ternary.pivot = pivot;
	expression.ternary.left = (ccv_nnc_micro_loop_expression_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_expression_t));
	*expression.ternary.left = left;
	expression.ternary.right = (ccv_nnc_micro_loop_expression_t*)ccmalloc(sizeof(ccv_nnc_micro_loop_expression_t));
	*expression.ternary.right = right;
	return expression;
}

static inline ccv_nnc_micro_loop_statement_t ccv_nnc_micro_loop_compound_assignment_of_id(const ccv_nnc_micro_id_t lvalue, const ccv_nnc_micro_loop_expression_t rvalue)
{
	return (ccv_nnc_micro_loop_statement_t){
		.type = CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_COMPOUND_ASSIGNMENT,
		.compound_assignment = {
			.lvalue = {
				.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_ID,
				.id = lvalue
			},
			.rvalue = rvalue
		}
	};
}

static inline ccv_nnc_micro_loop_statement_t ccv_nnc_micro_loop_compound_assignment_of_tensor(const ccv_nnc_micro_loop_variable_t lvalue, const ccv_nnc_micro_loop_expression_t rvalue)
{
	return (ccv_nnc_micro_loop_statement_t){
		.type = CCV_NNC_MICRO_LOOP_STATEMENT_TYPE_COMPOUND_ASSIGNMENT,
		.compound_assignment = {
			.lvalue = {
				.type = CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR,
				.variable = lvalue
			},
			.rvalue = rvalue
		}
	};
}

static inline ccv_nnc_micro_loop_carried_t ccv_nnc_micro_loop_carried(const uint8_t reduce_op, const int idx)
{
	return (ccv_nnc_micro_loop_carried_t){
		.id = {
			.type = CCV_NNC_MICRO_LOOP_CARRIED_ID,
			.d = reduce_op,
			.id = idx
		}
	};
}

// This method has to be mutable for efficiency reasons. Hence I kept it private.
void ccv_nnc_micro_program_simplify(ccv_nnc_micro_program_t* const program, const ccv_nnc_micro_io_t* const inputs, const int input_size, const ccv_nnc_micro_io_t* const outputs, const int output_size);
ccv_nnc_micro_loop_index_term_t ccv_nnc_micro_loop_index_deep_copy(const ccv_nnc_micro_loop_index_term_t* const term);
void ccv_nnc_micro_loop_index_free(ccv_nnc_micro_loop_index_term_t* const term);
void ccv_nnc_micro_loop_variable_free(ccv_nnc_micro_loop_variable_t* const var);
void ccv_nnc_micro_loop_statement_free(ccv_nnc_micro_loop_statement_t* const statement);
void ccv_nnc_micro_loop_statement_lvalue_free(ccv_nnc_micro_loop_statement_t* const statement);
void ccv_nnc_micro_loops_free(ccv_nnc_micro_loop_t* const loops, const int loop_count);

#endif
