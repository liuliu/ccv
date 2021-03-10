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
	CCV_NNC_MICRO_CARRY_OVER_ID,
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
	// These could be much more unary ops.
	CCV_NNC_MICRO_UNARY_OP_LOG,
	CCV_NNC_MICRO_UNARY_OP_EXP,
};

enum {
	CCV_NNC_MICRO_BINARY_OP_PLUS,
	CCV_NNC_MICRO_BINARY_OP_MINUS,
	CCV_NNC_MICRO_BINARY_OP_MUL,
	CCV_NNC_MICRO_BINARY_OP_DIV,
	CCV_NNC_MICRO_BINARY_OP_MAX,
	CCV_NNC_MICRO_BINARY_OP_MIN,
};

enum {
	CCV_NNC_MICRO_REDUCE_OP_MAX,
	CCV_NNC_MICRO_REDUCE_OP_MIN,
	CCV_NNC_MICRO_REDUCE_OP_MEAN, // Mean is complicated, we need a way to compute total for loops after this. It has to be done statically, and that is "interesting".
	CCV_NNC_MICRO_REDUCE_OP_SUM,
	CCV_NNC_MICRO_REDUCE_OP_PROD,
};

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
	ccv_nnc_micro_loop_index_term_t index[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_micro_loop_variable_t;

enum {
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_ID,
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR,
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_UNARY,
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_BINARY,
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

struct ccv_nnc_micro_loop_expression_s  {
	int type;
	union {
		ccv_nnc_micro_id_t id; // If this is a compound assignment, the id to that.
		ccv_nnc_micro_loop_variable_t variable;
		ccv_nnc_micro_loop_unary_t unary;
		ccv_nnc_micro_loop_binary_t binary;
	};
};

typedef struct {
	ccv_nnc_micro_loop_variable_t lvalue;
	ccv_nnc_micro_loop_expression_t rvalue;
} ccv_nnc_micro_loop_assignment_t;

typedef struct {
	ccv_nnc_micro_id_t id;
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
	int reduce_op;
	ccv_nnc_micro_id_t id;
} ccv_nnc_micro_loop_carry_over_t; // The accumulating register.

// A loop is identified with a loop counter id, some blocks inside this loop, some carry overs within
// this loop and can be used outside of this loop.
// If this loop has another loop nested (represented as the next one in the ccv_nnc_micro_nested_loop_t)
// all blocks are executed after the nested loop finished.
typedef struct {
	ccv_nnc_micro_id_t id; // Loop counter's id, this will be used for indexing.
	int carry_over_count;
	int statement_count;
	ccv_nnc_micro_loop_index_term_t start_index;
	ccv_nnc_micro_loop_index_term_t end_index;
	ccv_nnc_micro_loop_carry_over_t* carry_overs;
	ccv_nnc_micro_loop_statement_t* statements;
} ccv_nnc_micro_loop_t;

// A loop block contains many loops within each other.
typedef struct {
	int loop_count;
	ccv_nnc_micro_loop_t* loops;
} ccv_nnc_micro_loop_block_t;

typedef struct {
	int input; // The one it derives its shape from. If shape is nullptr, it has the same shape as input. -1 means it is an input.
	int sibling; // The sibling that has the same shape.
	int dimensions;
	int id;
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

// A combined op is constructed with many nested loops. These loops may have data dependencies
// between each other, but they are ordered in topological order to make sure one is finished
// after the another.
struct ccv_nnc_micro_combine_s {
	// Combined ops only have global vars, there is no local vars. All vars are tensors.
	int var_count;
	// loops are our constructs of IR ops really. It is hierarchical.
	int function_count;
	ccv_nnc_micro_tensor_t* vars;
	ccv_nnc_micro_function_t* functions;
};

typedef uint32_t(*ccv_nnc_micro_scalar_lookup_f)(const void* const context, const char* const name);

/**
 * This is the virtual table for micro op.
 */
struct ccv_nnc_micro_io_vtab_s {
	void (*bind_scalars)(const ccv_nnc_micro_io_t self, ccv_nnc_micro_scalar_lookup_f lookup, const void* const context); /**< Bind scalar name to a scoped id. */
	void (*numbering)(const ccv_nnc_micro_io_t self, const int id); /**< Assign id to the output of this micro op. */
	ccv_nnc_micro_function_t (*emit)(const ccv_nnc_micro_io_t self); /**< Emit instructions for this micro op. */
	ccv_nnc_micro_function_t (*emit_grad)(const ccv_nnc_micro_io_t self); /**< Emit backward instructions for this micro op. */
	ccv_nnc_micro_tensor_t (*return_shape)(const ccv_nnc_micro_io_t self); /**< The shape of the returned tensor. */
};

extern const ccv_nnc_micro_io_vtab_t ccv_nnc_micro_io_input_isa;

#define CCV_NNC_IS_MICRO_IO_INPUT(x) ((x)->isa == &ccv_nnc_micro_io_input_isa)

static inline void ccv_nnc_micro_numbering(const ccv_nnc_micro_io_t self, const int id)
{
	const ccv_nnc_micro_io_vtab_t* const isa = self->isa;
	if (isa->numbering)
		isa->numbering(self, id);
	else
		self->id = id;
}

static inline void ccv_nnc_micro_bind_scalars(const ccv_nnc_micro_io_t self, ccv_nnc_micro_scalar_lookup_f lookup, const void* const context)
{
	const ccv_nnc_micro_io_vtab_t* const isa = self->isa;
	if (isa->bind_scalars)
		isa->bind_scalars(self, lookup, context);
}

static inline CCV_WARN_UNUSED(ccv_nnc_micro_function_t) ccv_nnc_micro_emit(const ccv_nnc_micro_io_t self)
{
	const ccv_nnc_micro_io_vtab_t* const isa = self->isa;
	return isa->emit(self);
}

static inline CCV_WARN_UNUSED(ccv_nnc_micro_function_t) ccv_nnc_micro_emit_grad(const ccv_nnc_micro_io_t self)
{
	const ccv_nnc_micro_io_vtab_t* const isa = self->isa;
	return isa->emit_grad(self);
}

static inline CCV_WARN_UNUSED(ccv_nnc_micro_tensor_t) ccv_nnc_micro_return_shape(const ccv_nnc_micro_io_t self)
{
	const ccv_nnc_micro_io_vtab_t* const isa = self->isa;
	return isa->return_shape(self);
}

#endif
