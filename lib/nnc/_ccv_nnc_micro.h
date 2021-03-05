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
	CCV_NNC_MICRO_INPUT,
	CCV_NNC_MICRO_REINDEX,
	CCV_NNC_MICRO_BINARY,
	CCV_NNC_MICRO_REDUCE,
	CCV_NNC_MICRO_SELECT,
};

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
	CCV_NNC_MICRO_LOOP_INDEX_TYPE_NONE,
	CCV_NNC_MICRO_LOOP_INDEX_TYPE_ID,
	CCV_NNC_MICRO_LOOP_INDEX_TYPE_VAL,
	CCV_NNC_MICRO_LOOP_INDEX_TYPE_EXPR,
	CCV_NNC_MICRO_LOOP_INDEX_TYPE_UNBOUND_VAR, // Unbounded variable, shouldn't be there after fully-evaluated.
};

enum {
	CCV_NNC_MICRO_BINARY_OP_PLUS,
	CCV_NNC_MICRO_BINARY_OP_MINUS,
	CCV_NNC_MICRO_BINARY_OP_MUL,
	CCV_NNC_MICRO_BINARY_OP_DIV,
};

typedef struct {
	int type;
	union {
		char* name; // binding variable name.
		ccv_nnc_micro_id_t id;
		int immediate_value;
		struct ccv_nnc_micro_loop_index_expression_s* expression;
	};
} ccv_nnc_micro_loop_index_term_t;

typedef struct ccv_nnc_micro_loop_index_expression_s {
	int op;
	ccv_nnc_micro_loop_index_term_t left;
	ccv_nnc_micro_loop_index_term_t right;
} ccv_nnc_micro_loop_index_expression_t;

typedef struct {
	ccv_nnc_micro_id_t id;
	int index_count;
	ccv_nnc_micro_loop_index_term_t index[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_micro_loop_variable_t;

enum {
	CCV_NNC_MICRO_LOOP_BINARY_TYPE_VAL,
	CCV_NNC_MICRO_LOOP_BINARY_TYPE_VAR,
	CCV_NNC_MICRO_LOOP_BINARY_TYPE_EXPR,
};

typedef struct ccv_nnc_micro_loop_binary_expression_s {
	int op;
	struct {
		int type;
		union {
			double immediate_value;
			ccv_nnc_micro_loop_variable_t variable;
			struct ccv_nnc_micro_loop_binary_expression_s* expression;
		};
	} left;
	struct {
		int type;
		union {
			double immediate_value; // We need a method to identify whether this immediate value is an integer.
			ccv_nnc_micro_loop_variable_t variable;
			struct ccv_nnc_micro_loop_binary_expression_s* expression;
		};
	} right;
} ccv_nnc_micro_loop_binary_expression_t;

enum {
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_ID,
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_VAR,
	CCV_NNC_MICRO_LOOP_EXPR_TYPE_BINARY,
};

typedef struct {
	int type;
	union {
		ccv_nnc_micro_id_t id; // If this is a compound assignment, the id to that.
		ccv_nnc_micro_loop_variable_t variable;
		ccv_nnc_micro_loop_binary_expression_t expression;
	};
} ccv_nnc_micro_loop_expression_t;

typedef struct {
	ccv_nnc_micro_loop_variable_t lvalue;
	ccv_nnc_micro_loop_expression_t rvalue;
} ccv_nnc_micro_loop_assignment_t;

typedef struct {
	ccv_nnc_micro_id_t id;
	ccv_nnc_micro_loop_expression_t rvalue;
} ccv_nnc_micro_loop_compound_assignment_t;

enum {
	CCV_NNC_MICRO_LOOP_BLOCK_TYPE_ASSIGNMENT,
	CCV_NNC_MICRO_LOOP_BLOCK_TYPE_COMPOUND,
};

// A generic statement within a loop.
// For our purpose, there will be two types of generic statement:
// an assignment statement (for tensors).
// an compound assignment statement (for loop carry overs).
typedef struct {
	int type;
	union {
		ccv_nnc_micro_loop_assignment_t assignment;
		ccv_nnc_micro_loop_compound_assignment_t compound;
	};
} ccv_nnc_micro_loop_block_t;

enum {
	CCV_NNC_MICRO_REDUCE_OP_MAX,
	CCV_NNC_MICRO_REDUCE_OP_MIN,
	CCV_NNC_MICRO_REDUCE_OP_MEAN, // Mean is complicated, we need a way to compute total for loops after this. It has to be done statically, and that is "interesting".
	CCV_NNC_MICRO_REDUCE_OP_SUM,
	CCV_NNC_MICRO_REDUCE_OP_PROD,
};

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
	int block_count;
	ccv_nnc_micro_loop_index_term_t start_index;
	ccv_nnc_micro_loop_index_term_t end_index;
	ccv_nnc_micro_loop_carry_over_t* carry_overs;
	ccv_nnc_micro_loop_block_t* blocks;
} ccv_nnc_micro_loop_t;

// A nested loop contains many loops within each other.
typedef struct {
	int dep_count;
	int loop_count;
	int* deps; // Depend on previous loops, what's their index.
	ccv_nnc_micro_loop_t* loops;
} ccv_nnc_micro_nested_loop_t;

// A combined op is constructed with many nested loops. These loops may have data dependencies
// between each other, but they are ordered in topological order to make sure one is finished
// after the another.
struct ccv_nnc_micro_combine_s {
	int loop_count;
	ccv_nnc_micro_nested_loop_t* loops;
};

#endif
