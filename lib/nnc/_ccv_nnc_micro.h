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

typedef struct ccv_nnc_micro_loop_index_expression_s {
	struct {
		int type;
		union {
			int id;
			int immediate_value;
			struct ccv_nnc_micro_loop_index_expression_s* expression;
		};
	} left;
	struct {
		int type;
		union {
			int id;
			int immediate_value;
			struct ccv_nnc_micro_loop_index_expression_s* expression;
		};
	} right;
} ccv_nnc_micro_loop_index_expression_t;

typedef struct {
	ccv_nnc_micro_loop_index_expression_t expression;
} ccv_nnc_micro_loop_index_t;

typedef struct {
	int id;
	int index_count;
	ccv_nnc_micro_loop_index_t index[CCV_NNC_MAX_DIM_ALLOC];
} ccv_nnc_micro_loop_variable_t;

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

typedef struct {
	int type;
	union {
		int id; // If this is a compound assignment, the id to that.
		ccv_nnc_micro_loop_variable_t variable;
		ccv_nnc_micro_loop_binary_expression_t expression;
	};
} ccv_nnc_micro_loop_expression_t;

typedef struct {
	ccv_nnc_micro_loop_variable_t lvalue;
	ccv_nnc_micro_loop_expression_t rvalue;
} ccv_nnc_micro_loop_assignment_t;

typedef struct {
	int id;
	ccv_nnc_micro_loop_expression_t rvalue;
} ccv_nnc_micro_loop_compound_assignment_t;

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

typedef struct {
	int reduce_op;
	int id;
} ccv_nnc_micro_loop_carry_over_t; // The accumulating register.

typedef struct {
	int id; // Loop counter's id, this will be used for indexing.
	int start_index;
	int end_index;
	int carry_over_count;
	int block_count;
	ccv_nnc_micro_loop_carry_over_t* carry_overs;
	ccv_nnc_micro_loop_block_t* blocks;
} ccv_nnc_micro_loop_t;

typedef struct {
	int loop_count;
	ccv_nnc_micro_loop_t* loops;
} ccv_nnc_micro_nested_loop_t;

struct ccv_nnc_micro_combine_s {
	int loop_count;
	ccv_nnc_micro_nested_loop_t* loops;
};

#endif
