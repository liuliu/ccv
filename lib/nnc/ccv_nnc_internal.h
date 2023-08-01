/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_internal_h
#define GUARD_ccv_nnc_internal_h

#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"

// Define some internal constraints

#define CCV_NNC_STACK_BITMASK_ALLOC (2)
#define CCV_NNC_TENSOR_PLACEHOLDER ((ccv_nnc_tensor_t*)(intptr_t)(0x10))

typedef void (*ccv_nnc_cmd_tensor_auto_f)(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size);
typedef int (*ccv_nnc_cmd_bitmask_f)(const ccv_nnc_cmd_param_t cmd, const int input_size, const int output_size, const uint64_t* const input_bitmasks, const int input_bitmask_size, const uint64_t* const output_bitmasks, const int output_bitmask_size);
typedef int (*ccv_nnc_cmd_inplace_f)(const ccv_nnc_cmd_param_t cmd, const int input_idx, const int input_size, const int output_idx, const int output_size);

typedef struct {
	int flags;
	ccv_nnc_cmd_bitmask_f bitmask;
	ccv_nnc_cmd_tensor_auto_f tensor_auto;
	ccv_nnc_cmd_inplace_f allow_inplace;
	ccv_nnc_cmd_inplace_f enforce_inplace;
} ccv_nnc_cmd_registry_t;

typedef struct {
	int tensor_formats; /**< [formats] The supported formats for this API implementation. */
	int tensor_datatypes; /**< [datatypes] The supported data types for this API implementation. */
	int tensor_memory; /**< [memory] The supported tensor memory type for this API implementation. */
	int algorithms; /**< [algorithms] Number of algorithms variation. */
	ccv_nnc_cmd_exec_f exec; /**< [exec] The function for command execution. */
	ccv_nnc_cmd_autotune_f autotune; /**< [autotune] The function to find the best algorithm to apply. */
	void* aux; /**< [aux] The additional information available for a particular command under a particular backend. */
} ccv_nnc_cmd_backend_registry_t;

static inline int ccv_nnc_tensor_hw(const ccv_nnc_tensor_param_t a, const int nd)
{
	if ((a.format == CCV_TENSOR_FORMAT_CHWN) ||
		(a.format == CCV_TENSOR_FORMAT_NHWC && nd == CCV_NNC_MAX_DIM + 1))
		return 0;
	else if ((a.format == CCV_TENSOR_FORMAT_NHWC && nd == CCV_NNC_MAX_DIM + 2) ||
			 (a.format == CCV_TENSOR_FORMAT_NCHW && nd == CCV_NNC_MAX_DIM + 1))
		return 1;
	else if (a.format == CCV_TENSOR_FORMAT_NCHW && nd == CCV_NNC_MAX_DIM + 2)
		return 2;
	return -1;
}

static inline void ccv_nnc_hint_tensor_forward(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* b)
{
	int i;
	assert(a.format == b->format);
	const int nd = ccv_nnc_tensor_nd(a.dim);
	assert(nd == CCV_NNC_MAX_DIM + 1 || nd == CCV_NNC_MAX_DIM + 2);
	int hw = ccv_nnc_tensor_hw(a, nd);
	assert(hw >= 0);
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
	{
		int stride = ccv_max(1, hint.stride.dim[i]);
		b->dim[i + hw] = (a.dim[i + hw] + hint.border.begin[i] + hint.border.end[i] - cmd.size.dim[i]) / stride + 1;
	}
}

static inline void ccv_nnc_hint_tensor_backward(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t a, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* b)
{
	int i;
	assert(a.format == b->format);
	const int nd = ccv_nnc_tensor_nd(a.dim);
	assert(nd == CCV_NNC_MAX_DIM + 1 || nd == CCV_NNC_MAX_DIM + 2);
	int hw = ccv_nnc_tensor_hw(a, nd);
	assert(hw >= 0);
	for (i = 0; i < CCV_NNC_MAX_DIM; i++)
	{
		int stride = ccv_max(1, hint.stride.dim[i]);
		b->dim[i + hw] = (a.dim[i + hw] - 1) * stride - hint.border.begin[i] - hint.border.end[i] + cmd.size.dim[i];
	}
}

void ccv_nnc_hint_tensor_auto_forward_from_inputs(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size);
void ccv_nnc_hint_tensor_auto_backward_from_gradient(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size);
void ccv_nnc_hint_tensor_auto_backward_from_inputs(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size);
void ccv_nnc_hint_tensor_auto_backward_from_gradient_and_inputs(const ccv_nnc_cmd_param_t cmd, const ccv_nnc_tensor_param_t* const inputs, const int input_size, const ccv_nnc_hint_t hint, ccv_nnc_tensor_param_t* const outputs, const int output_size);
int ccv_nnc_device_ids_for_io(ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, const int tensor_type, int* const device_ids, const int max_device_id_size);
void ccv_nnc_print_tensor_info(const ccv_nnc_tensor_t* const tensor);

static inline off_t ccv_nnc_tensor_view_offset(const int datatype, const int stride[CCV_NNC_MAX_DIM_ALLOC], const int ofs[CCV_NNC_MAX_DIM_ALLOC])
{
	int i;
	int nd = ccv_nnc_tensor_nd(stride);
	off_t offset = 0;
	for (i = nd - 1; i >= 0; i--)
		offset += ofs[i] * stride[i] * CCV_GET_DATA_TYPE_SIZE(datatype);
	return offset;
}

static inline void ccv_nnc_tensor_get_stride(const int dim[CCV_NNC_MAX_DIM_ALLOC], int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	int x;
	const int nd = ccv_nnc_tensor_nd(dim);
	if (nd < CCV_NNC_MAX_DIM_ALLOC)
		stride[nd] = 0;
	int cstride = 1;
	for (x = nd - 1; x >= 0; x--)
	{
		stride[x] = cstride;
		cstride *= dim[x];
	}
}

static inline int ccv_nnc_tensor_view_is_contiguous(const int dim[CCV_NNC_MAX_DIM_ALLOC], const int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	// Check if a tensor view is contiguous.
	const int nd = ccv_nnc_tensor_nd(dim);
	int first_none_one_dim_idx = -1;
	int i;
	for (i = 0; first_none_one_dim_idx < 0 && i < nd; i++)
		if (dim[i] > 1)
			first_none_one_dim_idx = i;
	// If it is all 1, it is contiguous.
	if (first_none_one_dim_idx < 0)
		return 1;
	// Check if from 0 to first_none_one_dim_idx, it is 1.
	assert(first_none_one_dim_idx < CCV_NNC_MAX_DIM_ALLOC);
	int cstride = 1;
	for (i = nd - 1; i >= first_none_one_dim_idx; i--)
	{
		if (stride[i] != cstride)
			return 0;
		else
			cstride *= dim[i];
	}
	return 1;
}

static inline void ccv_nnc_tensor_data(const ccv_nnc_tensor_param_t params, unsigned char* const data, const off_t off, ccv_numeric_data_t* const data_ref, off_t* const dataof_ref)
{
#ifdef HAVE_MPS
	if (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY)
	{
		(*data_ref).u8 = data;
		*dataof_ref = off;
		return;
	}
#endif
	(*data_ref).u8 = data + off;
}

static inline void ccv_nnc_tensor_data_add(const ccv_nnc_tensor_param_t params, const off_t off, ccv_numeric_data_t* const data_ref, off_t* const dataof_ref)
{
#ifdef HAVE_MPS
	if (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY)
	{
		(*dataof_ref) += off;
		return;
	}
#endif
	(*data_ref).u8 += off;
}

static inline void ccv_array_add_unique_int(ccv_array_t* ints, const int idx)
{
	int i;
	for (i = 0; i < ints->rnum; i++)
		if (*(int*)ccv_array_get(ints, i) == idx)
			return;
	ccv_array_push(ints, &idx);
}

static inline void ccv_array_add_unique_uint(ccv_array_t* ints, const uint32_t idx)
{
	int i;
	for (i = 0; i < ints->rnum; i++)
		if (*(uint32_t*)ccv_array_get(ints, i) == idx)
			return;
	ccv_array_push(ints, &idx);
}

#ifdef __cplusplus
#define REGISTER_COMMAND_BACKEND(x, y) extern "C" void _register_command_ ## x ## _backend_ ## y
#define REGISTER_COMMAND(x) extern "C" void _register_command_ ## x
#else
#define REGISTER_COMMAND_BACKEND(x, y) void _register_command_ ## x ## _backend_ ## y
#define REGISTER_COMMAND(x) void _register_command_ ## x
#endif
#define FIND_BACKEND(...)
#define FIND_FILE(...)

// x is the dimension.
// n[x] is the start point for the filter on y axis, so that we can avoid computing the padding.
// m[x] shows how long we should loop for filter on y axis, avoid computing the padding too.
#define SET_BORDER_OFFSET_SIZE_FOR(x, i, hint, wd, ad, n, m) \
	do { \
		n[x] = ccv_max(i[x] * hint.stride.dim[x] - hint.border.begin[x], 0) - (i[x] * hint.stride.dim[x] - hint.border.begin[x]); \
		m[x] = (wd)[x] - n[x] - (i[x] * hint.stride.dim[x] - hint.border.begin[x] + (wd)[x] - ccv_min((ad)[x], i[x] * hint.stride.dim[x] - hint.border.begin[x] + (wd)[x])); \
	} while (0)

// Defines common graph visit macros

// The visitor function / macro takes parameter visitor(node_type* node, int index, int term);
#define CCV_NNC_GRAPH_VISIT(_graph, nodes, node_size, sources, source_size, destinations, destination_size, allow_subset, visitor) \
	do { \
		/* Use the same data structure to do topological ordering. */ \
		typedef struct { \
			int8_t d; /* tag if this is the destination node. */ \
			int8_t r; /* tag if this is reached as destination node. */ \
			uint16_t c; /* number of incoming edges. */ \
			int32_t edges; /* pointer to incoming edges list. */ \
		} ccv_nnc_incoming_t; \
		int _i_, _j_; \
		/* Statistics of how many incoming edges for all nodes of a graph. */ \
		int _incoming_edges_ = 0; \
		for (_i_ = 0; _i_ < (node_size); _i_++) /* assuming it is all reached */ \
			_incoming_edges_ += ((nodes)[_i_].outgoings) ? (nodes)[_i_].outgoings->rnum : 0; \
		const int _heap_mem_ = (node_size + _incoming_edges_ > 1024); \
		ccv_nnc_incoming_t* _incomings_; \
		if (_heap_mem_) \
			_incomings_ = (ccv_nnc_incoming_t*)ccmalloc(sizeof(ccv_nnc_incoming_t) * (node_size) + sizeof(int32_t) * ((node_size) * 2 + _incoming_edges_)); \
		else \
			_incomings_ = (ccv_nnc_incoming_t*)alloca(sizeof(ccv_nnc_incoming_t) * (node_size) + sizeof(int32_t) * ((node_size) * 2 + _incoming_edges_)); \
		memset(_incomings_, 0, sizeof(ccv_nnc_incoming_t) * (node_size)); \
		int32_t* _exists_[2] = { \
			(int32_t*)(_incomings_ + (node_size)), \
			(int32_t*)(_incomings_ + (node_size)) + (node_size), \
		}; \
		int32_t* const _edges_ = _exists_[1] + (node_size); \
		for (_i_ = 0; _i_ < (source_size); _i_++) \
		{ \
			assert((sources)[_i_].graph == _graph); \
			_incomings_[(sources)[_i_].d].r = 1; \
			_exists_[0][_i_] = (sources)[_i_].d; \
		} \
		int _exist_size_[2] = { \
			(source_size), \
			0, \
		}; \
		int _p_ = 0, _q_ = 1; /* ping, pong swap. */ \
		/* Gather statistics. */ \
		while (_exist_size_[_p_] > 0) \
		{ \
			_exist_size_[_q_] = 0; \
			for (_i_ = 0; _i_ < _exist_size_[_p_]; _i_++) \
			{ \
				const int32_t _idx_ = _exists_[_p_][_i_]; \
				if (_incomings_[_idx_].r != 1) \
					continue; \
				_incomings_[_idx_].r = 2; \
				/* mark as not reached */ \
				if ((nodes)[_idx_].outgoings) \
					for (_j_ = 0; _j_ < (nodes)[_idx_].outgoings->rnum; _j_++) \
					{ \
						const int d = *(int*)ccv_array_get((nodes)[_idx_].outgoings, _j_); \
						++_incomings_[d].c; \
						if (_incomings_[d].r != 0) \
							continue; \
						_incomings_[d].r = 1; \
						assert(_exist_size_[_q_] < node_size); \
						_exists_[_q_][_exist_size_[_q_]] = d; \
						++_exist_size_[_q_]; \
					} \
			} \
			/* swap p and q. */ \
			CCV_SWAP(_p_, _q_, _i_ /* using i as temp holder */); \
		} \
		/* Fill incoming edges. */ \
		for (_i_ = 0; _i_ < (source_size); _i_++) \
		{ \
			assert((sources)[_i_].graph == _graph); \
			_incomings_[(sources)[_i_].d].r = 3; \
			_exists_[0][_i_] = (sources)[_i_].d; \
		} \
		_exist_size_[0] = (source_size); \
		_exist_size_[1] = 0; \
		_p_ = 0, _q_ = 1; /* ping, pong swap. */ \
		int _bump_ = 1; \
		while (_exist_size_[_p_] > 0) \
		{ \
			_exist_size_[_q_] = 0; \
			for (_i_ = 0; _i_ < _exist_size_[_p_]; _i_++) \
			{ \
				const int32_t _idx_ = _exists_[_p_][_i_]; \
				if (_incomings_[_idx_].r != 3) \
					continue; \
				_incomings_[_idx_].r = 4; \
				/* mark as not reached */ \
				if ((nodes)[_idx_].outgoings) \
					for (_j_ = 0; _j_ < (nodes)[_idx_].outgoings->rnum; _j_++) \
					{ \
						const int d = *(int*)ccv_array_get((nodes)[_idx_].outgoings, _j_); \
						if (_incomings_[d].edges == 0) \
						{ \
							_incomings_[d].edges = _bump_; \
							_bump_ += _incomings_[d].c; \
							_incomings_[d].c = 0; \
						} \
						_edges_[_incomings_[d].edges - 1 + _incomings_[d].c] = _idx_; \
						++_incomings_[d].c; \
						if (_incomings_[d].r != 2) \
							continue; \
						_incomings_[d].r = 3; \
						assert(_exist_size_[_q_] < node_size); \
						_exists_[_q_][_exist_size_[_q_]] = d; \
						++_exist_size_[_q_]; \
					} \
			} \
			/* swap p and q. */ \
			CCV_SWAP(_p_, _q_, _i_ /* using i as temp holder */); \
		} \
		/* Traverse back and mark r if it can be marked */ \
		for (_i_ = 0; _i_ < (destination_size); _i_++) \
		{ \
			assert((destinations)[_i_].graph == _graph); \
			_incomings_[(destinations)[_i_].d].r = 5; \
			_exists_[0][_i_] = (destinations)[_i_].d; \
		} \
		_exist_size_[0] = (destination_size); \
		_exist_size_[1] = 0; \
		_p_ = 0, _q_ = 1; /* ping, pong swap. */ \
		while (_exist_size_[_p_] > 0) \
		{ \
			_exist_size_[_q_] = 0; \
			for (_i_ = 0; _i_ < _exist_size_[_p_]; _i_++) \
			{ \
				const int32_t _idx_ = _exists_[_p_][_i_]; \
				if (_incomings_[_idx_].r != 5) /* If cannot be traversed in forward pass, cannot in backward pass. */ \
					continue; \
				_incomings_[_idx_].r = 6; \
				/* mark as not reached */ \
				if (_incomings_[_idx_].edges > 0) \
					for (_j_ = 0; _j_ < _incomings_[_idx_].c; _j_++) \
					{ \
						const int d = _edges_[_incomings_[_idx_].edges - 1 + _j_]; \
						if (_incomings_[d].r != 4) \
							continue; \
						_incomings_[d].r = 5; \
						assert(_exist_size_[_q_] < node_size); \
						_exists_[_q_][_exist_size_[_q_]] = d; \
						++_exist_size_[_q_]; \
					} \
			} \
			/* swap p and q. */ \
			CCV_SWAP(_p_, _q_, _i_ /* using i as temp holder */); \
		} \
		for (_i_ = 0; _i_ < (destination_size); _i_++) \
		{ \
			assert((destinations)[_i_].graph == _graph); \
			/* tagging destination nodes. */ \
			_incomings_[(destinations)[_i_].d].d = 1; \
		} \
		for (_i_ = 0; _i_ < (source_size); _i_++) \
		{ \
			assert((sources)[_i_].graph == _graph); \
			_exists_[0][_i_] = (sources)[_i_].d; \
		} \
		_p_ = 0; \
		_q_ = 1; \
		_exist_size_[0] = (source_size); \
		_exist_size_[1] = 0; \
		int _d_ = 0; \
		/* After we have that statistics, we can do topsort and run the command. */ \
		while (_exist_size_[_p_] > 0) \
		{ \
			_exist_size_[_q_] = 0; \
			for (_i_ = 0; _i_ < _exist_size_[_p_];) \
			{ \
				const int32_t _idx_ = _exists_[_p_][_i_]; \
				visitor(((nodes) + _idx_), (_idx_), (_incomings_[_idx_].d)); \
				/* mark as reached */ \
				if (_incomings_[_idx_].d) \
				{ \
					++_d_; \
					_incomings_[_idx_].r = 7; \
				} \
				if ((nodes)[_idx_].outgoings) \
				{ \
					if ((nodes)[_idx_].outgoings->rnum == 1) \
					{ \
						/* Optimizing for the case have only one child. Go through that directly. */ \
						const int d = *(int*)ccv_array_get((nodes)[_idx_].outgoings, 0); \
						--_incomings_[d].c; \
						if (_incomings_[d].c == 0 && _incomings_[d].r == 6 && _d_ < (destination_size)) \
						{ \
							_exists_[_p_][_i_] = d; \
							continue; \
						} \
					} else \
						for (_j_ = 0; _j_ < (nodes)[_idx_].outgoings->rnum; _j_++) \
						{ \
							const int d = *(int*)ccv_array_get((nodes)[_idx_].outgoings, _j_); \
							--_incomings_[d].c; \
							/* If all incoming edges are consumed, and not all destination node are computed, push it into next round */ \
							if (_incomings_[d].c == 0 && _incomings_[d].r == 6 && _d_ < (destination_size)) \
							{ \
								assert(_exist_size_[_q_] < node_size); \
								_exists_[_q_][_exist_size_[_q_]] = d; \
								++_exist_size_[_q_]; \
							} \
						} \
				} \
				++_i_; \
			} \
			/* swap p and q. */ \
			CCV_SWAP(_p_, _q_, _i_ /* using i as temp holder */); \
		} \
		for (_i_ = 0; _i_ < (destination_size); _i_++) \
		{ \
			assert((destinations)[_i_].graph == _graph); \
			/* skip if this is already reached. */ \
			if (_incomings_[(destinations)[_i_].d].r == 7) \
				continue; \
			/* this destination node should have every incoming nodes consumed. */ \
			if (!(allow_subset)) \
				{ assert(_incomings_[(destinations)[_i_].d].c == 0); } \
			else if (_incomings_[(destinations)[_i_].d].c > 0) /* Otherwise if incoming is not satisfied, no need to execute (allow subset to get executed, that is). */ \
				continue; \
			/* fetch the info for destination node and exec current node. */ \
			visitor(((nodes) + (destinations)[_i_].d), ((destinations)[_i_].d), (_incomings_[(destinations)[_i_].d].d)); \
		} \
		if (_heap_mem_) \
			ccfree(_incomings_); \
	} while (0);

typedef struct {
	int size;
	struct {
		int index;
		int term;
	} node[1];
} ccv_nnc_graph_visit_t;

static inline void ccv_nnc_graph_visit_free(ccv_nnc_graph_visit_t* graph_visit)
{
	ccfree(graph_visit);
}

#define CCV_NNC_GRAPH_VISIT_FOR1(graph_visit, nodes, _node_, _index_, _term_, ...) { \
	int _i_; \
	for (_i_ = 0; _i_ < (graph_visit)->size; _i_++) { \
		const int _index_ __attribute__((unused)) = (graph_visit)->node[_i_].index; \
		const int _term_ __attribute__((unused)) = (graph_visit)->node[_i_].term; \
		typeof ((nodes)) const _node_ __attribute__((unused)) = (nodes) + _index_; \

#define ccv_nnc_graph_visit_for(graph_visit, nodes, ...) \
	CCV_NNC_GRAPH_VISIT_FOR1(graph_visit, nodes, ##__VA_ARGS__, _node_unused_, _index_unused_, _term_unused_)

#define ccv_nnc_graph_visit_endfor } }

#define CCV_NNC_GRAPH_VISIT_NEW_VISITOR1(_, _index_, _term_) \
	_visit_->node[_visit_->size].index = (_index_); \
	_visit_->node[_visit_->size].term = (_term_); \
	++_visit_->size;

#define ccv_nnc_graph_visit_new(_graph, nodes, node_size, sources, source_size, destinations, destination_size, allow_subset) ({\
	ccv_nnc_graph_visit_t* _visit_ = (ccv_nnc_graph_visit_t*)ccmalloc(sizeof(ccv_nnc_graph_visit_t) + sizeof(_visit_->node[0]) * ((node_size) - 1)); \
	_visit_->size = 0; \
	CCV_NNC_GRAPH_VISIT(_graph, nodes, node_size, sources, source_size, destinations, destination_size, allow_subset, CCV_NNC_GRAPH_VISIT_NEW_VISITOR1); \
	assert(_visit_->size <= (node_size)); \
	_visit_; \
})

#endif
