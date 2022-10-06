/**********************************************************
 * C-based/Cached/Core Computer Vision Library
 * Liu Liu, 2010-02-01
 **********************************************************/

/**********************************************************
 * CCV - Neural Network Collection
 **********************************************************/

#ifndef GUARD_ccv_nnc_easy_h
#define GUARD_ccv_nnc_easy_h

#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"

/**
 * Convenience API
 *
 * This header provides convenience APIs for nnc usage. Being convenience API,
 * it is optimized for shorthand coding, and may collide the naming space with
 * others.
 *
 */
// c99 only, make sure your compiler supports that.

#define NOOP_GRAPH_WHILE_EXPR (ccv_nnc_graph_while_f)(1)
#define NOOP_GRAPH_CASE_OF_EXPR (ccv_nnc_graph_case_of_f)(1)

// This is a better LIST_COUNT macro, it generates a list of 1+1+0+0+0 where it is 1 if the parameter presents, and 0 otherwise.
// This works better for cases such as LIST_COUNT(1, 2, 3,) where previous macro will get 4 and this one will have correctly
// computed result.
#define LIST_COUNT_01(_0,_1,_2,...) _2
#define LIST_COUNT_E(...) LIST_COUNT_01(_0,##__VA_ARGS__,1,0)
#define LIST_COUNT_N(_0,_1,_2,_3,_4,_5,_6,_7,_8,_9,_10,_11,_12,_13,_14,_15,_16,_17,_18,_19,_20,_21,_22,_23,_24,_25,_26,_27,_28,_29,_30,_31,_32,_33,_34,_35,_36,_37,_38,_39,_40,_41,_42,_43,_44,_45,_46,_47,_48,_49,_50,_51,_52,_53,_54,_55,_56,_57,_58,_59,_60,_61,_62,_63,...) (LIST_COUNT_E(_0)+LIST_COUNT_E(_1)+LIST_COUNT_E(_2)+LIST_COUNT_E(_3)+LIST_COUNT_E(_4)+LIST_COUNT_E(_5)+LIST_COUNT_E(_6)+LIST_COUNT_E(_7)+LIST_COUNT_E(_8)+LIST_COUNT_E(_9)+LIST_COUNT_E(_10)+LIST_COUNT_E(_11)+LIST_COUNT_E(_12)+LIST_COUNT_E(_13)+LIST_COUNT_E(_14)+LIST_COUNT_E(_15)+LIST_COUNT_E(_16)+LIST_COUNT_E(_17)+LIST_COUNT_E(_18)+LIST_COUNT_E(_19)+LIST_COUNT_E(_20)+LIST_COUNT_E(_21)+LIST_COUNT_E(_22)+LIST_COUNT_E(_23)+LIST_COUNT_E(_24)+LIST_COUNT_E(_25)+LIST_COUNT_E(_26)+LIST_COUNT_E(_27)+LIST_COUNT_E(_28)+LIST_COUNT_E(_29)+LIST_COUNT_E(_30)+LIST_COUNT_E(_31)+LIST_COUNT_E(_32)+LIST_COUNT_E(_33)+LIST_COUNT_E(_34)+LIST_COUNT_E(_35)+LIST_COUNT_E(_36)+LIST_COUNT_E(_37)+LIST_COUNT_E(_38)+LIST_COUNT_E(_39)+LIST_COUNT_E(_40)+LIST_COUNT_E(_41)+LIST_COUNT_E(_42)+LIST_COUNT_E(_43)+LIST_COUNT_E(_44)+LIST_COUNT_E(_45)+LIST_COUNT_E(_46)+LIST_COUNT_E(_47)+LIST_COUNT_E(_48)+LIST_COUNT_E(_49)+LIST_COUNT_E(_50)+LIST_COUNT_E(_51)+LIST_COUNT_E(_52)+LIST_COUNT_E(_53)+LIST_COUNT_E(_54)+LIST_COUNT_E(_55)+LIST_COUNT_E(_56)+LIST_COUNT_E(_57)+LIST_COUNT_E(_58)+LIST_COUNT_E(_59)+LIST_COUNT_E(_60)+LIST_COUNT_E(_61)+LIST_COUNT_E(_62)+LIST_COUNT_E(_63)-1)
#define LIST_COUNT(...) LIST_COUNT_N(_0,##__VA_ARGS__,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,)

#define LIST_X(_type, ...) (_type []){__VA_ARGS__}

#define KV_X_2(_x, _y, ...) {(_x), (_y)}
#define KV_X_1(_x, ...) {(_x)}
#define KV_X_SEL(_1, _2, _FX, ...) _FX
#define KV(...) KV_X_SEL(__VA_ARGS__, KV_X_2, KV_X_1)(__VA_ARGS__)

#define LIST_SIZEOF_COUNT(_type, ...) (sizeof(LIST_X(_type, __VA_ARGS__)) / sizeof(_type))

/**
 * @defgroup convenience_api Convenience API
 * @{
 */
/**
 * Pass a list of tensors to NNC functions that accepts (tensor array, tensor array size).
 * This method effectively gives two parameters as one.
 */
#define TENSOR_LIST(...) LIST_X(ccv_nnc_tensor_t*, __VA_ARGS__), LIST_COUNT(__VA_ARGS__)
/**
 * Pass a list of tensor parameters to NNC functions that accepts (parameter array, parameter array size).
 * This method effectively gives two parameters as one.
 */
#define TENSOR_PARAM_LIST(...) LIST_X(const ccv_nnc_tensor_param_t, __VA_ARGS__), LIST_COUNT(__VA_ARGS__)
/**
 * This represents a tensor symbol that is empty (tensor = nil)
 */
#define NO_TENSOR_SYMBOL (const ccv_nnc_tensor_symbol_t){.d = CCV_NNC_NO_TENSOR_SYMBOL}
/**
 * This represents a graph exec symbol that is empty (exec = nil)
 */
#define NO_GRAPH_EXEC_SYMBOL (const ccv_nnc_graph_exec_symbol_t){.d = CCV_NNC_NO_GRAPH_EXEC_SYMBOL}
/**
 * Pass a list of tensor symbols to NNC functions that accepts (tensor symbol array, tensor symbol array size).
 * This method effectively gives two parameters as one.
 */
#define TENSOR_SYMBOL_LIST(...) LIST_X(const ccv_nnc_tensor_symbol_t, __VA_ARGS__), LIST_COUNT(__VA_ARGS__)
/**
 * Pass a list of tensor variables to NNC functions that accepts (tensor variable array, tensor variable array size).
 * This method effectively gives two parameters as one.
 */
#define TENSOR_VARIABLE_LIST(...) LIST_X(ccv_nnc_tensor_variable_t, __VA_ARGS__), LIST_COUNT(__VA_ARGS__)
/**
 * Pass a list of tensor bindings to NNC functions that accepts (tensor binding array, tensor binding array size).
 * This method effectively gives two parameters as one. Since tensor binding requires two: symbol and a tensor,
 * you should use this like: TENSOR_BIND_MAP(KV(symbol1, tensor1), KV(symbol2, tensor2)).
 */
#define TENSOR_BIND_MAP(...) LIST_X(const ccv_nnc_tensor_bind_t, __VA_ARGS__), LIST_SIZEOF_COUNT(ccv_nnc_tensor_bind_t, __VA_ARGS__)
/**
 * Pass a list of tensor symbol pairs to NNC functions that accepts (tensor symbol pair array, tensor symbol pair array size).
 * This method effectively gives two parameters as one. Since tensor symbol pair requires two: source symbol and destination symbol,
 * you should use this like: TENSOR_SYMBOL_MAP(KV(symbol1, symbol2), KV(symbol3, symbol4)).
 */
#define TENSOR_SYMBOL_MAP(...) LIST_X(const ccv_nnc_tensor_symbol_map_t, __VA_ARGS__), LIST_SIZEOF_COUNT(ccv_nnc_tensor_symbol_map_t, __VA_ARGS__)
/**
 * Pass a list of execution nodes to NNC functions that accepts (execution node array, execution node array size).
 * This method effectively gives two parameters as one.
 */
#define GRAPH_EXEC_LIST(...) LIST_X(const ccv_nnc_graph_exec_t, __VA_ARGS__), LIST_COUNT(__VA_ARGS__)
/**
 * Pass a list of execution node symbols to NNC functions that accepts (execution node symbol array, execution node symbol array size).
 * This method effectively gives two parameters as one.
 */
#define GRAPH_EXEC_SYMBOL_LIST(...) LIST_X(const ccv_nnc_graph_exec_symbol_t, __VA_ARGS__), LIST_COUNT(__VA_ARGS__)
/**
 * Pass both default sources and default sources size to function that accepts (sources, source size).
 * @param x A given symbolic graph.
 */
#define SYMBOLIC_GRAPH_SOURCES(x) ccv_nnc_symbolic_graph_sources(x), ccv_nnc_symbolic_graph_source_size(x)
/**
 * Pass both default destinations and default destinations size to function that accepts (destinations, destination size).
 * @param x A given symbolic graph.
 */
#define SYMBOLIC_GRAPH_DESTINATIONS(x) ccv_nnc_symbolic_graph_destinations(x), ccv_nnc_symbolic_graph_destination_size(x)
/**
 * Pass a list of simplification passes to NNC functions that accepts (pass array, pass array size).
 * This method effectively gives two parameters as one.
 */
#define SYMBOLIC_GRAPH_PASSES(...) LIST_X(const int, __VA_ARGS__), LIST_COUNT(__VA_ARGS__)
/**
 * Pass a list of CNNP models to NNC functions that accepts (model array, model array size).
 * This method effectively gives two parameters as one.
 */
#define MODEL_LIST(...) LIST_X(ccv_cnnp_model_t*, __VA_ARGS__), LIST_COUNT(__VA_ARGS__)
/**
 * Pass a list of CNNP model IOs to NNC functions that accepts (model IO array, model IO array size).
 * This method effectively gives two parameters as one.
 */
#define MODEL_IO_LIST(...) LIST_X(const ccv_cnnp_model_io_t, __VA_ARGS__), LIST_COUNT(__VA_ARGS__)
/**
 * Pass a list of CNNP tensor params to ccv_cnnp_cmd_exec which accepts (tensor params array, tensor params array size).
 * This method effectively gives two parameters as one.
 */
#define MODEL_CMD_EXEC_IO_MAP(...) LIST_X(const ccv_cnnp_cmd_exec_io_t, __VA_ARGS__), LIST_SIZEOF_COUNT(ccv_cnnp_cmd_exec_io_t, __VA_ARGS__)
/**
 * Pass a list of CNNP tensor type to ccv_cnnp_cmd_exec which accepts (tensor type array, tensor type array size).
 * This method effectively gives two parameters as one.
 */
#define MODEL_CMD_EXEC_IO_LIST(...) LIST_X(const int, __VA_ARGS__), LIST_COUNT(__VA_ARGS__)
/**
 * Pass a list of dataframe column ids to iteration function that accepts (column id array, column id array size).
 * This method effectively gives two parameters as one.
 */
#define COLUMN_ID_LIST(...) LIST_X(const int, __VA_ARGS__), LIST_COUNT(__VA_ARGS__)

#define TRAVERSE_FULL 0,0,0,0

#define ALL_PARAMETERS -1

// We will support NUMA allocation on CPU in the future. Currently, this is not very meaningful (except enforce no memory reuse between tensors).
#define CPU_NUMA_TENSOR_NHWC(device_id, dt, ...) ((ccv_nnc_tensor_param_t){.type=(CCV_COMPUTE_DEVICE_##device_id) | CCV_TENSOR_CPU_MEMORY,.format=CCV_TENSOR_FORMAT_NHWC,.datatype=CCV_##dt,.dim={__VA_ARGS__}})
#define CPU_NUMA_TENSOR_NCHW(device_id, dt, ...) ((ccv_nnc_tensor_param_t){.type=(CCV_COMPUTE_DEVICE_##device_id) | CCV_TENSOR_CPU_MEMORY,.format=CCV_TENSOR_FORMAT_NCHW,.datatype=CCV_##dt,.dim={__VA_ARGS__}})
#define CPU_NUMA_TENSOR_CHWN(device_id, dt, ...) ((ccv_nnc_tensor_param_t){.type=(CCV_COMPUTE_DEVICE_##device_id) | CCV_TENSOR_CPU_MEMORY,.format=CCV_TENSOR_FORMAT_CHWN,.datatype=CCV_##dt,.dim={__VA_ARGS__}})
#define CPU_TENSOR_NHWC(dt, ...) CPU_NUMA_TENSOR_NHWC(ANY, dt, __VA_ARGS__)
#define CPU_TENSOR_NCHW(dt, ...) CPU_NUMA_TENSOR_NCHW(ANY, dt, __VA_ARGS__)
#define CPU_TENSOR_CHWN(dt, ...) CPU_NUMA_TENSOR_CHWN(ANY, dt, __VA_ARGS__)
// This way, we can do error check on the device type :)
#define GPU_TENSOR_NHWC(device_id, dt, ...) ((ccv_nnc_tensor_param_t){.type=(CCV_COMPUTE_DEVICE_##device_id) | CCV_TENSOR_GPU_MEMORY,.format=CCV_TENSOR_FORMAT_NHWC,.datatype=CCV_##dt,.dim={__VA_ARGS__}})
#define GPU_TENSOR_NCHW(device_id, dt, ...) ((ccv_nnc_tensor_param_t){.type=(CCV_COMPUTE_DEVICE_##device_id) | CCV_TENSOR_GPU_MEMORY,.format=CCV_TENSOR_FORMAT_NCHW,.datatype=CCV_##dt,.dim={__VA_ARGS__}})
#define GPU_TENSOR_CHWN(device_id, dt, ...) ((ccv_nnc_tensor_param_t){.type=(CCV_COMPUTE_DEVICE_##device_id) | CCV_TENSOR_GPU_MEMORY,.format=CCV_TENSOR_FORMAT_CHWN,.datatype=CCV_##dt,.dim={__VA_ARGS__}})
/** @} */

#define DIM_ALLOC(...) (int [CCV_NNC_MAX_DIM_ALLOC]){__VA_ARGS__}

#define ESCAPE_X(...) __VA_ARGS__
#define HINT_X_1(_stride_) ((ccv_nnc_hint_t){.stride={.dim={ESCAPE_X _stride_}}, .border={.begin={0},.end={0}}})
#define HINT_X_2(_stride_, _border_) ((ccv_nnc_hint_t){.stride={.dim={ESCAPE_X _stride_}}, .border={.begin={ESCAPE_X _border_},.end={ESCAPE_X _border_}}})
#define HINT_X_3(_stride_, _begin_, _end_) ((ccv_nnc_hint_t){.stride={.dim={ESCAPE_X _stride_}}, .border={.begin={ESCAPE_X _begin_},.end={ESCAPE_X _end_}}})
#define HINT_X_SEL(_1, _2, _3, _FX, ...) _FX
/**
 * @ingroup convenience_api
 * Simpler method to create hint.
 * HINT(stride), HINT(stride, border), HINT(stride, border begin, border end)
 */
#define HINT(...) HINT_X_SEL(__VA_ARGS__, HINT_X_3, HINT_X_2, HINT_X_1)(__VA_ARGS__)

static inline size_t ccv_nnc_dimension_count(const int dim[CCV_NNC_MAX_DIM_ALLOC])
{
	if (dim[0] == 0)
		return 0;
	int i;
	size_t count = 1;
	for (i = 0; i < CCV_NNC_MAX_DIM_ALLOC && dim[i] > 0; i++)
		count *= dim[i];
	return count;
}

static inline size_t ccv_nnc_tensor_count(const ccv_nnc_tensor_param_t params)
{
	return ccv_nnc_dimension_count(params.dim);
}

static inline size_t ccv_nnc_tensor_data_size(const ccv_nnc_tensor_param_t params)
{
#ifdef HAVE_CUDA // For CUDA, we align to 128-bytes.
	if (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY)
		return ((CCV_GET_DATA_TYPE_SIZE(params.datatype) * (ssize_t)ccv_nnc_tensor_count(params) + 127) & -128);
	else
#elif defined(HAVE_MPS) // For MPS, we have to align to PAGE_SIZE (4096).
	if (CCV_TENSOR_GET_MEMORY(params.type) == CCV_TENSOR_GPU_MEMORY)
		return ((CCV_GET_DATA_TYPE_SIZE(params.datatype) * (ssize_t)ccv_nnc_tensor_count(params) + 4095) & -4096);
	else
#endif
	return ((CCV_GET_DATA_TYPE_SIZE(params.datatype) * (ssize_t)ccv_nnc_tensor_count(params) + 63) & -64);
}

static inline void ccv_nnc_tensor_view_get_dim(const ccv_nnc_tensor_view_t* const tv, int dim[CCV_NNC_MAX_DIM_ALLOC])
{
	int x;
	const int nd = ccv_nnc_tensor_nd(tv->info.dim);
	const int offset = CCV_NNC_MAX_DIM + 2 - nd;
	for (x = 0; x < offset; x++)
		dim[x] = 1;
	for (x = offset; x < CCV_NNC_MAX_DIM + 2; x++)
		dim[x] = tv->info.dim[x - offset];
	dim[CCV_NNC_MAX_DIM + 2] = 0;
}

static inline CCV_WARN_UNUSED(int) ccv_nnc_is_tensor_stride_packed(const int stride[CCV_NNC_MAX_DIM_ALLOC], const int dim[CCV_NNC_MAX_DIM_ALLOC])
{
	const int nd = ccv_nnc_tensor_nd(stride);
	int i;
	int cstride = 1;
	for (i = nd - 1; i >= 0; i--)
	{
		if (stride[i] != cstride)
			return 0;
		cstride *= dim[i];
	}
	return 1;
}

static inline CCV_WARN_UNUSED(int) ccv_nnc_tensor_view_check_dim(const ccv_nnc_tensor_view_t* const tv, const int dim[CCV_NNC_MAX_DIM_ALLOC])
{
	int x;
	const int nd = ccv_nnc_tensor_nd(tv->info.dim);
	const int offset = CCV_NNC_MAX_DIM + 2 - nd;
	for (x = 0; x < offset; x++)
		if (dim[x] != 1)
			return 0;
	for (x = offset; x < CCV_NNC_MAX_DIM + 2; x++)
		if (dim[x] != tv->info.dim[x - offset])
			return 0;
	return 1;
}

static inline void ccv_nnc_tensor_view_get_broadcast_dim(const ccv_nnc_tensor_view_t* const tv, int dim[CCV_NNC_MAX_DIM_ALLOC])
{
	int x;
	const int nd = ccv_nnc_tensor_nd(tv->info.dim);
	const int offset = CCV_NNC_MAX_DIM + 2 - nd;
	for (x = 0; x < offset; x++)
		dim[x] = ccv_max(1, dim[x]);
	for (x = offset; x < CCV_NNC_MAX_DIM + 2; x++)
		dim[x] = ccv_max(dim[x], tv->info.dim[x - offset]);
}

static inline CCV_WARN_UNUSED(int) ccv_nnc_tensor_view_check_broadcast_dim(const ccv_nnc_tensor_view_t* const tv, int dim[CCV_NNC_MAX_DIM_ALLOC])
{
	int x;
	const int nd = ccv_nnc_tensor_nd(tv->info.dim);
	const int offset = CCV_NNC_MAX_DIM + 2 - nd;
	for (x = offset; x < CCV_NNC_MAX_DIM + 2; x++)
		if (dim[x] != tv->info.dim[x - offset] && tv->info.dim[x - offset] != 1)
			return 0;
	return 1;
}

static inline void ccv_nnc_tensor_view_get_stride(const ccv_nnc_tensor_view_t* const tv, int stride[CCV_NNC_MAX_DIM_ALLOC])
{
	int x;
	const int nd = ccv_nnc_tensor_nd(tv->info.dim);
	const int offset = CCV_NNC_MAX_DIM + 2 - nd;
	stride[nd] = stride[CCV_NNC_MAX_DIM + 2] = 0;
	if (CCV_IS_TENSOR_VIEW(tv))
	{
		for (x = offset; x < CCV_NNC_MAX_DIM + 2; x++)
			stride[x] = tv->stride[x - offset];
		for (x = 0; x < offset; x++)
			stride[x] = stride[offset];
	} else {
		int cstride = 1;
		for (x = CCV_NNC_MAX_DIM + 1; x >= offset; x--)
		{
			stride[x] = cstride;
			cstride *= tv->info.dim[x - offset];
		}
		for (x = 0; x < offset; x++)
			stride[x] = cstride;
	}
}

static inline int ccv_nnc_tensor_get_n(const ccv_nnc_tensor_param_t params)
{
	switch (params.format)
	{
		case CCV_TENSOR_FORMAT_NHWC:
		case CCV_TENSOR_FORMAT_NCHW:
			if (ccv_nnc_tensor_nd(params.dim) == CCV_NNC_MAX_DIM + 1)
				return 1;
			else
				return params.dim[0];
		case CCV_TENSOR_FORMAT_CHWN:
			return params.dim[CCV_NNC_MAX_DIM + 1];
	}
	return 0;
}

static inline int ccv_nnc_tensor_get_c(const ccv_nnc_tensor_param_t params)
{
	switch (params.format)
	{
		case CCV_TENSOR_FORMAT_NHWC:
			return params.dim[ccv_nnc_tensor_nd(params.dim) - 1];
		case CCV_TENSOR_FORMAT_NCHW:
			if (ccv_nnc_tensor_nd(params.dim) == CCV_NNC_MAX_DIM + 1)
				return params.dim[0];
			else
				return params.dim[1];
		case CCV_TENSOR_FORMAT_CHWN:
			return params.dim[0];
	}
	return 0;
}

static inline void ccv_nnc_tensor_set_n(ccv_nnc_tensor_param_t* const params, const int n)
{
	switch (params->format)
	{
		case CCV_TENSOR_FORMAT_NHWC:
		case CCV_TENSOR_FORMAT_NCHW:
			params->dim[0] = n;
			break;
		case CCV_TENSOR_FORMAT_CHWN:
			params->dim[CCV_NNC_MAX_DIM + 1] = n;
			break;
	}
}

static inline void ccv_nnc_tensor_set_c(ccv_nnc_tensor_param_t* const params, const int nd, const int c)
{
	switch (params->format)
	{
		case CCV_TENSOR_FORMAT_NHWC:
			params->dim[nd - 1] = c;
			break;
		case CCV_TENSOR_FORMAT_NCHW:
			if (nd == CCV_NNC_MAX_DIM + 1)
				params->dim[0] = c;
			else
				params->dim[1] = c;
			break;
		case CCV_TENSOR_FORMAT_CHWN:
			params->dim[0] = c;
			break;
	}
}

static inline int ccv_nnc_is_matrix_transpose(const ccv_nnc_tensor_param_t params, const int transpose[2])
{
	const int nd = ccv_nnc_tensor_nd(params.dim);
	assert(nd >= 1);
	if (transpose[0] != transpose[1])
	{
		assert(nd > 1);
		assert(((transpose[0] == ((nd == 2) ? 0 : nd - 2)) && (transpose[1] == ((nd == 2) ? 1 : nd - 1))) ||
			((transpose[1] == ((nd == 2) ? 0 : nd - 2)) && (transpose[0] == ((nd == 2) ? 1 : nd - 1))));
		return 1;
	}
	return 0;
}

// Assuming this is batched matrix. Getting relevant parameters.
static inline void ccv_nnc_tensor_get_matrix_params(const ccv_nnc_tensor_param_t params, const int* const stride, const int* const dim, const int transpose[2], int* const batch_size_ref, int* const rows_ref, int* const cols_ref, int* const batch_inc_ref, int* const rows_inc_ref, int* const cols_inc_ref)
{
	const int nd = ccv_nnc_tensor_nd(params.dim);
	assert(nd >= 1);
	*batch_size_ref = nd < 3 ? 1 : params.dim[nd - 3];
	*batch_inc_ref = nd < 3 ? 0 : stride ? stride[nd - 3] : dim[nd - 2] * dim[nd - 1];
	int rows = nd == 1 ? 1 : (nd == 2 ? params.dim[0] : params.dim[nd - 2]);
	int rows_inc = stride ? (nd >= 2 ? stride[nd - 2] : stride[0] * dim[0]) : dim[nd - 1];
	int cols = params.dim[nd - 1];
	int cols_inc = 1;
	if (transpose[0] != transpose[1])
	{
		assert(nd > 1);
		assert(((transpose[0] == ((nd == 2) ? 0 : nd - 2)) && (transpose[1] == ((nd == 2) ? 1 : nd - 1))) ||
			((transpose[1] == ((nd == 2) ? 0 : nd - 2)) && (transpose[0] == ((nd == 2) ? 1 : nd - 1))));
		int t;
		CCV_SWAP(rows, cols, t);
		CCV_SWAP(rows_inc, cols_inc, t);
	}
	*rows_ref = rows;
	*cols_ref = cols;
	*rows_inc_ref = rows_inc;
	*cols_inc_ref = cols_inc;
}

static inline CCV_WARN_UNUSED(ccv_nnc_tensor_view_t) ccv_nnc_get_tensor_view(const ccv_nnc_tensor_t* const tensor)
{
	if (CCV_IS_TENSOR_VIEW(tensor))
		return (ccv_nnc_tensor_view_t)*(ccv_nnc_tensor_view_t*)tensor;
	ccv_nnc_tensor_view_t tv;
	memcpy(&tv, tensor, sizeof(ccv_nnc_tensor_t));
	return tv;
}

static inline void ccv_nnc_tensor_view_alignment(ccv_nnc_tensor_view_t** const tvs, const int tv_size)
{
	int i, j;
	int max_nd = 0;
	for (i = 0; i < tv_size; i++)
		max_nd = ccv_max(ccv_nnc_tensor_nd(tvs[i]->info.dim), max_nd);
	for (i = 0; i < tv_size; i++)
	{
		const int nd = ccv_nnc_tensor_nd(tvs[i]->info.dim);
		for (j = max_nd - 1; j >= max_nd - nd; j--)
			tvs[i]->info.dim[j] = tvs[i]->info.dim[j - max_nd + nd];
		for (j = 0; j < max_nd - nd; j++)
			tvs[i]->info.dim[j] = 1;
		if (!CCV_IS_TENSOR_VIEW(tvs[i]))
			continue;
		for (j = max_nd - 1; j >= max_nd - nd; j--)
			tvs[i]->stride[j] = tvs[i]->stride[j - max_nd + nd];
		for (j = 0; j < max_nd - nd; j++)
			tvs[i]->stride[j] = tvs[i]->stride[max_nd - nd];
	}
}


#define TRANSPOSE(_X, _Y) ((int[]){(_X),(_Y)})
#define NO_TRANSPOSE TRANSPOSE(0, 0)
#define CMD_GEMM_X(_0, _TA, _TB, ...) ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.blas={.a={1,1},.transpose_a={_TA[0],_TA[1]},.transpose_b={_TB[0],_TB[1]},}}) // We default to alpha = 1 and beta = 1
#define CMD_GEMM(...) CMD_GEMM_X(_0, ##__VA_ARGS__, NO_TRANSPOSE, NO_TRANSPOSE)
#define CMD_GENERIC_X_0() ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}}})
#define CMD_GENERIC_X_F(...) ("This should not be used, you should have either 0 parameter or 3 parameters for CMD_GENERIC")
#define CMD_GENERIC_X_3(...) ((ccv_nnc_cmd_param_t){.size={.dim={__VA_ARGS__}}})
#define CMD_GENERIC_X_SEL(_0, _1, _2, _3, _FX, ...) _FX
// Using ## so that if it is empty, we omit one comma.
#define CMD_GENERIC(...) CMD_GENERIC_X_SEL(CMD_GENERIC_X_F, ##__VA_ARGS__, CMD_GENERIC_X_3, CMD_GENERIC_X_F, CMD_GENERIC_X_F, CMD_GENERIC_X_0)(__VA_ARGS__)
#define CMD_REDUCE(...) ((ccv_nnc_cmd_param_t){.size={.dim={1,1,1}},.reduce={.count=LIST_COUNT(__VA_ARGS__),.axis={__VA_ARGS__}}})
/**
 * @defgroup available_commands Available Commands
 * @{
 */
#define CMD_NOOP() ccv_nnc_cmd(CCV_NNC_NOOP, 0, ccv_nnc_cmd_auto, 0)
#define CMD_CUSTOM_FORWARD(f) ccv_nnc_cmd(CCV_NNC_CUSTOM_FORWARD, f, ccv_nnc_cmd_auto, 0)
/** @} */

int ccv_nnc_is_no_hint(const ccv_nnc_hint_t hint);
int ccv_nnc_is_cmd_auto(const ccv_nnc_cmd_param_t params);
int ccv_nnc_is_tensor_auto(const ccv_nnc_tensor_param_t params);

/**
 * @addtogroup convenience_api
 * @{
 */
/**
 * Offsets all zero.
 */
extern const int ccv_nnc_no_ofs[CCV_NNC_MAX_DIM_ALLOC];
/**
 * No hint available.
 */
extern const ccv_nnc_hint_t ccv_nnc_no_hint;
/**
 * The default symbolic graph compile parameters.
 */
extern const ccv_nnc_symbolic_graph_compile_param_t ccv_nnc_default_compile_params;
/**
 * Derive the command parameters automatically if possible.
 */
extern const ccv_nnc_cmd_param_t ccv_nnc_cmd_auto;
/**
 * Derive the tensor parameters automatically if possible.
 */
extern const ccv_nnc_tensor_param_t ccv_nnc_tensor_auto;
/** @} */

// Generated command flags for easy creation of ccv_nnc_cmd_t objects.
#include "cmd/ccv_nnc_cmd_easy.h"

#endif
