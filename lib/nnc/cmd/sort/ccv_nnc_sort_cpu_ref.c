#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_DISPATCH
#include <dispatch/dispatch.h>
#endif

#define less_than(a, b, aux) ((a) < (b))
#define greater_than(a, b, aux) ((a) > (b))
#define swap_func(a, b, array, aux, t) do { \
	(t) = (a); \
	(a) = (b); \
	(b) = (t); \
	int _t = aux[&(a) - array]; \
	aux[&(a) - array] = aux[&(b) - array]; \
	aux[&(b) - array] = _t; \
} while (0)
static CCV_IMPLEMENT_QSORT_EX(_ccv_nnc_sort_less_than_f32, float, less_than, swap_func, int*)
static CCV_IMPLEMENT_QSORT_EX(_ccv_nnc_sort_less_than_i32, int, less_than, swap_func, int*)
static CCV_IMPLEMENT_QSORT_EX(_ccv_nnc_sort_greater_than_f32, float, greater_than, swap_func, int*)
static CCV_IMPLEMENT_QSORT_EX(_ccv_nnc_sort_greater_than_i32, int, greater_than, swap_func, int*)
#undef less_than
#undef greater_than
#undef swap_func
typedef struct {
	float* array;
	int stride;
	int* idx;
} ccv_nnc_sort_aux_f32_t;
typedef struct {
	int* array;
	int stride;
	int* idx;
} ccv_nnc_sort_aux_i32_t;
#define less_than(a, b, aux) (aux.array[(&(a) - aux.array) * aux.stride] < aux.array[(&(b) - aux.array) * aux.stride])
#define greater_than(a, b, aux) (aux.array[(&(a) - aux.array) * aux.stride] > aux.array[(&(b) - aux.array) * aux.stride])
#define swap_func(a, b, array, aux, t) do { \
	(t) = aux.array[(&(a) - array) * aux.stride]; \
	aux.array[(&(a) - array) * aux.stride] = aux.array[(&(b) - array) * aux.stride]; \
	aux.array[(&(b) - array) * aux.stride] = (t); \
	int _t = aux.idx[(&(a) - array) * aux.stride]; \
	aux.idx[(&(a) - array) * aux.stride] = aux.idx[(&(b) - array) * aux.stride]; \
	aux.idx[(&(b) - array) * aux.stride] = _t; \
} while (0)
static CCV_IMPLEMENT_QSORT_EX(_ccv_nnc_sort_with_stride_less_than_f32, float, less_than, swap_func, ccv_nnc_sort_aux_f32_t)
static CCV_IMPLEMENT_QSORT_EX(_ccv_nnc_sort_with_stride_less_than_i32, int, less_than, swap_func, ccv_nnc_sort_aux_i32_t)
static CCV_IMPLEMENT_QSORT_EX(_ccv_nnc_sort_with_stride_greater_than_f32, float, greater_than, swap_func, ccv_nnc_sort_aux_f32_t)
static CCV_IMPLEMENT_QSORT_EX(_ccv_nnc_sort_with_stride_greater_than_i32, int, greater_than, swap_func, ccv_nnc_sort_aux_i32_t)
#undef less_than
#undef swap_func

static int _ccv_nnc_sort_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 2);
	const ccv_nnc_tensor_view_t* const a = (ccv_nnc_tensor_view_t*)inputs[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	const ccv_nnc_tensor_view_t* const b = (ccv_nnc_tensor_view_t*)outputs[0];
	assert(ccv_nnc_tensor_nd(b->info.dim) == a_nd);
	const ccv_nnc_tensor_view_t* const indices = (ccv_nnc_tensor_view_t*)outputs[1];
	assert(ccv_nnc_tensor_nd(indices->info.dim) == a_nd);
	assert(indices->info.datatype == CCV_32S);
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(CCV_IS_TENSOR_CONTIGUOUS(indices));
	const int count = ccv_nnc_tensor_count(a->info);
	int i;
	if (a_nd == 1)
	{
		// This is the fast path, we just do a regular sort and extract the index.
		assert(ccv_nnc_tensor_count(b->info) == count);
		assert(ccv_nnc_tensor_count(indices->info) == count);
		for (i = 0; i < count; i++)
			indices->data.i32[i] = i;
		if (a->info.datatype == CCV_32F)
		{
			memcpy(b->data.f32, a->data.f32, sizeof(float) * count);
			if (cmd.info.sort.descending)
				_ccv_nnc_sort_greater_than_f32(b->data.f32, count, indices->data.i32);
			else
				_ccv_nnc_sort_less_than_f32(b->data.f32, count, indices->data.i32);
		} else {
			assert(a->info.datatype == CCV_32S);
			memcpy(b->data.i32, a->data.i32, sizeof(int) * count);
			if (cmd.info.sort.descending)
				_ccv_nnc_sort_greater_than_i32(b->data.i32, count, indices->data.i32);
			else
				_ccv_nnc_sort_less_than_i32(b->data.i32, count, indices->data.i32);
		}
	} else {
		const int count = ccv_nnc_tensor_count(a->info);
		assert(ccv_nnc_tensor_count(b->info) == count);
		assert(ccv_nnc_tensor_count(indices->info) == count);
		if (a->info.datatype == CCV_32F)
			memcpy(b->data.f32, a->data.f32, sizeof(float) * count);
		else
			memcpy(b->data.i32, a->data.i32, sizeof(float) * count);
		int sort_runs = 1;
		int sort_stride = 1;
		for (i = 0; i < a_nd; i++)
		{
			if (i < cmd.info.sort.along_axis) // Skip this.
				sort_runs *= a->info.dim[i];
			else if (i > cmd.info.sort.along_axis)
				sort_stride *= a->info.dim[i];
		}
		const int skip_stride = sort_stride * a->info.dim[cmd.info.sort.along_axis];
		int j, k;
		const int dim = a->info.dim[cmd.info.sort.along_axis];
		if (a->info.datatype == CCV_32F)
		{
			ccv_nnc_sort_aux_f32_t aux = {
				.array = b->data.f32,
				.stride = sort_stride,
				.idx = indices->data.i32,
			};
			if (cmd.info.sort.descending)
				for (i = 0; i < sort_runs; i++)
					for (j = 0; j < sort_stride; j++)
					{
						aux.array = b->data.f32 + skip_stride * i + j;
						aux.idx = indices->data.i32 + skip_stride * i + j;
						for (k = 0; k < dim; k++)
							indices->data.i32[i * skip_stride + k * sort_stride + j] = k;
						_ccv_nnc_sort_with_stride_greater_than_f32(aux.array, dim, aux);
					}
			else
				for (i = 0; i < sort_runs; i++)
					for (j = 0; j < sort_stride; j++)
					{
						aux.array = b->data.f32 + skip_stride * i + j;
						aux.idx = indices->data.i32 + skip_stride * i + j;
						for (k = 0; k < dim; k++)
							indices->data.i32[i * skip_stride + k * sort_stride + j] = k;
						_ccv_nnc_sort_with_stride_less_than_f32(aux.array, dim, aux);
					}
		} else {
			assert(a->info.datatype == CCV_32S);
			ccv_nnc_sort_aux_i32_t aux = {
				.array = b->data.i32,
				.stride = sort_stride,
				.idx = indices->data.i32,
			};
			if (cmd.info.sort.descending)
				for (i = 0; i < sort_runs; i++)
					for (j = 0; j < sort_stride; j++)
					{
						aux.array = b->data.i32 + skip_stride * i + j;
						aux.idx = indices->data.i32 + skip_stride * i + j;
						for (k = 0; k < dim; k++)
							indices->data.i32[i * skip_stride + k * sort_stride + j] = k;
						_ccv_nnc_sort_with_stride_greater_than_i32(aux.array, dim, aux);
					}
			else
				for (i = 0; i < sort_runs; i++)
					for (j = 0; j < sort_stride; j++)
					{
						aux.array = b->data.i32 + skip_stride * i + j;
						aux.idx = indices->data.i32 + skip_stride * i + j;
						for (k = 0; k < dim; k++)
							indices->data.i32[i * skip_stride + k * sort_stride + j] = k;
						_ccv_nnc_sort_with_stride_less_than_i32(aux.array, dim, aux);
					}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_sort_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SORT_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sort_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_SORT_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_sort_back;
}
