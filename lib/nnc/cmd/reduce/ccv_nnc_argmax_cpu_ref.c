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

// Shared methods.
#include "../_ccv_nnc_cpu_ref.h"

static int _ccv_nnc_argmax_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	ccv_nnc_tensor_t* const a = inputs[0];
	assert(!CCV_IS_TENSOR_VIEW(a));
	assert(output_size == 1);
	ccv_nnc_tensor_t* const b = outputs[0];
	assert(!CCV_IS_TENSOR_VIEW(b));
	const int axis = cmd.info.reduce.axis[0];
	assert(cmd.info.reduce.count == 1);
	assert(axis >= 0);
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(axis < a_nd);
	const int b_nd = ccv_nnc_tensor_nd(b->info.dim);
	int i, j, k;
	for (i = axis; i < a_nd; i++)
		{ assert(a->info.dim[i] == (b_nd - (a_nd - i) >= 0) ? b->info.dim[b_nd - (a_nd - i)] : 1); }
	assert(1 == (b_nd - (a_nd - axis) >= 0) ? b->info.dim[b_nd - (a_nd - axis)] : 1);
	for (i = 0; i < axis; i++)
		{ assert(a->info.dim[i] == (b_nd - (a_nd - i) >= 0) ? b->info.dim[b_nd - (a_nd - i)] : 1); }
	const int tensor_count = ccv_nnc_tensor_count(a->info);
	const int axis_dim = a->info.dim[axis];
	int dim_after_axis = 1;
	for (i = axis + 1; i < a_nd; i++)
		dim_after_axis *= a->info.dim[i];
	const int dim_before_axis = tensor_count / axis_dim / dim_after_axis;
	assert(ccv_nnc_tensor_count(b->info) == tensor_count / axis_dim);
	const float* const ap = a->data.f32;
	assert(b->info.datatype == CCV_32F || b->info.datatype == CCV_32S);
	if (b->info.datatype == CCV_32S)
	{
		int* const bp = b->data.i32;
		for (i = 0; i < dim_before_axis; i++)
		{
			const float* const ap0 = ap + i * dim_after_axis * axis_dim;
			int* const bp0 = bp + i * dim_after_axis;
			for (j = 0; j < dim_after_axis; j++)
			{
				float max = ap0[j];
				int idx = 0;
				for (k = 1; k < axis_dim; k++)
					if (ap0[j + k * dim_after_axis] > max)
						max = ap0[j + k * dim_after_axis], idx = k;
				bp0[j] = idx;
			}
		}
	} else {
		float* const bp = b->data.f32;
		for (i = 0; i < dim_before_axis; i++)
		{
			const float* const ap0 = ap + i * dim_after_axis * axis_dim;
			float* const bp0 = bp + i * dim_after_axis;
			for (j = 0; j < dim_after_axis; j++)
			{
				float max = ap0[j];
				int idx = 0;
				for (k = 1; k < axis_dim; k++)
					if (ap0[j + k * dim_after_axis] > max)
						max = ap0[j + k * dim_after_axis], idx = k;
				bp0[j] = idx;
			}
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_argmax_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ARGMAX_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_argmax_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_ARGMAX_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_argmax_back;
}
