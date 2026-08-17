#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"
#include "3rdparty/dsfmt/dSFMT.h"

static int _ccv_nnc_gumbel_argmax_forw(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == 1);
	assert(output_size == 1);
	const ccv_nnc_tensor_t* const a = inputs[0];
	ccv_nnc_tensor_t* const b = outputs[0];
	assert(CCV_IS_TENSOR_CONTIGUOUS(a));
	assert(CCV_IS_TENSOR_CONTIGUOUS(b));
	assert(a->info.datatype == CCV_32F);
	assert(b->info.datatype == CCV_32F || b->info.datatype == CCV_32S);
	assert(cmd.info.reduce.count == 1);
	const int axis = cmd.info.reduce.axis[0];
	const int a_nd = ccv_nnc_tensor_nd(a->info.dim);
	assert(axis >= 0 && axis < a_nd);
	const int tensor_count = ccv_nnc_tensor_count(a->info);
	const int axis_dim = a->info.dim[axis];
	assert(axis_dim > 0);
	int dim_after_axis = 1;
	int i;
	for (i = axis + 1; i < a_nd; i++)
		dim_after_axis *= a->info.dim[i];
	const int dim_before_axis = tensor_count / axis_dim / dim_after_axis;
	assert(ccv_nnc_tensor_count(b->info) == tensor_count / axis_dim);
	const float scale = cmd.info.reduce.scale;

	dsfmt_t dsfmt;
	dsfmt_init_gen_rand(&dsfmt, ccv_nnc_stream_context_genrand_uint32(stream_context));
	const float* const ap = a->data.f32;
	int j, k;
	for (i = 0; i < dim_before_axis; i++)
	{
		const float* const ap0 = ap + i * dim_after_axis * axis_dim;
		for (j = 0; j < dim_after_axis; j++)
		{
			const double uniform = dsfmt_genrand_open_open(&dsfmt);
			float max_value = ap0[j] - scale * (float)log(-log(uniform));
			int max_index = 0;
			for (k = 1; k < axis_dim; k++)
			{
				const double next_uniform = dsfmt_genrand_open_open(&dsfmt);
				const float value = ap0[j + k * dim_after_axis] - scale * (float)log(-log(next_uniform));
				if (value > max_value)
					max_value = value, max_index = k;
			}
			const int output_index = i * dim_after_axis + j;
			if (b->info.datatype == CCV_32S)
				b->data.i32[output_index] = max_index;
			else
				b->data.f32[output_index] = max_index;
		}
	}
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_gumbel_argmax_back(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return CCV_NNC_EXEC_INVALID;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GUMBEL_ARGMAX_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gumbel_argmax_forw;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_GUMBEL_ARGMAX_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_32F | CCV_32S;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_gumbel_argmax_back;
}
