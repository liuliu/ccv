#include "ccv.h"
#include "ccv_internal.h"
#include "nnc/ccv_nnc.h"
#include "nnc/ccv_nnc_easy.h"
#include "nnc/ccv_nnc_internal.h"

static int _ccv_nnc_all_to_all_forw_cpu_ref(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	assert(input_size == output_size);
	assert(input_size > 0);
	const int rank_count = input_size;
	assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[0]));
	const int tensor_nd = ccv_nnc_tensor_nd(inputs[0]->info.dim);
	const int axis = cmd.info.all_to_all.axis;
	assert(axis >= 0 && axis < tensor_nd);
	assert(inputs[0]->info.dim[axis] % rank_count == 0);
	const size_t datatype_size = CCV_GET_DATA_TYPE_SIZE(inputs[0]->info.datatype);
	int i, j;
	size_t k;
	size_t inner_count = 1;
	for (i = axis + 1; i < tensor_nd; i++)
		inner_count *= inputs[0]->info.dim[i];
	size_t outer_count = 1;
	for (i = 0; i < axis; i++)
		outer_count *= inputs[0]->info.dim[i];
	const size_t axis_dim_count = inputs[0]->info.dim[axis] * inner_count;
	const size_t chunk_count = inputs[0]->info.dim[axis] / rank_count * inner_count;
	const size_t chunk_size = chunk_count * datatype_size;
	for (i = 0; i < rank_count; i++)
	{
		assert(CCV_IS_TENSOR_CONTIGUOUS(inputs[i]));
		assert(CCV_IS_TENSOR_CONTIGUOUS(outputs[i]));
		assert(inputs[i]->info.format == inputs[0]->info.format);
		assert(outputs[i]->info.format == inputs[0]->info.format);
		assert(inputs[i]->info.datatype == inputs[0]->info.datatype);
		assert(outputs[i]->info.datatype == inputs[0]->info.datatype);
		assert(memcmp(inputs[i]->info.dim, inputs[0]->info.dim, sizeof(inputs[0]->info.dim)) == 0);
		assert(memcmp(outputs[i]->info.dim, inputs[0]->info.dim, sizeof(inputs[0]->info.dim)) == 0);
	}
	for (i = 0; i < rank_count; i++)
		for (j = 0; j < rank_count; j++)
			assert(inputs[i] != outputs[j]);
	for (i = 0; i < rank_count; i++)
		for (j = 0; j < rank_count; j++)
			for (k = 0; k < outer_count; k++)
				memcpy(outputs[j]->data.u8 + (k * axis_dim_count + i * chunk_count) * datatype_size, inputs[i]->data.u8 + (k * axis_dim_count + j * chunk_count) * datatype_size, chunk_size);
	return CCV_NNC_EXEC_SUCCESS;
}

static int _ccv_nnc_all_to_all_back_cpu_ref(const ccv_nnc_cmd_t cmd, const ccv_nnc_hint_t hint, const int flags, ccv_nnc_tensor_t* const* const inputs, const int input_size, ccv_nnc_tensor_t* const* const outputs, const int output_size, ccv_nnc_stream_context_t* const stream_context)
{
	return _ccv_nnc_all_to_all_forw_cpu_ref(cmd, hint, flags, inputs, input_size, outputs, output_size, stream_context);
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_ALL_TO_ALL_FORWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F | CCV_32S | CCV_8U;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_all_to_all_forw_cpu_ref;
}

REGISTER_COMMAND_BACKEND(CCV_NNC_COMM_ALL_TO_ALL_BACKWARD, CCV_NNC_BACKEND_CPU_REF)(ccv_nnc_cmd_backend_registry_t* const registry)
{
	registry->tensor_formats = CCV_TENSOR_FORMAT_NHWC | CCV_TENSOR_FORMAT_NCHW | CCV_TENSOR_FORMAT_CHWN;
	registry->tensor_datatypes = CCV_64F | CCV_32F | CCV_16F | CCV_32S | CCV_8U;
	registry->tensor_memory = CCV_TENSOR_CPU_MEMORY;
	registry->algorithms = 1;
	registry->exec = _ccv_nnc_all_to_all_back_cpu_ref;
}
